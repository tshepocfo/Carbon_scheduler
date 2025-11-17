#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import uuid
import shutil
import logging
import textwrap
import time
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Dict, Optional

import boto3
import requests
import matplotlib.pyplot as plt
import numpy as np
from botocore.config import Config
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from math import radians, sin, cos, sqrt, atan2

# --------------------------------------------------------------------------- #
# Load env + basic setup
# --------------------------------------------------------------------------- #
load_dotenv()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("CarbonSight")

app = Flask(__name__)

# CORS – adjust if you deploy elsewhere
CORS(app, resources={r"/*": {"origins": "*"}})

# --------------------------------------------------------------------------- #
# Rate limiter & API key (fixed – was missing in your last version)
# --------------------------------------------------------------------------- #
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))
_requests: Dict[str, int] = {}

def rate_limiter(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        ip = request.remote_addr or "unknown"
        now = int(time.time())
        window = now // 60
        key = f"{ip}:{window}"
        _requests[key] = _requests.get(key, 0) + 1
        _requests.pop(f"{ip}:{window-1}", None)
        if _requests[key] > RATE_LIMIT:
            return jsonify({"ok": False, "error": "rate_limited"}), 429
        return func(*args, **kwargs)
    return wrapper

def require_api_key(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        required = os.getenv("API_KEY")
        if not required:
            return jsonify({"ok": False, "error": "server_misconfig"}), 500
        provided = request.headers.get("X-API-KEY")
        if provided != required:
            return jsonify({"ok": False, "error": "unauthorized"}), 401
        return func(*args, **kwargs)
    return wrapper

# --------------------------------------------------------------------------- #
# AWS & constants
# --------------------------------------------------------------------------- #
boto_config = Config(retries={"max_attempts": 3, "mode": "standard"}, read_timeout=10)
pricing_client = boto3.client("pricing", region_name="us-east-1", config=boto_config)

regions = {  # only the ones that work reliably
    "us-east-1": {"lat": 38.9940541, "lon": -77.4524237, "location": "US East (N. Virginia)"},
    "us-west-2": {"lat": 45.9174667, "lon": -119.2684488, "location": "US West (Oregon)"},
    "eu-west-1": {"lat": 53.4056545, "lon": -6.224503, "location": "EU (Ireland)"},
    "eu-north-1": {"lat": 59.326242, "lon": 17.8419717, "location": "EU (Stockholm)"},
    "ap-southeast-2": {"lat": -33.9117717, "lon": 151.1907535, "location": "Asia Pacific (Sydney)"},
}

workload_to_instance = {"inference": "g4dn.xlarge", "training": "p3.2xlarge"}
instance_powers = {"g4dn.xlarge": 0.2, "p3.2xlarge": 0.4}
baseline_intensity = 500.0
user_lat, user_lon = 40.7128, -74.0060

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    a = sin(radians(lat2 - lat1)/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(radians(lon2 - lon1)/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))

def estimate_latency(dist_km): return (2 * dist_km / 200) + 15

# --------------------------------------------------------------------------- #
# Core calculation
# --------------------------------------------------------------------------- #
def compute_metrics(company, workload, priorities, gpu_hours, region):
    r = regions[region]
    lat, lon = r["lat"], r["lon"]

    # Carbon intensity
    token = os.getenv("ELECTRICITYMAPS_TOKEN")
    resp = requests.get(
        "https://api-access.electricitymaps.com/free-tier/carbon-intensity/latest",
        headers={"auth-token": token},
        params={"lat": lat, "lon": lon},
        timeout=10,
    )
    if not resp.ok:
        resp = requests.get("https://api-access.electricitymaps.com/free-tier/carbon-intensity/latest?zone=SE",
                            headers={"auth-token": token}, timeout=10)
    carbon_intensity = float(resp.json()["carbonIntensity"])

    # Pricing
    instance = workload_to_instance.get(workload, "g4dn.xlarge")
    price_resp = pricing_client.get_products(
        ServiceCode="AmazonEC2",
        Filters=[
            {"Type": "TERM_MATCH", "Field": "instanceType", "Value": instance},
            {"Type": "TERM_MATCH", "Field": "location", "Value": r["location"]},
            {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "shared"},
            {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
            {"Type": "TERM_MATCH", "Field": "capacitystatus", "Value": "Used"},
        ],
        MaxResults=1,
    )
    prod = json.loads(price_resp["PriceList"][0])
    on_demand = float(prod["terms"]["OnDemand"].popitem()[1]["priceDimensions"].popitem()[1]["pricePerUnit"]["USD"])

    # Spot (fallback 30% of on-demand if no history)
    ec2 = boto3.client("ec2", region_name=region, config=boto_config)
    hist = ec2.describe_spot_price_history(
        InstanceTypes=[instance],
        ProductDescriptions=["Linux/UNIX"],
        StartTime=datetime.utcnow() - timedelta(hours=1),
        EndTime=datetime.utcnow(),
        MaxResults=10,
    )
    spot = on_demand * 0.3 if not hist["SpotPriceHistory"] else float(hist["SpotPriceHistory"][0]["SpotPrice"])

    # Final numbers
    distance = haversine(user_lat, user_lon, lat, lon)
    latency = estimate_latency(distance)
    saved = (on_demand - spot) * gpu_hours
    reduced_co2 = (baseline_intensity - carbon_intensity) * instance_powers[instance] * gpu_hours / 1000

    norm_cost = (on_demand - spot) / 1.0
    norm_carbon = (baseline_intensity - carbon_intensity) / 300.0
    norm_speed = max(0, 1 - latency/200)
    total_w = sum(priorities.values()) or 1
    score = (
        priorities.get("cost", 0)/total_w * norm_cost +
        priorities.get("carbon", 0)/total_w * norm_carbon +
        priorities.get("speed", 0)/total_w * norm_speed
    )

    return {
        "company_name": company,
        "workload_type": workload,
        "gpu_hours": gpu_hours,
        "cloud_region": region,
        "region_location": r["location"],
        "on_demand_price_per_hour": on_demand,
        "spot_price_per_hour": spot,
        "saved_money": round(saved, 2),
        "carbon_intensity_gco2_kwh": carbon_intensity,
        "reduced_emissions_kg_co2": round(reduced_co2, 1),
        "latency_ms": round(latency),
        "score": round(max(0, min(1, score)), 3),
    }

# --------------------------------------------------------------------------- #
# Charts (fixed function name)
# --------------------------------------------------------------------------- #
def create_chart(metrics: Dict, base_path: str):
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    paths = []

    # 1 – Cost
    p = f"{base_path}_cost.png"
    fig, ax = plt.subplots(figsize=(8.5, 3.8), facecolor="#0b0b0b")
    ax.set_facecolor("#0b0b0b")
    ax.bar(["On-demand", "Spot"], [metrics["on_demand_price_per_hour"]*metrics["gpu_hours"], metrics["spot_price_per_hour"]*metrics["gpu_hours"]],
           color=["#ff6b6b", "#7BE200"])
    ax.set_title("Total Cost", color="white", fontsize=14, pad=20)
    ax.set_ylabel("USD", color="white")
    ax.tick_params(colors="white")
    for i, v in enumerate([metrics["on_demand_price_per_hour"]*metrics["gpu_hours"], metrics["spot_price_per_hour"]*metrics["gpu_hours"]]):
        ax.text(i, v*1.02, f"${v:,.0f}", ha="center", color="white")
    plt.tight_layout()
    plt.savefig(p, dpi=200, bbox_inches="tight", facecolor="#0b0b0b")
    plt.close()
    paths.append(p)

    # 2 – Emissions
    p = f"{base_path}_emissions.png"
    fig, ax = plt.subplots(figsize=(8.5, 3.8), facecolor="#0b0b0b")
    ax.set_facecolor("#0b0b0b")
    baseline = metrics["reduced_emissions_kg_co2"] + (metrics["carbon_intensity_gco2_kwh"] * instance_powers[workload_to_instance[metrics["workload_type"]]] * metrics["gpu_hours"] / 1000)
    ax.bar(["Baseline", "Optimised"], [baseline/1000, (baseline - metrics["reduced_emissions_kg_co2"])/1000],
           color=["#444", "#00c2ff"])
    ax.set_title("CO₂ Emissions", color="white", fontsize=14, pad=20)
    ax.set_ylabel("tonnes", color="white")
    ax.tick_params(colors="white")
    plt.tight_layout()
    plt.savefig(p, dpi=200, bbox_inches="tight", facecolor="#0b0b0b")
    plt.close()
    paths.append(p)

    return paths[:2]  # we only need 2 for the PDF

# --------------------------------------------------------------------------- #
# Minimal PDF (Playwright fallback to WeasyPrint)
# --------------------------------------------------------------------------- #
def create_pdf(metrics, chart_files, pdf_path, summary=""):
    workdir = Path("artifacts") / uuid.uuid4().hex[:10]
    workdir.mkdir(parents=True, exist_ok=True)
    assets = workdir / "assets"
    assets.mkdir(exist_ok=True)

    for i, src in enumerate(chart_files, 1):
        shutil.copy(src, assets / f"chart{i}.png")

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>CarbonSight Report</title>
        <style>
            body {{ background:#000; color:white; font-family:system-ui; padding:40px; }}
            .page {{ width:297mm; height:210mm; background:#0b0b0b; margin:0 auto 40px; padding:20mm; box-shadow:0 0 30px rgba(0,0,0,0.8); }}
            img {{ width:100%; border-radius:12px; margin:20px 0; }}
            .big {{ font-size:48px; color:#7BE200; }}
        </style>
    </head>
    <body>
    <div class="page">
        <h1>CarbonSight Scheduler</h1>
        <p class="big">Saved ${metrics['saved_money']:,}</p>
        <p class="big">Avoided {metrics['reduced_emissions_kg_co2']:,} kg CO₂</p>
        <p>Region: {metrics['region_location']} • Latency: {metrics['latency_ms']} ms</p>
        <img src="assets/chart1.png">
        <img src="assets/chart2.png">
        <p><strong>Summary:</strong><br>{summary.replace('\n', '<br>')}</p>
    </div>
    </body></html>
    """
    (workdir / "index.html").write_text(html)

    # Try Playwright → fallback WeasyPrint
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(f"file://{workdir / 'index.html'}", wait_until="networkidle")
            page.pdf(path=pdf_path, format="A4", landscape=True, print_background=True)
            browser.close()
    except Exception as e:
        log.warning(f"Playwright failed ({e}), using WeasyPrint")
        from weasyprint import HTML
        HTML(filename=str(workdir / "index.html")).write_pdf(pdf_path)

# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/")
def health(): return "CarbonSight live"

@app.post("/calculate")
@rate_limiter
@require_api_key
def calculate():
    data = request.get_json() or {}
    required = ["company_name", "workload_type", "priorities", "gpu_hours", "cloud_region"]
    if not all(k in data for k in required):
        return jsonify({"ok": False, "error": "missing fields"}), 400

    try:
        gpu_hours = float(data["gpu_hours"])
        priorities = {k: float(v) for k, v in data["priorities"].items()}
    except:
        return jsonify({"ok": False, "error": "invalid numbers"}), 400

    metrics = compute_metrics(
        data["company_name"],
        data["workload_type"],
        priorities,
        gpu_hours,
        data["cloud_region"],
    )

    # AI summary (optional)
    summary = "Carbon-aware scheduling delivered significant cost and emissions savings."
    if os.getenv("GEMINI_API_KEY"):
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model = genai.GenerativeModel("gemini-1.5-flash")
            resp = model.generate_content(
                f"Write a 120-word positive sustainability summary for:\n"
                f"Company: {metrics['company_name']}\n"
                f"Saved ${metrics['saved_money']:,} and avoided {metrics['reduced_emissions_kg_co2']:,} kg CO₂\n"
                f"Region: {metrics['region_location']}"
            )
            summary = resp.text.strip()
        except Exception as e:
            log.warning(f"Gemini failed: {e}")

    # Charts + PDF
    run_id = uuid.uuid4().hex[:12]
    chart_base = f"artifacts/{run_id}_chart"
    chart_files = create_chart(metrics, chart_base)
    pdf_path = f"artifacts/{run_id}_report.pdf"
    create_pdf(metrics, chart_files, pdf_path, summary)

    return jsonify({
        "ok": True,
        "metrics": metrics,
        "summary": summary,
        "chart_url": f"/artifact/{os.path.basename(chart_files[0])}",
        "pdf_url": f"/artifact/{run_id}_report.pdf",
    })

@app.get("/artifact/<path:filename>")
@rate_limiter
def artifact(filename):
    return send_from_directory("artifacts", filename)

# --------------------------------------------------------------------------- #
# Render needs PORT env var
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
