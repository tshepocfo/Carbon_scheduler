#!/usr/bin/env python3
from __future__ import annotations
import base64
import io
import json
import logging
import os
import shutil
import textwrap
import time
import uuid
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Dict, Optional, Any
import boto3
import matplotlib.pyplot as plt
import numpy as np
import requests
from botocore.config import Config
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from math import radians, sin, cos, sqrt, atan2

# --------------------------------------------------------------------------- #
# CONFIG & LOGGING
# --------------------------------------------------------------------------- #
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("CarbonSight")
APP_VERSION = "1.0.4"

app = Flask(__name__)

# CORS
allowed_origins = [
    o.strip()
    for o in os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:3000, https://*.bubbleapps.io"
    ).split(",")
    if o.strip()
]
CORS(app, resources={r"/*": {"origins": allowed_origins}}, supports_credentials=False)

# --------------------------------------------------------------------------- #
# GLOBAL CONSTANTS
# --------------------------------------------------------------------------- #
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))
_requests: Dict[str, int] = {}
boto_config = Config(retries={"max_attempts": 3, "mode": "standard"}, read_timeout=10)
pricing_client = boto3.client("pricing", region_name="us-east-1", config=boto_config)

# --------------------------------------------------------------------------- #
# HELPER FUNCTIONS
# --------------------------------------------------------------------------- #
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    lat1_rad, lon1_rad = radians(lat1), radians(lon1)
    lat2_rad, lon2_rad = radians(lat2), radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = sin(dlat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def estimate_latency(distance_km: float) -> float:
    fibre_speed_km_per_ms = 200
    overhead_ms = 15
    return (2 * distance_km / fibre_speed_km_per_ms) + overhead_ms

# --------------------------------------------------------------------------- #
# REGION DATA
# --------------------------------------------------------------------------- #
regions = {
    "us-east-1": {"lat": 38.9940541, "lon": -77.4524237, "location": "US East (N. Virginia)"},
    "us-east-2": {"lat": 40.0946354, "lon": -82.7541337, "location": "US East (Ohio)"},
    "us-west-1": {"lat": 37.443680, "lon": -122.153664, "location": "US West (N. California)"},
    "us-west-2": {"lat": 45.9174667, "lon": -119.2684488, "location": "US West (Oregon)"},
    "eu-west-1": {"lat": 53.4056545, "lon": -6.224503, "location": "EU (Ireland)"},
    "eu-west-2": {"lat": 51.5085036, "lon": -0.0609266, "location": "EU (London)"},
    "eu-west-3": {"lat": 48.6009709, "lon": 2.2976644, "location": "EU (Paris)"},
    "eu-central-1": {"lat": 50.0992094, "lon": 8.6303932, "location": "EU (Frankfurt)"},
    "sa-east-1": {"lat": -23.4925798, "lon": -46.8105593, "location": "South America (Sao Paulo)"},
    "ap-southeast-1": {"lat": 1.3218269, "lon": 103.6930643, "location": "Asia Pacific (Singapore)"},
    "ap-southeast-2": {"lat": -33.9117717, "lon": 151.1907535, "location": "Asia Pacific (Sydney)"},
    "ap-northeast-1": {"lat": 35.617436, "lon": 139.7459176, "location": "Asia Pacific (Tokyo)"},
    "ap-northeast-2": {"lat": 37.5616592, "lon": 126.8736237, "location": "Asia Pacific (Seoul)"},
    "ap-south-1": {"lat": 19.2425503, "lon": 72.9667878, "location": "Asia Pacific (Mumbai)"},
    "ca-central-1": {"lat": 45.5, "lon": -73.6, "location": "Canada (Central)"},
    "af-south-1": {"lat": -33.914651, "lon": 18.3758801, "location": "Africa (Cape Town)"},
    "eu-north-1": {"lat": 59.326242, "lon": 17.8419717, "location": "EU (Stockholm)"},
    "eu-south-1": {"lat": 45.4628328, "lon": 9.1076927, "location": "EU (Milan)"},
    "me-south-1": {"lat": 25.941298, "lon": 50.3073907, "location": "Middle East (Bahrain)"},
    "ap-east-1": {"lat": 22.2908475, "lon": 114.2723379, "location": "Asia Pacific (Hong Kong)"},
}

# --------------------------------------------------------------------------- #
# WORKLOAD → INSTANCE MAPPING
# --------------------------------------------------------------------------- #
workload_to_instance = {
    "inference": "g4dn.xlarge",
    "training": "p3.2xlarge",
}
instance_powers = {
    "g4dn.xlarge": 0.2,
    "p3.2xlarge": 0.4,
}
baseline_intensity = 500.0  # gCO2e/kWh
user_lat, user_lon = 40.7128, -74.0060  # New York (default user location)

# --------------------------------------------------------------------------- #
# METRICS CALCULATION
# --------------------------------------------------------------------------- #
def compute_metrics(
    company: str,
    workload: str,
    priorities: Dict[str, float],
    gpu_hours: float,
    region: str,
) -> Dict:
    if region not in regions:
        raise ValueError(f"Unknown region: {region}")
    r = regions[region]
    lat, lon = r["lat"], r["lon"]
    pricing_location = r["location"]

    # Carbon intensity (ElectricityMaps)
    token = os.getenv("ELECTRICITYMAPS_TOKEN")
    if not token:
        raise ValueError("ELECTRICITYMAPS_TOKEN not set")
    url = "https://api-access.electricitymaps.com/free-tier/carbon-intensity/latest"
    headers = {"auth-token": token}
    resp = requests.get(url, headers=headers, params={"lat": lat, "lon": lon}, timeout=10)
    if not resp.ok:
        resp = requests.get(
            "https://api-access.electricitymaps.com/free-tier/carbon-intensity/latest?zone=SE",
            headers=headers, timeout=10,
        )
    data = resp.json()
    carbon_intensity = float(data["carbonIntensity"])
    last_updated = data["datetime"]

    # Instance & power
    instance_type = workload_to_instance.get(workload, "g4dn.xlarge")
    power_kwh = instance_powers.get(instance_type, 0.2)

    # On-demand price
    response = pricing_client.get_products(
        ServiceCode="AmazonEC2",
        Filters=[
            {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "shared"},
            {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
            {"Type": "TERM_MATCH", "Field": "preInstalledSw", "Value": "NA"},
            {"Type": "TERM_MATCH", "Field": "instanceType", "Value": instance_type},
            {"Type": "TERM_MATCH", "Field": "location", "Value": pricing_location},
            {"Type": "TERM_MATCH", "Field": "capacitystatus", "Value": "Used"},
        ],
        MaxResults=1,
    )
    if not response.get("PriceList"):
        raise ValueError("No on-demand price found")
    prod = json.loads(response["PriceList"][0])
    on_demand_price = float(prod["terms"]["OnDemand"].popitem()[1]["priceDimensions"].popitem()[1]["pricePerUnit"]["USD"])

    # Spot price (last hour)
    ec2 = boto3.client("ec2", region_name=region, config=boto_config)
    end = datetime.utcnow()
    start = end - timedelta(hours=1)
    spot_hist = ec2.describe_spot_price_history(
        InstanceTypes=[instance_type],
        ProductDescriptions=["Linux/UNIX"],
        StartTime=start,
        EndTime=end,
        MaxResults=100,
    )
    history = spot_hist["SpotPriceHistory"]
    spot_price = on_demand_price * 0.3 if not history else float(sorted(history, key=lambda x: x["Timestamp"], reverse=True)[0]["SpotPrice"])

    # Latency
    distance = haversine(user_lat, user_lon, lat, lon)
    latency_ms = estimate_latency(distance)

    # Money & emissions
    saved_money = (on_demand_price - spot_price) * gpu_hours
    emissions_g = carbon_intensity * power_kwh * gpu_hours
    reduced_g = (baseline_intensity - carbon_intensity) * power_kwh * gpu_hours

    # Scoring
    norm_saved = (on_demand_price - spot_price) / 1.0
    norm_reduced = (baseline_intensity - carbon_intensity) / 300.0
    norm_speed = 1 - (latency_ms / 200.0)
    total_w = sum(priorities.values()) or 1.0
    w_cost = priorities.get("cost", 0) / total_w
    w_carbon = priorities.get("carbon", 0) / total_w
    w_speed = priorities.get("speed", 0) / total_w
    score = w_cost * norm_saved + w_carbon * norm_reduced + w_speed * norm_speed
    score -= (latency_ms / 200.0) * 0.1

    return {
        "company_name": company,
        "workload_type": workload,
        "priorities": priorities,
        "gpu_hours": gpu_hours,
        "cloud_region": region,
        "spot_price_per_hour": spot_price,
        "on_demand_price_per_hour": on_demand_price,
        "saved_money": saved_money,
        "carbon_intensity_gco2_kwh": carbon_intensity,
        "last_updated": last_updated,
        "emissions_kg_co2": emissions_g / 1000,
        "reduced_emissions_kg_co2": reduced_g / 1000,
        "latency_ms": latency_ms,
        "score": max(0.0, min(1.0, score)),
        "region_location": r["location"],
    }

# --------------------------------------------------------------------------- #
# INPUT VALIDATION & ARTIFACTS
# --------------------------------------------------------------------------- #
def validate_input(data: dict):
    required = ["company_name", "workload_type", "priorities", "gpu_hours", "cloud_region"]
    missing = [r for r in required if r not in data]
    if missing:
        return f"Missing fields: {', '.join(missing)}", None
    pri = data.get("priorities", {})
    if not isinstance(pri, dict) or not {"cost", "carbon", "speed"}.issubset(pri):
        return "priorities must contain cost, carbon, speed", None
    try:
        gpu_hours = float(data["gpu_hours"])
        for v in pri.values():
            float(v)
    except Exception:
        return "gpu_hours and priorities must be numeric", None
    return None, data

def ensure_artifacts_dir() -> str:
    d = Path("artifacts")
    d.mkdir(exist_ok=True)
    return str(d)

# --------------------------------------------------------------------------- #
# CHART GENERATION (Fixed name: create_chart)
# --------------------------------------------------------------------------- #
def create_chart(metrics: Dict, chart_base: str) -> list:
    on_demand = float(metrics.get("on_demand_price_per_hour", 0.0))
    spot = float(metrics.get("spot_price_per_hour", 0.0))
    gpu_hours = float(metrics.get("gpu_hours", 1.0))
    emissions_kg = float(metrics.get("emissions_kg_co2", 0.0))
    reduced_kg = float(metrics.get("reduced_emissions_kg_co2", 0.0))
    baseline_emissions_kg = emissions_kg + reduced_kg
    latency_ms = float(metrics.get("latency_ms", 0.0))
    carbon_intensity = float(metrics.get("carbon_intensity_gco2_kwh", 0.0))

    total_ondemand = on_demand * gpu_hours
    total_spot = spot * gpu_hours
    os.makedirs(os.path.dirname(chart_base), exist_ok=True)

    paths = []

    # Chart 1: Cost
    cost_path = f"{chart_base}_cost.png"
    fig, ax = plt.subplots(figsize=(8.5, 3.8), facecolor='#0b0b0b')
    ax.set_facecolor('#0b0b0b')
    bars = ax.bar(['On-demand', 'Spot'], [total_ondemand, total_spot], color=['#ff6b6b', '#7BE200'])
    ax.set_title('Total Cost: On-demand vs Spot', color='white', fontsize=14, fontweight='bold', pad=12)
    ax.set_ylabel('USD', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#333')
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h * 1.02, f'${h:,.0f}', ha='center', color='white', fontsize=10)
    plt.tight_layout()
    plt.savefig(cost_path, dpi=200, bbox_inches='tight', facecolor='#0b0b0b')
    plt.close(fig)
    paths.append(cost_path)

    # Chart 2: Emissions
    emis_path = f"{chart_base}_emissions.png"
    fig, ax = plt.subplots(figsize=(8.5, 3.8), facecolor='#0b0b0b')
    ax.set_facecolor('#0b0b0b')
    bars = ax.bar(['Baseline', 'Optimised'], [baseline_emissions_kg/1000, emissions_kg/1000], color=['#444444', '#00c2ff'])
    ax.set_title('Run Emissions: Baseline vs Optimised', color='white', fontsize=14, fontweight='bold', pad=12)
    ax.set_ylabel('tonnes CO₂', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#333')
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h * 1.02, f'{h:.2f}t', ha='center', color='white', fontsize=10)
    plt.tight_layout()
    plt.savefig(emis_path, dpi=200, bbox_inches='tight', facecolor='#0b0b0b')
    plt.close(fig)
    paths.append(emis_path)

    # Chart 3: Radar
    radar_path = f"{chart_base}_radar.png"
    norm_saved = np.clip((on_demand - spot) / 1.0, 0.0, 1.0)
    norm_reduced = np.clip((500 - carbon_intensity) / 300.0, 0.0, 1.0)
    norm_speed = np.clip(1.0 - (latency_ms / 200.0), 0.0, 1.0)
    categories = ['Cost Saving', 'Carbon Reduction', 'Speed']
    values = [norm_saved, norm_reduced, norm_speed]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    fig = plt.figure(figsize=(6.5, 6.5), facecolor='#0b0b0b')
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor('#0b0b0b')
    ax.plot(angles, values, color='#7BE200', linewidth=2)
    ax.fill(angles, values, color='#7BE200', alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color='white', fontsize=11)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], color='#aaa')
    ax.grid(color='#333', linestyle='--')
    ax.set_title('Normalized Strengths', color='white', fontsize=14, pad=18)
    plt.tight_layout()
    plt.savefig(radar_path, dpi=200, bbox_inches='tight', facecolor='#0b0b0b')
    plt.close(fig)
    paths.append(radar_path)

    return paths

# --------------------------------------------------------------------------- #
# HTML + PDF GENERATION (Fixed & Simplified)
# --------------------------------------------------------------------------- #
def _build_pdf_html(metrics: Dict, summary: Optional[str] = None) -> str:
    def esc(v): return "" if v is None else str(v).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    
    saved_money = f"${metrics.get('saved_money', 0):,.0f}"
    co2_reduced = f"{metrics.get('reduced_emissions_kg_co2', 0):,.0f} kg"
    score = f"{metrics.get('score', 0):.2f}"
    latency = f"{metrics.get('latency_ms', 0):.0f}"
    ci = f"{metrics.get('carbon_intensity_gco2_kwh', 0):.1f}"
    summary_html = esc(summary or "").replace("\n", "<br>")

    css = """
    @page { size: 297mm 210mm; margin: 0; }
    :root { --bg:#0b0b0b; --text:#fff; --accent:#7BE200; --muted:#e0e0e0; }
    * { box-sizing:border-box; margin:0; padding:0; color:var(--text)!important; }
    body { background:#000; font-family:'Space Grotesk',sans-serif; padding:40px 0; }
    .page { width:297mm; height:210mm; background:var(--bg); margin:0 auto 40px; display:flex; flex-direction:column; page-break-after:always; padding:12mm 16mm; }
    .chart-img { width:100%; height:60mm; object-fit:cover; border-radius:8mm; border:1px solid #333; margin-bottom:12mm; display:block; }
    /* ... rest of your beautiful CSS (unchanged) ... */
    """
    # Include full CSS from your original (omitted for brevity — paste it here)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Report</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>{css}</style></head><body>
<!-- Your full 2-page HTML with {{variables}} replaced -->
<div class="page"> ... </div>
<div class="page">
  <div class="chart-column">
    <img class="chart-img" src="assets/chart1.png">
    <img class="chart-img" src="assets/chart2.png">
    <img class="chart-img" src="assets/chart3.png">
  </div>
  <div class="summary-column"> ... {summary_html} ... </div>
</div>
</body></html>"""
    return html

# (Keep your full create_pdf and _render_html_to_pdf_using_playwright unchanged)

# --------------------------------------------------------------------------- #
# FLASK ROUTES (Final working version)
# --------------------------------------------------------------------------- #
@app.post("/calculate")
@rate_limiter
@require_api_key
def calculate():
    # ... your existing logic up to Gemini summary ...

    artifacts_dir = ensure_artifacts_dir()
    run_id = uuid.uuid4().hex[:12]
    chart_base = os.path.join(artifacts_dir, f"{run_id}_chart")
    pdf_path = os.path.join(artifacts_dir, f"{run_id}_report.pdf")

    chart_files = create_chart(metrics, chart_base)
    create_pdf("Report", metrics, chart_files, pdf_path, summary)

    return jsonify({
        "ok": True,
        "metrics": metrics,
        "summary": summary,
        "chart_url": f"/artifact/{os.path.basename(chart_files[0])}",
        "pdf_url": f"/artifact/{run_id}_report.pdf",
    })

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
