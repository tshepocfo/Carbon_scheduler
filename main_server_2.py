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
import requests
from botocore.config import Config
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from math import radians, sin, cos, sqrt, atan2

# --------------------------------------------------------------------------- #
#                              CONFIG & LOGGING                              #
# --------------------------------------------------------------------------- #
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("CarbonSight")

APP_VERSION = "1.0.3"
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
#                              GLOBAL CONSTANTS                               #
# --------------------------------------------------------------------------- #
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))
_requests: Dict[str, int] = {}

# AWS clients – timeout + retry
boto_config = Config(retries={"max_attempts": 3, "mode": "standard"}, read_timeout=10)
pricing_client = boto3.client("pricing", region_name="us-east-1", config=boto_config)

# --------------------------------------------------------------------------- #
#                              HELPER FUNCTIONS                               #
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
#                               REGION DATA                                 #
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
    "cn-north-1": {"lat": 39.8094478, "lon": 116.5783234, "location": "China (Beijing)"},
    "cn-northwest-1": {"lat": 37.5024418, "lon": 105.1627193, "location": "China (Ningxia)"},
}

# --------------------------------------------------------------------------- #
#                         WORKLOAD → INSTANCE MAPPING                         #
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
user_lat, user_lon = 40.7128, -74.0060


# --------------------------------------------------------------------------- #
#                         METRICS CALCULATION                                 #
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

    # ---- Carbon intensity (ElectricityMaps) ----
    token = os.getenv("ELECTRICITYMAPS_TOKEN")
    if not token:
        raise ValueError("ELECTRICITYMAPS_TOKEN not set")

    url = "https://api-access.electricitymaps.com/free-tier/carbon-intensity/latest"
    headers = {"auth-token": token}
    resp = requests.get(url, headers=headers, params={"lat": lat, "lon": lon}, timeout=10)

    if not resp.ok:
        zone = "SE"
        resp = requests.get(
            f"https://api-access.electricitymaps.com/free-tier/carbon-intensity/latest?zone={zone}",
            headers=headers,
            timeout=10,
        )
        if not resp.ok:
            raise ValueError(f"Carbon intensity fetch failed: {resp.text}")

    data = resp.json()
    carbon_intensity = float(data["carbonIntensity"])
    last_updated = data["datetime"]

    # ---- Instance & power ----
    instance_type = workload_to_instance.get(workload, "g4dn.xlarge")
    power_kwh = instance_powers.get(instance_type, 0.2)

    # ---- On-demand price (Pricing API) ----
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
    on_demand = prod["terms"]["OnDemand"]
    price_id = list(on_demand.keys())[0]
    dim_id = list(on_demand[price_id]["priceDimensions"].keys())[0]
    on_demand_price = float(
        on_demand[price_id]["priceDimensions"][dim_id]["pricePerUnit"]["USD"]
    )

    # ---- Spot price (latest hour) ----
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
    if not history:
        spot_price = on_demand_price * 0.3
    else:
        history.sort(key=lambda x: x["Timestamp"], reverse=True)
        spot_price = float(history[0]["SpotPrice"])

    # ---- Latency ----
    distance = haversine(user_lat, user_lon, lat, lon)
    latency_ms = estimate_latency(distance)

    # ---- Money & emissions ----
    saved_money = (on_demand_price - spot_price) * gpu_hours
    emissions_g = carbon_intensity * power_kwh * gpu_hours
    reduced_g = (baseline_intensity - carbon_intensity) * power_kwh * gpu_hours

    # ---- Scoring ----
    max_saved_per_hour = 1.0
    max_reduced_per_hour = 300.0
    max_latency = 200.0

    norm_saved = (on_demand_price - spot_price) / max_saved_per_hour
    norm_reduced = (baseline_intensity - carbon_intensity) / max_reduced_per_hour
    norm_speed = 1 - (latency_ms / max_latency)

    total_w = sum(priorities.values()) or 1.0
    w_cost = priorities.get("cost", 0) / total_w
    w_carbon = priorities.get("carbon", 0) / total_w
    w_speed = priorities.get("speed", 0) / total_w

    score = w_cost * norm_saved + w_carbon * norm_reduced + w_speed * norm_speed
    score -= (latency_ms / max_latency) * 0.1

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
        "score": score,
        "region_location": r["location"],
    }


# --------------------------------------------------------------------------- #
#                         INPUT VALIDATION & ARTIFACTS                         #
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
#                         CHART (Matplotlib → PNG)                           #
# --------------------------------------------------------------------------- #
def create_charts(metrics: Dict, chart_base: str) -> list:
    """
    Create three charts:
      1) Cost comparison: total on-demand vs total spot
      2) Emissions comparison: baseline vs optimized (tonnes)
      3) Radar: normalized cost/carbon/speed strengths

    chart_base is a path prefix. This function will create:
      {chart_base}_cost.png, {chart_base}_emissions.png, {chart_base}_radar.png
    and return a list of those three paths.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # metrics (safely coerce to float)
    on_demand = float(metrics.get("on_demand_price_per_hour", 0.0))
    spot = float(metrics.get("spot_price_per_hour", 0.0))
    gpu_hours = float(metrics.get("gpu_hours", 1.0))

    emissions_kg = float(metrics.get("emissions_kg_co2", 0.0))          # optimized emissions (kg)
    reduced_kg = float(metrics.get("reduced_emissions_kg_co2", 0.0))    # reduction vs baseline (kg)
    baseline_emissions_kg = emissions_kg + reduced_kg                  # baseline total (kg)

    latency_ms = float(metrics.get("latency_ms", 0.0))
    carbon_intensity = float(metrics.get("carbon_intensity_gco2_kwh", 0.0))

    # Derived totals
    total_ondemand = on_demand * gpu_hours
    total_spot = spot * gpu_hours

    os.makedirs(os.path.dirname(chart_base), exist_ok=True)

    # ---------------- Chart 1: Cost comparison (On-demand vs Spot) ----------------
    cost_path = f"{chart_base}_cost.png"
    fig, ax = plt.subplots(figsize=(8.5, 3.8), facecolor='#0b0b0b')
    ax.set_facecolor('#0b0b0b')

    labels = ['On-demand', 'Spot']
    values = [total_ondemand, total_spot]
    bars = ax.bar(labels, values, color=['#ff6b6b', '#7BE200'], edgecolor='#222', linewidth=0.8)
    ax.set_title('Total Cost: On-demand vs Spot', color='white', fontsize=14, fontweight='bold', pad=12)
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#333'); ax.spines['left'].set_color('#333')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_ylabel('GBP', color='white')

    # labels
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h * 1.02, f'£{h:,.0f}', ha='center', color='white', fontsize=10, fontweight='600')

    plt.tight_layout()
    plt.savefig(cost_path, dpi=200, bbox_inches='tight', facecolor='#0b0b0b')
    plt.close(fig)

    # ---------------- Chart 2: Emissions comparison (Baseline vs Optimised) ----------------
    emis_path = f"{chart_base}_emissions.png"
    fig, ax = plt.subplots(figsize=(8.5, 3.8), facecolor='#0b0b0b')
    ax.set_facecolor('#0b0b0b')

    labels = ['Baseline', 'Optimised']
    # convert kg -> tonnes for display
    baseline_t = baseline_emissions_kg / 1000.0
    optim_t = emissions_kg / 1000.0
    values = [baseline_t, optim_t]
    bars = ax.bar(labels, values, color=['#444444', '#00c2ff'], edgecolor='#222', linewidth=0.8)
    ax.set_title('Run Emissions: Baseline vs Optimised', color='white', fontsize=14, fontweight='bold', pad=12)
    ax.set_ylabel('tonnes CO₂', color='white')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#333'); ax.spines['left'].set_color('#333')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h * 1.02, f'{h:,.2f} t', ha='center', color='white', fontsize=10, fontweight='600')

    plt.tight_layout()
    plt.savefig(emis_path, dpi=200, bbox_inches='tight', facecolor='#0b0b0b')
    plt.close(fig)

    # ---------------- Chart 3: Radar – normalized strengths ----------------
    radar_path = f"{chart_base}_radar.png"
    # Normalization constants (same used in compute_metrics)
    max_saved_per_hour = 1.0
    max_reduced_per_hour = 300.0
    max_latency = 200.0

    # compute normalized values (clipped 0..1)
    saved_per_hour = max(0.0, on_demand - spot)
    norm_saved = np.clip(saved_per_hour / max_saved_per_hour, 0.0, 1.0)

    # baseline CI used in compute_metrics assumed baseline_intensity = 500 (but we used 500 global constant)
    baseline_intensity_val = globals().get("baseline_intensity", 500.0)
    # norm reduction based on intensity delta (same idea)
    norm_reduced = np.clip((baseline_intensity_val - carbon_intensity) / max_reduced_per_hour, 0.0, 1.0)

    norm_speed = np.clip(1.0 - (latency_ms / max_latency), 0.0, 1.0)

    categories = ['Cost Saving', 'Carbon Reduction', 'Speed']
    values = [norm_saved, norm_reduced, norm_speed]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    # close the loop
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
    ax.spines['polar'].set_color('#333')
    ax.grid(color='#333', linestyle='--', linewidth=0.5)
    ax.set_title('Normalized Strengths (0–1)', color='white', fontsize=14, pad=18)

    plt.tight_layout()
    plt.savefig(radar_path, dpi=200, bbox_inches='tight', facecolor='#0b0b0b')
    plt.close(fig)

    return [cost_path, emis_path, radar_path]



# --------------------------------------------------------------------------- #
#               HTML + PLAYWRIGHT PDF (FINAL)                                 #
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
#               HTML + PLAYWRIGHT PDF (FINAL)                                 #
# --------------------------------------------------------------------------- #
def _build_pdf_html(metrics: Dict, chart_count: int = 0, summary: Optional[str] = None) -> Dict[str, str]:
    # THIS IS YOUR EXACT HTML — NO CHANGES AT ALL
    html = 

    # Replace placeholders
    saved_money = f"£{metrics.get('saved_money', 0):,.0f}".replace(",", " ")
    reduced_co2 = f"{metrics.get('reduced_emissions_kg_co2', 0):,.0f}".replace(",", " ")
    score = f"{metrics.get('score', 0):.2f}"
    ci = f"{metrics.get('carbon_intensity_gco2_kwh', 0):.1f}"
    latency = f"{metrics.get('latency_ms', 0):.0f}"

    html = html.replace("{{saved_money}}", saved_money)
    html = html.replace("{{reduced_co2}}", reduced_co2)
    html = html.replace("{{score}}", score)
    html = html.replace("{{company}}", metrics.get("company_name", "Test Corp"))
    html = html.replace("{{region}}", metrics.get("region_location", "EU (Stockholm)"))
    html = html.replace("{{latency}}", latency)
    html = html.replace("{{workload}}", metrics.get("workload_type", "inference").title())
    html = html.replace("{{gpu_hours}}", f"{metrics.get('gpu_hours', 0):,.0f}".replace(",", " "))
    html = html.replace("{{ci}}", ci)
    html = html.replace("{{date}}", datetime.now().strftime("%Y-%m-%d"))
    html = html.replace("{{summary}}", (summary or "").replace("\n", "<br>"))

    # Insert chart (you can keep QuickChart or your PNG)
    chart_url = f"https://quickchart.io/chart?c={metrics.get('chart_quickchart_url', 'YOUR_DEFAULT_CHART')}&backgroundColor=%230b0b0b&w=1200&h=500"
    chart_img = f'<img class="chart-img" src="{chart_url}" alt="Chart">'
    html = html.replace("{{chart_image}}", chart_img)

    return {"html": html, "css": ""}  # css is already inside




def _render_html_to_pdf_using_playwright(html_path: str, pdf_path: str) -> None:
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(f"file://{html_path}", wait_until="networkidle")
            page.add_style_tag(content="*{color:white !important;}")
            page.pdf(
    path=pdf_path,
    format="A4",
    landscape=True,
    print_background=True,
    prefer_css_page_size=True,
    margin={"top": "0mm", "bottom": "0mm", "left": "0mm", "right": "0mm"}
)
            browser.close()
    except Exception as e:
        log.warning(f"Playwright failed: {e}. Falling back to WeasyPrint.")
        try:
            from weasyprint import HTML
            HTML(filename=html_path).write_pdf(pdf_path, presentational_hints=True)
        except Exception as e2:
            log.error(f"WeasyPrint failed: {e2}")
            raise


def create_pdf(title: str, metrics: Dict, chart_files: list, output_pdf_path: str, summary: Optional[str] = None) -> None:
    """
    chart_files: list of local PNG files created earlier (paths)
    """
    # build workdir & assets
    artifacts_dir = ensure_artifacts_dir()
    run_id = uuid.uuid4().hex[:10]
    workdir = Path(artifacts_dir) / f"html_report_{run_id}"
    assets_dir = workdir / "assets"
    workdir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(exist_ok=True)

    # Copy provided chart files into assets/chart1.png, chart2.png, chart3.png ...
    for i, src in enumerate(chart_files, start=1):
        if src and os.path.exists(src):
            dst = assets_dir / f"chart{i}.png"
            shutil.copy(src, dst)

    # copy other static images (hero, etc.)
    repo_assets = Path(__file__).parent / "static" / "assets"
    for img in ["hero-datacenter.jpg", "laptop-dark-ui.jpg"]:
        src = repo_assets / img
        dst = assets_dir / img
        if src.exists():
            shutil.copy(src, dst)

    # Build HTML now that assets exist (pass chart count)
    built = _build_pdf_html(metrics, chart_count=len(chart_files), summary=summary)
    html_str = built["html"]

    (workdir / "index.html").write_text(html_str, encoding="utf-8")

    # Render to PDF (same Playwright helper you have)
    _render_html_to_pdf_using_playwright(str(workdir / "index.html"), output_pdf_path)



# --------------------------------------------------------------------------- #
#                         FLASK DECORATORS (ADDED)                            #
# --------------------------------------------------------------------------- #
def rate_limiter(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        ip = request.headers.get("X-Forwarded-For", request.remote_addr) or "unknown"
        now = int(time.time())
        window = now // 60
        key = f"{ip}:{window}"
        _requests[key] = _requests.get(key, 0) + 1
        _requests.pop(f"{ip}:{window - 1}", None)
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
#                         FLASK ROUTES                                        #
# --------------------------------------------------------------------------- #
@app.post("/calculate")
@rate_limiter
@require_api_key
def calculate():
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"ok": False, "error": "invalid_json"}), 400

        err, clean = validate_input(data)
        if err:
            return jsonify({"ok": False, "error": "bad_request", "detail": err}), 400

        metrics = compute_metrics(
            clean["company_name"],
            clean["workload_type"],
            clean["priorities"],
            float(clean["gpu_hours"]),
            clean["cloud_region"],
        )

        # ------------------------------------------------------------------- #
        # === AI SUMMARY – GEMINI ONLY ======================================= #
        # ------------------------------------------------------------------- #
        summary = "AI summary temporarily unavailable."
        ai_error = None

        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            ai_error = "GEMINI_API_KEY not set"
            log.warning(ai_error)
        else:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)

                prompt = f"""
                Write a concise, professional 120–150 word sustainability impact report for:
                Company: {metrics['company_name']}
                Workload: {metrics['workload_type'].title()} ({metrics['gpu_hours']} GPU hours)
                Region: {metrics['region_location']}
                Financial Savings: £{metrics['saved_money']:,.0f}
                CO₂ Reduction: {metrics['reduced_emissions_kg_co2']:,.0f} kg
                Carbon Intensity: {metrics['carbon_intensity_gco2_kwh']:.1f} gCO₂e/kWh
                Optimization Score: {metrics['score']:.2f}/1.0

                Highlight: cost savings via spot instances, carbon reduction vs. baseline, and latency impact.
                Use positive, forward-looking tone. No disclaimers.
                """

                model = genai.GenerativeModel('gemini-2.5-flash')  # Free, fast model
                response = model.generate_content(prompt)          # Plain text prompt
                raw_summary = response.text.strip()
                summary = textwrap.fill(raw_summary, width=90)

            except Exception as e:
                ai_error = str(e)
                log.warning(f"Gemini failed: {e}")

        # ------------------------------------------------------------------- #
        # === CHART + PDF ===
        # ------------------------------------------------------------------- #
        artifacts_dir = ensure_artifacts_dir()
        run_id = uuid.uuid4().hex[:12]
        pdf_path = os.path.join(artifacts_dir, f"{run_id}_report.pdf")

        # create 3 charts (chart_base -> artifacts/<runid>_chart)
        chart_base = os.path.join(artifacts_dir, f"{run_id}_chart")
        chart_files = create_charts(metrics, chart_base)    # returns list of 3 files

        # create PDF; create_pdf expects chart_files list
        create_pdf("Report", metrics, chart_files, pdf_path, summary)

        # expose first chart for quick preview; pdf url for download
        # chart_files typically: <runid>_chart_cost.png, _emissions.png, _radar.png
        chart_preview_name = os.path.basename(chart_files[0]) if chart_files else f"{run_id}_chart_cost.png"
        chart_url = f"/artifact/{chart_preview_name}"
        pdf_url = f"/artifact/{run_id}_report.pdf"

        return jsonify({
            "ok": True,
            "metrics": metrics,
            "summary": summary,
            "ai_error": ai_error,          # <-- tells Bubble *why* it failed
            "chart_url": chart_url,
            "pdf_url": pdf_url,
        })

    except Exception as e:
        log.exception("Error")
        return jsonify({"ok": False, "error": "internal_error"}), 500



@app.get("/artifact/<path:filename>")
@rate_limiter
def artifact(filename: str):
    return send_from_directory(ensure_artifacts_dir(), filename)


# --------------------------------------------------------------------------- #
#                         ENTRY POINT                                         #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
