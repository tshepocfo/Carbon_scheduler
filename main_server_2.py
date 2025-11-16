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
def create_chart(metrics: Dict, chart_path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5), dpi=120)
    labels = ["Saved Money (USD)", "CO₂ Reduced (kg)"]
    values = [metrics["saved_money"], metrics["reduced_emissions_kg_co2"]]
    colors = ["#7BE200", "#00C2FF"]
    ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.2)
    ax.set_title("Optimization Impact", color="white", fontsize=16, pad=20)
    ax.set_ylabel("Value", color="white")
    ax.tick_params(colors="white", labelsize=12)
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.patch.set_facecolor("#0b0b0b")
    ax.set_facecolor("#0b0b0b")
    plt.tight_layout(pad=4.0)
    plt.savefig(chart_path, format="png", facecolor="#0b0b0b", dpi=180, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
#               HTML + PLAYWRIGHT PDF (FINAL)                                 #
# --------------------------------------------------------------------------- #
def _build_pdf_html(metrics: Dict, chart_path: Optional[str], summary: Optional[str]) -> Dict[str, str]:
    def esc(v: Any) -> str:
        s = "" if v is None else str(v)
        return (
            s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#x27;")
        )

    # ----- dynamic values ----------------------------------------------------
    company            = esc(metrics.get("company_name", ""))
    saved_money        = f"{metrics.get('saved_money', 0.0):,.2f}"
    co2_reduced        = f"{metrics.get('reduced_emissions_kg_co2', 0.0):,.2f}"
    latency_ms         = f"{metrics.get('latency_ms', 0):.0f}"
    score_fmt          = f"{metrics.get('score', 0.0):.2f}"
    region_loc         = esc(metrics.get("region_location", metrics.get("cloud_region", "")))
    workload           = esc(metrics.get("workload_type", ""))
    gpu_hours          = esc(metrics.get("gpu_hours", ""))
    carbon_intensity   = metrics.get("carbon_intensity_gco2_kwh")
    carbon_intensity_fmt = "" if carbon_intensity is None else f"{float(carbon_intensity):.1f}"
    last_updated       = esc(metrics.get("last_updated", ""))
    summary_html       = esc(summary or "").replace("\n", "<br/>")
    chart_img_html     = "<img class='chart-img' src='assets/chart.png' alt='Chart'/>" if chart_path else ""

    # ------------------------------------------------------------------------
    css = """
    @page { size: 297mm 210mm; margin: 0; }
    :root {
      --bg: #0b0b0b;
      --glass: rgba(255,255,255,0.02);
      --glass-border: rgba(255,255,255,0.18);
      --text: #ffffff;
      --muted: #e0e0e0;
      --accent: #7BE200;
      --footer: #0e0e0e;
      --divider: rgba(255,255,255,0.08);
      --font: 'Space Grotesk', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
    }
    * { box-sizing: border-box; color: var(--text) !important; margin:0; padding:0; }
    html, body { background:var(--bg); font-family:var(--font); width:297mm; height:210mm; }

    .page      { width:297mm; height:210mm; display:flex; flex-direction:column; }
    .container { padding:12mm 16mm 0 16mm; flex:1; display:flex; flex-direction:column; }
    .nav       { display:flex; justify-content:space-between; margin-bottom:6mm; font-size:3.6mm; }
    .logo .text { font-weight:700; letter-spacing:0.4px; }

    /* ───── HERO ───── */
    .hero      { display:grid; grid-template-columns:1.4fr 1fr; gap:10mm; margin-bottom:8mm; }
    .glass     { background:var(--glass); border:1px solid var(--glass-border); border-radius:5mm; padding:10mm; }
    .headline  { font-size:15mm; font-weight:700; line-height:1.1; margin-bottom:4mm; }
    .sub       { font-size:4.4mm; color:var(--muted); margin-bottom:6mm; }

    /* ───── STATS (centred, two-line labels) ───── */
    .stats     { display:grid; grid-template-columns:repeat(3,1fr); gap:5mm; }
    .stat      { padding:4mm 2mm; border:1px solid var(--divider); border-radius:4mm; text-align:center; background:rgba(255,255,255,0.015); }
    .stat .label { font-size:3.2mm; line-height:1.2; margin-bottom:1.5mm; }
    .stat .value { font-size:6.5mm; font-weight:600; color:var(--accent); line-height:1.1; }

    /* ───── HERO RIGHT – image first, meta card underneath ───── */
    .hero-right { display:flex; flex-direction:column; height:70mm; }
    .hero-img   { flex:1; width:100%; object-fit:cover; border-radius:4mm; border:1px solid var(--glass-border); }
    .meta-card  {
      margin-top:20mm; padding:5mm; background:var(--glass);
      border:1px solid var(--glass-border); border-radius:4mm;
      font-size:3.3mm; line-height:1.4;
    }

    /* ───── FEATURES ───── */
    .features   { display:grid; grid-template-columns:1fr 1fr; gap:8mm; margin-bottom:6mm; }
    .feature-img { width:100%; height:auto; max-height:38mm; object-fit:cover; border-radius:4mm; margin-top:4mm; border:1px solid var(--glass-border); }

    /* ───── CHART ───── */
    .chart-section { margin:0 16mm 6mm 16mm; flex-shrink:0; }
    .chart-title   { font-size:4.6mm; color:var(--muted); margin-bottom:3mm; }
    .chart-img     { width:100%; max-width:260mm; height:auto; border-radius:3mm; border:1px solid var(--divider); }

    /* ───── AI SUMMARY ───── */
    .ai-summary { margin-top:5mm; padding:5mm; border:1px solid var(--divider); border-radius:4mm;
                  background:rgba(255,255,255,0.015); font-size:3.5mm; line-height:1.45; max-height:42mm; overflow:auto; word-wrap:break-word; }

    /* ───── FOOTER ───── */
    .footer    { background:var(--footer); padding:5mm 16mm; border-top:1px solid var(--divider);
                 display:grid; grid-template-columns:1.2fr 1fr 1fr 1fr; gap:8mm; font-size:3.5mm; flex-shrink:0; }
    .copyright { text-align:center; padding:3mm 0; font-size:3.2mm; color:var(--muted);
                 border-top:1px solid var(--divider); margin:0 16mm; }
    """

    # ------------------------------------------------------------------------
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Report</title>
<style>@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');</style>
<style>{css}</style></head><body>
<div class="page">
  <div class="container">
    <!-- NAV -->
    <div class="nav">
      <div class="logo"><div class="text">CARBONSIGHT SCHEDULER</div></div>
      <div class="nav-menu">Solutions | About Us | Blog | Support</div>
    </div>

    <!-- HERO -->
    <div class="hero">
      <div class="glass hero-left">
        <div class="headline">Smarter AI. Lower Cost. Less Carbon.</div>
        <div class="sub">Precision scheduling aligned to carbon intensity and spot market efficiency.</div>

        <!-- STATS – two-line, perfectly centred -->
        <div class="stats">
          <div class="stat">
            <div class="label">Financial<br>Savings</div>
            <div class="value">£{saved_money}</div>
          </div>
          <div class="stat">
            <div class="label">CO₂<br>Reduction</div>
            <div class="value">{co2_reduced} kg</div>
          </div>
          <div class="stat">
            <div class="label">Optimisation<br>Score</div>
            <div class="value">{score_fmt}/1.0</div>
          </div>
        </div>

        <div class="ai-summary">{summary_html}</div>
      </div>

      <!-- RIGHT SIDE – image first, meta card below -->
      <div class="hero-right">
        <img class="hero-img" src="assets/hero-datacenter.jpg" alt="Data Center"/>
        <div class="meta-card">
          <h4 style="margin:0 0 3mm;font-size:3.8mm;">Deployment Meta</h4>
          <div class="meta">Company: {company} • Region: {region_loc} • Latency: {latency_ms} ms</div>
          <div class="meta">Workload: {workload} • GPU Hours: {gpu_hours} • CI: {carbon_intensity_fmt} gCO₂e/kWh</div>
          <div class="meta">Last Updated: {last_updated}</div>
        </div>
      </div>
    </div>

    <!-- FEATURES -->
    <div class="features">
      <div class="glass feature">
        <h3>Carbon-Aware Orchestration</h3>
        <p>Dynamically shifts workloads to lower carbon regions and optimises for spot pricing.</p>
      </div>
      <div class="glass feature">
        <h3>Operator Experience</h3>
        <p>Visual scheduling, proactive insights and real-time alerts.</p>
        <img class="feature-img" src="assets/laptop-dark-ui.jpg" alt="UI"/>
      </div>
    </div>

    <!-- CHART -->
    <div class="chart-section">
      <div class="chart-title">Run Comparison: Savings and Emissions</div>
      {chart_img_html}
    </div>
  </div>

  <!-- FOOTER -->
  <div class="footer">
    <div class="brand"><div><div style="font-weight:700;">CARBONSIGHT SCHEDULER</div><div style="font-size:3.2mm;">Smarter AI. Lower Cost. Less Carbon.</div></div></div>
    <div><h5>Products</h5><ul><li>Scheduler</li><li>Carbon Intelligence</li><li>Cost Insights</li></ul></div>
    <div><h5>Company</h5><ul><li>About</li><li>Blog</li><li>Careers</li></ul></div>
    <div><h5>Connect</h5><div class="social"><div class="dot">Instagram</div><div class="dot">LinkedIn</div><div class="dot">Github</div></div></div>
  </div>
  <div class="copyright">© 2025 CarbonSight Scheduler | Terms | Privacy | Cookies</div>
</div>
</body></html>"""

    return {"html": html, "css": css.strip()}


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


def create_pdf(title: str, metrics: Dict, chart_path: str, output_pdf_path: str, summary: Optional[str] = None) -> None:
    built = _build_pdf_html(metrics, chart_path, summary)
    html_str = built["html"]

    artifacts_dir = ensure_artifacts_dir()
    run_id = uuid.uuid4().hex[:10]
    workdir = Path(artifacts_dir) / f"html_report_{run_id}"
    assets_dir = workdir / "assets"
    workdir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(exist_ok=True)

    if chart_path and os.path.exists(chart_path):
        shutil.copy(chart_path, assets_dir / "chart.png")

    repo_assets = Path(__file__).parent / "static" / "assets"
    for img in ["hero-datacenter.jpg", "laptop-dark-ui.jpg"]:
        src = repo_assets / img
        dst = assets_dir / img
        if src.exists():
            shutil.copy(src, dst)

    (workdir / "index.html").write_text(html_str, encoding="utf-8")
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
        # === AI SUMMARY (ROBUST + DEBUGGABLE) ============================== #
        # ------------------------------------------------------------------- #
        summary   = "AI summary temporarily unavailable."
        ai_error  = None

        if os.getenv("OPENAI_API_KEY"):
            try:
                import openai
                client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=280,
                    temperature=0.7,
                )
                raw_summary = response.choices[0].message.content.strip()
                summary = textwrap.fill(raw_summary, width=90)

            except Exception as e:
                ai_error = str(e)
                log.warning(f"OpenAI failed: {e}")
                summary = "AI summary temporarily unavailable."
        else:
            ai_error = "OPENAI_API_KEY not set"
            log.warning("OpenAI skipped: API key missing")

        # ------------------------------------------------------------------- #
        # === CHART + PDF ===
        # ------------------------------------------------------------------- #
        artifacts_dir = ensure_artifacts_dir()
        run_id        = uuid.uuid4().hex[:12]
        chart_path    = os.path.join(artifacts_dir, f"{run_id}_chart.png")
        pdf_path      = os.path.join(artifacts_dir, f"{run_id}_report.pdf")

        create_chart(metrics, chart_path)
        create_pdf("Report", metrics, chart_path, pdf_path, summary)

        chart_url = f"/artifact/{run_id}_chart.png"
        pdf_url   = f"/artifact/{run_id}_report.pdf"

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
