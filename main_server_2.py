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
from typing import Dict, Optional

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

APP_VERSION = "1.0.2"
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
    """Distance in km between two lat/lon points."""
    R = 6371.0
    lat1_rad, lon1_rad = radians(lat1), radians(lon1)
    lat2_rad, lon2_rad = radians(lat2), radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = sin(dlat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def estimate_latency(distance_km: float) -> float:
    """Realistic RTT (ms) – light in fibre + overhead."""
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

# Default user location (NYC)
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
        # fallback to a zone (simple)
        zone = "SE"  # Sweden for eu-north-1 – you can improve this
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
        # Fallback to 30 % of on-demand if no spot data
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

    # ---- Scoring (normalised) ----
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
    score -= (latency_ms / max_latency) * 0.1  # latency penalty

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
    """Save a dark-theme bar chart to `chart_path`."""
    fig, ax = plt.subplots(figsize=(9, 5), dpi=100)  # ← FIXED SIZE
    labels = ["Saved Money (£)", "CO₂ Reduced (kg)"]
    values = [metrics["saved_money"], metrics["reduced_emissions_kg_co2"]]
    colors = ["#7BE200", "#00C2FF"]
    ax.bar(labels, values, color=colors)
    ax.set_title("Optimization Impact", color="white", fontsize=14)
    ax.set_ylabel("Value", color="white")
    ax.tick_params(colors="white")
    fig.patch.set_facecolor("#0b0b0b")
    ax.set_facecolor("#0b0b0b")
    plt.tight_layout(pad=3.0)  # ← PREVENT OVERFLOW
    plt.savefig(chart_path, format="png", facecolor="#0b0b0b", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
#               HTML + PLAYWRIGHT PDF (Qodo.ai + Fixes)                        #
# --------------------------------------------------------------------------- #
def _html_escape(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _safe_copy(src: Optional[str], dst: str) -> Optional[str]:
    try:
        if src and os.path.exists(src):
            _ensure_dir(os.path.dirname(dst))
            shutil.copyfile(src, dst)
            return dst
    except Exception:
        pass
    return None


def _build_pdf_html(metrics: Dict, chart_path: Optional[str], summary: Optional[str]) -> Dict[str, str]:
    """
    Build HTML and CSS for a print-perfect A4 landscape PDF.
    Returns a dict with keys: "html" and "css".
    """
    def esc(v: Any) -> str:
        s = "" if v is None else str(v)
        return (
            s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#x27;")
        )
    # Dynamic fields with formatting
    company = esc(metrics.get("company_name", ""))
    saved_money_val = metrics.get("saved_money", 0.0)
    saved_money = f"{float(saved_money_val):,.2f}"
    co2_val = metrics.get("reduced_emissions_kg_co2", 0.0)
    co2_reduced = f"{float(co2_val):,.2f}"
    latency_val = metrics.get("latency_ms", 0)
    latency_ms = f"{float(latency_val):.0f}"
    score_val = metrics.get("score", 0.0)
    score_fmt = f"{float(score_val):.2f}"
    region_loc = esc(metrics.get("region_location", metrics.get("cloud_region", "")))
    workload = esc(metrics.get("workload_type", ""))
    gpu_hours = esc(metrics.get("gpu_hours", ""))
    carbon_intensity = metrics.get("carbon_intensity_gco2_kwh")
    carbon_intensity_fmt = "" if carbon_intensity is None else f"{float(carbon_intensity):.1f}"
    last_updated = esc(metrics.get("last_updated", ""))
    summary_html = esc(summary or "").replace("\n", "<br/>")
    chart_img_html = "<img class='chart-img' src='assets/chart.png' alt='Chart'/>" if chart_path else ""
    css = """
    @page {
      size: 297mm 210mm; /* A4 Landscape */
      margin: 0;
    }
    :root {
      --bg: #0b0b0b;
      --glass: rgba(255,255,255,0.02);
      --glass-border: rgba(255,255,255,0.18);
      --text: #ffffff;
      --muted: #bdbdbd;
      --accent: #7BE200;
      --footer: #0e0e0e;
      --divider: rgba(255,255,255,0.06);
      --font: 'Space Grotesk', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
    }
    * { box-sizing: border-box; -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; }
    html, body {
      background: var(--bg);
      color: var(--text);
      font-family: var(--font);
      width: 297mm;
      height: 210mm;
      overflow: hidden;
      line-height: 1.5;
    }
    .page {
      position: relative;
      width: 297mm;
      height: 210mm;
      display: flex;
      flex-direction: column;
    }
    .container {
      padding: 16mm 18mm 0 18mm;
      flex: 1;
    }
    /* Top nav */
    .nav {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 10mm;
    }
    .logo {
      display: flex;
      align-items: center;
      gap: 6mm;
    }
    .logo .mark {
      width: 12mm; height: 12mm; border-radius: 3mm;
      background: linear-gradient(180deg, rgba(255,255,255,0.18), rgba(255,255,255,0.08));
      border: 1px solid var(--glass-border);
      display: grid; place-items: center;
      box-shadow: 0 4px 24px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.06);
    }
    .logo .mark:before {
      content: "★";
      color: #fff; font-size: 6mm; line-height: 1;
      filter: drop-shadow(0 1px 1px rgba(0,0,0,0.4));
    }
    .logo .text {
      font-weight: 700; letter-spacing: 0.5px; opacity: 0.95;
    }
    .nav-menu {
      display: flex; gap: 10mm;
      color: var(--muted);
      font-size: 3.5mm;
    }
    /* Hero split */
    .hero {
      display: grid;
      grid-template-columns: 1.2fr 1fr;
      gap: 10mm;
      align-items: stretch;
      position: relative;
    }
    .glass {
      background: var(--glass);
      border: 1px solid var(--glass-border);
      border-radius: 6mm;
      box-shadow: 0 8px 40px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.05);
      position: relative;
      overflow: hidden;
    }
    .glass:after {
      content: "";
      position: absolute; inset: 0;
      background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0));
      pointer-events: none;
    }
    .hero-left {
      padding: 12mm;
      display: flex; flex-direction: column; justify-content: center;
      gap: 6mm;
    }
    .headline {
      font-size: 18mm;
      font-weight: 700;
      line-height: 1.05;
      letter-spacing: -0.2mm;
      max-height: 3.2em;
    }
    .sub {
      font-size: 4.8mm;
      color: var(--muted);
      max-width: 140mm;
    }
    .stats {
      display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 6mm;
    }
    .stat {
      padding: 6mm;
      border-radius: 4mm;
      border: 1px solid var(--divider);
      background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.00));
      text-align: center;
    }
    .stat .label { color: var(--muted); font-size: 3.5mm; margin-bottom: 2mm; }
    .stat .value { font-size: 7mm; font-weight: 600; color: var(--accent); }
    .hero-right {
      position: relative;
    }
    .hero-img {
      width: 100%; height: 100%;
      border-radius: 4.5mm;
      object-fit: cover;
      border: 1px solid var(--glass-border);
      box-shadow: 0 10px 40px rgba(0,0,0,0.5);
    }
    .overlap-card {
      position: absolute;
      left: -12%; top: 12%;
      width: 60%;
      z-index: 2;
      padding: 6mm;
      border-radius: 4mm;
      background: var(--glass);
      border: 1px solid var(--glass-border);
      box-shadow: 0 8px 40px rgba(0,0,0,0.45);
      backdrop-filter: blur(4px);
    }
    .overlap-card h4 { margin: 0 0 2mm 0; font-size: 5mm; }
    .meta { display: flex; gap: 6mm; color: var(--muted); font-size: 3.5mm; flex-wrap: wrap; }
    /* Middle feature cards */
    .features {
      margin-top: 10mm;
      display: grid; grid-template-columns: 1fr 1fr; gap: 8mm;
    }
    .feature { padding: 8mm; }
    .feature h3 { font-size: 6mm; margin: 0 0 2mm 0; color: #fff; }
    .feature p { color: var(--muted); font-size: 4mm; line-height: 1.45; }
    .feature .feature-img {
      width: 100%; height: 45mm; border-radius: 4mm; object-fit: cover;
      border: 1px solid var(--glass-border);
      margin-top: 4mm;
      box-shadow: 0 6px 24px rgba(0,0,0,0.35);
    }
    .contact {
      position: absolute;
      right: 18mm; top: 20mm;
      padding: 5mm 6mm;
      border-radius: 3.5mm;
      border: 1px solid var(--glass-border);
      background: var(--glass);
      color: var(--muted);
      font-size: 3.7mm;
      box-shadow: 0 8px 28px rgba(0,0,0,0.35);
    }
    .divider {
      height: 1px; background: var(--divider); margin: 8mm 0;
      width: calc(100% - 36mm); margin-left: 18mm; margin-right: 18mm;
    }
    /* Chart section */
    .chart-wrap {
      margin-top: 8mm;
      padding: 0 18mm;
      text-align: center;
    }
    .chart-title { font-size: 5mm; margin-bottom: 4mm; color: var(--muted); text-align: left; }
    .chart-img {
      width: 100%; max-width: 240mm; height: auto; border-radius: 3mm; border: 1px solid var(--divider);
      box-shadow: 0 6px 24px rgba(0,0,0,0.35); margin: 0 auto; display: block;
    }
    /* Footer */
    .footer {
      background: var(--footer);
      margin-top: 8mm;
      padding: 8mm 18mm;
      border-top: 1px solid var(--divider);
      display: grid;
      grid-template-columns: 1.2fr 1fr 1fr 1fr;
      gap: 10mm;
      align-items: start;
    }
    .footer .brand {
      display: flex; gap: 5mm; align-items: center;
    }
    .footer .brand .foot-mark {
      width: 10mm; height: 10mm; border-radius: 3mm;
      background: linear-gradient(180deg, rgba(255,255,255,0.18), rgba(255,255,255,0.08));
      border: 1px solid var(--glass-border);
      display: grid; place-items: center;
    }
    .footer .brand .foot-mark:before { content: "★"; color: #fff; font-size: 5mm; }
    .footer h5 { margin: 0 0 3mm 0; font-size: 4.2mm; color: #fff; }
    .footer ul { list-style: none; padding: 0; margin: 0; }
    .footer li { color: var(--muted); font-size: 3.7mm; margin: 1.5mm 0; }
    .footer .social { display: flex; gap: 3mm; }
    .footer .social .dot {
      width: 7mm; height: 7mm; border-radius: 50%;
      background: var(--bg);
      border: 1px solid var(--divider);
      display: grid; place-items: center; color: var(--muted); font-size: 3.5mm;
    }
    .copyright {
      color: var(--muted); font-size: 3.5mm; text-align: center;
      padding: 4mm 0 6mm 0;
      border-top: 1px solid var(--divider);
      width: calc(100% - 36mm); margin-left: 18mm; margin-right: 18mm;
    }
    .ai-summary {
      margin-top: 6mm;
      padding: 6mm;
      border: 1px solid var(--divider);
      border-radius: 4mm;
      color: var(--muted);
      line-height: 1.5;
      background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0));
      font-size: 3.8mm;
    }
    ul { padding-left: 20px; margin: 8px 0; }
    li { margin: 6px 0; position: relative; }
    li:before { content: "•"; color: var(--accent); position: absolute; left: -16px; font-weight: bold; }
    img { max-width: 100%; height: auto; display: block; border-radius: 4mm; }
    """
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CarbonSight Scheduler Report</title>
  <meta name="viewport" content="width=297mm, initial-scale=1">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <style>@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');</style>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <div class="page">
    <div class="container">
      <div class="nav">
        <div class="logo">
          <div class="mark"></div>
          <div class="text">CARBONSIGHT SCHEDULER</div>
        </div>
        <div class="nav-menu">Solutions | About Us | Blog | Support</div>
      </div>
      <div class="hero">
        <div class="glass hero-left">
          <div class="headline">Smarter AI. Lower Cost. Less Carbon.</div>
          <div class="sub">
            Precision scheduling aligned to carbon intensity and spot market efficiency. Delivering measurable savings and sustainable performance for modern AI workloads.
          </div>
          <div class="stats">
            <div class="stat">
              <div class="label">Financial Savings</div>
              <div class="value">£<span class="dynamic-savings">{saved_money}</span></div>
            </div>
            <div class="stat">
              <div class="label">CO₂ Reduction</div>
              <div class="value"><span class="dynamic-co2">{co2_reduced} kg</span></div>
            </div>
            <div class="stat">
              <div class="label">Optimisation Score</div>
              <div class="value"><span class="dynamic-score">{score_fmt}/1.0</span></div>
            </div>
          </div>
          <div class="ai-summary">{summary_html}</div>
        </div>
        <div class="hero-right">
          <img class="hero-img" src="assets/hero-datacenter.jpg" alt="Data Center"/>
          <div class="overlap-card">
            <h4>Deployment Meta</h4>
            <div class="meta">
              <div>Company: <span class="dynamic-company">{company}</span></div>
              <div>Region: <span class="dynamic-region">{region_loc}</span></div>
              <div>Latency: <span class="dynamic-latency">{latency_ms} ms</span></div>
            </div>
            <div class="meta" style="margin-top:3mm;">
              <div>Workload: <span class="dynamic-workload">{workload}</span></div>
              <div>GPU Hours: <span class="dynamic-gpu">{gpu_hours}</span></div>
              <div>CI: {carbon_intensity_fmt} gCO₂e/kWh</div>
            </div>
            <div class="meta" style="margin-top:3mm;">
              <div>Last Updated: {last_updated}</div>
            </div>
          </div>
        </div>
      </div>
      <div class="features">
        <div class="glass feature">
          <h3>Carbon-Aware Orchestration</h3>
          <p>
            Dynamically shifts workloads to lower carbon regions and optimises for spot pricing without compromising SLAs.
            Policy-driven controls ensure compliance and predictable outcomes.
          </p>
        </div>
        <div class="glass feature">
          <h3>Operator Experience</h3>
          <p>
            Visual scheduling, proactive insights and real-time alerts improve operational efficiency and visibility across fleets.
          </p>
          <img class="feature-img" src="assets/laptop-dark-ui.jpg" alt="Dark UI"/>
        </div>
      </div>
      <div class="contact">
        Need a custom rollout? Contact our delivery team.
      </div>
    </div>
    <div class="divider"></div>
    <div class="chart-wrap">
      <div class="chart-title">Run Comparison: Savings and Emissions</div>
      {chart_img_html}
    </div>
    <div class="footer">
      <div class="brand">
        <div class="foot-mark"></div>
        <div>
          <div style="font-weight:700;">CARBONSIGHT SCHEDULER</div>
          <div style="color:var(--muted); font-size:3.7mm;">Smarter AI. Lower Cost. Less Carbon.</div>
        </div>
      </div>
      <div>
        <h5>Products</h5>
        <ul>
          <li>Scheduler</li>
          <li>Carbon Intelligence</li>
          <li>Cost Insights</li>
        </ul>
      </div>
      <div>
        <h5>Company</h5>
        <ul>
          <li>About</li>
          <li>Blog</li>
          <li>Careers</li>
        </ul>
      </div>
      <div>
        <h5>Connect</h5>
        <div class="social">
          <div class="dot">in</div>
          <div class="dot">x</div>
          <div class="dot">gh</div>
        </div>
      </div>
    </div>
    <div class="copyright">
      © 2025 CarbonSight Scheduler | Terms | Privacy | Cookies
    </div>
  </div>
</body>
</html>
""".strip()
    return {"html": html, "css": css.strip()}


def _render_html_to_pdf_using_playwright(html_path: str, pdf_path: str) -> None:
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"],
            )
            page = browser.new_page()
            page.goto(f"file://{html_path}", wait_until="load")
            page.pdf(
                path=pdf_path,
                format="A4",
                landscape=True,
                print_background=True,
                prefer_css_page_size=True,
                margin=0,
            )
            browser.close()
    except Exception:
        # Fallback to weasyprint
        try:
            from weasyprint import HTML

            HTML(filename=html_path).write_pdf(pdf_path)
        except Exception:
            # Ultimate fallback – tiny ReportLab message
            from reportlab.lib.pagesizes import landscape, A4
            from reportlab.pdfgen import canvas

            c = canvas.Canvas(pdf_path, pagesize=landscape(A4))
            c.setFont("Helvetica", 12)
            c.drawString(50, 550, "PDF generation failed – install playwright or weasyprint.")
            c.save()


def create_pdf(
    title: str,
    metrics: Dict,
    chart_path: str,
    output_pdf_path: str,
    summary: Optional[str] = None,
) -> None:
    """Public entry point – builds HTML + renders PDF."""
    built = _build_pdf_html(metrics, chart_path, summary)
    html_str, css_str = built["html"], built["css"]

    artifacts_dir = ensure_artifacts_dir()
    run_id = uuid.uuid4().hex[:10]
    workdir = Path(artifacts_dir) / f"html_report_{run_id}"
    assets_dir = workdir / "assets"
    workdir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(exist_ok=True)

    # copy chart
    if chart_path:
        _safe_copy(chart_path, assets_dir / "chart.png")

    # --- FIX: Copy assets from repo's static/assets ---
    static_assets = Path(__file__).parent / "static" / "assets"
    _safe_copy(static_assets / "hero-datacenter.jpg", assets_dir / "hero-datacenter.jpg")
    _safe_copy(static_assets / "laptop-dark-ui.jpg", assets_dir / "laptop-dark-ui.jpg")

    # write files
    (workdir / "index.html").write_text(html_str, encoding="utf-8")
    (workdir / "styles.css").write_text(css_str, encoding="utf-8")

    _render_html_to_pdf_using_playwright(str(workdir / "index.html"), output_pdf_path)


# --------------------------------------------------------------------------- #
#                         S3 UPLOAD (optional)                               #
# --------------------------------------------------------------------------- #
def s3_upload(file_path: str, key: str, bucket: str, region: str) -> str:
    s3 = boto3.client("s3", region_name=region, config=boto_config)
    s3.upload_file(file_path, bucket, key)
    return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"


# --------------------------------------------------------------------------- #
#                         FLASK DECORATORS                                    #
# --------------------------------------------------------------------------- #
def rate_limiter(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        ip = request.headers.get("X-Forwarded-For", request.remote_addr) or "unknown"
        now = int(time.time())
        window = now // 60
        key = f"{ip}:{window}"
        _requests[key] = _requests.get(key, 0) + 1
        # clean old window
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
@app.get("/health")
@rate_limiter
def health():
    return jsonify({"ok": True, "version": APP_VERSION})


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

        user_email = clean.get("user_email", "tshepotrustin@outlook.com")
        metrics = compute_metrics(
            clean["company_name"],
            clean["workload_type"],
            clean["priorities"],
            float(clean["gpu_hours"]),
            clean["cloud_region"],
        )

        # ---- AI Summary (OpenAI) ----
        summary = "OPENAI_API_KEY not set."
        if os.getenv("OPENAI_API_KEY"):
            try:
                import requests

                prompt = f"""
                Write a professional AI-optimisation report (200-350 words) for {metrics['company_name']}.
                Include: savings £{metrics['saved_money']:.2f}, CO₂ reduction {metrics['reduced_emissions_kg_co2']:.2f} kg,
                carbon intensity {metrics['carbon_intensity_gco2_kwh']:.1f} gCO₂e/kWh,
                latency {metrics['latency_ms']:.0f} ms, score {metrics['score']:.2f}/1.0.
                Structure: Executive Summary, Financial Impact, Environmental Impact,
                Performance & Latency, Recommendations.
                """
                resp = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"},
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "system", "content": "You are a senior sustainability consultant."},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.6,
                        "max_tokens": 600,
                    },
                    timeout=20,
                )
                if resp.ok:
                    summary = resp.json()["choices"][0]["message"]["content"].strip()
                else:
                    summary = f"AI failed: {resp.text[:200]}"
            except Exception as e:
                summary = f"AI exception: {e}"

        # ---- Artifacts (chart + PDF) ----
        artifacts_dir = ensure_artifacts_dir()
        run_id = uuid.uuid4().hex[:12]
        chart_name = f"{run_id}_chart.png"
        pdf_name = f"{run_id}_report.pdf"
        chart_path = os.path.join(artifacts_dir, chart_name)
        pdf_path = os.path.join(artifacts_dir, pdf_name)

        create_chart(metrics, chart_path)
        create_pdf("CarbonSight Report", metrics, chart_path, pdf_path, summary)

        # ---- Optional S3 upload ----
        if os.getenv("USE_S3", "false").lower() == "true":
            bucket = os.getenv("S3_BUCKET")
            region = os.getenv("AWS_REGION", "us-east-1")
            if not bucket:
                return jsonify({"ok": False, "error": "S3_BUCKET missing"}), 500
            chart_url = s3_upload(chart_path, f"charts/{chart_name}", bucket, region)
            pdf_url = s3_upload(pdf_path, f"reports/{pdf_name}", bucket, region)
        else:
            chart_url = f"/artifact/{chart_name}"
            pdf_url = f"/artifact/{pdf_name}"

        # ---- Response ----
        resp = {
            "ok": True,
            "metrics": metrics,
            "summary": summary,
            "chart_url": chart_url,
            "pdf_url": pdf_url,
        }

        # ---- Optional n8n webhook for email ----
        if os.getenv("N8N_WEBHOOK_URL"):
            try:
                requests.post(
                    os.getenv("N8N_WEBHOOK_URL"),
                    json={"pdf_url": pdf_url, "metrics": metrics, "user_email": user_email},
                    timeout=5,
                )
            except Exception:
                pass

        return jsonify(resp)

    except Exception as e:
        log.exception("Unhandled error in /calculate")
        return jsonify({"ok": False, "error": "internal_error", "detail": str(e)}), 500


@app.get("/artifact/<path:filename>")
@rate_limiter
def artifact(filename: str):
    artifacts_dir = ensure_artifacts_dir()
    return send_from_directory(artifacts_dir, filename, as_attachment=False)


# --------------------------------------------------------------------------- #
#                         ENTRY POINT                                         #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
