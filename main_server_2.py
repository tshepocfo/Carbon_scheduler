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
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>CarbonSight Scheduler – Live PDF Preview (2 Pages)</title>
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    @page { size: 297mm 210mm; margin: 0; }
    :root{
      --bg:#0b0b0b;
      --glass:rgba(255,255,255,0.02);
      --glass-border:rgba(255,255,255,0.18);
      --text:#fff;
      --muted:#e0e0e0;
      --accent:#7BE200;
      --footer-bg:#0e0e0e;
      --divider:rgba(255,255,255,0.08);
      --footer-height:22mm;
      --page-padding-vertical:12mm;
      --page-padding-horizontal:16mm;
    }
    *{box-sizing:border-box;margin:0;padding:0;color:var(--text)!important}
    body{background:#000;font-family:'Space Grotesk',sans-serif;padding:40px 0}
    .page{
      width:297mm;height:210mm;background:var(--bg);margin:0 auto 40px auto;
      box-shadow:0 0 30px rgba(0,0,0,0.8);display:flex;flex-direction:column;
      position:relative;page-break-after:always;overflow:hidden;
      padding: calc(var(--page-padding-vertical)) calc(var(--page-padding-horizontal));
    }
    .page .content { flex: 1 1 auto; overflow: hidden; display:flex; flex-direction:column; gap:8mm; }
    .page .content-inner { overflow:auto; padding-right:4mm; }
    .footer { flex: 0 0 var(--footer-height); background:var(--footer-bg); padding:6mm 12mm; border-top:1px solid var(--divider); display:grid; grid-template-columns:1.6fr 1fr 1fr 1fr; gap:10mm; align-items:start; font-size:3.0mm; }
    .footer .brand{font-weight:700}
    .footer h5{font-size:3.8mm;margin-bottom:6px;color:var(--text)}
    .footer ul{list-style:none;margin:0;padding:0;line-height:1.8;font-size:3mm;color:var(--muted)}
    .footer ul li{margin-bottom:3px}
    .social{display:flex;gap:8px;flex-wrap:wrap}
    .dot{background:rgba(255,255,255,0.03);padding:6px 8px;border-radius:999px;font-size:3mm;color:var(--muted)}
    .footer .copyright { grid-column: 1 / -1; text-align:center; font-size:2.9mm; color:var(--muted); margin-top:6px }
    .nav{display:flex;justify-content:space-between;margin-bottom:6mm;font-size:3.6mm;font-weight:600}
    .hero{display:grid;grid-template-columns:1.3fr 1fr;gap:8mm;margin-bottom:6mm}
    .glass{background:var(--glass);border:1px solid var(--glass-border);border-radius:7mm;padding:12mm}
    .headline{font-size:13mm;font-weight:700;line-height:1.05;margin-bottom:3mm}
    .sub{font-size:4.1mm;color:var(--muted);margin-bottom:6mm}
    .stats{display:grid;grid-template-columns:repeat(3,1fr);gap:7mm;margin-top:5mm}
    .stat{background:rgba(255,255,255,0.04);backdrop-filter:blur(8px);border:1px solid rgba(255,255,255,0.12);border-radius:7mm;padding:8mm 6mm;text-align:center;display:flex;flex-direction:column;justify-content:center;min-height:34mm}
    .stat .label{font-size:3.2mm;color:var(--muted);margin-bottom:3mm}
    .stat .value{font-size:8.4mm;font-weight:700;color:var(--accent);white-space:nowrap}
    .hero-right img{width:100%;height:60mm;object-fit:cover;border-radius:4mm;border:1px solid var(--glass-border);display:block}
    .meta-card{margin-top:10mm;padding:6mm;background:var(--glass);border:1px solid var(--glass-border);border-radius:5mm;font-size:3.2mm;line-height:1.45}
    .features{display:grid;grid-template-columns:1fr 1fr;gap:6mm;margin-top:6mm}
    .feature-card{background:var(--glass);border:1px solid var(--glass-border);border-radius:7mm;padding:8mm;min-height:36mm;display:flex;flex-direction:column;justify-content:flex-start}
    .feature-card h3{font-size:4.6mm;margin-bottom:3mm}
    .feature-card p{font-size:3.6mm;color:var(--muted);line-height:1.45}
    .chart-and-summary { display:flex; gap:16mm; align-items:flex-start; }
    .chart-column { flex: 1 1 60%; }
    .summary-column { flex: 1 1 40%; min-width:90mm; }
    .chart-title{font-size:5.2mm;color:var(--muted);margin-bottom:10mm;font-weight:600;text-align:left; padding-left:6mm}
    .chart-img{width:100%;height:60mm;border-radius:8mm;border:1px solid var(--divider);box-shadow:0 4mm 16mm rgba(0,0,0,0.5);display:block;object-fit:cover}
    .ai-summary-full{background:rgba(255,255,255,0.04);border:1px solid var(--divider);border-radius:8mm;padding:12mm;font-size:4.1mm;line-height:1.56;backdrop-filter:blur(6px);height:100%;box-sizing:border-box;overflow:hidden}
    .ai-summary-full strong{font-size:5.2mm;color:var(--accent);display:block;margin-bottom:8mm}
    .page:first-of-type .footer { display:none; }
    img{display:block;max-width:100%;height:auto;-webkit-print-color-adjust:exact;print-color-adjust:exact;image-rendering:-webkit-optimize-contrast;page-break-inside:avoid}
    .glass, .stat, .feature-card, .ai-summary-full{page-break-inside:avoid}
    @media print { body{padding:0} .page{box-shadow:none;margin:0} }
  </style>
</head>
<body>
<!-- PAGE 1 -->
<div class="page">
  <div class="content">
    <div class="content-inner">
      <div class="nav">
        <div class="logo">CARBONSIGHT SCHEDULER</div>
        <div>Solutions | About Us | Blog | Support</div>
      </div>
      <div class="hero">
        <div class="glass">
          <div class="headline">Smarter AI. Lower Cost. Less Carbon.</div>
          <div class="sub">Precision scheduling aligned to carbon intensity and spot market efficiency.</div>
          <div class="stats">
            <div class="stat"><div class="label">Financial<br>Savings</div><div class="value">£{{saved_money}}</div></div>
            <div class="stat"><div class="label">CO₂<br>Reduction</div><div class="value">{{reduced_co2}} kg</div></div>
            <div class="stat"><div class="label">Optimisation<br>Score</div><div class="value">{{score}}/1.0</div></div>
          </div>
        </div>
        <div class="hero-right">
          <img crossorigin="anonymous" referrerpolicy="no-referrer" src="https://images.unsplash.com/photo-1518770660439-4636190af475?w=800&h=600&fit=crop&auto=format" alt="Data Center">
          <div class="meta-card">
            <strong style="font-size:3.8mm;">Deployment Meta</strong><br>
            Company: {{company}} • Region: {{region}} • Latency: {{latency}} ms<br>
            Workload: {{workload}} • GPU Hours: {{gpu_hours}} • CI: {{ci}} gCO₂e/kWh<br>
            Last Updated: {{date}}
          </div>
        </div>
      </div>
      <div class="features">
        <div class="feature-card">
          <h3>Carbon-Aware Orchestration</h3>
          <p>Dynamically shifts workloads to lower carbon regions and optimises for spot pricing.</p>
        </div>
        <div class="feature-card">
          <h3>Operator Experience</h3>
          <p>Visual scheduling, proactive insights and real-time alerts.</p>
        </div>
      </div>
    </div>
    <div class="footer" aria-hidden="true">
      <div class="brand">
        <div style="font-weight:700;">CARBONSIGHT SCHEDULER</div>
        <div style="font-size:3.2mm;color:var(--muted);margin-top:4px">Smarter AI. Lower Cost. Less Carbon.</div>
      </div>
      <div><h5>Products</h5><ul><li>Scheduler</li><li>Carbon Intelligence</li><li>Cost Insights</li></ul></div>
      <div><h5>Company</h5><ul><li>About</li><li>Blog</li><li>Careers</li></ul></div>
      <div><h5>Connect</h5><div class="social"><div class="dot">Instagram</div><div class="dot">LinkedIn</div><div class="dot">Github</div></div></div>
    </div>
  </div>
</div>
<!-- PAGE 2 -->
<div class="page">
  <div class="content">
    <div class="content-inner">
      <div style="text-align:center;margin-bottom:8mm;">
        <h2 style="font-size:6.5mm;color:var(--accent);font-weight:700;margin:0;">Sustainability Impact Report</h2>
      </div>
      <div class="chart-and-summary">
        <div class="chart-column">
          <div class="chart-title">Run Comparison: Savings & Emissions</div>
          {{chart_image}}
        </div>
        <div class="summary-column">
          <div class="ai-summary-full">
            <strong>Sustainability Impact Summary</strong>
            {{summary}}
          </div>
        </div>
      </div>
    </div>
    <div class="footer">
      <div class="brand">
        <div style="font-weight:700;">CARBONSIGHT SCHEDULER</div>
        <div style="font-size:3.2mm;color:var(--muted);margin-top:4px">Smarter AI. Lower Cost. Less Carbon.</div>
      </div>
      <div><h5>Products</h5><ul><li>Scheduler</li><li>Carbon Intelligence</li><li>Cost Insights</li></ul></div>
      <div><h5>Company</h5><ul><li>About</li><li>Blog</li><li>Careers</li></ul></div>
      <div><h5>Connect</h5><div class="social"><div class="dot">Instagram</div><div class="dot">LinkedIn</div><div class="dot">Github</div></div></div>
      <div class="copyright">© 2025 CarbonSight Scheduler | Terms | Privacy | Cookies</div>
    </div>
  </div>
</div>
</body>
</html>"""

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
