from __future__ import annotations
import os
import time
import uuid
from functools import wraps
from typing import Dict, Optional
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import boto3
from datetime import datetime, timedelta
import json
import requests
from math import radians, sin, cos, sqrt, atan2

# Assuming utils.helpers are modified or stubbed. I'll provide a sample implementation for compute_metrics and others.
# For completeness, I'll define stub functions here. In practice, move them to utils/helpers.py.

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in km
    lat1_rad, lon1_rad = radians(lat1), radians(lon1)
    lat2_rad, lon2_rad = radians(lat2), radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

# Sample regions dict with lat, lon, location for pricing
regions = {
    "us-east-1": {"lat": 38.9940541, "long": -77.4524237, "location": "US East (N. Virginia)"},
    "us-east-2": {"lat": 40.0946354, "long": -82.7541337, "location": "US East (Ohio)"},
    "us-west-1": {"lat": 37.443680, "long": -122.153664, "location": "US West (N. California)"},
    "us-west-2": {"lat": 45.9174667, "long": -119.2684488, "location": "US West (Oregon)"},
    "eu-west-1": {"lat": 53.4056545, "long": -6.224503, "location": "EU (Ireland)"},
    "eu-west-2": {"lat": 51.5085036, "long": -0.0609266, "location": "EU (London)"},
    "eu-west-3": {"lat": 48.6009709, "long": 2.2976644, "location": "EU (Paris)"},
    "eu-central-1": {"lat": 50.0992094, "long": 8.6303932, "location": "EU (Frankfurt)"},
    "sa-east-1": {"lat": -23.4925798, "long": -46.8105593, "location": "South America (Sao Paulo)"},
    "ap-southeast-1": {"lat": 1.3218269, "long": 103.6930643, "location": "Asia Pacific (Singapore)"},
    "ap-southeast-2": {"lat": -33.9117717, "long": 151.1907535, "location": "Asia Pacific (Sydney)"},
    "ap-northeast-1": {"lat": 35.617436, "long": 139.7459176, "location": "Asia Pacific (Tokyo)"},
    "ap-northeast-2": {"lat": 37.5616592, "long": 126.8736237, "location": "Asia Pacific (Seoul)"},
    "ap-south-1": {"lat": 19.2425503, "long": 72.9667878, "location": "Asia Pacific (Mumbai)"},
    "ca-central-1": {"lat": 45.5, "long": -73.6, "location": "Canada (Central)"},
    "af-south-1": {"lat": -33.914651, "long": 18.3758801, "location": "Africa (Cape Town)"},
    "eu-north-1": {"lat": 59.326242, "long": 17.8419717, "location": "EU (Stockholm)"},
    "eu-south-1": {"lat": 45.4628328, "long": 9.1076927, "location": "EU (Milan)"},
    "me-south-1": {"lat": 25.941298, "long": 50.3073907, "location": "Middle East (Bahrain)"},
    "ap-east-1": {"lat": 22.2908475, "long": 114.2723379, "location": "Asia Pacific (Hong Kong)"},
    "cn-north-1": {"lat": 39.8094478, "long": 116.5783234, "location": "China (Beijing)"},
    "cn-northwest-1": {"lat": 37.5024418, "long": 105.1627193, "location": "China (Ningxia)"},
    "eu-north-1": {"lat": 59.326242, "long": 17.8419717, "location": "EU (Stockholm)"},
    # Add more regions as needed
}

# Sample workload to instance_type
workload_to_instance = {
    "inference": "g4dn.xlarge",
    "training": "p3.2xlarge",
    # Add more
}

# Instance power consumption estimate in kWh per hour
instance_powers = {
    "g4dn.xlarge": 0.2,
    "p3.2xlarge": 0.4,
}

# Baseline carbon intensity for reduction calculation (gCO2e/kWh)
baseline_intensity = 500.0

# Default user location for latency (New York)
user_lat = 40.7128
user_lon = -74.0060

def compute_metrics(company: str, workload: str, priorities: Dict[str, float], gpu_hours: float, region: str) -> Dict:
    if region not in regions:
        raise ValueError(f"Unknown region: {region}")
    
    region_info = regions[region]
    lat = region_info["lat"]
    lon = region_info["long"]
    pricing_location = region_info["location"]
    
    # Fetch carbon intensity
        # Fetch carbon intensity from ElectricityMaps (free tier)
    token = os.getenv('ELECTRICITYMAPS_TOKEN')
    if not token:
        raise ValueError("ELECTRICITYMAPS_TOKEN not set")
    
    url = f"https://api-access.electricitymaps.com/free-tier/carbon-intensity/latest"
    headers = {"auth-token": token}
    params = {"lat": lat, "lon": lon}
    
    resp = requests.get(url, headers=headers, params=params, timeout=10)
    if not resp.ok:
        # Fallback: use zone-based lookup if lat/lon fails
        zone_url = "https://api-access.electricitymaps.com/free-tier/zones"
        zone_resp = requests.get(zone_url, headers=headers, timeout=10)
        if zone_resp.ok:
            zones = zone_resp.json()
            # Simple nearest zone fallback (you can improve this)
            fallback_zone = "SE"  # Sweden for eu-north-1
            intensity_url = f"https://api-access.electricitymaps.com/free-tier/carbon-intensity/latest?zone={fallback_zone}"
            resp = requests.get(intensity_url, headers=headers, timeout=10)
            if not resp.ok:
                raise ValueError(f"Failed to fetch carbon intensity: {resp.text}")
            data = resp.json()
            carbon_intensity = float(data['carbonIntensity'])
            last_updated = data['datetime']
        else:
            raise ValueError(f"Failed to fetch carbon intensity: {resp.text}")
    else:
        data = resp.json()
        carbon_intensity = float(data['carbonIntensity'])
        last_updated = data['datetime']
    
    # Get instance_type and power
    instance_type = workload_to_instance.get(workload, "g4dn.xlarge")
    power_kwh_per_hour = instance_powers.get(instance_type, 0.2)
    
    # Get on-demand price
    pricing = boto3.client('pricing', region_name='us-east-1')
    response = pricing.get_products(
        ServiceCode='AmazonEC2',
        Filters=[
            {'Type': 'TERM_MATCH', 'Field': 'tenancy', 'Value': 'shared'},
            {'Type': 'TERM_MATCH', 'Field': 'operatingSystem', 'Value': 'Linux'},
            {'Type': 'TERM_MATCH', 'Field': 'preInstalledSw', 'Value': 'NA'},
            {'Type': 'TERM_MATCH', 'Field': 'instanceType', 'Value': instance_type},
            {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': pricing_location},
            {'Type': 'TERM_MATCH', 'Field': 'capacitystatus', 'Value': 'Used'},
        ],
        MaxResults=1
    )
    if not response['PriceList']:
        raise ValueError("No on-demand price found")
    prod = json.loads(response['PriceList'][0])
    on_demand_data = prod['terms']['OnDemand']
    price_id = list(on_demand_data.keys())[0]
    dimension_id = list(on_demand_data[price_id]['priceDimensions'].keys())[0]
    on_demand_price = float(on_demand_data[price_id]['priceDimensions'][dimension_id]['pricePerUnit']['USD'])
    
    # Get spot price (latest)
    ec2 = boto3.client('ec2', region_name=region)
    end = datetime.utcnow()
    start = end - timedelta(hours=1)
    response = ec2.describe_spot_price_history(
        InstanceTypes=[instance_type],
        ProductDescriptions=['Linux/UNIX'],
        StartTime=start,
        EndTime=end,
        MaxResults=100
    )
    history = response['SpotPriceHistory']
    if not history:
        raise ValueError("No spot price history found")
    history.sort(key=lambda x: x['Timestamp'], reverse=True)
    spot_price = float(history[0]['SpotPrice'])
    
    # Calculate latency
    distance = haversine(user_lat, user_lon, lat, lon)
    latency_ms = (distance / 100) * 1.5  # Approximate RTT in ms
    
    # Calculations
    saved_money = (on_demand_price - spot_price) * gpu_hours
    emissions_g = carbon_intensity * power_kwh_per_hour * gpu_hours
    emissions_kg = emissions_g / 1000
    reduced_emissions_g = (baseline_intensity - carbon_intensity) * power_kwh_per_hour * gpu_hours
    reduced_emissions_kg = reduced_emissions_g / 1000
    
    # Normalize for score (assumed max values)
    max_saved_per_hour = 1.0  # Example $
    max_reduced_per_hour = 300.0  # Example gCO2/h
    max_latency = 200.0  # ms
    normalized_saved = (on_demand_price - spot_price) / max_saved_per_hour
    normalized_reduced = (baseline_intensity - carbon_intensity) / max_reduced_per_hour
    normalized_speed = 1 - (latency_ms / max_latency)
    
    total_weight = sum(priorities.values())
    w_cost = priorities.get('cost', 0) / total_weight if total_weight else 0.333
    w_carbon = priorities.get('carbon', 0) / total_weight if total_weight else 0.333
    w_speed = priorities.get('speed', 0) / total_weight if total_weight else 0.333
    
    score = w_cost * normalized_saved + w_carbon * normalized_reduced + w_speed * normalized_speed
    
    # Latency penalty
    latency_penalty = latency_ms / max_latency
    score -= latency_penalty * 0.1  # Penalty for high latency
    
    metrics = {
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
        "emissions_kg_co2": emissions_kg,
        "reduced_emissions_kg_co2": reduced_emissions_kg,
        "latency_ms": latency_ms,
        "score": score
    }
    return metrics

# Stub for other helpers (implement as needed)
def validate_input(data):
    required = ["company_name", "workload_type", "priorities", "gpu_hours", "cloud_region"]
    missing = [r for r in required if r not in data]
    if missing:
        return f"Missing fields: {', '.join(missing)}", None

    priorities = data.get("priorities", {})
    if not isinstance(priorities, dict):
        return "Invalid priorities: must be an object", None
    expected_keys = {"cost", "carbon", "speed"}
    if not expected_keys.issubset(priorities.keys()):
        return f"Priorities must include: {', '.join(expected_keys)}", None

    # Optional: validate types
    try:
        float(data["gpu_hours"])
        for v in priorities.values():
            float(v)
    except (ValueError, TypeError):
        return "gpu_hours and priorities must be numbers", None

    # user_email is OPTIONAL → allowed but not required
    return None, data

def ensure_artifacts_dir():
    dir = "artifacts"
    os.makedirs(dir, exist_ok=True)
    return dir

# =========================
# ENHANCED PDF + CHART (HTML/CSS + PLAYWRIGHT)
# Replaces old ReportLab + matplotlib
# =========================
import base64
import shutil
import textwrap

def create_chart(gpu_hours, metrics) -> str:
    """Generate chart → return base64 string"""
    import matplotlib.pyplot as plt
    import io

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Saved Money (£)", "CO₂ Reduced (kg)"]
    values = [metrics["saved_money"], metrics["reduced_emissions_kg_co2"]]
    colors = ['#7BE200', '#00C2FF']
    ax.bar(labels, values, color=colors)
    ax.set_title("Optimization Impact", color='white', fontsize=14)
    ax.set_ylabel("Value", color='white')
    ax.tick_params(colors='white')
    fig.patch.set_facecolor('#0b0b0b')
    ax.set_facecolor('#0b0b0b')
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', facecolor='#0b0b0b', dpi=150)
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

def _html_escape(s: str) -> str:
    return (str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#x27;"))

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

def _build_pdf_html(metrics: Dict, chart_base64: Optional[str], summary: Optional[str]) -> Dict[str, str]:
    company = _html_escape(metrics.get('company_name', ''))
    saved_money = f"{metrics.get('saved_money', 0.0):.2f}"
    co2_reduced = f"{metrics.get('reduced_emissions_kg_co2', 0.0):.2f}"
    latency_ms = f"{metrics.get('latency_ms', 0.0):.0f}"
    score_val = f"{metrics.get('score', 0.0):.2f}"
    region_code = metrics.get('cloud_region', '')
    region_loc = regions.get(region_code, {}).get('location', region_code)
    workload = _html_escape(str(metrics.get('workload_type', '')).title())
    gpu_hours = _html_escape(str(metrics.get('gpu_hours', '')))
    carbon_intensity = f"{metrics.get('carbon_intensity_gco2_kwh', 0.0):.1f}"
    last_updated = _html_escape(metrics.get('last_updated', ''))

    summary_html = _html_escape(summary or "").replace("\n", "<br/>")

    chart_img = f'<img class="chart-img" src="data:image/png;base64,{chart_base64}" alt="Chart"/>' if chart_base64 else ""

    css = textwrap.dedent(""" 
        @page { size: 297mm 210mm; margin: 0; }
        :root { --bg: #0b0b0b; --glass: rgba(255,255,255,0.02); --glass-border: rgba(255,255,255,0.18); --text: #ffffff; --muted: #bdbdbd; --accent: #7BE200; --footer: #0e0e0e; --divider: rgba(255,255,255,0.06); }
        * { box-sizing: border-box; -webkit-font-smoothing: antialiased; }
        html, body { background: var(--bg); color: var(--text); font-family: 'Space Grotesk', system-ui; width: 297mm; height: 210mm; overflow: hidden; }
        .page { position: relative; width: 297mm; height: 210mm; display: flex; flex-direction: column; }
        .container { padding: 16mm 18mm 0 18mm; flex: 1; }
        .nav { display: flex; justify-content: space-between; margin-bottom: 10mm; }
        .logo { display: flex; align-items: center; gap: 6mm; }
        .logo .mark { width: 12mm; height: 12mm; border-radius: 3mm; background: linear-gradient(180deg, rgba(255,255,255,0.18), rgba(255,255,255,0.08)); border: 1px solid var(--glass-border); display: grid; place-items: center; }
        .logo .mark:before { content: "★"; color: #fff; font-size: 6mm; }
        .logo .text { font-weight: 700; font-size: 5mm; }
        .nav-menu { color: var(--muted); font-size: 3.5mm; }
        .hero { display: grid; grid-template-columns: 1.2fr 1fr; gap: 10mm; }
        .glass { background: var(--glass); border: 1px solid var(--glass-border); border-radius: 6mm; box-shadow: 0 8px 40px rgba(0,0,0,0.45); }
        .hero-left { padding: 12mm; display: flex; flex-direction: column; gap: 6mm; }
        .headline { font-size: 18mm; font-weight: 700; line-height: 1.05; }
        .sub { font-size: 4.8mm; color: var(--muted); }
        .stats { display: grid; grid-template-columns: repeat(3,1fr); gap: 6mm; }
        .stat { padding: 6mm; border-radius: 4mm; border: 1px solid var(--divider); background: rgba(255,255,255,0.02); }
        .stat .label { color: var(--muted); font-size: 3.5mm; }
        .stat .value { font-size: 7mm; font-weight: 600; color: var(--accent); }
        .hero-right { position: relative; }
        .hero-img { width: 100%; height: 100%; border-radius: 4.5mm; object-fit: cover; }
        .overlap-card { position: absolute; left: -12%; top: 12%; width: 60%; z-index: 2; padding: 6mm; border-radius: 4mm; background: var(--glass); border: 1px solid var(--glass-border); }
        .overlap-card h4 { margin: 0 0 2mm 0; font-size: 5mm; }
        .meta { display: flex; gap: 6mm; color: var(--muted); font-size: 3.5mm; }
        .features { margin-top: 10mm; display: grid; grid-template-columns: 1fr 1fr; gap: 8mm; }
        .feature { padding: 8mm; }
        .feature h3 { font-size: 6mm; margin: 0 0 2mm 0; }
        .feature p { color: var(--muted); font-size: 4mm; }
        .feature .feature-img { width: 100%; height: 45mm; border-radius: 4mm; object-fit: cover; margin-top: 4mm; }
        .contact { position: absolute; right: 18mm; top: 20mm; padding: 5mm 6mm; border-radius: 3.5mm; border: 1px solid var(--glass-border); background: var(--glass); font-size: 3.7mm; }
        .divider { height: 1px; background: var(--divider); margin: 8mm 0; width: calc(100% - 36mm); margin-left: 18mm; }
        .chart-wrap { margin-top: 8mm; padding: 0 18mm; }
        .chart-title { font-size: 5mm; color: var(--muted); }
        .chart-img { width: 120mm; height: auto; border-radius: 3mm; border: 1px solid var(--divider); }
        .footer { background: var(--footer); margin-top: 8mm; padding: 8mm 18mm; border-top: 1px solid var(--divider); display: grid; grid-template-columns: 1.2fr 1fr 1fr 1fr; gap: 10mm; }
        .footer .brand { display: flex; gap: 5mm; align-items: center; }
        .footer .brand .foot-mark { width: 10mm; height: 10mm; border-radius: 3mm; background: linear-gradient(180deg, rgba(255,255,255,0.18), rgba(255,255,255,0.08)); border: 1px solid var(--glass-border); display: grid; place-items: center; }
        .footer .brand .foot-mark:before { content: "★"; color: #fff; font-size: 5mm; }
        .footer h5 { font-size: 4.2mm; color: #fff; }
        .footer li { color: var(--muted); font-size: 3.7mm; }
        .footer .social .dot { width: 7mm; height: 7mm; border-radius: 50%; background: var(--bg); border: 1px solid var(--divider); display: grid; place-items: center; font-size: 3.5mm; }
        .copyright { color: var(--muted); font-size: 3.5mm; text-align: center; padding: 4mm 0; border-top: 1px solid var(--divider); width: calc(100% - 36mm); margin-left: 18mm; }
        .ai-summary { margin-top: 6mm; padding: 6mm; border: 1px solid var(--divider); border-radius: 4mm; color: var(--muted); font-size: 3.8mm; line-height: 1.5; background: rgba(255,255,255,0.02); }
    """).strip()

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>CarbonSight Report</title>
<style>@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');{css}</style>
</head><body><div class="page"><div class="container">
<div class="nav"><div class="logo"><div class="mark"></div><div class="text">CARBONSIGHT SCHEDULER</div></div><div class="nav-menu">Solutions | About Us | Blog | Support</div></div>
<div class="hero"><div class="glass hero-left">
<div class="headline">Smarter AI. Lower Cost. Less Carbon.</div>
<div class="sub">Precision scheduling aligned to carbon intensity and spot market efficiency.</div>
<div class="stats">
<div class="stat"><div class="label">Financial Savings</div><div class="value">£{saved_money}</div></div>
<div class="stat"><div class="label">CO₂ Reduction</div><div class="value">{co2_reduced} kg</div></div>
<div class="stat"><div class="label">Optimisation Score</div><div class="value">{score_val}/1.0</div></div>
</div><div class="ai-summary">{summary_html}</div>
</div><div class="hero-right"><img class="hero-img" src="assets/hero-datacenter.jpg" alt="Data Center"/>
<div class="overlap-card"><h4>Deployment Meta</h4>
<div class="meta"><div>Company: {company}</div><div>Region: {_html_escape(region_loc)}</div><div>Latency: {latency_ms} ms</div></div>
<div class="meta" style="margin-top:3mm;"><div>Workload: {workload}</div><div>GPU Hours: {gpu_hours}</div><div>CI: {carbon_intensity} gCO₂e/kWh</div></div>
<div class="meta" style="margin-top:3mm;"><div>Last Updated: {last_updated}</div></div>
</div></div></div>
<div class="features"><div class="glass feature"><h3>Carbon-Aware Orchestration</h3><p>Dynamically shifts workloads to lower carbon regions.</p></div>
<div class="glass feature"><h3>Operator Experience</h3><p>Visual scheduling and real-time alerts.</p><img class="feature-img" src="assets/laptop-dark-ui.jpg" alt="UI"/></div></div>
<div class="contact">Contact delivery team</div></div>
<div class="divider"></div>
<div class="chart-wrap"><div class="chart-title">Run Comparison</div>{chart_img}</div>
<div class="footer"><div class="brand"><div class="foot-mark"></div><div><div style="font-weight:700;">CARBONSIGHT SCHEDULER</div><div style="color:var(--muted);font-size:3.7mm;">Smarter AI. Lower Cost. Less Carbon.</div></div></div>
<div><h5>Products</h5><ul><li>Scheduler</li><li>Carbon Intelligence</li></ul></div>
<div><h5>Company</h5><ul><li>About</li><li>Blog</li></ul></div>
<div><h5>Connect</h5><div class="social"><div class="dot">in</div><div class="dot">x</div></div></div></div>
<div class="copyright">© 2025 CarbonSight Scheduler | Terms | Privacy | Cookies</div></div></body></html>""".strip()

    return {"html": html, "css": css}

def _render_html_to_pdf(html_path: str, pdf_path: str):
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"]
            )
            page = browser.new_page()
            page.goto(f"file://{html_path}", wait_until="load")
            page.pdf(path=pdf_path, format="A4", landscape=True, print_background=True, prefer_css_page_size=True, margin=0)
            browser.close()
    except Exception:
        try:
            from weasyprint import HTML
            HTML(filename=html_path).write_pdf(pdf_path)
        except Exception:
            from reportlab.lib.pagesizes import landscape, A4
            from reportlab.pdfgen import canvas
            c = canvas.Canvas(pdf_path, pagesize=landscape(A4))
            c.setFont("Helvetica", 12)
            c.drawString(50, 550, "PDF generation failed. Install Playwright or WeasyPrint.")
            c.save()

def create_pdf(title: str, metrics: Dict, chart_base64: str, output_pdf_path: str, summary: Optional[str] = None) -> None:
    built = _build_pdf_html(metrics, chart_base64, summary)
    html_str = built["html"]

    artifacts_dir = ensure_artifacts_dir()
    run_id = uuid.uuid4().hex[:10]
    workdir = os.path.join(artifacts_dir, f"report_{run_id}")
    assets_dir = os.path.join(workdir, "assets")
    _ensure_dir(workdir); _ensure_dir(assets_dir)

    project_assets = os.path.join(os.getcwd(), "assets")
    _safe_copy(os.path.join(project_assets, "hero-datacenter.jpg"), os.path.join(assets_dir, "hero-datacenter.jpg"))
    _safe_copy(os.path.join(project_assets, "laptop-dark-ui.jpg"), os.path.join(assets_dir, "laptop-dark-ui.jpg"))

    html_path = os.path.join(workdir, "index.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_str)

    _render_html_to_pdf(html_path, output_pdf_path)


def s3_upload(file_path, key, bucket, region):
    # Stub: Upload to S3
    s3 = boto3.client('s3', region_name=region)
    s3.upload_file(file_path, bucket, key)
    return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"

# Rest of the original code
load_dotenv()
APP_VERSION = "1.0.0"
app = Flask(__name__)
# CORS configuration
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000, https://*.bubbleapps.io")
allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
CORS(app, resources={r"/*": {"origins": allowed_origins}}, supports_credentials=False)
# In-memory rate limiting (per-IP)
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))
_requests: Dict[str, int] = {}
def rate_limiter(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        ip = request.headers.get("X-Forwarded-For", request.remote_addr) or "unknown"
        now = int(time.time())
        window = now // 60
        key = f"{ip}:{window}"
        count = _requests.get(key, 0) + 1
        _requests[key] = count
        if count == 1:
            _requests.pop(f"{ip}:{window-1}", None)
        if count > RATE_LIMIT:
            return jsonify({"ok": False, "error": "rate_limited", "detail": "Too many requests"}), 429
        return func(*args, **kwargs)
    return wrapper
def require_api_key(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        required = os.getenv("API_KEY")
        if not required:
            return jsonify({"ok": False, "error": "server_misconfig", "detail": "API_KEY not set"}), 500
        provided = request.headers.get("X-API-KEY")
        if provided != required:
            return jsonify({"ok": False, "error": "unauthorized", "detail": "Invalid API key"}), 401
        return func(*args, **kwargs)
    return wrapper
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
        if data is None:
            return jsonify({"ok": False, "error": "invalid_json", "detail": "Expected JSON body"}), 400
        err, clean = validate_input(data)
        if err:
            return jsonify({"ok": False, "error": "bad_request", "detail": err}), 400
        # NEW: Extract user email
        user_email = clean.get("user_email", "tshepotrustin@outlook.com")
        company = clean["company_name"]
        workload = clean["workload_type"]
        priorities = clean["priorities"]
        gpu_hours = clean["gpu_hours"]
        region = clean["cloud_region"]
        metrics = compute_metrics(company, workload, priorities, gpu_hours, region)

        # AI-Powered Formal Report using OpenAI GPT-4o-mini (FREE TRIAL + DEBUG)
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        if OPENAI_API_KEY:
            try:
                import requests
                print(f"[DEBUG] OPENAI_API_KEY found: {OPENAI_API_KEY[:8]}...")

                prompt = f"""
                Write a professional, detailed AI optimization report (200-350 words) for {metrics['company_name']}.
                Use formal British business language. Include these key results:
                - Financial savings: £{metrics['saved_money']:.2f} (Spot vs On-Demand)
                - CO₂ reduction: {metrics['reduced_emissions_kg_co2']:.2f} kg
                - Carbon intensity: {metrics['carbon_intensity_gco2_kwh']:.1f} gCO₂e/kWh
                - Latency: {metrics['latency_ms']:.0f} ms
                - Optimization score: {metrics['score']:.2f}/1.0
                - Region: {regions[metrics['cloud_region']]['location']}
                - Workload: {metrics['workload_type'].title()}
                - GPU hours: {metrics['gpu_hours']}

                Structure:
                1. Executive Summary
                2. Financial Impact
                3. Environmental Impact
                4. Performance & Latency
                5. Recommendations
                End with a positive, actionable closing.
                """

                print(f"[DEBUG] Sending prompt to OpenAI (first 200 chars): {prompt[:200]}...")

                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {OPENAI_API_KEY}"
                    },
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "system", "content": "You are a senior sustainability consultant writing formal reports for UK tech companies."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.6,
                        "max_tokens": 600,
                        "top_p": 0.95
                    },
                    timeout=20
                )

                print(f"[DEBUG] OpenAI HTTP Status: {response.status_code}")
                print(f"[DEBUG] OpenAI Response Body: {response.text[:500]}")

                if response.status_code == 200:
                    result = response.json()
                    print(f"[DEBUG] OpenAI response keys: {list(result.keys())}")

                    if "choices" in result and result["choices"]:
                        message = result["choices"][0]["message"]
                        if "content" in message:
                            summary = message["content"].strip()
                            print(f"[DEBUG] AI report generated successfully. Length: {len(summary)} chars")
                        else:
                            summary = "No content in OpenAI response."
                    else:
                        summary = f"No choices in response: {result}"
                else:
                    try:
                        error_data = response.json()
                        error_msg = error_data.get("error", {}).get("message", "Unknown error")
                    except:
                        error_msg = response.text[:200]
                    summary = f"AI report failed (HTTP {response.status_code}): {error_msg}\n\nFallback: Saved £{metrics['saved_money']:.2f}, reduced CO₂ by {metrics['reduced_emissions_kg_co2']:.2f} kg."
                    print(f"[DEBUG] AI report failed: {error_msg}")

            except Exception as e:
                summary = f"AI report exception: {str(e)}\n\nFallback: Saved £{metrics['saved_money']:.2f}, reduced CO₂ by {metrics['reduced_emissions_kg_co2']:.2f} kg."
                print(f"[DEBUG] Exception in OpenAI call: {str(e)}")
        else:
            summary = "OPENAI_API_KEY not set in environment. Using fallback."
            print("[DEBUG] OPENAI_API_KEY missing")


        # Prepare artifacts
        artifacts_dir = ensure_artifacts_dir()
        run_id = uuid.uuid4().hex[:12]
        chart_name = f"{run_id}_chart.png"
        pdf_name = f"{run_id}_report.pdf"
        pdf_path = os.path.join(artifacts_dir, pdf_name)
        chart_base64 = create_chart(gpu_hours, metrics)
        create_pdf("AI Report", metrics, chart_base64, pdf_path, summary=summary)
        
        # Optional S3 upload
        if os.getenv("USE_S3", "false").lower() == "true":
            bucket = os.getenv("S3_BUCKET")
            if not bucket:
                return jsonify({"ok": False, "error": "server_misconfig", "detail": "S3_BUCKET not set"}), 500
            region = os.getenv("AWS_REGION")
            chart_key = f"charts/{chart_name}"
            pdf_key = f"reports/{pdf_name}"
            chart_url = s3_upload(chart_path, chart_key, bucket, region)
            pdf_url = s3_upload(pdf_path, pdf_key, bucket, region)
        else:
            chart_url = f"/artifact/{chart_name}"
            pdf_url = f"/artifact/{pdf_name}"
        resp: Dict[str, Optional[str] | Dict[str, float]] = {
            "ok": True,
            "metrics": metrics,
            "summary": summary,
            "chart_url": chart_url,
            "pdf_url": pdf_url,
        }
        # Optional n8n webhook for email
        n8n = os.getenv("N8N_WEBHOOK_URL")
        if n8n:
            try:
                import requests
                requests.post(n8n, json={
                    "pdf_url": pdf_url,
                    "metrics": metrics,
                    "user_email": user_email
                }, timeout=5)
            except Exception:
                pass
        return jsonify(resp)
    except Exception as e:
        return jsonify({"ok": False, "error": "internal_error", "detail": str(e)}), 500
@app.get("/artifact/<path:filename>")
@rate_limiter
def artifact(filename: str):
    artifacts_dir = ensure_artifacts_dir()
    return send_from_directory(artifacts_dir, filename, as_attachment=False)
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
