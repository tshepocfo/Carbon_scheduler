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

def create_chart(gpu_hours, metrics, chart_path):
    # Stub: Use matplotlib to create a bar chart
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.bar(["Saved Money", "Reduced Emissions"], [metrics["saved_money"], metrics["reduced_emissions_kg_co2"]])
    plt.savefig(chart_path)

def create_pdf(title: str, metrics: Dict, chart_path: str, output_pdf_path: str, summary: Optional[str] = None) -> None:
    """Create a beautiful PDF report."""
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    from reportlab.lib import colors

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=18, spaceAfter=20)
    heading_style = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=14, spaceAfter=10)
    body_style = styles['BodyText']
    
    story = []
    
    # Title
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 0.3 * inch))
    
    # Company and Details
    story.append(Paragraph(f"Company: {metrics['company_name']}", heading_style))
    story.append(Paragraph(f"Workload Type: {metrics['workload_type']}", body_style))
    story.append(Paragraph(f"Cloud Region: {metrics['cloud_region']} ({regions[metrics['cloud_region']]['location']})", body_style))
    story.append(Paragraph(f"GPU Hours: {metrics['gpu_hours']}", body_style))
    story.append(Spacer(1, 0.2 * inch))
    
    # Metrics Table
    data = [
        ["Metric", "Value"],
        ["Saved Money ($)", f"${metrics['saved_money']:.2f}"],
        ["Emissions (kg CO2)", f"{metrics['emissions_kg_co2']:.2f}"],
        ["Reduced Emissions (kg CO2)", f"{metrics['reduced_emissions_kg_co2']:.2f}"],
        ["Latency (ms)", f"{metrics['latency_ms']:.2f}"],
        ["Carbon Intensity (gCO2e/kWh)", f"{metrics['carbon_intensity_gco2_kwh']:.2f}"],
        ["Last Updated", metrics['last_updated']],
        ["Optimization Score", f"{metrics['score']:.2f}"],
    ]
    table = Table(data, colWidths=[3*inch, 3*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.3 * inch))
    
    # Chart
    if os.path.exists(chart_path):
        story.append(Paragraph("Visualization", heading_style))
        story.append(Image(chart_path, width=6 * inch, height=4 * inch))
        story.append(Spacer(1, 0.2 * inch))
    
    # Summary
    if summary:
        story.append(Paragraph("AI Summary", heading_style))
        story.append(Paragraph(summary, body_style))
    
    doc = SimpleDocTemplate(output_pdf_path, pagesize=LETTER)
    doc.build(story)


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

        # AI-Powered Formal Report using Google Gemini
                # AI-Powered Formal Report using Google Gemini (WITH FULL DEBUG LOGS)
        GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
        if GEMINI_API_KEY:
            try:
                import requests
                print(f"[DEBUG] GEMINI_API_KEY found: {GEMINI_API_KEY[:10]}...")  # DEBUG: Confirm key exists

                prompt = f"""
                Write a professional, detailed AI optimization report (200-350 words) for {metrics['company_name']}.
                Use formal business language. Include these key results:
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

                print(f"[DEBUG] Sending prompt to Gemini (first 200 chars): {prompt[:200]}...")  # DEBUG: Show prompt

                response = requests.post(
                    "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
                    headers={"Content-Type": "application/json"},
                    params={"key": GEMINI_API_KEY},
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "temperature": 0.6,
                            "maxOutputTokens": 1024,
                            "topP": 0.95
                        }
                    },
                    timeout=15
                )

                print(f"[DEBUG] Gemini HTTP Status: {response.status_code}")  # DEBUG: HTTP code
                print(f"[DEBUG] Gemini Response Body: {response.text[:500]}")  # DEBUG: First 500 chars of response

                if response.status_code == 200:
                    result = response.json()
                    summary = result["candidates"][0]["content"]["parts"][0]["text"]
                    summary = summary.replace("```markdown", "").replace("```", "").strip()
                    print(f"[DEBUG] AI report generated successfully. Length: {len(summary)} chars")  # DEBUG: Success
                else:
                    error_msg = response.json().get("error", {}).get("message", "Unknown error")
                    summary = f"AI report failed (HTTP {response.status_code}): {error_msg}\n\nFallback: Saved £{metrics['saved_money']:.2f}, reduced CO₂ by {metrics['reduced_emissions_kg_co2']:.2f} kg."
                    print(f"[DEBUG] AI report failed: {error_msg}")  # DEBUG: Failure

            except Exception as e:
                summary = f"AI report exception: {str(e)}\n\nFallback: Saved £{metrics['saved_money']:.2f}, reduced CO₂ by {metrics['reduced_emissions_kg_co2']:.2f} kg."
                print(f"[DEBUG] Exception in Gemini call: {str(e)}")  # DEBUG: Exception
        else:
            summary = "GEMINI_API_KEY not set in environment. Using fallback."
            print("[DEBUG] GEMINI_API_KEY missing")  # DEBUG: No key
        # Prepare artifacts
        artifacts_dir = ensure_artifacts_dir()
        run_id = uuid.uuid4().hex[:12]
        chart_name = f"{run_id}_chart.png"
        pdf_name = f"{run_id}_report.pdf"
        chart_path = os.path.join(artifacts_dir, chart_name)
        pdf_path = os.path.join(artifacts_dir, pdf_name)
        create_chart(gpu_hours, metrics, chart_path)
        create_pdf("AI Report", metrics, chart_path, pdf_path, summary=summary)
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
