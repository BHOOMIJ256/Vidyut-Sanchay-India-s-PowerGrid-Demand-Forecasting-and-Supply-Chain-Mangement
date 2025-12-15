import os
import json
import joblib
import pandas as pd
import sys
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List

# --- LANGCHAIN IMPORTS ---
from langchain_core.messages import HumanMessage

# --- 1. SAFE IMPORTS ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from price_agent.price_agent import PriceAgent
    print("✅ Price Agent Imported")
except ImportError:
    PriceAgent = None

try:
    from logistics_agent.logistics_agent import LogisticsAgent
    print("✅ Logistics Agent Imported")
except ImportError:
    LogisticsAgent = None

try:
    from risk_agent.risk_agent import chatbot as risk_bot
    print("✅ Risk Agent Imported")
except ImportError:
    risk_bot = None

# --- 2. SETUP ---
ML_MODEL_PATH = "full_ml_pipeline_1.pkl"
app = FastAPI(title="Vidyut Sanchay Orchestrator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

price_engine = PriceAgent(use_live_apis=True) if PriceAgent else None
logistics_engine = LogisticsAgent() if LogisticsAgent else None

try:
    ml_model = joblib.load(ML_MODEL_PATH)
    print("✅ ML Model loaded successfully.")
except Exception:
    ml_model = None

# --- 3. DATA MODELS ---
class ProjectInput(BaseModel):
    project_type: str
    region: str
    project_city: str
    soil_type: str
    terrain_type: str
    voltage_kv: int
    circuit_type: str
    conductor_type: str
    length_km: float
    num_towers: int

# --- 4. HELPERS ---
def extract_json_from_text(text: str):
    try:
        match = re.search(r"(\[.*\])", text, re.DOTALL)
        if match: return json.loads(match.group(1))
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match: return json.loads(match.group(1))
        return json.loads(text)
    except json.JSONDecodeError:
        return None

def map_supplier_to_logistics_key(price_agent_name: str) -> str:
    """
    Maps the Price Agent's supplier names to Logistics Agent's location keys.
    """
    mapping = {
        "SKIPPER": "SKIPPER_KOLKATA",
        "TATA_STEEL": "TATA_JAMSHEDPUR",
        "SAIL": "SAIL_BHILAI",
        "JSW_STEEL": "JSW_MUMBAI",
        "KEC_INTL": "KEC_NAGPUR",
        "GUPTA_PWR": "GUPTA_BHUBANESWAR"
    }
    # Default behavior: try to find a partial match or return as-is
    for key, value in mapping.items():
        if key in price_agent_name:
            return value
    return price_agent_name.replace(" ", "_").upper() + "_HQ"

def get_ml_prediction(input_data: ProjectInput) -> Dict:
    if ml_model:
        try:
            # Note: We strip whitespace to avoid ' Upgradation' errors
            p_type = input_data.project_type.strip()
            
            df = pd.DataFrame([{
                "project_type": p_type,
                "region": input_data.region,
                "soil_type": input_data.soil_type,
                "terrain_type": input_data.terrain_type,
                "voltage_kv": input_data.voltage_kv,
                "circuit_type": input_data.circuit_type,
                "conductor_type": input_data.conductor_type,
                "Length_km": input_data.length_km,
                "num_towers": input_data.num_towers
            }])
            raw = ml_model.predict(df)[0]
            return {
                "steel_tonnes": {"value": float(raw[0])},
                "conductor_km": {"value": float(raw[1])},
                "insulators_unit": {"value": float(raw[2])},
                "concrete_cubic_meter": {"value": float(raw[3])},
                "transformers_count": {"value": float(raw[5])}, 
                "circuit_breaker_count": {"value": float(raw[6])},
                "num_towers": {"value": input_data.num_towers}
            }
        except Exception as e:
            print(f"ML Error: {e}")
            
    return {
        "steel_tonnes": {"value": 5975.25},
        "conductor_km": {"value": 215.42},
        "insulators_unit": {"value": 5564.0},
        "concrete_cubic_meter": {"value": 3375.52},
        "transformers_count": {"value": 2.0},
        "circuit_breaker_count": {"value": 5.0},
        "num_towers": {"value": input_data.num_towers}
    }

# --- 5. ENDPOINTS ---

@app.get("/api/market-prices")
def get_live_prices():
    if price_engine: return price_engine.get_current_prices()
    return {"aluminum_price_per_tonne": 2500.0, "usd_to_inr": 84.0}

@app.post("/api/generate-plan")
def generate_procurement_plan(project: ProjectInput):
    results = {}
    
    # 1. ML
    print(f"\n1️⃣  Running ML Model for {project.project_city}...")
    quantities = get_ml_prediction(project)
    results["engineering_estimates"] = quantities

    # 2. PRICE
    print("2️⃣  Running Price Agent...")
    price_report = price_engine.calculate_project_cost(quantities) if price_engine else {"grand_total": 50000000}
    results["procurement"] = price_report
    
    steel_qty = quantities['steel_tonnes']['value']
    grand_total_str = f"₹ {price_report.get('grand_total', 0):,.2f}"

    # --- DYNAMIC SUPPLIER SELECTION ---
    # 1. Get the Winner from Price Agent
    winner_name = price_report.get('steel_supplier', 'Unknown')
    winner_key = map_supplier_to_logistics_key(winner_name)
    
    # 2. Add Competitors for comparison
    competitors = ["TATA_JAMSHEDPUR", "SAIL_BHILAI"]
    
    # 3. Create final list (Winner first, then competitors, removing duplicates)
    potential_suppliers = [winner_key] + [c for c in competitors if c != winner_key]
    
    destination = project.project_city.upper().replace(" ", "_") + "_SITE"
    
    print(f"3️⃣  Running Logistics for: {potential_suppliers}...")
    
    logistics_routes = []
    if logistics_engine:
        for supp in potential_suppliers:
            try:
                report = logistics_engine.calculate_delivery(supp, destination, steel_qty)
                logistics_routes.append(report)
            except Exception:
                # Fallback if key not in DB
                logistics_routes.append(_mock_logistics(supp, destination))
    else:
        for supp in potential_suppliers:
            logistics_routes.append(_mock_logistics(supp, destination))

    results["logistics"] = {"routes": logistics_routes}

    # 4. RISK (BATCH ANALYSIS)
    print(f"\n4️⃣  Running Risk Agent (Batch Analysis for {len(logistics_routes)} companies)...")
    
    routes_summary = json.dumps(logistics_routes, indent=2)
    
    risk_prompt = f"""
    You are the RISK SCORING AGENT.
    
    TASKS:
    Analyze the risk for these Logistics Routes:
    {routes_summary}
    
    Total Project Budget: {grand_total_str}
    
    CRITICAL:
    Return a JSON LIST of objects.
    Format:
    [
      {{ "company": "NAME", "risk_score": 3, "reason": "Short reason" }}
    ]
    """
    
    risk_results = []
    if risk_bot:
        try:
            response = risk_bot.invoke(
                {'messages': [HumanMessage(content=risk_prompt)]},
                config={"recursion_limit": 60}
            )
            ai_content = response['messages'][-1].content
            parsed_data = extract_json_from_text(ai_content)
            
            if isinstance(parsed_data, list):
                risk_results = parsed_data
            elif isinstance(parsed_data, dict):
                risk_results = [parsed_data]
            
        except Exception as e:
            print(f"⚠️ Risk Agent Inference Failed: {e}")
            risk_results = [_mock_risk(s) for s in potential_suppliers]
    else:
        risk_results = [_mock_risk(s) for s in potential_suppliers]

    results["risk_analysis"] = {"reports": risk_results}
    return results

def _mock_logistics(supp, dest):
    return {
        'origin_supplier': supp, 'destination_project': dest,
        'quantity_tonnes': 5000, 'distance_km': 1200,
        'transit_time_days': 4.5, 'est_arrival_date': '2025-12-20',
        'transport_cost_inr': 450000
    }

def _mock_risk(supp):
    return {"company": supp, "status": "LOW RISK", "reason": "Agent Offline", "risk_score": 2}