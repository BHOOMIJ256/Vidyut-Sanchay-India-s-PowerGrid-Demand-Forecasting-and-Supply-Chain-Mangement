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
from dotenv import load_dotenv
from vidyut_brain import get_vidyut_agent

# --- 0. LOAD ENV & DEBUG ---
# Force load .env file
load_dotenv() 

print("\n🔍 --- STARTUP DIAGNOSTICS ---")
groq_key = os.getenv("GROQ_API_KEY")
alpha_key = os.getenv("ALPHA_VANTAGE_API_KEY")

print(f"🔑 GROQ_API_KEY Found? : {'✅ Yes' if groq_key else '❌ NO (Risk Agent will fail)'}")
print(f"🔑 ALPHAVANTAGE_KEY?   : {'✅ Yes' if alpha_key else '❌ NO (Price Agent will mock)'}")

# --- LANGCHAIN IMPORTS ---
from langchain_core.messages import HumanMessage, SystemMessage

# --- 1. SAFE IMPORTS ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from price_agent.price_agent import PriceAgent
    print("✅ Price Agent Imported")
except ImportError as e:
    print(f"❌ Price Agent Import Failed: {e}")
    PriceAgent = None

try:
    from logistics_agent.logistics_agent import LogisticsAgent
    print("✅ Logistics Agent Imported")
except ImportError as e:
    print(f"❌ Logistics Agent Import Failed: {e}")
    LogisticsAgent = None

try:
    from risk_agent.risk_agent import chatbot as risk_bot
    print("✅ Risk Agent Imported")
except ImportError as e:
    print(f"❌ Risk Agent Import Failed: {e}")
    risk_bot = None

try:
    vidyut_brain = get_vidyut_agent()
    print("🧠 Vidyut Brain Initialized")
except Exception as e:
    print(f"❌ Failed to load Vidyut Brain: {e}")
    vidyut_brain = None

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

# Initialize Engines
print("\n🤖 --- INITIALIZING AGENTS ---")
try:
    price_engine = PriceAgent(use_live_apis=True) if PriceAgent else None
    print("   -> Price Engine Ready")
except Exception as e:
    print(f"   -> ⚠️ Price Engine Error: {e}")
    price_engine = None

try:
    logistics_engine = LogisticsAgent() if LogisticsAgent else None
    print("   -> Logistics Engine Ready")
except Exception as e:
    print(f"   -> ⚠️ Logistics Engine Error: {e}")
    logistics_engine = None

try:
    ml_model = joblib.load(ML_MODEL_PATH)
    print("✅ ML Model loaded.")
except Exception:
    print("⚠️ ML Model Missing. Using fallbacks.")
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

class ChatRequest(BaseModel):
    message: str
    context: Dict[str, Any] | None = None

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
    mapping = {
        "SKIPPER": "SKIPPER_KOLKATA",
        "TATA_STEEL": "TATA_JAMSHEDPUR",
        "SAIL": "SAIL_BHILAI",
        "JSW_STEEL": "JSW_MUMBAI",
        "KEC_INTL": "KEC_NAGPUR",
        "GUPTA_PWR": "GUPTA_BHUBANESWAR"
    }
    for key, value in mapping.items():
        if key in price_agent_name: return value
    return price_agent_name.replace(" ", "_").upper() + "_HQ"

def get_ml_prediction(input_data: ProjectInput) -> Dict:
    if ml_model:
        try:
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
            print(f"⚠️ ML Prediction Failed: {e}")
            
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

# Paste this INSIDE backend/main.py, replacing the existing generate_procurement_plan function

@app.post("/api/generate-plan")
def generate_procurement_plan(project: ProjectInput):
    results = {}
    
    # === STEP 1: ML ===
    print(f"\n1️⃣  Running ML Model for {project.project_city}...")
    quantities = get_ml_prediction(project)
    results["engineering_estimates"] = quantities

    # === STEP 2: PRICE ===
    print("2️⃣  Running Price Agent...")
    if price_engine:
        try:
            price_report = price_engine.calculate_project_cost(quantities)
        except Exception as e:
            print(f"❌ Price Agent Failed: {e}")
            price_report = {"grand_total": 50000000, "steel_supplier": "ERROR_FALLBACK"}
    else:
        price_report = {"grand_total": 50000000, "steel_supplier": "AGENT_MISSING"}
    results["procurement"] = price_report
    
    steel_qty = quantities['steel_tonnes']['value']
    grand_total_str = f"₹ {price_report.get('grand_total', 0):,.2f}"

    # === STEP 3: LOGISTICS (THE FIX IS HERE) ===
    # We map supplier names to REAL ADDRESSES, not Database Keys
    city_mapping = {
        "SKIPPER": "Kolkata, India",
        "TATA": "Jamshedpur, India",
        "SAIL": "Bhilai, India",
        "JSW": "Mumbai, India",
        "GUPTA": "Bhubaneswar, India",
        "KEC": "Nagpur, India"
    }
    
    winner_name = price_report.get('steel_supplier', 'Unknown').upper()
    
    # Find the city for the winner (Default to Kolkata if not found)
    winner_loc = "Kolkata, India"
    for key, city in city_mapping.items():
        if key in winner_name:
            winner_loc = city
            break
            
    # Define Real Competitor Cities
    competitors = ["Jamshedpur, India", "Bhilai, India"]
    # Build list: Winner + Competitors (removing duplicates)
    potential_origins = [winner_loc] + [c for c in competitors if c != winner_loc]
    
    # Destination is also a Real Address
    dest_city = f"{project.project_city}, India"
    
    print(f"3️⃣  Running Logistics for: {potential_origins} -> {dest_city}...")
    
    logistics_routes = []
    
    if logistics_engine:
        for origin in potential_origins:
            try:
                # Passing 'origin' as the name ensures the agent Geocodes it!
                report = logistics_engine.calculate_delivery(
                    supplier_name=origin,       # e.g., "Kolkata, India"
                    project_site_key=dest_city, # e.g., "Mumbai, India"
                    quantity_tonnes=steel_qty
                )
                
                # Make the output name pretty (e.g., "Kolkata, India" -> "KOLKATA_SUPPLIER")
                clean_name = origin.split(",")[0].upper().strip() + "_SUPPLIER"
                report['origin_supplier'] = clean_name
                report['destination_project'] = project.project_city.upper() + "_SITE"
                
                logistics_routes.append(report)
            except Exception as e:
                print(f"   ⚠️ Route {origin}->{dest_city} Failed: {e}")
                # Fallback
                logistics_routes.append(_mock_logistics(origin, dest_city))
    else:
        for origin in potential_origins:
            logistics_routes.append(_mock_logistics(origin, dest_city))

    results["logistics"] = {"routes": logistics_routes}

    # === STEP 4: RISK ===
    print(f"\n4️⃣  Running Risk Agent...")
    risk_results = []
    
    if risk_bot:
        try:
            # We summarize the routes for the AI
            routes_summary = json.dumps(logistics_routes, indent=2)
            
            risk_prompt = f"""
            Analyze the Logistics Risks for these options:
            {routes_summary}
            
            Total Project Value: {grand_total_str}
            
            OUTPUT ONLY A JSON LIST: 
            [
              {{ "company": "NAME", "risk_score": 3, "reason": "Short reason" }}
            ]
            """
            
            response = risk_bot.invoke(
                {'messages': [HumanMessage(content=risk_prompt)]}, 
                config={"recursion_limit": 60}
            )
            
            # Extract JSON
            ai_content = response['messages'][-1].content
            parsed_data = extract_json_from_text(ai_content)
            
            if isinstance(parsed_data, list):
                risk_results = parsed_data
            elif isinstance(parsed_data, dict):
                risk_results = [parsed_data]
            else:
                risk_results = [] # Parsing failed
                
        except Exception as e:
            print(f"❌ Risk Agent Failed: {e}")
            risk_results = [_mock_risk(s) for s in potential_origins]
    else:
        risk_results = [_mock_risk(s) for s in potential_origins]

    results["risk_analysis"] = {"reports": risk_results}
    return results

from langchain_core.messages import SystemMessage, HumanMessage # Make sure these are imported!

# Define the Persona HERE instead
VIDYUT_SYSTEM_PROMPT = """
You are 'Vidyut Sanchay', an advanced AI Orchestrator for Power Grid Procurement.

YOUR CAPABILITIES:
1. You have access to real-time market prices (Tools).
2. You can calculate logistics routes and costs (Tools).
3. You can analyze risks and engineering estimates (Logic).

GUIDELINES:
- If the user asks about the "Current Plan" or "Dashboard", USE THE CONTEXT provided in the message history.
- If the user asks a general question (e.g., "Current copper price"), USE YOUR TOOLS.
- Be professional, concise, and data-driven.
"""

@app.post("/api/chat")
def chat_with_agent(request: ChatRequest):
    user_query = request.message
    plan_context = request.context
    
    # 1. Start with the Persona (System Message)
    messages = [SystemMessage(content=VIDYUT_SYSTEM_PROMPT)]
    
    # 2. Add Project Context (if available)
    if plan_context:
        context_str = json.dumps(plan_context, indent=2)
        messages.append(SystemMessage(content=f"ACTIVE PROJECT PLAN DATA:\n{context_str}"))
    
    # 3. Add User Question
    messages.append(HumanMessage(content=user_query))

    if vidyut_brain:
        try:
            # invoke the agent with the list of messages
            result = vidyut_brain.invoke(
                {"messages": messages},
                config={"recursion_limit": 20}
            )
            return {"response": result['messages'][-1].content}
        except Exception as e:
            print(f"Chat Error: {e}")
            return {"response": f"I encountered an error: {e}"}
    else:
        return {"response": "Vidyut Brain is offline."}

def _mock_logistics(supp, dest):
    return {'origin_supplier': supp, 'destination_project': dest, 'quantity_tonnes': 5000, 'distance_km': 1200, 'transit_time_days': 4.5, 'est_arrival_date': '2025-12-20', 'transport_cost_inr': 450000}

def _mock_risk(supp):
    return {"company": supp, "status": "LOW RISK", "reason": "Agent Error / Offline", "risk_score": 2}