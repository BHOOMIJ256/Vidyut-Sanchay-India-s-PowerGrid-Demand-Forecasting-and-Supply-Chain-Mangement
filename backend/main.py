import os
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional

# --- IMPORT AGENTS ---
from price_agent.price_agent import PriceAgent
from langchain_agent import agent as langchain_brain
# from risk_agent.risk_agent import RiskAgent       # Uncomment when ready
# from logistics_agent.logistics_agent import LogisticsAgent # Uncomment when ready

# --- CONFIG ---
# Ensure this file exists in your backend folder!
ML_MODEL_PATH = "./full_ml_pipeline_1.pkl" 

app = FastAPI(title="Vidyut Sanchay Orchestrator")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Logic Engine
# We use live APIs for the Price Agent
price_engine = PriceAgent(use_live_apis=True)

# --- LOAD ML MODEL ---
try:
    print("🔮 Loading ML Model from disk...")
    ml_model = joblib.load(ML_MODEL_PATH)
    print("✅ ML Model loaded successfully.")
except Exception as e:
    print(f"⚠️ WARNING: Could not load '{ML_MODEL_PATH}'. Check file path.")
    print(f"Error details: {e}")
    ml_model = None

# --- INPUT MODELS ---
class ProjectInput(BaseModel):
    project_type: str = "Transmission Line"
    region: str         # e.g., "South"
    state: str          # e.g., "Kerala" (Used for Logistics, not ML)
    soil_type: str      # e.g., "Rocky"
    terrain_type: str   # e.g., "Plains"
    voltage_kv: int     # e.g., 132
    circuit_type: str   # e.g., "Single Circuit"
    conductor_type: str # e.g., "ACSR Panther"
    length_km: float    # e.g., 71.79
    num_towers: int     # e.g., 282

class ChatRequest(BaseModel):
    message: str

# --- HELPER: ML ENGINE ---
def get_ml_prediction(input_data: ProjectInput) -> Dict:
    """
    Connects User Inputs -> Real ML Model -> Predicted Quantities
    """
    if ml_model is None:
        # Fallback if model failed to load
        print("⚠️ Using Simulation Fallback (Model not loaded)")
        return _simulate_prediction(input_data)

    try:
        # 1. Prepare Dataframe for Model
        # Note: We map Pydantic fields to the EXACT column names your model expects
        # We explicitly exclude 'state' because your model didn't use it in training
        input_df = pd.DataFrame([{
            "project_type": input_data.project_type,
            "region": input_data.region,
            "soil_type": input_data.soil_type,
            "terrain_type": input_data.terrain_type,
            "voltage_kv": input_data.voltage_kv,
            "circuit_type": input_data.circuit_type,
            "conductor_type": input_data.conductor_type,
            "Length_km": input_data.length_km,  # Note Capital 'L' based on your previous output
            "num_towers": input_data.num_towers
        }])

        # 2. Predict
        raw_pred = ml_model.predict(input_df)[0]

        # 3. Map Output Array to Dictionary
        # Mapping based on your Test.py output: 
        # [0]=Steel, [1]=Conductor, [2]=Insulators, [3]=Concrete, [4]=Reactor, [5]=Transformer, [6]=Breaker
        return {
            "steel_tonnes": {"value": float(raw_pred[0]), "unit": "tonnes"},
            "conductor_km": {"value": float(raw_pred[1]), "unit": "km"},
            "insulators_unit": {"value": float(raw_pred[2]), "unit": "units"},
            "concrete_cubic_meter": {"value": float(raw_pred[3]), "unit": "cubic_meter"},
            "bus_reactor_count": {"value": float(raw_pred[4]), "unit": "count"},
            "transformers_count": {"value": float(raw_pred[5]), "unit": "count"},
            "circuit_breaker_count": {"value": float(raw_pred[6]), "unit": "count"},
            # Pass through towers for Earthing calc in Price Agent
            "num_towers": {"value": input_data.num_towers, "unit": "count"} 
        }

    except Exception as e:
        print(f"❌ ML Prediction Error: {e}")
        # Fallback so the app doesn't crash during demo
        return _simulate_prediction(input_data)

def _simulate_prediction(input_data: ProjectInput) -> Dict:
    """Fallback simulation if model fails"""
    return {
        "steel_tonnes": {"value": 5975.25, "unit": "tonnes"},
        "conductor_km": {"value": 215.42, "unit": "km"},
        "insulators_unit": {"value": 5564.0, "unit": "units"},
        "concrete_cubic_meter": {"value": 3375.52, "unit": "cubic_meter"},
        "transformers_count": {"value": 2.0, "unit": "count"},
        "circuit_breaker_count": {"value": 5.0, "unit": "count"},
        "bus_reactor_count": {"value": 1.0, "unit": "count"},
        "num_towers": {"value": input_data.num_towers, "unit": "count"}
    }

# --- ENDPOINTS ---

@app.get("/")
def home():
    return {"status": "Vidyut Sanchay System Online"}

@app.get("/api/market-prices")
def get_live_prices():
    """Ticker Tape Data"""
    try:
        return price_engine.get_current_prices()
    except Exception as e:
        # Return fallback if API fails
        return {"error": str(e), "usd_to_inr": 84.0}

@app.post("/api/generate-plan")
def generate_procurement_plan(project: ProjectInput):
    """
    THE MASTER FLOW:
    User Input -> ML Model -> Price Agent -> (Risk/Logistics) -> Final Report
    """
    results = {}
    
    # 1. ML PREDICTION (Real Model)
    print(f"📥 Received Project: {project.length_km}km in {project.state}")
    quantities = get_ml_prediction(project)
    results["engineering_estimates"] = quantities

    # 2. PRICE AGENT (Real Logic)
    print("💰 Calling Price Agent...")
    try:
        price_report = price_engine.calculate_project_cost(quantities)
        results["procurement"] = price_report
    except Exception as e:
        print(f"❌ Price Agent Error: {e}")
        price_report = {} # Handle gracefully

    # Extract Winners for Next Steps
    steel_winner = price_report.get('steel_supplier', 'Unknown')
    cond_winner = price_report.get('conductor_supplier', 'Unknown')

    # 3. RISK AGENT (Mocked - Connect Friend's Code Here)
    # TODO: Import RiskAgent and call risk_agent.analyze(steel_winner)
    print(f"⛈️ Checking Risk for {steel_winner}...")
    results["risk_analysis"] = {
        "steel_supplier": {
            "name": steel_winner, 
            "status": "LOW RISK", 
            "alert": "None",
            "details": "No active strikes or financial alerts found."
        },
        "conductor_supplier": {
            "name": cond_winner, 
            "status": "MEDIUM RISK", 
            "alert": "⚠️ Moderate Rain Forecast",
            "details": "Weather warning in region for next 3 days."
        }
    }

    # 4. LOGISTICS AGENT (Mocked - Connect Friend's Code Here)
    # TODO: Import LogisticsAgent and call logistics_agent.calculate(steel_winner, project.state)
    print(f"🚚 Calculating Logistics to {project.state}...")
    results["logistics"] = {
        "steel_route": {
            "origin": f"{steel_winner} HQ",
            "dest": project.state,
            "eta_days": 4.5,
            "cost_inr": 450000,
            "distance_km": 1200
        },
        "conductor_route": {
            "origin": f"{cond_winner} HQ",
            "dest": project.state,
            "eta_days": 2.0,
            "cost_inr": 120000,
            "distance_km": 450
        }
    }

    return results

@app.post("/api/chat")
def chat_with_ai(request: ChatRequest):
    """LangChain Chatbot (Groq)"""
    try:
        # Invoking the new v1.x agent syntax
        response = langchain_brain.invoke(
            {"messages": [{"role": "user", "content": request.message}]}
        )
        return {"response": response['messages'][-1].content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))