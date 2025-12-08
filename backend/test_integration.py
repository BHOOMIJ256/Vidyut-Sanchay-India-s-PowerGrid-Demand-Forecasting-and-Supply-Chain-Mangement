"""
Integration Test Script for Vidyut Sanchay
Simulates the flow from ML Model Prediction -> Price Agent -> Final Cost Estimation
"""

import json
from price_agent import PriceAgent

def run_integration_test():
    # 1. Initialize the Price Agent
    # This will trigger the connection to AlphaVantage and ExchangeRate API
    print("🚀 STARTING INTEGRATION TEST...")
    agent = PriceAgent(use_live_apis=True)

    # 2. SIMULATE ML MODEL OUTPUT
    # This matches the exact JSON structure you showed me earlier
    # (In the real app, this variable comes from: prediction = full_pipeline.predict(input_data))
    print("\n🔮 SIMULATING ML MODEL OUTPUT (Predicted Quantities)...")
    
    ml_output_raw = """
    {
        "steel_tonnes": {
            "value": 5975.25,
            "unit": "tonnes"
        },
        "conductor_km": {
            "value": 215.42,
            "unit": "km"
        },
        "insulators_unit": {
            "value": 5564.0,
            "unit": "units"
        },
        "concrete_cubic_meter": {
            "value": 3375.52,
            "unit": "cubic_meter"
        },
        "bus_reactor_count": {
            "value": 1.0,
            "unit": "count"
        },
        "transformers_count": {
            "value": 2.0,
            "unit": "count"
        },
        "circuit_breaker_count": {
            "value": 5.0,
            "unit": "count"
        }
    }
    """
    
    # Parse JSON string into a Python Dictionary
    ml_data = json.loads(ml_output_raw)
    print("✅ ML Data Loaded Successfully.")

    # 3. CALCULATE COST
    # The Agent will now merge these quantities with live market prices
    final_report = agent.calculate_project_cost(ml_data)

    # ... inside run_integration_test() ...

    # 4. VERIFICATION
    print("📋 TEST VERIFICATION:")
    
    # FIX: Access 'grand_total' directly instead of 'raw_values'['total']
    if final_report.get('grand_total', 0) > 0:
        print("✅ SUCCESS: Final Total calculated.")
        
        # We can now print who won the contracts!
        print(f"   Steel Supplier Selected: {final_report.get('steel_supplier')}")
        print(f"   Conductor Supplier Selected: {final_report.get('conductor_supplier')}")
    else:
        print("❌ FAILURE: Total is zero. Check API connections.")

    # FIX: Use 'grand_total' key for formatting
    # NEW (Correct)
    # ✅ CORRECT (Looks inside the price_agent folder)
    from price_agent.utils import format_currency # Helper to format the final number
    formatted_total = format_currency(final_report.get('grand_total', 0))
    
    print(f"\n💰 FINAL QUOTATION FOR TENDER: {formatted_total}")

if __name__ == "__main__":
    run_integration_test()