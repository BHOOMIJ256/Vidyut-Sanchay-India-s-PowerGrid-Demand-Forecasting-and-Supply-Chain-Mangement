import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# Use safe imports
try:
    from price_agent.price_agent import PriceAgent
    from logistics_agent.logistics_agent import LogisticsAgent
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from price_agent.price_agent import PriceAgent
    from logistics_agent.logistics_agent import LogisticsAgent

load_dotenv()

# Initialize Engines
price_engine = PriceAgent(use_live_apis=True)
logistics_engine = LogisticsAgent()

# --- TOOLS ---

@tool
def get_live_market_rates():
    """Fetches real-time market prices for Aluminum, Copper, Steel, and USD/INR."""
    try:
        return str(price_engine.get_current_prices())
    except Exception as e:
        return f"Error fetching prices: {e}"

@tool
def check_logistics_feasibility(origin_city: str, destination_city: str, tonnes: float = 100.0):
    """
    Calculates distance, ETA, and transport cost. 
    Inputs: origin_city (e.g. "Kolkata, India"), destination_city (e.g. "Mumbai, India"), tonnes.
    """
    try:
        if "India" not in origin_city: origin_city += ", India"
        if "India" not in destination_city: destination_city += ", India"
        return str(logistics_engine.calculate_delivery(origin_city, destination_city, tonnes))
    except Exception as e:
        return f"Error calculating logistics: {e}"

def get_vidyut_agent():
    """
    Returns the compiled LangGraph Agent.
    NOTE: We do NOT pass the system prompt here to avoid version conflicts.
    We will inject the prompt in main.py instead.
    """
    llm = ChatGroq(
        temperature=0.2, 
        model_name="llama-3.3-70b-versatile", 
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    
    tools = [get_live_market_rates, check_logistics_feasibility]
    
    # Simple creation - No extra keywords to crash the system
    return create_react_agent(llm, tools)