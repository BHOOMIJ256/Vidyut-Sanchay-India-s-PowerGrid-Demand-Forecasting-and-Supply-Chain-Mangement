import os
import json
from dotenv import load_dotenv

# --- UPDATED IMPORTS FOR LANGCHAIN v1.x ---
from langchain.agents import create_agent  # The new standard
from langchain_groq import ChatGroq
from langchain.tools import tool
# ------------------------------------------

from price_agent.price_agent import PriceAgent

# Load Env for API Keys
load_dotenv()

# 1. Initialize your existing Logic Agent (THE ENGINE)
backend_agent = PriceAgent(use_live_apis=True)

# 2. Define Tools
@tool
def check_current_market_prices(dummy_input: str = "") -> str:
    """
    Fetches the latest live market prices for commodities like Steel, Aluminum, 
    Copper, and Exchange Rates. Returns a raw dictionary of prices.
    """
    try:
        prices = backend_agent.get_current_prices()
        return str(prices)
    except Exception as e:
        return f"Error fetching prices: {e}"

@tool
def calculate_project_cost_estimate(ml_json_string: str) -> str:
    """
    Calculates the full project cost, selects suppliers (SAIL vs Tata, etc.), 
    and returns a detailed budget report.
    
    IMPORTANT: Input must be a JSON string with keys: 
    'steel_tonnes', 'conductor_km', 'transformers_count', etc.
    """
    try:
        data = json.loads(ml_json_string)
        report = backend_agent.calculate_project_cost(data)
        return str(report)
    except Exception as e:
        return f"Error calculating cost: {e}"

# 3. Initialize the Brain (Groq LLM)
llm = ChatGroq(
    temperature=0, 
    model_name="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# 4. Create the Agent (New v1.x Syntax)
# No more AgentExecutor! The agent handles the loop itself.
tools = [check_current_market_prices, calculate_project_cost_estimate]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are the Vidyut Sanchay AI Procurement Manager. Use your tools to fetch real-time prices and calculate project costs. Be precise and professional."
)

# ================= DEMO MODE =================
if __name__ == "__main__":
    print("🤖 VIDYUT SANCHAY AI MANAGER (Modern v1.x): Ready.")
    
    while True:
        user_input = input("\nAdmin: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        try:
            # New Invocation Syntax: Takes a list of messages
            response = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
            
            # The last message in the response is the AI's answer
            print(f"AI: {response['messages'][-1].content}")
            
        except Exception as e:
            print(f"Error: {e}")