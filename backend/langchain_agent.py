import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from price_agent.price_agent import PriceAgent

# Load Env for OpenAI Key
load_dotenv()

# 1. Initialize your existing Logic Agent (THE ENGINE)
# We use your robust Python code as the backend
backend_agent = PriceAgent(use_live_apis=True)

# 2. Define "Tools" (This lets GPT use your code)
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
    import json
    try:
        # GPT might send a string, we need to convert it to Dict for your code
        data = json.loads(ml_json_string)
        report = backend_agent.calculate_project_cost(data)
        return str(report)
    except Exception as e:
        return f"Error calculating cost: {e}"

# 3. Initialize the Brain (OpenAI)
# You need OPENAI_API_KEY in your .env file
llm = ChatOpenAI(temperature=0, model="gpt-4o") # Use 3.5-turbo if 4o is not available

# 4. Give the Tools to the Brain
tools = [check_current_market_prices, calculate_project_cost_estimate]

agent_executor = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.OPENAI_FUNCTIONS, 
    verbose=True,
    handle_parsing_errors=True
)

# ================= DEMO MODE =================
if __name__ == "__main__":
    print("🤖 VIDYUT SANCHAY AI MANAGER: I am ready.")
    
    while True:
        user_input = input("\nAdmin: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        try:
            response = agent_executor.invoke(user_input)
            print(f"AI: {response['output']}")
        except Exception as e:
            print(f"Error: {e}")