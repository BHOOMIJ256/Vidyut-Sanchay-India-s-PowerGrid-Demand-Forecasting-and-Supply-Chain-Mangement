
import os
import re
import json
import time
import sqlite3
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import re

from dotenv import load_dotenv
load_dotenv()
from groq import Groq
from groq import BadRequestError, APIError
# LangChain / LangGraph imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq


SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
if not SERPAPI_KEY:
    raise RuntimeError("Set SERPAPI_KEY env var before running")

CHECKPOINT_DB = os.getenv("CHECKPOINT_DB", "./langgraph_checkpoints.db")
ALPHAVANTAGE_KEY = os.getenv("ALPHAVANTAGE_KEY", "BR1UI7DQQ6WKTV5N")  # keep original if present


def serp_search_news(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    
    
    url = "https://google.serper.dev/search"

    payload = {
        "q": query ,
        "tbs": "qdr:m"
    }
    headers = {
        'X-API-KEY': SERPAPI_KEY,
        'Content-Type': 'application/json'
    }

    response = requests.post(url, headers=headers, json=payload)

    # Parse and pretty print JSON
    data = response.json()
    res = []
    for i in data['organic']:
        res.append(i['link'])
    return res[0]


HEADERS = {
    'User-Agent': 'Mozilla/5.0 (compatible; RiskAgent/1.0; +https://example.com/bot)'
}

def scrape_article_content(url: str, timeout: int = 10) -> Dict[str, Any]:
    try:
        domain = urlparse(url).netloc.lower()
        skip_domains = ['youtube.com', 'facebook.com', 'twitter.com', 'instagram.com', 'linkedin.com', 'reddit.com']
        if any(skip in domain for skip in skip_domains) or url.lower().endswith(".pdf"):
            return {"success": False, "reason": "Non-article domain"}

        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        if resp.status_code != 200:
            return {"success": False, "reason": f"HTTP {resp.status_code}"}

        soup = BeautifulSoup(resp.content, "html.parser")
        for s in soup(["script", "style", "nav", "footer", "header"]):
            s.decompose()

        title = ""
        if soup.title:
            title = soup.title.get_text().strip()
        h1 = soup.find("h1")
        if h1 and len(h1.get_text().strip()) > len(title):
            title = h1.get_text().strip()

        article_selectors = [
            'article', '.article-content', '.post-content', '.entry-content',
            '.story-body', '.article-body', 'main'
        ]
        content = ""
        for sel in article_selectors:
            node = soup.select_one(sel)
            if node:
                ps = node.find_all("p")
                content = " ".join([p.get_text().strip() for p in ps if len(p.get_text().strip()) > 40])
                if len(content) > 200:
                    break

        if not content or len(content) < 200:
            ps = soup.find_all("p")
            content = " ".join([p.get_text().strip() for p in ps if len(p.get_text().strip()) > 40])

        content = re.sub(r'\s+', ' ', content).strip()
        if len(content) < 100:
            return {"success": False, "reason": "Content too short"}

        return {"content": content[:5000]}

    except Exception as e:
        return {"success": False, "reason": str(e)[:200]}


@tool
def serp_tool(query: str, num_results: int = 5) -> Dict[str, Any]:
    """Tool does google search and shows the single most relevant search for the query."""
    try:
        results = serp_search_news(query, num_results=num_results)
        return {"success": True, "results": results}
    except Exception as e:
        return {"success": False, "error": str(e)}

@tool
def scrape_tool(url: str) -> Dict[str, Any]:
    """Tool scrapes the content accurately of the provided link and this data can be used by the LLM for risk scoring the company."""
    try:
        res = scrape_article_content(url)
        return res
    except Exception as e:
        return {"success": False, "error": str(e)}
    
    
import requests

# -------------------------------
# 1. SETUP
# -------------------------------
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")
if not WEATHER_API_KEY:
    raise RuntimeError("Set WEATHER_API_KEY env var before running")
 # Replace with your actual key
BASE_URL = "http://api.weatherapi.com/v1"


def get_forecast(city, days=3):
    """
    Fetches forecast for the next X days (max: 14 days in free plan)
    """
    endpoint = f"{BASE_URL}/forecast.json"
    
    params = {
        "key": WEATHER_API_KEY,
        "q": city,
        "days": days,
        "aqi": "yes",
        "alerts": "yes"      # shows storm / weather alerts
    }

    response = requests.get(endpoint, params=params)
    data = response.json()
    return data


@tool 
def weather_summary(city_name:str):
    """This tools helps in gathering the weather summary of the city and also provides the forecast of the next 3 days"""
    # Get forecast (next 3 days)
    forecast_data = get_forecast(city_name, days=3)
    result = []
    for j in range(0,3):
        for i in range(0,24,5):
            result.append(["time:",forecast_data['forecast']['forecastday'][j]['hour'][i]['time'],"condition:",forecast_data['forecast']['forecastday'][0]['hour'][i]['condition']['text'],"temp_c:",forecast_data['forecast']['forecastday'][0]['hour'][0]['temp_c']])

    return result

# Tools list that LLM can choose from

TOOLS = [serp_tool, scrape_tool, weather_summary] 

# LLM instance (Gemini)
#llm = ChatGroq(temperature=0.4, model="qwen/qwen3-32b")
llm = ChatGroq(temperature=0.4, model="moonshotai/kimi-k2-instruct-0905")
llm_bound = llm.bind_tools(TOOLS) 

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage] ,  add_messages]

def chat_node(state:ChatState):
    """
    LLM node that may answer or request a tool call."""
    messages = state['messages']
    response = llm_bound.invoke(messages)
    return {'messages':[response]}

tool_node = ToolNode(TOOLS)
  
  
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge('tools', 'chat_node') 

chatbot = graph.compile()


logistic_agent = {'supplier': 'JAMSHEDPUR_TATA', 'project_site': 'NAGPUR_SITE', 
 'quantity_tonnes': 1500.0, 'road_distance_km': 983.28, 
 'ETA_days': 4, 'estimated_arrival_date': '2025-12-15', 
 'transport_cost_inr': 9587005.0}

price_agent = {"grand_total":"51 cr"}


company_name = logistic_agent['supplier']
price_agent_info = price_agent['grand_total']

logistic_agent_info = str(logistic_agent)




try:
    
    out = chatbot.invoke({'messages':[HumanMessage(content=f"""You are the RISK SCORING AGENT.

Your purpose:
Given:
- Company names,
- Price Agent data ,
- Logistics Agent data (distance, delivery time, region, transport duration),
- Weather Summary Tool results for supplier regions,
- Web news about the company's supplier performance,

You must compute a final RISK SCORE for each company.

---------------------------------------------------------
YOUR WORKFLOW (DO NOT DEVIATE FROM THESE STEPS)
---------------------------------------------------------

For each company, FOLLOW THIS EXACT ORDER:

### STEP 1 — Fetch Latest Supplier-Relevant News
Use ONLY "serp_tool" to search for ONE highly relevant article URL about the company.
(This tool provides only one url)
Your search query MUST focus on:
- company name
- supplier performance
- raw material delivery reliability
- manufacturing stability
- supply chain disruptions
(Example query: “Tata Steel supplier performance latest news”)

MUST: Only ONE URL per company.
MUST NOT use browser.search or browser.open.

### STEP 2 — Scrape the Article
Use ONLY "scrape_tool" on the URL obtained from serp_tool.
If scraping fails → skip to Step 6 with status: "unable_to_fetch".

### STEP 3 — Interpret the Article
Analyze the scraped text and classify sentiment regarding:
- supply reliability,
- production stability,
- financial stability,
- delivery delays,
- raw-material-side risks.

Classification MUST be:
- "good"
- "neutral"
- "bad"

### STEP 4 — Weather-Based Risk (Low Weight)
You will be provided Weather Summary Tool output from the Logistics Agent.
Weather corresponds to the supplier’s location.

Rules:
- Severe storms, blizzards, floods → add +1 to risk
- Mild/partly cloudy/low impact → +0.2 to risk
- Weather should NEVER dominate (max 10% influence)

### STEP 5 — Combine ML Model + Price Agent + Logistics Agent Data
Use all agent inputs with these weights:

- Supplier News Sentiment → **50%**
- Logistics_agent_info → **25%**
- Price_agent_info  → **15%**
- Weather impact → **10%**

### STEP 6 — Final Risk Score (1–10)
Strict rules:
- Bad News → Base Risk 8–10  
- Neutral → Base 4–7  
- Good → Base 1–3  
Then modify using weighted factors above.

If news scraping failed:
- return:
    "status": "unable_to_fetch",
    "risk_score": null

### STEP 7 — Output JSON for Each Company
The final output MUST be a JSON object:


  "company": "<company_name>",
  "news_status": "good / neutral / bad / unable_to_fetch",
  "risk_score": "<1-10 or null>",
  "reason": "<one short explanation>",


### STEP 8 — After All Companies Are Processed
Return a JSON LIST containing all company results.

---------------------------------------------------------
HARD RULES (DO NOT BREAK THESE)
---------------------------------------------------------
- Do NOT use browser.search, browser.open, or any tool besides serp_tool and scrape_tool.
- No recursive loops.
- No retry loops.
- Process companies sequentially: finish one → then next.
- If any tool fails, DO NOT repeat the call.
- Search queries MUST be relevant to supplier performance.
- Keep reasoning internal; only return final JSON.
- Weather influence must stay low-weight.



User will provide:

COMPANIES: {company_name},
logistic_agent_info : {logistic_agent_info},
price_agent_info: {price_agent_info},

Follow the workflow above exactly.
""") ]})
except BadRequestError as e:
    print("run again:",e)
print(out)

