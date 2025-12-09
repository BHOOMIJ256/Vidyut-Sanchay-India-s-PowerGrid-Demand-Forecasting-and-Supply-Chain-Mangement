
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
    
    


# Tools list that LLM can choose from
TOOLS = [serp_tool, scrape_tool]

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
try:
    
    out = chatbot.invoke({'messages':[HumanMessage(content="""You are a Risk Scoring Agent.

    Your ONLY job:
    Given one or more company names, for each company:
    1. Use "serp_tool" to fetch one news article URL about the company (latest or most relevant).
    2. Then use "scrape_tool" on that URL to get the article content.
    3. Read the scraped content and classify the sentiment as Good / Neutral / Bad.
    4. Assign a Risk Score:
        Bad news   → High Risk (8–10)
        Neutral    → Medium Risk (4–7)
        Positive   → Low Risk (1–3)
    5. Output results in this JSON format:



    ### VERY IMPORTANT RULES
    - You MUST NOT use browser.search or browser.open.
    - You must ONLY use:
        • serp_tool → to get URL
        • scrape_tool → to extract content from URL
    - One URL per company only.
    - No loops, no recursion, no re-thinking steps.
    - Process companies sequentially. Finish one → move to next.
    - If any tool fails, return "Unable to fetch" for that company instead of looping.

    After processing all companies, return the final JSON list like this,make sure it is json or dictionary:

    [
    {company1 , risk score/10},
    {company2 , risk score/10},
    {company3 , risk_score/10}
    ]

    Input:
    COMPANIES:   JSW steel 
    """) ]})
except BadRequestError as e:
    print("run again:",e)
print(out['messages'][-1].content)

