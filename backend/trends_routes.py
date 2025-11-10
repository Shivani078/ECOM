from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
import os
import json
import re
import random
import asyncio
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY missing in environment variables")

router = APIRouter()
groq_model = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7, api_key=GROQ_API_KEY)

# -------------------------------
# Prompt Template
# -------------------------------
template = """
You are an expert fashion and lifestyle trend analyst.

Analyze the following search results about **{category} trends in {city}**, 
and return logical insights based on local culture, season, and events.

Return STRICT JSON in this format:
[
  {{
    "city": "{city}",
    "trend": "Trend Name",
    "popularity": "High 🔥 / Medium ⚡ / Low ❄️",
    "change_pct": "45.2%",
    "features": ["Feature 1", "Feature 2"],
    "competitors": ["Competitor 1", "Competitor 2"],
    "local_hotspots": ["Market/Area 1", "Market/Area 2"],
    "tips": ["Tip 1", "Tip 2"]
  }}
]

Search Results:
{search_results}
"""
prompt = PromptTemplate.from_template(template)

# -------------------------------
# Request model
# -------------------------------
class TrendsRequest(BaseModel):
    cities: List[str]
    category: str

# -------------------------------
# Helpers
# -------------------------------
def clean_json_response(text: str):
    match = re.search(r"\[.*\]", text, re.DOTALL)
    return match.group(0) if match else "[]"

def assign_random_metrics():
    pct = round(random.uniform(3.0, 65.0), 1)
    if pct >= 35.0:
        label = "High 🔥"
        score = 85
    elif pct >= 15.0:
        label = "Medium ⚡"
        score = 55
    else:
        label = "Low ❄️"
        score = 20
    return pct, label, score

async def process_city(city: str, category: str, search_results: str = ""):
    """
    Processes a single city using Groq to generate trends.
    search_results can be provided or left empty.
    """
    try:
        def groq_call():
            runnable = prompt | groq_model | StrOutputParser()
            return runnable.invoke({
                "city": city,
                "category": category,
                "search_results": search_results,
            })

        response = await asyncio.to_thread(groq_call)
        cleaned = clean_json_response(response)
        parsed = json.loads(cleaned)

        for trend in parsed:
            pct = None
            if "change_pct" in trend:
                try:
                    pct_val = re.search(r"([-+]?\d+(\.\d+)?)", str(trend.get("change_pct")))
                    if pct_val:
                        pct = round(float(pct_val.group(1)), 1)
                except:
                    pct = None

            if pct is None:
                pct, label, score = assign_random_metrics()
            else:
                if pct >= 35.0:
                    label = "High 🔥"
                    score = 85
                elif pct >= 15.0:
                    label = "Medium ⚡"
                    score = 55
                else:
                    label = "Low ❄️"
                    score = 20

            trend["pct_change"] = pct
            trend["change_pct"] = f"{pct}%"
            trend["popularity_score"] = score
            trend["popularity"] = trend.get("popularity", label)

            for key in ("features", "competitors", "local_hotspots", "tips"):
                if key not in trend or not isinstance(trend[key], list):
                    trend[key] = trend.get(key, []) if isinstance(trend.get(key, list), list) else []

        return parsed
    except Exception as e:
        return [{"city": city, "error": str(e)}]

# -------------------------------
# Async endpoint
# -------------------------------
@router.post("/")
async def get_trends(request: TrendsRequest):
    tasks = [process_city(city, request.category) for city in request.cities]
    results = await asyncio.gather(*tasks)
    all_trends = [trend for city_trends in results for trend in city_trends]
    return {"trends": all_trends}
