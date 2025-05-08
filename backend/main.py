# backend/main.py
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
# fastapi.responses.FileResponse is no longer needed here
# fastapi.staticfiles.StaticFiles is no longer needed here
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import os
import traceback

# Use explicit relative import (important for Vercel structure)
from stock_analyzer import get_stock_analyzer_app, initialize_analyzer, StockAnalysisState, STATIC_DIR # type: ignore

# The FastAPI app instance is the default export Vercel looks for
app = FastAPI(title="Stock Analysis Chat API - Async")

# CORS configuration (adjust origins if needed, but '*' is often fine for Vercel)
origins = [
    # Add your deployed frontend URL here after deployment if needed,
    # or use "*" for wider access (less secure)
    "*" # Allows all origins - adjust for production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class StockQuery(BaseModel):
    user_stock_query: str = Field(..., description="The company name or stock symbol entered by the user.")
    user_question: Optional[str] = None
    user_position_shares: Optional[float] = None
    user_position_avg_price: Optional[float] = None

class AnalysisResponse(BaseModel):
    user_stock_query: Optional[str] = None
    stock_symbol: Optional[str] = None
    technical_analysis_summary: Optional[str] = None
    fundamental_analysis_summary: Optional[str] = None
    final_recommendation: Optional[str] = None
    position_specific_advice: Optional[str] = None
    error_message: Optional[str] = None
    generated_chart_urls: Optional[List[str]] = None

# --- Global App Variable ---
stock_analyzer_graph_app = None

# --- Vercel Specific Note ---
# Static file serving and homepage routing will be handled by vercel.json
# We don't need the path calculations or FileResponse here anymore.

@app.on_event("startup")
async def startup_event():
    """Initialize the analyzer on startup."""
    global stock_analyzer_graph_app
    print("INFO: FastAPI application startup (Vercel)...")
    if not initialize_analyzer():
        print("CRITICAL: Stock analyzer initialization failed. Check OpenAI API key.")
        # In a real deployment, you might want to raise an error
        # or implement a health check endpoint that reflects this state.
    else:
        print("INFO: Stock analyzer initialized successfully.")
        stock_analyzer_graph_app = get_stock_analyzer_app()
        if stock_analyzer_graph_app:
            print("INFO: LangGraph application compiled and ready.")
        else:
            print("ERROR: LangGraph application could not be compiled.")


# --- API Endpoints ---
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_stock(query: StockQuery = Body(...)):
    """Analyzes the stock based on user query."""
    print("--- DEBUG: Entered /analyze endpoint ---")
    global stock_analyzer_graph_app
    if not stock_analyzer_graph_app:
        print("ERROR: LangGraph app not available.")
        raise HTTPException(status_code=503, detail="Analysis service is not available.")
    if not query.user_stock_query:
        raise HTTPException(status_code=400, detail="Stock query is required.")

    print(f"INFO: Received async analysis request for query: {query.user_stock_query}")

    inputs = {
        "user_stock_query": query.user_stock_query,
        "user_question": query.user_question
    }
    inputs["user_position"] = { "shares": query.user_position_shares, "avg_price": query.user_position_avg_price } if query.user_position_shares is not None and query.user_position_avg_price is not None else None

    try:
        print(f"--- DEBUG: Invoking LangGraph app with inputs: {inputs} ---")
        final_state: StockAnalysisState = await stock_analyzer_graph_app.ainvoke(inputs)
        print(f"--- DEBUG: LangGraph app finished. Resolved Symbol: {final_state.get('stock_symbol')}, Error: {final_state.get('error_message')} ---")

        error_msg = final_state.get("error_message", "").strip()

        # Adjust chart URLs to be absolute or relative to the deployment root
        # Vercel serves static files from the root path defined in vercel.json
        chart_urls = final_state.get("generated_chart_urls")
        if chart_urls:
             # Assuming vercel.json routes /static/* to backend/static/*
             # The URL returned by the tool is already relative like /static/charts/file.png
             pass # No change needed if URL prefix is correct

        response_data = AnalysisResponse(
            user_stock_query=final_state.get("user_stock_query"),
            stock_symbol=final_state.get("stock_symbol"),
            technical_analysis_summary=final_state.get("technical_analysis_summary"),
            fundamental_analysis_summary=final_state.get("fundamental_analysis_summary"),
            final_recommendation=final_state.get("final_recommendation"),
            position_specific_advice=final_state.get("position_specific_advice"),
            generated_chart_urls=chart_urls, # Pass the URLs
            error_message=error_msg if error_msg else None,
        )
        print(f"INFO: Async analysis complete for query '{query.user_stock_query}'. Resolved to: {response_data.stock_symbol}. Errors: {response_data.error_message or 'None'}")
        return response_data

    except Exception as e:
        print(f"ERROR: Unhandled exception during async analysis for query '{query.user_stock_query}': {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal error occurred during analysis: {str(e)}")

# Remove the @app.get("/") endpoint, as vercel.json will handle serving index.html

