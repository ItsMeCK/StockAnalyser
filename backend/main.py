# backend/main.py

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import os
import traceback
from dotenv import load_dotenv

load_dotenv()
# Assuming stock_analyzer.py is in the same directory (backend)
# and __init__.py exists in the backend directory.
from stock_analyzer import get_stock_analyzer_app, initialize_analyzer, StockAnalysisState, CHARTS_DIR  # type: ignore

# The FastAPI app instance is the default export App Engine looks for
app = FastAPI(title="Stock Analysis Chat API - Async")

# CORS configuration
origins = ["*"]  # Allows all origins - adjust for production if needed

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
    original_user_question: Optional[str] = None
    analyzed_user_query_focus: Optional[str] = None
    stock_symbol: Optional[str] = None
    company_name: Optional[str] = None # Name of the company
    technical_analysis_summary: Optional[str] = None
    fundamental_analysis_summary: Optional[str] = None
    news_analysis_summary: Optional[str] = None # Field for News Analysis Summary
    direct_answer_to_user_question: Optional[str] = None
    final_recommendation: Optional[str] = None
    position_specific_advice: Optional[str] = None
    error_message: Optional[str] = None
    generated_chart_urls: Optional[List[str]] = None


# --- Global App Variable ---
stock_analyzer_graph_app = None

# --- Determine Frontend Path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
frontend_dir_name = "frontend"
# Primary expected path: 'frontend' is a subdirectory of where main.py is (e.g., backend/frontend/index.html)
index_html_path = os.path.join(current_dir, frontend_dir_name, 'index.html')

# Fallback logic for different structures (e.g. backend/ and frontend/ as siblings)
if not os.path.exists(index_html_path):
    project_root_candidate = os.path.dirname(current_dir) # Go up one level from 'backend'
    alternative_path = os.path.join(project_root_candidate, frontend_dir_name, 'index.html')
    if os.path.exists(alternative_path):
        index_html_path = alternative_path
    else:
        # Fallback for main.py in project root and frontend is a subdir
        root_frontend_path = os.path.join(os.getcwd(), frontend_dir_name, 'index.html')
        if os.path.exists(root_frontend_path) and current_dir == os.getcwd():
             index_html_path = root_frontend_path
        # If still not found, the initial path will be used, and a warning will be printed at startup.


@app.on_event("startup")
async def startup_event():
    """Initialize the analyzer on startup."""
    global stock_analyzer_graph_app
    print("INFO: FastAPI application startup...")
    if not initialize_analyzer():
        print("CRITICAL: Stock analyzer initialization failed. Check API keys and other configurations.")
    else:
        print("INFO: Stock analyzer initialized successfully.")
        stock_analyzer_graph_app = get_stock_analyzer_app()
        if stock_analyzer_graph_app:
            print("INFO: LangGraph application compiled and ready.")
        else:
            print("ERROR: LangGraph application could not be compiled.")

    print(f"INFO: Current working directory during startup: {os.getcwd()}")
    print(f"INFO: Directory of main.py (current_dir): {current_dir}")
    print(f"INFO: Calculated path for index.html: {index_html_path}")
    if not os.path.exists(index_html_path):
        print(f"WARNING: Frontend file (index.html) not found at calculated path: {index_html_path}")
        # Listing contents for debugging if path is problematic
        print(f"DEBUG: Contents of current_dir ({current_dir}): {os.listdir(current_dir) if os.path.exists(current_dir) else 'N/A'}")
        parent_dir = os.path.dirname(current_dir)
        print(f"DEBUG: Contents of parent_dir ({parent_dir}): {os.listdir(parent_dir) if os.path.exists(parent_dir) else 'N/A'}")
        project_root = os.path.dirname(parent_dir) # two levels up if main.py is in backend/src
        print(f"DEBUG: Contents of project_root_candidate ({project_root_candidate}): {os.listdir(project_root_candidate) if os.path.exists(project_root_candidate) else 'N/A'}")


    try:
        os.makedirs(CHARTS_DIR, exist_ok=True)
        print(f"INFO: Ensured charts directory exists at {CHARTS_DIR}")
    except Exception as e:
        print(f"WARNING: Could not create charts directory at {CHARTS_DIR}: {e}")


# --- API Endpoints ---
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_stock(query: StockQuery = Body(...)):
    """Analyzes the stock based on user query."""
    print("--- DEBUG: Entered /analyze endpoint ---")
    global stock_analyzer_graph_app
    if not stock_analyzer_graph_app:
        print("ERROR: LangGraph app not available.")
        raise HTTPException(status_code=503,
                            detail="Analysis service is temporarily unavailable. Please try again later.")
    if not query.user_stock_query:
        raise HTTPException(status_code=400, detail="Stock query (name or symbol) is required.")

    print(f"INFO: Received async analysis request for query: {query.user_stock_query}")

    inputs = {
        "user_stock_query": query.user_stock_query,
        "user_question": query.user_question,
        "user_position": {"shares": query.user_position_shares, "avg_price": query.user_position_avg_price}
                         if query.user_position_shares is not None and query.user_position_avg_price is not None else None
    }

    try:
        print(f"--- DEBUG: Invoking LangGraph app with inputs: {inputs} ---")
        final_state: StockAnalysisState = await stock_analyzer_graph_app.ainvoke(inputs) # type: ignore
        print(f"--- DEBUG: LangGraph app finished. Resolved Symbol: {final_state.get('stock_symbol')}, Error in state: {final_state.get('error_message')} ---")

        internal_error_msg = final_state.get("error_message", "").strip()
        user_facing_error = None

        if internal_error_msg:
            print(f"INFO: Internal error message from analysis: {internal_error_msg}")
            # Default user-facing error
            user_facing_error = "An error occurred during the analysis. Please check your input or try again later."
            # Specific error mappings
            if "Could not resolve" in internal_error_msg or "Could not find valid stock info" in internal_error_msg:
                user_facing_error = f"Could not find a valid stock matching '{query.user_stock_query}'. Please check the name/symbol and try again."
            elif any(e_msg in internal_error_msg for e_msg in ["Primary data fetch (yfinance) failed", "Alpha Vantage fallback also failed", "Failed to fetch/serialize data"]):
                user_facing_error = "There was an issue fetching market data for the requested stock. It might be temporarily unavailable or delisted."
            elif "News fetching disabled due to missing API key" in internal_error_msg:
                 user_facing_error = "News analysis is currently unavailable. Other analysis components may still be provided." # Non-critical error
            elif "News fetching via Brave API failed" in internal_error_msg:
                user_facing_error = "Could not fetch latest news for the analysis. Other components may still be available."


        chart_urls = final_state.get("generated_chart_urls")

        response_data = AnalysisResponse(
            user_stock_query=final_state.get("user_stock_query"),
            original_user_question=final_state.get("original_user_question"),
            analyzed_user_query_focus=final_state.get("analyzed_user_query_focus"),
            stock_symbol=final_state.get("stock_symbol"),
            company_name=final_state.get("company_name"), # Pass company_name
            technical_analysis_summary=final_state.get("technical_analysis_summary"),
            fundamental_analysis_summary=final_state.get("fundamental_analysis_summary"),
            news_analysis_summary=final_state.get("news_analysis_summary"), # Pass News Summary
            direct_answer_to_user_question=final_state.get("direct_answer_to_user_question"),
            final_recommendation=final_state.get("final_recommendation"),
            position_specific_advice=final_state.get("position_specific_advice"),
            generated_chart_urls=chart_urls,
            error_message=user_facing_error, # Use the mapped user_facing_error
        )
        print(f"INFO: Async analysis complete for query '{query.user_stock_query}'. Resolved to: {response_data.stock_symbol}. User facing error: {response_data.error_message or 'None'}")
        return response_data

    except HTTPException:
        raise
    except Exception as e:
        print(f"CRITICAL_ERROR: Unhandled exception during async analysis for query '{query.user_stock_query}': {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="An unexpected internal error occurred. Please try again later.")


# --- Serve Homepage ---
@app.get("/")
async def serve_homepage():
    """Serves the frontend index.html file."""
    print(f"INFO: Attempting to serve frontend from: {index_html_path}")
    if not os.path.exists(index_html_path):
        print(f"ERROR: index.html not found at {index_html_path}")
        raise HTTPException(status_code=404, detail=f"Homepage HTML file not found. Expected at: {index_html_path}. Current dir: {os.getcwd()}")
    return FileResponse(index_html_path)


# --- Endpoint to Serve Generated Charts from /tmp ---
@app.get("/charts_data/{filename:path}")
async def get_chart_image(filename: str):
    """Serves a chart image file from the temporary directory."""
    file_path = os.path.join(CHARTS_DIR, filename)
    print(f"INFO: Attempting to serve chart: {file_path}")
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        print(f"ERROR: Chart file not found: {file_path}")
        raise HTTPException(status_code=404, detail="Chart image not found.")

# To run locally (e.g. from project root, if main.py is in backend/):
# uvicorn backend.main:app --reload --port 8000
# Or from backend/ directory:
# uvicorn main:app --reload --port 8000
