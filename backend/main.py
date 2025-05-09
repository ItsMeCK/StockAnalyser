# backend/main.py
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
# StaticFiles is not explicitly used here anymore as GAE serves static assets via app.yaml or FastAPI serves dynamic ones
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import os
import traceback

# Assuming stock_analyzer.py is in the same directory (backend)
from stock_analyzer import get_stock_analyzer_app, initialize_analyzer, StockAnalysisState, CHARTS_DIR  # type: ignore

# The FastAPI app instance is the default export App Engine looks for
app = FastAPI(title="Stock Analysis Chat API - Async")

# CORS configuration
origins = ["*"]  # Allows all origins - adjust for production

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

# --- Determine Frontend Path ---
# When deployed to GAE, the 'backend' directory content is at the root of /workspace
# So, __file__ will be /workspace/main.py (or similar if in a subdirectory of workspace)
# We assume 'frontend' is now a subdirectory of where main.py is.
current_dir = os.path.dirname(os.path.abspath(__file__))
index_html_path = os.path.join(current_dir, 'frontend', 'index.html')


@app.on_event("startup")
async def startup_event():
    """Initialize the analyzer on startup."""
    global stock_analyzer_graph_app
    print("INFO: FastAPI application startup (GAE version)...")
    if not initialize_analyzer():
        print("CRITICAL: Stock analyzer initialization failed. Check OpenAI API key.")
    else:
        print("INFO: Stock analyzer initialized successfully.")
        stock_analyzer_graph_app = get_stock_analyzer_app()
        if stock_analyzer_graph_app:
            print("INFO: LangGraph application compiled and ready.")
        else:
            print("ERROR: LangGraph application could not be compiled.")

    print(f"INFO: Expected path for index.html: {index_html_path}")
    if not os.path.exists(index_html_path):
        print(f"WARNING: Frontend file not found at expected path: {index_html_path}")
        print(f"Current working directory: {os.getcwd()}")
        print(
            f"Contents of current_dir ({current_dir}): {os.listdir(current_dir) if os.path.exists(current_dir) else 'N/A'}")
        # Check one level up for frontend if backend is a subdir of workspace
        alt_frontend_dir = os.path.join(current_dir, '..', 'frontend')
        alt_index_path = os.path.join(alt_frontend_dir, 'index.html')
        if os.path.exists(alt_index_path):
            print(
                f"INFO: Alternative index.html path found at {alt_index_path} - this might indicate a structure mismatch for GAE deployment if app.yaml is in a subdir.")

    # Ensure CHARTS_DIR (which is /tmp/charts) exists on startup if possible
    # This might be recreated by each instance on GAE, but good to have the logic
    try:
        os.makedirs(CHARTS_DIR, exist_ok=True)  # CHARTS_DIR is imported from stock_analyzer
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
        raise HTTPException(status_code=503, detail="Analysis service is not available.")
    if not query.user_stock_query:
        raise HTTPException(status_code=400, detail="Stock query is required.")

    print(f"INFO: Received async analysis request for query: {query.user_stock_query}")

    inputs = {
        "user_stock_query": query.user_stock_query,
        "user_question": query.user_question
    }
    inputs["user_position"] = {"shares": query.user_position_shares,
                               "avg_price": query.user_position_avg_price} if query.user_position_shares is not None and query.user_position_avg_price is not None else None

    try:
        print(f"--- DEBUG: Invoking LangGraph app with inputs: {inputs} ---")
        final_state: StockAnalysisState = await stock_analyzer_graph_app.ainvoke(inputs)
        print(
            f"--- DEBUG: LangGraph app finished. Resolved Symbol: {final_state.get('stock_symbol')}, Error: {final_state.get('error_message')} ---")

        error_msg = final_state.get("error_message", "").strip()
        chart_urls = final_state.get("generated_chart_urls")

        response_data = AnalysisResponse(
            user_stock_query=final_state.get("user_stock_query"),
            stock_symbol=final_state.get("stock_symbol"),
            technical_analysis_summary=final_state.get("technical_analysis_summary"),
            fundamental_analysis_summary=final_state.get("fundamental_analysis_summary"),
            final_recommendation=final_state.get("final_recommendation"),
            position_specific_advice=final_state.get("position_specific_advice"),
            generated_chart_urls=chart_urls,
            error_message=error_msg if error_msg else None,
        )
        print(
            f"INFO: Async analysis complete for query '{query.user_stock_query}'. Resolved to: {response_data.stock_symbol}. Errors: {response_data.error_message or 'None'}")
        return response_data

    except Exception as e:
        print(f"ERROR: Unhandled exception during async analysis for query '{query.user_stock_query}': {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal error occurred during analysis: {str(e)}")


# --- Serve Homepage ---
@app.get("/")
async def serve_homepage():
    """Serves the frontend index.html file."""
    print(f"INFO: Attempting to serve frontend from: {index_html_path}")
    if not os.path.exists(index_html_path):
        print(f"ERROR: index.html not found at {index_html_path}")
        # Add more debug info about directory structure
        workspace_content = os.listdir(os.path.dirname(current_dir)) if os.path.exists(
            os.path.dirname(current_dir)) else "N/A (parent)"
        current_dir_content = os.listdir(current_dir) if os.path.exists(current_dir) else "N/A"
        print(f"DEBUG: Contents of /workspace (approx): {workspace_content}")
        print(f"DEBUG: Contents of current_dir ({current_dir}): {current_dir_content}")
        raise HTTPException(status_code=404, detail=f"Homepage HTML file not found. Expected at: {index_html_path}")
    return FileResponse(index_html_path)


# --- Endpoint to Serve Generated Charts from /tmp ---
@app.get("/charts_data/{filename:path}")
async def get_chart_image(filename: str):
    """Serves a chart image file from the temporary directory."""
    # CHARTS_DIR is imported from stock_analyzer and should point to /tmp/charts
    file_path = os.path.join(CHARTS_DIR, filename)
    print(f"INFO: Attempting to serve chart: {file_path}")
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        print(f"ERROR: Chart file not found: {file_path}")
        raise HTTPException(status_code=404, detail="Chart image not found.")

# To run locally (from backend directory): uvicorn main:app --reload --port 8000
