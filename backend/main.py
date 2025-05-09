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
    original_user_question: Optional[str] = None  # To echo back the user's exact question
    analyzed_user_query_focus: Optional[str] = None  # To show the AI's understanding of the question's intent
    stock_symbol: Optional[str] = None  # The resolved symbol used for analysis
    technical_analysis_summary: Optional[str] = None
    fundamental_analysis_summary: Optional[str] = None
    direct_answer_to_user_question: Optional[str] = None  # For the direct answer to the user's question
    final_recommendation: Optional[str] = None  # For the Buy/Sell/Hold recommendation
    position_specific_advice: Optional[str] = None
    error_message: Optional[str] = None
    generated_chart_urls: Optional[List[str]] = None


# --- Global App Variable ---
stock_analyzer_graph_app = None

# --- Determine Frontend Path ---
# Assumes 'frontend' folder is a sub-directory of the 'backend' directory
# where main.py and app.yaml are located for GAE deployment.
current_dir = os.path.dirname(os.path.abspath(__file__))
index_html_path = os.path.join(current_dir, 'frontend', 'index.html')


@app.on_event("startup")
async def startup_event():
    """Initialize the analyzer on startup."""
    global stock_analyzer_graph_app
    print("INFO: FastAPI application startup (GAE version)...")
    if not initialize_analyzer():  # This call initializes LLMs and tool descriptions in stock_analyzer.py
        print("CRITICAL: Stock analyzer initialization failed. Check OpenAI API key and other configurations.")
        # In a real deployment, you might want to raise an error or implement a health check.
    else:
        print("INFO: Stock analyzer initialized successfully.")
        stock_analyzer_graph_app = get_stock_analyzer_app()  # Compile the LangGraph app
        if stock_analyzer_graph_app:
            print("INFO: LangGraph application compiled and ready.")
        else:
            print("ERROR: LangGraph application could not be compiled.")

    # Debugging frontend path for GAE
    print(f"INFO: Expected path for index.html: {index_html_path}")
    if not os.path.exists(index_html_path):
        print(f"WARNING: Frontend file (index.html) not found at expected path: {index_html_path}")
        print(f"DEBUG: Current working directory during startup: {os.getcwd()}")
        print(
            f"DEBUG: Contents of current_dir ({current_dir}): {os.listdir(current_dir) if os.path.exists(current_dir) else 'N/A'}")

    # Ensure CHARTS_DIR (which is /tmp/charts on GAE) exists
    try:
        # CHARTS_DIR is imported from stock_analyzer
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

    # Prepare inputs for the LangGraph app
    inputs = {
        "user_stock_query": query.user_stock_query,
        "user_question": query.user_question  # This will be stored as 'original_user_question' in the state
    }
    inputs["user_position"] = {"shares": query.user_position_shares,
                               "avg_price": query.user_position_avg_price} if query.user_position_shares is not None and query.user_position_avg_price is not None else None

    try:
        print(f"--- DEBUG: Invoking LangGraph app with inputs: {inputs} ---")
        final_state: StockAnalysisState = await stock_analyzer_graph_app.ainvoke(inputs)
        print(
            f"--- DEBUG: LangGraph app finished. Resolved Symbol: {final_state.get('stock_symbol')}, Error in state: {final_state.get('error_message')} ---")

        internal_error_msg = final_state.get("error_message", "").strip()
        user_facing_error = None

        if internal_error_msg:
            print(f"INFO: Internal error message from analysis: {internal_error_msg}")
            user_facing_error = "An error occurred during the analysis. Please check your input or try again later."
            if "Could not resolve" in internal_error_msg or "Could not find valid stock info" in internal_error_msg:
                user_facing_error = f"Could not find a valid stock matching '{query.user_stock_query}'. Please check the name/symbol and try again."
            elif "Failed to fetch/serialize data" in internal_error_msg:
                user_facing_error = "There was an issue fetching data for the requested stock. It might be temporarily unavailable or delisted."

        chart_urls = final_state.get("generated_chart_urls")

        response_data = AnalysisResponse(
            user_stock_query=final_state.get("user_stock_query"),
            original_user_question=final_state.get("original_user_question"),
            analyzed_user_query_focus=final_state.get("analyzed_user_query_focus"),
            stock_symbol=final_state.get("stock_symbol"),
            technical_analysis_summary=final_state.get("technical_analysis_summary"),
            fundamental_analysis_summary=final_state.get("fundamental_analysis_summary"),
            direct_answer_to_user_question=final_state.get("direct_answer_to_user_question"),  # Pass direct answer
            final_recommendation=final_state.get("final_recommendation"),
            position_specific_advice=final_state.get("position_specific_advice"),
            generated_chart_urls=chart_urls,
            error_message=user_facing_error,
        )
        print(
            f"INFO: Async analysis complete for query '{query.user_stock_query}'. Resolved to: {response_data.stock_symbol}. User facing error: {response_data.error_message or 'None'}")
        return response_data

    except HTTPException:
        # Re-raise HTTPExceptions directly (e.g., 400, 503 from above checks)
        raise
    except Exception as e:
        # This catches unhandled exceptions from the LangGraph invocation or other logic
        print(f"CRITICAL_ERROR: Unhandled exception during async analysis for query '{query.user_stock_query}': {e}")
        traceback.print_exc()  # Log the full error to the server console
        # Return a generic error to the user
        raise HTTPException(status_code=500, detail="An unexpected internal error occurred. Please try again later.")


# --- Serve Homepage ---
@app.get("/")
async def serve_homepage():
    """Serves the frontend index.html file."""
    print(f"INFO: Attempting to serve frontend from: {index_html_path}")
    if not os.path.exists(index_html_path):
        print(f"ERROR: index.html not found at {index_html_path}")
        # Add more debug info about directory structure
        # This assumes main.py is in 'backend', and 'frontend' is a subdir of 'backend'
        current_dir_content = os.listdir(current_dir) if os.path.exists(current_dir) else "N/A (current_dir)"
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
