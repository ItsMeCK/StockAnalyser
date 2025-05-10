# backend/analyzer_workflow.py

import json
from typing import TypedDict, List, Optional, Dict, Any
from langgraph.graph import StateGraph, END
import asyncio
import pandas as pd
from io import StringIO
import re # For symbol resolution heuristics

# Langchain specific imports if needed directly by nodes (prefer to keep in utils if possible)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


# Import utility functions and variables from .analyzer_utils
# These are the building blocks for the nodes defined in this file.
from analyzer_utils import (
    initialize_analyzer,
    get_llms,
    ToolSelection, # Pydantic model for tool selection LLM
    QueryAnalysisOutput, # Pydantic model for query analysis LLM
    available_technical_indicator_tools, # Dictionary of callable tools
    tool_descriptions_for_llm, # Pre-formatted string for LLM prompt
    fetch_stock_data_yf_async,
    fetch_stock_data_alphavantage_async,
    _is_valid_ticker_async,
    get_technical_analysis_summary_content,
    get_fundamental_analysis_summary_content,
    generate_direct_answer_to_question_content,
    generate_recommendation_content,
    generate_position_advice_content,
    sanitize_value, # Utility for cleaning data for JSON
    ALPHAVANTAGE_API_KEY, # Constant, if needed for logic here (e.g. fallback decision)
    # CHARTS_DIR, CHARTS_URL_PREFIX # Not directly used by workflow logic, but by main.py
)


# --- LangGraph State Definition ---
class StockAnalysisState(TypedDict):
    user_stock_query: str
    user_question: Optional[str]
    original_user_question: Optional[str]
    analyzed_user_query_focus: Optional[str]
    question_data_type_needed: Optional[str] # 'technical', 'fundamental', 'both', 'general'
    user_position: Optional[dict] # {"shares": float, "avg_price": float}
    stock_symbol: Optional[str] # Resolved ticker
    error_message: Optional[str] # Accumulates errors
    raw_stock_data_json: Optional[Dict[str, Optional[str]]] # {"daily": json_str, "weekly": json_str, "monthly": json_str}
    stock_info_json: Optional[str] # JSON string of stock general info
    stock_financials_json: Optional[str]
    stock_balance_sheet_json: Optional[str]
    stock_cashflow_json: Optional[str]
    selected_technical_tools: Optional[List[str]] # List of tool names like ["moving_averages"]
    executed_technical_indicators: Optional[Dict[str, Dict[str, Dict]]] # {"daily": {"tool_name": {...results...}}}
    generated_chart_urls: Optional[List[str]] # List of URLs or error messages for charts
    key_technical_levels: Optional[Dict[str, Optional[float]]] # {"sma_50": val, "last_close": val}
    technical_analysis_summary: Optional[str]
    fundamental_analysis_summary: Optional[str]
    direct_answer_to_user_question: Optional[str]
    final_recommendation: Optional[str]
    position_specific_advice: Optional[str]


# --- LangGraph Nodes (Async) ---
async def start_node(state: StockAnalysisState) -> StockAnalysisState:
    """Initializes the state for a new analysis request."""
    print(f"LOG (workflow): Start node for query: {state['user_stock_query']}")
    # Initialize all relevant fields to ensure a clean state
    state["error_message"] = "" # Start with no errors
    state["stock_symbol"] = None
    state["original_user_question"] = state.get("user_question") # Preserve initial question
    state["analyzed_user_query_focus"] = None
    state["question_data_type_needed"] = None
    state["selected_technical_tools"] = []
    state["executed_technical_indicators"] = {}
    state["generated_chart_urls"] = [] # Initialize as empty list
    state["key_technical_levels"] = {}
    state["raw_stock_data_json"] = {"daily": None, "weekly": None, "monthly": None}
    state["stock_info_json"] = None
    state["stock_financials_json"] = None
    state["stock_balance_sheet_json"] = None
    state["stock_cashflow_json"] = None
    state["technical_analysis_summary"] = None
    state["fundamental_analysis_summary"] = None
    state["direct_answer_to_user_question"] = None
    state["final_recommendation"] = None
    state["position_specific_advice"] = None
    return state


async def analyze_user_query_focus_node(state: StockAnalysisState) -> StockAnalysisState:
    """Uses an LLM to analyze the focus and data type needed for the user's question."""
    print(f"LOG (workflow): Analyze user query focus node for: '{state.get('original_user_question')}'")
    _, _, _, _, current_llm_analyzer, _ = get_llms() # Get specific LLM from utils
    original_question = state.get("original_user_question")
    stock_query_context = state.get("user_stock_query", "the specified stock")

    if not original_question or original_question.strip() == "":
        print("LOG (workflow): No user question provided. Defaulting focus to general analysis.")
        state["analyzed_user_query_focus"] = "general analysis and recommendation"
        state["question_data_type_needed"] = "general" # Default data type
        return state

    try:
        # Use the QueryAnalysisOutput Pydantic model for structured output
        structured_llm = current_llm_analyzer.with_structured_output(QueryAnalysisOutput)
        prompt_template_str = (
            "You are an AI assistant. Analyze the user's question about a stock. "
            "Identify the primary focus (e.g., 'buy/sell decision', 'long-term outlook', 'dividend info', 'risk assessment', 'specific indicator like RSI'). "
            "Also, determine the primary type of data needed to answer it: 'technical', 'fundamental', 'both', or 'general' (if it's a broad question for overall analysis/recommendation). "
            "If the question is very generic like 'general analysis' or 'what do you think?', set focus to 'general analysis and recommendation' and data_type_needed to 'general'.\n"
            "User's question about {stock_context}: '{question}'"
        )
        # Await the LLM call
        analysis_output: QueryAnalysisOutput = await structured_llm.ainvoke(
            prompt_template_str.format(
                stock_context=state.get("stock_symbol") or stock_query_context, # Use resolved symbol if available
                question=original_question
            )
        )
        state["analyzed_user_query_focus"] = analysis_output.focus.strip()
        state["question_data_type_needed"] = analysis_output.data_type_needed.lower().strip()
        print(
            f"LOG (workflow): Original question: '{original_question}', Analyzed focus: '{state['analyzed_user_query_focus']}', Data needed: '{state['question_data_type_needed']}'")
    except Exception as e:
        print(f"ERROR (workflow) in analyze_user_query_focus_node: {e}")
        state["error_message"] = (state.get("error_message", "") + f"Failed to analyze user query focus: {str(e)}. ").strip()
        # Fallback to general if analysis fails
        state["analyzed_user_query_focus"] = "general analysis and recommendation"
        state["question_data_type_needed"] = "general"
    return state


async def resolve_stock_symbol_node(state: StockAnalysisState) -> StockAnalysisState:
    """Resolves user query to a yfinance ticker symbol using LLM and heuristics."""
    print(f"LOG (workflow): Resolve symbol node for query: '{state['user_stock_query']}'")
    user_query_original = state["user_stock_query"]
    _, _, _, current_llm_resolver, _, _ = get_llms() # Get the symbol resolver LLM
    resolved_symbol = None
    error_accumulator = [] # To gather reasons if resolution fails

    try:
        print(f"LOG (workflow): Attempting LLM based symbol resolution for '{user_query_original}' (yfinance context)...")
        # Define the prompt for the LLM
        prompt_template = ChatPromptTemplate.from_messages([
            ("system",
             "You are an expert financial assistant. Your task is to identify the most likely official stock ticker symbol based on the user's query, suitable for yfinance. "
             "If the company sounds Indian (e.g., Reliance, Infosys, Tata Steel), suggest the symbol with '.NS' suffix (e.g., RELIANCE.NS, INFY.NS). "
             "For well-known international companies (e.g., Apple, Microsoft, Google), provide their common US exchange ticker (e.g., AAPL, MSFT, GOOGL). "
             "If the query is already a valid-looking ticker, return it as is. "
             "If highly ambiguous or unclear, or if the query is nonsensical as a company name, return 'UNKNOWN'. "
             "Respond with ONLY the ticker symbol or 'UNKNOWN'."),
            ("human", "User query: {query}")
        ])
        resolve_chain = prompt_template | current_llm_resolver | StrOutputParser()
        llm_suggested_ticker = await resolve_chain.ainvoke({"query": user_query_original})
        llm_suggested_ticker = llm_suggested_ticker.strip().upper().replace("'", "").replace("\"", "") # Clean up LLM output
        print(f"LOG (workflow): LLM suggested ticker for yfinance: '{llm_suggested_ticker}'")

        if llm_suggested_ticker and llm_suggested_ticker not in ["UNKNOWN", "N/A", ""]:
            if await _is_valid_ticker_async(llm_suggested_ticker): # Validate with yfinance
                resolved_symbol = llm_suggested_ticker
                print(f"LOG (workflow): LLM suggested ticker '{resolved_symbol}' validated by yfinance.")
            else:
                msg = f"LLM suggestion '{llm_suggested_ticker}' was not validated by yfinance."
                print(f"LOG (workflow): {msg}"); error_accumulator.append(msg)
        else:
            msg = "LLM could not confidently suggest a ticker or returned UNKNOWN/N/A."
            print(f"LOG (workflow): {msg}"); error_accumulator.append(msg)
    except Exception as e:
        msg = f"Exception during LLM symbol resolution: {e}"; print(f"ERROR (workflow): {msg}"); error_accumulator.append(msg)

    # Fallback to heuristics if LLM fails or validation fails
    if not resolved_symbol:
        print(f"LOG (workflow): LLM resolution failed or invalid. Falling back to yfinance heuristics for '{user_query_original}'...")
        user_query_upper = user_query_original.strip().upper()
        potential_symbols_to_try = []
        # Heuristic 1: Already looks like a ticker (e.g., AAPL, RELIANCE.NS)
        if re.match(r"^[A-Z0-9.-]+$", user_query_upper) and ('.' in user_query_upper or len(user_query_upper) <= 5):
            potential_symbols_to_try.append(user_query_upper)
        # Heuristic 2: Indian company name without .NS
        if not '.' in user_query_upper and re.match(r"^[A-Z\s]+$", user_query_upper): # Check if it's all caps and spaces (company name like)
            potential_symbols_to_try.append(f"{user_query_upper.replace(' ', '')}.NS")
        # Heuristic 3: Add original query as is, if not already covered
        if user_query_upper not in potential_symbols_to_try:
            potential_symbols_to_try.append(user_query_upper)
        # Heuristic 4: If no space, no dot, try adding .NS (common for Indian symbols entered without it)
        if ' ' not in user_query_upper and '.' not in user_query_upper and not user_query_upper.endswith(".NS"):
            potential_symbols_to_try.append(f"{user_query_upper}.NS")

        unique_symbols_to_try = list(dict.fromkeys(potential_symbols_to_try)) # Remove duplicates
        print(f"LOG (workflow): yfinance Heuristic: Potential symbols to try: {unique_symbols_to_try}")

        for symbol_attempt in unique_symbols_to_try:
            if await _is_valid_ticker_async(symbol_attempt):
                resolved_symbol = symbol_attempt
                print(f"LOG (workflow): yfinance Heuristic resolved '{user_query_original}' to '{resolved_symbol}'")
                error_accumulator = [] # Clear previous errors if heuristic succeeds
                break
            else:
                print(f"LOG (workflow): yfinance Heuristic: '{symbol_attempt}' is not valid.")

    if resolved_symbol:
        state["stock_symbol"] = resolved_symbol
    else:
        final_error_msg = " ".join(error_accumulator) if error_accumulator else f"Could not resolve '{user_query_original}' to a known stock symbol."
        state["error_message"] = (state.get("error_message", "") + final_error_msg).strip()
        print(f"LOG (workflow): Failed to resolve symbol for '{user_query_original}'. Final error: {state['error_message']}")
    return state


async def fetch_data_node(state: StockAnalysisState) -> StockAnalysisState:
    """Fetches stock data using yfinance, with Alpha Vantage as fallback."""
    resolved_symbol = state.get("stock_symbol")
    if state.get("error_message") or not resolved_symbol: # Check for prior errors or no symbol
        state["error_message"] = (state.get("error_message", "") + " Skipping data fetch due to unresolved symbol or prior error.").strip()
        return state

    print(f"LOG (workflow): Attempting yfinance fetch for: {resolved_symbol}")
    try:
        fetched_data = await fetch_stock_data_yf_async(resolved_symbol) # From .analyzer_utils
        state["stock_symbol"] = fetched_data.get("resolved_symbol", resolved_symbol) # Update symbol if yfinance refines it (rare)

        info_data = fetched_data.get("info", {})
        if not isinstance(info_data, dict): info_data = {} # Ensure it's a dict
        # Sanitize all values in the info dictionary before JSON dumping
        sanitized_stock_info = {str(k): sanitize_value(v) for k, v in info_data.items()}
        state["stock_info_json"] = json.dumps(sanitized_stock_info)

        # Serialize DataFrames to JSON strings
        state["raw_stock_data_json"] = {
            tf: (data.to_json(orient='split', date_format='iso') if data is not None and not data.empty else None)
            for tf, data in [
                ("daily", fetched_data.get("daily")),
                ("weekly", fetched_data.get("weekly")),
                ("monthly", fetched_data.get("monthly"))
            ]
        }
        state["stock_financials_json"] = fetched_data.get("financials").to_json(orient='split', date_format='iso') if fetched_data.get("financials") is not None and not fetched_data.get("financials").empty else None
        state["stock_balance_sheet_json"] = fetched_data.get("balance_sheet").to_json(orient='split', date_format='iso') if fetched_data.get("balance_sheet") is not None and not fetched_data.get("balance_sheet").empty else None
        state["stock_cashflow_json"] = fetched_data.get("cashflow").to_json(orient='split', date_format='iso') if fetched_data.get("cashflow") is not None and not fetched_data.get("cashflow").empty else None

        if not state["raw_stock_data_json"]["daily"]: # Check if primary daily data is missing
            print(f"Warning: Daily historical price data is empty for {state['stock_symbol']} from yfinance.")
            if not state["raw_stock_data_json"]["weekly"] and not state["raw_stock_data_json"]["monthly"]:
                # If all historical data is missing, it's a more significant issue
                state["error_message"] = (state.get("error_message", "") + f" All historical price data is missing for {state['stock_symbol']} from yfinance. Analysis will be limited. ").strip()
        print(f"LOG (workflow): Successfully fetched and serialized data for {state['stock_symbol']} using yfinance.")

    except Exception as yf_error:
        print(f"ERROR (workflow) in yfinance fetch_data_node for {resolved_symbol}: {yf_error}")
        state["error_message"] = (state.get("error_message", "") + f" Primary data fetch (yfinance) failed for {resolved_symbol}: {str(yf_error)}. ").strip()

        # --- Alpha Vantage Fallback Logic ---
        if ALPHAVANTAGE_API_KEY: # Check if AV key is configured
            print(f"LOG (workflow): yfinance failed. Attempting Alpha Vantage fallback for {resolved_symbol}...")
            try:
                av_symbol = resolved_symbol.replace(".NS", "") # AV usually doesn't need .NS
                fetched_av_data = await fetch_stock_data_alphavantage_async(av_symbol) # From .analyzer_utils

                state["stock_symbol"] = fetched_av_data.get("resolved_symbol", av_symbol)
                info_data_av = fetched_av_data.get("info", {})
                if not isinstance(info_data_av, dict): info_data_av = {}
                sanitized_stock_info_av = {str(k): sanitize_value(v) for k, v in info_data_av.items()}
                state["stock_info_json"] = json.dumps(sanitized_stock_info_av)

                state["raw_stock_data_json"] = {
                    tf: (data.to_json(orient='split', date_format='iso') if data is not None and not data.empty else None)
                    for tf, data in [
                        ("daily", fetched_av_data.get("daily")),
                        ("weekly", fetched_av_data.get("weekly")),
                        ("monthly", fetched_av_data.get("monthly"))
                    ]
                }
                # Alpha Vantage free tier doesn't provide detailed financials/balance sheet/cashflow easily
                state["stock_financials_json"] = None
                state["stock_balance_sheet_json"] = None
                state["stock_cashflow_json"] = None

                if not state["raw_stock_data_json"]["daily"]: # Critical if AV also fails for daily data
                    raise ValueError(f"Alpha Vantage also returned no daily data for {av_symbol}.")

                print(f"LOG (workflow): Successfully fetched data for {state['stock_symbol']} using Alpha Vantage fallback.")
                # Update error message to reflect fallback success or partial success
                state["error_message"] = (state.get("error_message", "") + f" Used Alpha Vantage as fallback for {resolved_symbol}. ").strip()

            except Exception as av_error:
                print(f"ERROR (workflow) in Alpha Vantage fallback for {resolved_symbol}: {av_error}")
                state["error_message"] = (state.get("error_message", "") + f" Alpha Vantage fallback also failed for {resolved_symbol}: {str(av_error)}. ").strip()
        else:
            print(f"LOG (workflow): Alpha Vantage API key not available or is 'demo', skipping fallback for {resolved_symbol}.")
    return state


async def select_technical_tools_node(state: StockAnalysisState) -> StockAnalysisState:
    """Selects technical indicator tools based on query focus using an LLM."""
    print(f"LOG (workflow): Select technical INDICATOR tools node for {state.get('stock_symbol', 'N/A')}")
    _, current_tool_selector_llm, _, _, _, _ = get_llms() # Get the tool selector LLM
    print(tool_descriptions_for_llm)
    # Critical check for tool descriptions needed by the LLM
    if not tool_descriptions_for_llm or "Error:" in tool_descriptions_for_llm:
        state["error_message"] = (state.get("error_message", "") + "Critical error: Indicator tool descriptions not available. Defaulting to all tools. ").strip()
        state["selected_technical_tools"] = list(available_technical_indicator_tools.keys())
        return state

    # Skip if prior errors or no data to analyze
    if state.get("error_message") or not state.get("raw_stock_data_json") or not any(state["raw_stock_data_json"].values()):
        state["selected_technical_tools"] = [] # No tools if no data or error
        state["error_message"] = (state.get("error_message", "") + "Skipped indicator tool selection due to missing data or prior error. ").strip()
        return state

    query_context = state.get("analyzed_user_query_focus") or state.get("original_user_question") or "Provide a general technical analysis."
    print(f"LOG (workflow): Using query context for tool selection: '{query_context}'")

    # Prompt for the LLM to select tools
    prompt = (
        f"Based on the user's query focus: \"{query_context}\" for the stock {state['stock_symbol']}, "
        f"which of the following technical INDICATOR tools should be used?\n"
        f"Available indicator tools and their descriptions:\n{tool_descriptions_for_llm}\n\n"
        f"Respond with a JSON object containing a single key \"tool_names\" with a list of selected indicator tool names. "
        f"For a general analysis, select all indicator tools: [\"moving_averages\", \"oscillators\", \"volatility\"]. "
        f"Example response: {{\"tool_names\": [\"moving_averages\", \"oscillators\"]}}"
    )
    try:
        structured_llm = current_tool_selector_llm.with_structured_output(ToolSelection) # Use Pydantic model
        response_model = await structured_llm.ainvoke(prompt)
        selected_tools = response_model.tool_names

        # Validate selected tools against available tools
        valid_selected_tools = [t for t in selected_tools if t in available_technical_indicator_tools]
        if len(valid_selected_tools) != len(selected_tools):
            print(f"Warning: LLM selected some invalid indicator tools. Original: {selected_tools}, Validated: {valid_selected_tools}")
        state["selected_technical_tools"] = valid_selected_tools

        if not valid_selected_tools: # If LLM returns empty or all invalid, default to all
            print("Warning: No valid indicator tools selected by LLM. Defaulting to all indicator tools.")
            state["selected_technical_tools"] = list(available_technical_indicator_tools.keys())
        print(f"LOG (workflow): Selected indicator tools: {state['selected_technical_tools']}")
    except Exception as e:
        print(f"ERROR (workflow) in select_technical_tools_node: {e}. Defaulting to all indicator tools.")
        state["error_message"] = (state.get("error_message", "") + f"Indicator tool selection failed: {str(e)}. Defaulting. ").strip()
        state["selected_technical_tools"] = list(available_technical_indicator_tools.keys())
    return state


async def execute_technical_tools_node(state: StockAnalysisState) -> StockAnalysisState:
    """Executes the selected technical indicator tools."""
    print(f"LOG (workflow): Execute technical INDICATOR tools node for {state.get('stock_symbol', 'N/A')}")

    if state.get("error_message") or not state.get("selected_technical_tools") or not state.get("raw_stock_data_json"):
        state["error_message"] = (state.get("error_message", "") + "Skipped technical indicator tool execution due to prior error or missing data/tools. ").strip()
        state["executed_technical_indicators"] = {} # Ensure it's an empty dict
        return state

    selected_tools = state["selected_technical_tools"]
    if not selected_tools: # Should be caught by above, but defensive check
        print("LOG (workflow): No indicator tools selected to execute.")
        state["executed_technical_indicators"] = {}
        return state

    raw_data_map = state["raw_stock_data_json"] # {"daily": json_str, ...}
    indicator_results = {"daily": {}, "weekly": {}, "monthly": {}} # Initialize structure
    tool_invocations = [] # For asyncio.gather

    for timeframe in ["daily", "weekly", "monthly"]:
        df_json_str = raw_data_map.get(timeframe)
        if not df_json_str: # No data for this timeframe
            for tool_name in selected_tools:
                indicator_results[timeframe][tool_name] = {"error": f"No data available for {timeframe} timeframe."}
            continue # Move to next timeframe

        for tool_name in selected_tools:
            tool_func = available_technical_indicator_tools.get(tool_name) # Get tool func from utils
            if tool_func:
                # Add the async call to the list for concurrent execution
                tool_invocations.append(
                    (timeframe, tool_name, tool_func.ainvoke({"data_json": df_json_str}))
                )
            else:
                indicator_results[timeframe][tool_name] = {"error": "Indicator tool function not found."}

    # Execute all tool invocations concurrently
    gathered_results = await asyncio.gather(*(inv[2] for inv in tool_invocations), return_exceptions=True)

    result_index = 0
    key_levels = state.get("key_technical_levels", {}) # Preserve existing key levels if any

    for timeframe, tool_name, _ in tool_invocations: # Iterate through the original invocation list to map results
        current_result = gathered_results[result_index]
        if isinstance(current_result, Exception):
            error_str = str(current_result)
            print(f"    Error executing indicator tool '{tool_name}' for {timeframe}: {error_str}")
            indicator_results[timeframe][tool_name] = {"error": error_str}
        else:
            indicator_results[timeframe][tool_name] = current_result
            # Extract key levels from daily results for position advice
            if timeframe == 'daily' and isinstance(current_result, dict) and not current_result.get("error"):
                if tool_name == 'moving_averages':
                    key_levels['sma_50'] = current_result.get('sma_50')
                    key_levels['sma_200'] = current_result.get('sma_200')
                    key_levels['last_close'] = current_result.get('last_close') # Important for context
                elif tool_name == 'volatility': # e.g., Bollinger Bands
                    key_levels['bb_low'] = current_result.get('bb_low')
                    key_levels['bb_high'] = current_result.get('bb_high')
        result_index += 1

    state["executed_technical_indicators"] = indicator_results
    state["key_technical_levels"] = {k: v for k, v in key_levels.items() if v is not None} # Clean None values
    return state


async def generate_chart_node(state: StockAnalysisState) -> StockAnalysisState:
    """Placeholder for chart generation - currently disabled."""
    # This node is currently disabled in the graph as per user request in analyzer_utils.py comments
    # If re-enabled, it would call a chart generation utility.
    print(f"LOG (workflow): Generate chart node for {state.get('stock_symbol', 'N/A')} (currently disabled)")
    state["generated_chart_urls"] = ["Chart generation is currently disabled."] # Default message
    return state


async def summarize_technical_analysis_node(state: StockAnalysisState) -> StockAnalysisState:
    """Generates a summary of the technical analysis."""
    print(f"LOG (workflow): Summarize TA node for {state.get('stock_symbol', 'N/A')}")
    indicators = state.get("executed_technical_indicators")

    # If critical errors occurred before this node, or no indicators were executed.
    if state.get("error_message") and not indicators: # Check for prior errors
        state["technical_analysis_summary"] = "Technical analysis summary cannot be generated due to critical prior errors."
        return state
    if not indicators or all(not tf_data for tf_data in indicators.values()): # Check if indicators dict is empty or all timeframes are empty
        state["technical_analysis_summary"] = "No technical indicators were calculated or available to summarize."
        return state

    try:
        state["technical_analysis_summary"] = await get_technical_analysis_summary_content(
            state["stock_symbol"], indicators # Pass symbol and indicators from state
        )
    except Exception as e:
        print(f"ERROR (workflow) in summarize_technical_analysis_node: {e}")
        state["error_message"] = (state.get("error_message", "") + f"TA summary generation failed: {str(e)}. ").strip()
        state["technical_analysis_summary"] = "Failed to generate technical analysis summary due to an unexpected error."
    return state


async def fundamental_analysis_node(state: StockAnalysisState) -> StockAnalysisState:
    """Generates a summary of the fundamental analysis."""
    print(f"LOG (workflow): Fundamental analysis node for {state.get('stock_symbol', 'N/A')}")

    # Skip if prior errors or no stock info JSON
    if state.get("error_message") or not state.get("stock_info_json"):
        state["error_message"] = (state.get("error_message", "") + "Skipped fundamental analysis due to missing stock info or prior error. ").strip()
        state["fundamental_analysis_summary"] = "Fundamental analysis skipped due to missing stock information or prior errors."
        return state

    try:
        state["fundamental_analysis_summary"] = await get_fundamental_analysis_summary_content(
            state["stock_symbol"],
            state["stock_info_json"],
            state.get("stock_financials_json"),
            state.get("stock_balance_sheet_json"),
            state.get("stock_cashflow_json")
        )
    except Exception as e:
        print(f"ERROR (workflow) in fundamental_analysis_node: {e}")
        state["error_message"] = (state.get("error_message", "") + f"Fundamental analysis failed: {str(e)}. ").strip()
        state["fundamental_analysis_summary"] = "Failed to generate fundamental analysis summary due to an unexpected error."
    return state


async def generate_direct_answer_node(state: StockAnalysisState) -> StockAnalysisState:
    """Generates a direct answer to the user's specific question if provided."""
    print(f"LOG (workflow): Generate direct answer node for {state.get('stock_symbol', 'N/A')}")
    original_question = state.get("original_user_question")

    if not original_question or original_question.strip() == "": # No question, no direct answer needed
        state["direct_answer_to_user_question"] = None
        return state

    # Check for critical missing pieces of information needed for a good answer
    ta_summary = state.get("technical_analysis_summary")
    fa_summary = state.get("fundamental_analysis_summary")
    critical_summaries_missing = not ta_summary or not fa_summary or \
                                 any(s in (ta_summary or "").lower() for s in ["failed", "cannot be generated", "skipped", "n/a", "no technical indicators"]) or \
                                 any(s in (fa_summary or "").lower() for s in ["failed", "cannot be generated", "skipped", "n/a"])


    if state.get("error_message") and critical_summaries_missing:
        state["direct_answer_to_user_question"] = "Could not answer your specific question due to missing analysis data or prior errors."
        state["error_message"] = (state.get("error_message", "") + " Skipped direct question answering due to missing summaries or prior errors. ").strip()
        return state

    try:
        state["direct_answer_to_user_question"] = await generate_direct_answer_to_question_content(
            state["stock_symbol"],
            original_question,
            state.get("analyzed_user_query_focus"),
            ta_summary,
            fa_summary,
            state.get("executed_technical_indicators"),
            state.get("stock_info_json"),
            state.get("question_data_type_needed")
        )
        print(f"LOG (workflow): Direct answer generated for: '{original_question}'")
    except Exception as e:
        print(f"ERROR (workflow) in generate_direct_answer_node: {e}")
        state["error_message"] = (state.get("error_message", "") + f"Failed to generate direct answer: {str(e)}. ").strip()
        state["direct_answer_to_user_question"] = "Sorry, I encountered an issue trying to answer your specific question."
    return state


async def recommendation_node(state: StockAnalysisState) -> StockAnalysisState:
    """Generates overall Buy/Sell/Hold recommendation."""
    print(f"LOG (workflow): Recommendation node for {state.get('stock_symbol', 'N/A')}")
    ta_summary = state.get("technical_analysis_summary", "Technical analysis information was not available.")
    fa_summary = state.get("fundamental_analysis_summary", "Fundamental analysis information was not available.")

    # Check if summaries are critically flawed
    critical_ta = any(s in ta_summary.lower() for s in ["cannot be generated", "skipped", "n/a", "failed", "no technical indicators"])
    critical_fa = any(s in fa_summary.lower() for s in ["cannot be generated", "skipped", "n/a", "failed"])

    if state.get("error_message") and (critical_ta or critical_fa):
        state["final_recommendation"] = "Cannot generate overall recommendation due to prior errors or missing essential analysis."
        return state
    try:
        state["final_recommendation"] = await generate_recommendation_content(
            state["stock_symbol"], ta_summary, fa_summary
        )
        print("LOG (workflow): Overall recommendation generated.")
    except Exception as e:
        print(f"ERROR (workflow) in recommendation_node: {e}")
        state["error_message"] = (state.get("error_message", "") + f"Overall recommendation generation failed: {str(e)}. ").strip()
        state["final_recommendation"] = "Failed to generate overall recommendation due to an unexpected error."
    return state


async def position_advice_node(state: StockAnalysisState) -> StockAnalysisState:
    """Generates advice for an existing user position, if provided."""
    print(f"LOG (workflow): Position advice node for {state.get('stock_symbol', 'N/A')}")
    final_rec = state.get("final_recommendation")
    user_pos = state.get("user_position")
    key_levels = state.get("key_technical_levels", {}) # Default to empty dict

    if not user_pos: # No position provided by user
        state["position_specific_advice"] = None # Or "No user position provided for advice."
        return state

    # If recommendation failed or is not conclusive
    if not final_rec or any(s in final_rec.lower() for s in ["cannot generate", "failed"]):
        state["position_specific_advice"] = "Position advice cannot be generated as the overall recommendation is unavailable or inconclusive."
        return state

    try:
        state["position_specific_advice"] = await generate_position_advice_content(
            state["stock_symbol"], final_rec, user_pos, key_levels
        )
        print("LOG (workflow): Position-specific advice generated.")
    except Exception as e:
        print(f"ERROR (workflow) in position_advice_node: {e}")
        state["error_message"] = (state.get("error_message", "") + f"Position advice generation failed: {str(e)}. ").strip()
        state["position_specific_advice"] = "Failed to generate position-specific advice due to an unexpected error."
    return state


# --- Compile LangGraph App ---
def get_stock_analyzer_app():
    """Compiles and returns the LangGraph application."""
    # Ensure LLMs are initialized (though initialize_analyzer() in main.py's startup does this globally)
    # get_llms() # Called by each node as needed, or initialize_analyzer() can be called here too.

    workflow = StateGraph(StockAnalysisState)

    # Add all nodes defined in this file
    workflow.add_node("start", start_node)
    workflow.add_node("resolve_symbol", resolve_stock_symbol_node)
    workflow.add_node("analyze_query_focus", analyze_user_query_focus_node)
    workflow.add_node("fetch_data", fetch_data_node)
    workflow.add_node("select_technical_tools", select_technical_tools_node)
    workflow.add_node("execute_technical_tools", execute_technical_tools_node)
    workflow.add_node("generate_chart", generate_chart_node) # Add even if disabled, can be conditional later
    workflow.add_node("summarize_technical_analysis", summarize_technical_analysis_node)
    workflow.add_node("fundamental_analysis", fundamental_analysis_node)
    workflow.add_node("generate_direct_answer", generate_direct_answer_node)
    workflow.add_node("generate_recommendation", recommendation_node)
    workflow.add_node("generate_position_advice", position_advice_node)

    # Define the workflow graph
    workflow.set_entry_point("start")
    workflow.add_edge("start", "resolve_symbol")
    workflow.add_edge("resolve_symbol", "analyze_query_focus") # Analyze focus after knowing the symbol
    workflow.add_edge("analyze_query_focus", "fetch_data")
    workflow.add_edge("fetch_data", "select_technical_tools")
    workflow.add_edge("select_technical_tools", "execute_technical_tools")
    workflow.add_edge("execute_technical_tools", "generate_chart") # Chart after indicators
    workflow.add_edge("generate_chart", "summarize_technical_analysis") # Summarize after indicators/chart
    workflow.add_edge("summarize_technical_analysis", "fundamental_analysis")
    workflow.add_edge("fundamental_analysis", "generate_direct_answer")
    workflow.add_edge("generate_direct_answer", "generate_recommendation")
    workflow.add_edge("generate_recommendation", "generate_position_advice")
    workflow.add_edge("generate_position_advice", END) # End of the graph

    return workflow.compile()

# --- Standalone Test (Optional, for testing this workflow file directly) ---
async def main_test_workflow():
    """For testing the workflow independently."""
    # Initialize analyzer (which also initializes LLMs)
    if initialize_analyzer(): # From .analyzer_utils
        app = get_stock_analyzer_app() # Defined in THIS file
        print("Stock Analyzer Workflow App compiled for standalone async testing.")
        # Example inputs
        inputs = {
            "user_stock_query": "Infosys",
            "user_question": "What are the short term risks for INFY?",
            "user_position": {"shares": 10, "avg_price": 1400.00}
        }
        # inputs = {"user_stock_query": "Apple", "user_question": "Is it good for long term?"}


        try:
            final_result_state = await app.ainvoke(inputs)
            print("\n--- FINAL STATE (for standalone workflow async test) ---")
            # Pretty print the final state
            print(json.dumps(final_result_state, indent=2, default=str))

            # Simulate API response structure for clarity
            api_response_data = {
                "user_stock_query": final_result_state.get("user_stock_query"),
                "original_user_question": final_result_state.get("original_user_question"),
                "analyzed_user_query_focus": final_result_state.get("analyzed_user_query_focus"),
                "question_data_type_needed": final_result_state.get("question_data_type_needed"),
                "stock_symbol": final_result_state.get("stock_symbol"),
                "generated_chart_urls": final_result_state.get("generated_chart_urls"),
                "technical_analysis_summary": final_result_state.get("technical_analysis_summary"),
                "fundamental_analysis_summary": final_result_state.get("fundamental_analysis_summary"),
                "direct_answer_to_user_question": final_result_state.get("direct_answer_to_user_question"),
                "final_recommendation": final_result_state.get("final_recommendation"),
                "position_specific_advice": final_result_state.get("position_specific_advice"),
                "error_message": final_result_state.get("error_message") if final_result_state.get("error_message", "").strip() else None,
            }
            print("\n--- SIMULATED API RESPONSE DATA (for standalone workflow async test) ---")
            print(json.dumps(api_response_data, indent=2, default=str))

        except Exception as e:
            print(f"Error during standalone workflow async test invocation: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Failed to initialize analyzer for workflow testing (check API keys).")

if __name__ == "__main__":
    # This allows running `python backend/analyzer_workflow.py` from the project root for testing
    # Make sure your PYTHONPATH is set up correctly if you run it from elsewhere,
    # or that your IDE understands the project structure.
    asyncio.run(main_test_workflow())
