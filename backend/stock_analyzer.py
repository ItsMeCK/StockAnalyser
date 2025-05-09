# backend/stock_analyzer.py

import os
import json
from typing import TypedDict, List, Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate  # For structured prompts
from langchain_core.output_parsers import StrOutputParser  # For simple string output
from langgraph.graph import StateGraph, END
import yfinance as yf
import pandas as pd
import numpy as np
from ta.volatility import BollingerBands
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
import warnings
from io import StringIO
import asyncio
import re
from dotenv import load_dotenv

load_dotenv()

warnings.simplefilter(action='ignore', category=FutureWarning)

ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_LANGCHAIN_TRACING_V2 = "LANGCHAIN_TRACING_V2"
ENV_LANGCHAIN_ENDPOINT = "LANGCHAIN_ENDPOINT"
ENV_LANGCHAIN_API_KEY = "LANGCHAIN_API_KEY"
ENV_LANGCHAIN_PROJECT = "LANGCHAIN_PROJECT"

OPENAI_API_KEY = os.getenv(ENV_OPENAI_API_KEY)
LANGCHAIN_TRACING_V2_RAW = os.getenv("LANGSMITH_TRACING", os.getenv(ENV_LANGCHAIN_TRACING_V2, "false"))
LANGCHAIN_TRACING_V2 = LANGCHAIN_TRACING_V2_RAW.lower() == "true"
LANGCHAIN_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", os.getenv(ENV_LANGCHAIN_ENDPOINT))
LANGCHAIN_API_KEY = os.getenv("LANGSMITH_API_KEY", os.getenv(ENV_LANGCHAIN_API_KEY))
LANGCHAIN_PROJECT = os.getenv("LANGSMITH_PROJECT", os.getenv(ENV_LANGCHAIN_PROJECT, "Stock Analyzer Chat App Async"))

LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o")  # Changed to gpt-4o for better reasoning
LLM_SYMBOL_RESOLUTION_MODEL = os.getenv("LLM_SYMBOL_RESOLUTION_MODEL",
                                        "gpt-3.5-turbo")  # Cheaper/faster for symbol resolution
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.4))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 500))
POSITION_ADVICE_MAX_TOKENS = int(os.getenv("POSITION_ADVICE_MAX_TOKENS", 700))
TOOL_SELECTOR_LLM_MAX_TOKENS = int(os.getenv("TOOL_SELECTOR_LLM_MAX_TOKENS", 250))

CHARTS_DIR = "/tmp/charts"
try:
    os.makedirs(CHARTS_DIR, exist_ok=True)
except OSError as e:
    print(f"Warning: Could not create CHARTS_DIR ({CHARTS_DIR}) at module level: {e}")
CHARTS_URL_PREFIX = "/charts_data"

tool_descriptions_for_llm = ""


class ToolSelection(BaseModel):
    tool_names: List[str] = Field(
        description="List of technical indicator tool names to be called (e.g., moving_averages, oscillators).")


def initialize_analyzer():
    global tool_descriptions_for_llm
    if not OPENAI_API_KEY:
        print(f"CRITICAL ERROR: The environment variable {ENV_OPENAI_API_KEY} is not set.")
        return False
    if LANGCHAIN_TRACING_V2 and LANGCHAIN_API_KEY and LANGCHAIN_ENDPOINT:
        os.environ[ENV_LANGCHAIN_TRACING_V2] = "true"
        os.environ[ENV_LANGCHAIN_ENDPOINT] = LANGCHAIN_ENDPOINT
        os.environ[ENV_LANGCHAIN_API_KEY] = LANGCHAIN_API_KEY
        if LANGCHAIN_PROJECT:
            os.environ[ENV_LANGCHAIN_PROJECT] = LANGCHAIN_PROJECT
        print(f"LangSmith tracing enabled. Project: {LANGCHAIN_PROJECT}")
    else:
        print("LangSmith tracing is not configured or missing some required environment variables.")
    try:
        descriptions = []
        for name, tool_func_obj in available_technical_indicator_tools.items():
            desc = getattr(tool_func_obj, 'description', None) or getattr(tool_func_obj, '__doc__', None)
            if desc:
                first_line_doc = next((line for line in desc.strip().splitlines() if line.strip()),
                                      f"Tool to calculate {name.replace('_', ' ')}.")
            else:
                first_line_doc = f"Tool to calculate {name.replace('_', ' ')}."
            descriptions.append(f"- {name}: {first_line_doc}")
        tool_descriptions_for_llm = "\n".join(descriptions)
        if not tool_descriptions_for_llm:
            print("Warning: tool_descriptions_for_llm is empty.")
        else:
            print("Indicator tool descriptions for LLM constructed successfully.")
    except Exception as e:
        print(f"ERROR: Failed to construct tool_descriptions_for_llm: {e}")
        tool_descriptions_for_llm = "Error: Could not load indicator tool descriptions."
    return True


llm = None
tool_selector_llm = None
llm_pos_advice = None
llm_symbol_resolver = None  # New LLM instance for symbol resolution


def get_llms():
    global llm, tool_selector_llm, llm_pos_advice, llm_symbol_resolver
    if llm is None: llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=LLM_TEMPERATURE, max_tokens=LLM_MAX_TOKENS)
    if tool_selector_llm is None: tool_selector_llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0.2,
                                                                 max_tokens=TOOL_SELECTOR_LLM_MAX_TOKENS)
    if llm_pos_advice is None: llm_pos_advice = ChatOpenAI(model=LLM_MODEL_NAME, temperature=LLM_TEMPERATURE,
                                                           max_tokens=POSITION_ADVICE_MAX_TOKENS)
    if llm_symbol_resolver is None:
        llm_symbol_resolver = ChatOpenAI(
            model=LLM_SYMBOL_RESOLUTION_MODEL,
            temperature=0.0,  # Low temperature for factual recall
            max_tokens=50  # Ticker symbols are short
        )
    return llm, tool_selector_llm, llm_pos_advice, llm_symbol_resolver


def _sanitize_indicator_value(value: Any) -> Any:
    if isinstance(value, pd.Series):
        if value.empty: return None
        if value.isna().all(): return None
        return value.to_json(orient='split', date_format='iso')
    elif pd.isna(value):
        return None
    elif isinstance(value, (np.generic, int, float)):
        return float(value)
    elif isinstance(value, str):
        return value
    return value


@tool
async def calculate_moving_averages(data_json: str) -> Dict[str, Optional[Any]]:
    """
    Calculates Simple Moving Averages (SMA 20, 50, 200) and Exponential Moving Averages (EMA 12, 26)
    for the provided stock data. Input data_json is a JSON string of a pandas DataFrame (orient='split').
    Returns latest values and full series as JSON strings.
    """
    try:
        df = pd.read_json(StringIO(data_json), orient='split')
        if df.empty or 'Close' not in df.columns: return {"error": "Invalid or empty data for moving averages"}
        close = df["Close"]
        indicators = {
            "sma_20": SMAIndicator(close, window=20).sma_indicator().iloc[-1] if len(close) >= 20 else None,
            "sma_50": SMAIndicator(close, window=50).sma_indicator().iloc[-1] if len(close) >= 50 else None,
            "sma_200": SMAIndicator(close, window=200).sma_indicator().iloc[-1] if len(close) >= 200 else None,
            "ema_12": EMAIndicator(close, window=12).ema_indicator().iloc[-1] if len(close) >= 12 else None,
            "ema_26": EMAIndicator(close, window=26).ema_indicator().iloc[-1] if len(close) >= 26 else None,
            "last_close": close.iloc[-1] if not close.empty else None,
            "sma_20_series": SMAIndicator(close, window=20).sma_indicator() if len(close) >= 20 else pd.Series(
                dtype=float),
            "sma_50_series": SMAIndicator(close, window=50).sma_indicator() if len(close) >= 50 else pd.Series(
                dtype=float),
            "sma_200_series": SMAIndicator(close, window=200).sma_indicator() if len(close) >= 200 else pd.Series(
                dtype=float),
        }
        return {k: _sanitize_indicator_value(v) for k, v in indicators.items()}
    except Exception as e:
        return {"error": f"Failed to calculate moving averages: {str(e)}"}


@tool
async def calculate_oscillators(data_json: str) -> Dict[str, Optional[Any]]:
    """
    Calculates MACD (Moving Average Convergence Divergence) and RSI (Relative Strength Index).
    Input data_json is a JSON string of a pandas DataFrame (orient='split').
    Returns latest values and full series as JSON strings.
    """
    try:
        df = pd.read_json(StringIO(data_json), orient='split')
        if df.empty or 'Close' not in df.columns: return {"error": "Invalid data for oscillators"}
        close = df["Close"]
        indicators = {}
        if len(close) >= 26:
            macd_indicator = MACD(close)
            indicators["macd"] = macd_indicator.macd().iloc[-1];
            indicators["macd_signal"] = macd_indicator.macd_signal().iloc[-1];
            indicators["macd_diff"] = macd_indicator.macd_diff().iloc[-1]
            indicators["macd_series"] = macd_indicator.macd();
            indicators["macd_signal_series"] = macd_indicator.macd_signal();
            indicators["macd_hist_series"] = macd_indicator.macd_diff()
        else:
            indicators["macd"], indicators["macd_signal"], indicators["macd_diff"] = None, None, None
            indicators["macd_series"], indicators["macd_signal_series"], indicators["macd_hist_series"] = pd.Series(
                dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
        if len(close) >= 14:
            rsi_series = RSIIndicator(close).rsi();
            indicators["rsi"] = rsi_series.iloc[-1];
            indicators["rsi_series"] = rsi_series
        else:
            indicators["rsi"] = None; indicators["rsi_series"] = pd.Series(dtype=float)
        return {k: _sanitize_indicator_value(v) for k, v in indicators.items()}
    except Exception as e:
        return {"error": f"Failed to calculate oscillators: {str(e)}"}


@tool
async def calculate_volatility_indicators(data_json: str) -> Dict[str, Optional[Any]]:
    """
    Calculates Bollinger Bands (High, Low, Moving Average).
    Input data_json is a JSON string of a pandas DataFrame (orient='split').
    Returns latest values and full series as JSON strings.
    """
    try:
        df = pd.read_json(StringIO(data_json), orient='split')
        if df.empty or 'Close' not in df.columns: return {"error": "Invalid data for volatility"}
        close = df["Close"]
        indicators = {}
        if len(close) >= 20:
            bb_indicator = BollingerBands(close)
            indicators["bb_high"] = bb_indicator.bollinger_hband().iloc[-1];
            indicators["bb_low"] = bb_indicator.bollinger_lband().iloc[-1];
            indicators["bb_ma"] = bb_indicator.bollinger_mavg().iloc[-1]
            indicators["bb_high_series"] = bb_indicator.bollinger_hband();
            indicators["bb_low_series"] = bb_indicator.bollinger_lband();
            indicators["bb_ma_series"] = bb_indicator.bollinger_mavg()
        else:
            indicators["bb_high"], indicators["bb_low"], indicators["bb_ma"] = None, None, None
            indicators["bb_high_series"], indicators["bb_low_series"], indicators["bb_ma_series"] = pd.Series(
                dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
        return {k: _sanitize_indicator_value(v) for k, v in indicators.items()}
    except Exception as e:
        return {"error": f"Failed to calculate volatility: {str(e)}"}


available_technical_indicator_tools = {
    "moving_averages": calculate_moving_averages,
    "oscillators": calculate_oscillators,
    "volatility": calculate_volatility_indicators,
}


# --- Chart Generation Function (Internal Helper, Not a Tool) ---
# def _generate_chart_with_indicators(daily_data_json: str, daily_indicators: Dict[str, Dict], stock_symbol: str) -> Dict[str, Optional[str]]:
#     # ... chart generation logic ...
#     pass
#     return {"error": "Chart generation is currently disabled."}


# --- Helper Functions (Data Fetching & Summaries - Async LLM Calls) ---
def sanitize_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp): return value.isoformat()
    if isinstance(value, (np.datetime64, np.timedelta64)): return str(value)
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, pd.Series): return [sanitize_value(item) for item in value.tolist()]
    if isinstance(value, dict): return {str(k): sanitize_value(v) for k, v in value.items()}
    if isinstance(value, list): return [sanitize_value(item) for item in value]
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)): return None
    return value


async def _is_valid_ticker_async(ticker_symbol: str) -> bool:
    """Asynchronously checks if a ticker symbol is valid by fetching its info."""
    try:
        loop = asyncio.get_event_loop()
        stock_info = await loop.run_in_executor(None, lambda t: yf.Ticker(t).info, ticker_symbol)
        return bool(stock_info and (stock_info.get('regularMarketPrice') is not None or stock_info.get('shortName')))
    except Exception as e:
        print(f"LOG: Validation check failed for {ticker_symbol}: {e}")
        return False


async def fetch_stock_data_yf_async(ticker_symbol: str):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, fetch_stock_data_yf_sync, ticker_symbol)


def fetch_stock_data_yf_sync(ticker_symbol: str):
    """Synchronous helper to fetch data using yfinance. Assumes ticker_symbol is already validated."""
    stock = yf.Ticker(ticker_symbol)
    info = stock.info
    # This was part of the old resolve_symbol node. Now fetch_stock_data_yf_sync assumes a valid ticker.
    # If info is still None after yf.Ticker(valid_ticker).info, then it's truly no info.
    if not info:
        print(f"Warning: No info dictionary returned by yfinance for already validated ticker '{ticker_symbol}'.")
        # raise ValueError(f"Could not retrieve info for validated ticker '{ticker_symbol}'.") # Or handle as warning

    daily_data = stock.history(period="1y")
    if daily_data.empty:
        # This could happen if a ticker is valid (e.g., exists) but has no recent trading data.
        print(f"Warning: No daily historical data found for ticker '{ticker_symbol}'.")
        # Depending on requirements, this could be an error or just a state to note.

    return {
        "resolved_symbol": ticker_symbol,
        "info": info or {},  # Ensure info is dict
        "daily": daily_data,
        "weekly": stock.history(period="5y", interval="1wk"),
        "monthly": stock.history(period="max", interval="1mo"),
        "financials": stock.financials,
        "balance_sheet": stock.balance_sheet,
        "cashflow": stock.cashflow,
    }


async def get_technical_analysis_summary_content(stock_symbol: str, executed_indicators: Dict[str, Dict[str, Dict]]):
    current_llm, _, _, _ = get_llms()
    prompt_parts = [
        f"Analyze the technical outlook for {stock_symbol} based ONLY on the following calculated indicator values."]
    prompt_parts.append(
        f"Provide a VERY BRIEF summary (2-3 bullet points MAX). Focus ONLY on the main trend and key signals (e.g., RSI overbought/oversold, MACD cross). STRICTLY ADHERE to the {LLM_MAX_TOKENS} token limit.")
    has_data = False
    for timeframe in ["daily", "weekly", "monthly"]:
        indicators_by_tool = executed_indicators.get(timeframe, {})
        if not indicators_by_tool: continue
        timeframe_data_exists = False
        timeframe_prompt_parts = [f"\n{timeframe.capitalize()} Indicators:"]
        for tool_name, values in indicators_by_tool.items():
            if values.get("error"):
                timeframe_prompt_parts.append(
                    f"  - {tool_name.replace('_', ' ').capitalize()}: Error ({values['error']})")
                continue
            filtered_values = {k: v for k, v in values.items() if
                               not k.endswith('_series') and k != 'last_close' and v is not None and not isinstance(v,
                                                                                                                    str)}
            if filtered_values:
                values_str = ", ".join(
                    [f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}" for k, v in filtered_values.items()])
                timeframe_prompt_parts.append(f"  - {tool_name.replace('_', ' ').capitalize()}: {values_str}")
                timeframe_data_exists = True
        if timeframe_data_exists:
            prompt_parts.extend(timeframe_prompt_parts)
            has_data = True
    if not has_data:
        return "No valid technical indicator data points were available to generate a summary."
    prompt_parts.append("\nSummary:")
    prompt = "\n".join(prompt_parts)
    response = await current_llm.ainvoke(prompt)
    return response.content


async def get_fundamental_analysis_summary_content(stock_symbol: str, stock_info_json: str,
                                                   financials_json: Optional[str], balance_sheet_json: Optional[str],
                                                   cashflow_json: Optional[str]):
    current_llm, _, _, _ = get_llms()
    stock_info = json.loads(stock_info_json) if stock_info_json else {}
    financials = pd.read_json(StringIO(financials_json), orient='split') if financials_json else pd.DataFrame()
    balance_sheet = pd.read_json(StringIO(balance_sheet_json), orient='split') if balance_sheet_json else pd.DataFrame()
    cashflow = pd.read_json(StringIO(cashflow_json), orient='split') if cashflow_json else pd.DataFrame()
    market_cap = stock_info.get('marketCap');
    pe_ratio = stock_info.get('trailingPE') or stock_info.get('forwardPE');
    eps = stock_info.get('trailingEps');
    dividend_yield = stock_info.get('dividendYield');
    sector = stock_info.get('sector');
    industry = stock_info.get('industry');
    summary = stock_info.get('longBusinessSummary', "");
    summary_preview = summary[:500] if summary else "N/A"
    latest_annual_revenue = financials.loc['Total Revenue'].iloc[
        0] if not financials.empty and 'Total Revenue' in financials.index and not financials.loc[
        'Total Revenue'].empty else 'N/A'
    latest_annual_net_income = financials.loc['Net Income'].iloc[
        0] if not financials.empty and 'Net Income' in financials.index and not financials.loc[
        'Net Income'].empty else 'N/A'
    latest_total_assets = balance_sheet.loc['Total Assets'].iloc[
        0] if not balance_sheet.empty and 'Total Assets' in balance_sheet.index and not balance_sheet.loc[
        'Total Assets'].empty else 'N/A'
    latest_total_liabilities = balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[
        0] if not balance_sheet.empty and 'Total Liabilities Net Minority Interest' in balance_sheet.index and not \
    balance_sheet.loc['Total Liabilities Net Minority Interest'].empty else 'N/A'
    prompt = f"Analyze the fundamental data for {stock_symbol}. Provide a VERY BRIEF summary (2-3 bullet points MAX) highlighting ONLY the most critical strengths and weaknesses based on the provided metrics. STRICTLY ADHERE to the {LLM_MAX_TOKENS} token limit.\n\nCompany Information:\nSector: {sector}\nIndustry: {industry}\nMarket Cap: {market_cap}\nBusiness Summary Preview: {summary_preview}...\n\nKey Metrics:\nP/E Ratio: {pe_ratio}\nEPS (Trailing): {eps}\nDividend Yield: {dividend_yield}\n\nFinancial Highlights (Latest Annual):\nTotal Revenue: {latest_annual_revenue}\nNet Income: {latest_annual_net_income}\n\nBalance Sheet Highlights (Latest Annual):\nTotal Assets: {latest_total_assets}\nTotal Liabilities: {latest_total_liabilities}\n\nBrief Summary:"
    response = await current_llm.ainvoke(prompt)
    return response.content


async def generate_recommendation_content(stock_symbol: str, ta_summary: str, fa_summary: str,
                                          user_question: Optional[str]):
    current_llm, _, _, _ = get_llms()
    default_question = "What is the overall recommendation (Buy, Sell, Hold) for this stock considering a medium-term investment horizon (6-12 months)?"
    is_specific_question = bool(user_question) and user_question.lower().strip() != default_question.lower().strip()
    prompt_parts = [f"Stock: {stock_symbol}"]
    if is_specific_question: prompt_parts.append(f"User Question: {user_question}")
    prompt_parts.append(f"Technical Analysis Summary:\n{ta_summary}");
    prompt_parts.append(f"Fundamental Analysis Summary:\n{fa_summary}");
    prompt_parts.append("\nInstructions:")
    if is_specific_question:
        prompt_parts.append(
            "1. First, directly answer the User Question based on the provided summaries (1-2 sentences).")
        prompt_parts.append("2. Then, provide the overall recommendation (Buy/Sell/Hold).")
        prompt_parts.append("3. Finally, list ONLY 2-3 main bullet points justifying the recommendation.")
    else:
        prompt_parts.append("1. Provide the overall recommendation (Buy/Sell/Hold).")
        prompt_parts.append("2. List ONLY 2-3 main bullet points justifying the recommendation.")
    prompt_parts.append(f"Be extremely concise. STRICTLY ADHERE to the {LLM_MAX_TOKENS} token limit.");
    prompt_parts.append("\nResponse:")
    prompt = "\n".join(prompt_parts)
    response = await current_llm.ainvoke(prompt)
    return response.content


async def generate_position_advice_content(stock_symbol: str, recommendation: str, user_position: dict,
                                           key_levels: Dict):
    _, _, current_llm_pos_advice, _ = get_llms()
    if not user_position or not user_position.get("shares") or not user_position.get(
        "avg_price"): return "No position information provided or incomplete information."
    shares = user_position["shares"];
    avg_price = user_position["avg_price"]
    levels_str = "Key technical levels (daily): "
    levels_found = []
    if key_levels.get('sma_50') is not None: levels_found.append(f"SMA 50: {key_levels['sma_50']:.2f}")
    if key_levels.get('sma_200') is not None: levels_found.append(f"SMA 200: {key_levels['sma_200']:.2f}")
    if key_levels.get('bb_low') is not None: levels_found.append(f"Bollinger Low: {key_levels['bb_low']:.2f}")
    if key_levels.get('last_close') is not None: levels_found.append(f"Last Close: {key_levels['last_close']:.2f}")
    if not levels_found:
        levels_str += "N/A"
    else:
        levels_str += ", ".join(levels_found)
    prompt = f"""
    Stock: {stock_symbol}
    User Position: {shares} shares @ avg price ${avg_price:.2f}
    Overall Recommendation: {recommendation}
    {levels_str}

    Instructions:
    Provide CONCISE advice for the user's existing position based on the recommendation and key levels.
    1.  Identify 1-2 key support levels based on the provided technical levels (e.g., nearest SMA, Bollinger Low). State the level clearly.
    2.  Suggest a potential stop-loss level based on the support levels (e.g., slightly below a key support). State the level clearly and mention this is a suggestion for risk management, not financial advice.
    3.  Briefly outline Conservative Next Steps (1-2 sentences).
    4.  Briefly outline Aggressive Next Steps (1-2 sentences), highlighting risks.

    Keep the entire response concise and actionable. STRICTLY ADHERE to the {POSITION_ADVICE_MAX_TOKENS} token limit.
    """
    response = await current_llm_pos_advice.ainvoke(prompt)
    return response.content


# --- LangGraph State Definition ---
class StockAnalysisState(TypedDict):
    user_stock_query: str
    user_question: Optional[str]
    user_position: Optional[dict]
    stock_symbol: Optional[str]
    error_message: Optional[str]
    raw_stock_data_json: Optional[Dict[str, Optional[str]]]
    stock_info_json: Optional[str]
    stock_financials_json: Optional[str]
    stock_balance_sheet_json: Optional[str]
    stock_cashflow_json: Optional[str]
    selected_technical_tools: Optional[List[str]]
    executed_technical_indicators: Optional[Dict[str, Dict[str, Dict]]]
    generated_chart_urls: Optional[List[str]]
    key_technical_levels: Optional[Dict[str, Optional[float]]]
    technical_analysis_summary: Optional[str]
    fundamental_analysis_summary: Optional[str]
    final_recommendation: Optional[str]
    position_specific_advice: Optional[str]


# --- LangGraph Nodes (Async) ---
async def start_node(state: StockAnalysisState) -> StockAnalysisState:
    print(f"LOG: Start node for query: {state['user_stock_query']}")
    state["error_message"] = ""
    state["stock_symbol"] = None
    state["selected_technical_tools"] = []
    state["executed_technical_indicators"] = {}
    state["generated_chart_urls"] = []
    state["key_technical_levels"] = {}
    state["raw_stock_data_json"] = {"daily": None, "weekly": None, "monthly": None}
    state["stock_info_json"] = None
    state["stock_financials_json"] = None
    state["stock_balance_sheet_json"] = None
    state["stock_cashflow_json"] = None
    state["technical_analysis_summary"] = None
    state["fundamental_analysis_summary"] = None
    state["final_recommendation"] = None
    state["position_specific_advice"] = None
    return state


async def resolve_stock_symbol_node(state: StockAnalysisState) -> StockAnalysisState:
    """
    Attempts to resolve the user's stock query to a valid yfinance ticker symbol
    using an LLM and fallback heuristics.
    """
    print(f"LOG: Resolve symbol node for query: '{state['user_stock_query']}'")
    user_query_original = state["user_stock_query"]
    user_query_upper = user_query_original.strip().upper()

    _, _, _, current_llm_resolver = get_llms()  # Get the symbol resolver LLM
    resolved_symbol = None
    error_msg = ""

    # 1. Attempt resolution with LLM
    try:
        print(f"LOG: Attempting LLM based symbol resolution for '{user_query_original}'...")
        # Simple prompt for ticker suggestion
        # You can make this more sophisticated with examples or specific instructions
        # about common exchanges (NYSE, NASDAQ, NSE for India etc.)
        prompt_template = ChatPromptTemplate.from_messages([
            ("system",
             "You are an expert financial assistant. Your task is to identify the most likely official stock ticker symbol based on the user's query. "
             "If the company sounds Indian (e.g., Reliance, Infosys, Tata, HDFC), append '.NS' to the suggested symbol. "
             "For well-known international companies (e.g., Apple, Microsoft, Google), provide their common US exchange ticker. "
             "If the query is already a valid-looking ticker, return it as is. "
             "If highly ambiguous or unclear, return 'UNKNOWN'. "
             "Respond with ONLY the ticker symbol or 'UNKNOWN'."),
            ("human", "User query: {query}")
        ])

        # Create a simple chain for this
        resolve_chain = prompt_template | current_llm_resolver | StrOutputParser()
        llm_suggested_ticker = await resolve_chain.ainvoke({"query": user_query_original})
        llm_suggested_ticker = llm_suggested_ticker.strip().upper()
        print(f"LOG: LLM suggested ticker: '{llm_suggested_ticker}'")

        if llm_suggested_ticker and llm_suggested_ticker != "UNKNOWN":
            if await _is_valid_ticker_async(llm_suggested_ticker):
                resolved_symbol = llm_suggested_ticker
                print(f"LOG: LLM suggested ticker '{resolved_symbol}' validated.")
            else:
                print(f"LOG: LLM suggested ticker '{llm_suggested_ticker}' was NOT valid.")
                error_msg += f"LLM suggestion '{llm_suggested_ticker}' was not a valid ticker. "
        else:
            print("LOG: LLM could not suggest a ticker or returned UNKNOWN.")
            error_msg += "LLM could not confidently resolve the stock name. "

    except Exception as e:
        print(f"ERROR: Exception during LLM symbol resolution: {e}")
        error_msg += "Error during LLM symbol resolution. "

    # 2. Fallback to heuristic if LLM fails or suggestion is invalid
    if not resolved_symbol:
        print(f"LOG: LLM resolution failed or invalid. Falling back to heuristics for '{user_query_original}'...")
        potential_symbols_to_try = []
        if re.match(r"^[A-Z0-9.-]+$", user_query_upper) and ('.' in user_query_upper or len(user_query_upper) <= 5):
            potential_symbols_to_try.append(user_query_upper)
        if not '.' in user_query_upper and re.match(r"^[A-Z\s]+$", user_query_upper):
            potential_symbols_to_try.append(f"{user_query_upper.replace(' ', '')}.NS")
        if user_query_upper not in potential_symbols_to_try:
            potential_symbols_to_try.append(user_query_upper)
        if ' ' not in user_query_upper and '.' not in user_query_upper and not user_query_upper.endswith(".NS"):
            potential_symbols_to_try.append(f"{user_query_upper}.NS")

        unique_symbols_to_try = list(dict.fromkeys(potential_symbols_to_try))
        print(f"LOG: Heuristic: Potential symbols to try: {unique_symbols_to_try}")

        for symbol_attempt in unique_symbols_to_try:
            if await _is_valid_ticker_async(symbol_attempt):
                resolved_symbol = symbol_attempt
                print(f"LOG: Heuristic resolved '{user_query_original}' to '{resolved_symbol}'")
                error_msg = ""  # Clear previous error if heuristic succeeds
                break
            else:
                print(f"LOG: Heuristic: '{symbol_attempt}' is not valid.")

    if resolved_symbol:
        state["stock_symbol"] = resolved_symbol
    else:
        final_error = error_msg if error_msg else f"Could not resolve '{user_query_original}' to a known stock symbol."
        state["error_message"] = (state.get("error_message", "") + final_error).strip()
        print(f"LOG: Failed to resolve symbol for '{user_query_original}'. Final error: {state['error_message']}")

    return state


async def fetch_data_node(state: StockAnalysisState) -> StockAnalysisState:
    resolved_symbol = state.get("stock_symbol")
    if state["error_message"] or not resolved_symbol:
        state["error_message"] = (state.get("error_message",
                                            "") + " Skipping data fetch due to unresolved symbol or prior error.").strip()
        return state
    print(f"LOG: Fetch data node for resolved symbol: {resolved_symbol}")
    try:
        fetched_data = await fetch_stock_data_yf_async(resolved_symbol)
        state["stock_symbol"] = fetched_data.get("resolved_symbol", resolved_symbol)
        info_data = fetched_data.get("info", {});
        if not isinstance(info_data, dict): info_data = {}
        sanitized_stock_info = {str(k): sanitize_value(v) for k, v in info_data.items()}
        state["stock_info_json"] = json.dumps(sanitized_stock_info)
        state["raw_stock_data_json"] = {
            tf: (data.to_json(orient='split', date_format='iso') if data is not None and not data.empty else None) for
            tf, data in [("daily", fetched_data.get("daily")), ("weekly", fetched_data.get("weekly")),
                         ("monthly", fetched_data.get("monthly"))]}
        state["stock_financials_json"] = fetched_data.get("financials").to_json(orient='split',
                                                                                date_format='iso') if fetched_data.get(
            "financials") is not None and not fetched_data.get("financials").empty else None
        state["stock_balance_sheet_json"] = fetched_data.get("balance_sheet").to_json(orient='split',
                                                                                      date_format='iso') if fetched_data.get(
            "balance_sheet") is not None and not fetched_data.get("balance_sheet").empty else None
        state["stock_cashflow_json"] = fetched_data.get("cashflow").to_json(orient='split',
                                                                            date_format='iso') if fetched_data.get(
            "cashflow") is not None and not fetched_data.get("cashflow").empty else None
        if not state["raw_stock_data_json"]["daily"]:
            print(
                f"Warning: Daily historical price data is empty for {state['stock_symbol']}. Some features might be affected.")
            if not state["raw_stock_data_json"]["weekly"] and not state["raw_stock_data_json"]["monthly"]:
                # If all historical data is missing, this becomes a more significant issue.
                state["error_message"] = (state.get("error_message",
                                                    "") + f" All historical price data is missing for {state['stock_symbol']}. Analysis will be limited. ").strip()
        print(f"LOG: Successfully fetched and serialized data for {state['stock_symbol']}.")
    except Exception as e:
        print(f"ERROR in fetch_data_node for {resolved_symbol}: {e}")
        state["error_message"] = (state.get("error_message",
                                            "") + f" Failed to fetch/serialize data for {resolved_symbol}: {str(e)}. ").strip()
    return state


async def select_technical_tools_node(state: StockAnalysisState) -> StockAnalysisState:
    global tool_descriptions_for_llm
    print(f"LOG: Select technical INDICATOR tools node for {state['stock_symbol']}")
    _, current_tool_selector_llm, _, _ = get_llms()
    if not tool_descriptions_for_llm or "Error:" in tool_descriptions_for_llm:
        state["error_message"] += "Critical error: Indicator tool descriptions not available. Defaulting. "
        state["selected_technical_tools"] = list(available_technical_indicator_tools.keys())
        return state
    if state["error_message"] or not state["raw_stock_data_json"] or not any(state["raw_stock_data_json"].values()):
        state["selected_technical_tools"] = []
        state["error_message"] += "Skipped indicator tool selection. "
        return state
    user_query = state.get("user_question", "Provide a general technical analysis.")
    prompt = f"Based on the user's query: \"{user_query}\" for the stock {state['stock_symbol']}, which of the following technical INDICATOR tools should be used?\nAvailable indicator tools and their descriptions:\n{tool_descriptions_for_llm}\n\nRespond with a JSON object containing a single key \"tool_names\" with a list of selected indicator tool names. For a general analysis, select all indicator tools: [\"moving_averages\", \"oscillators\", \"volatility\"]. Example response: {{\"tool_names\": [\"moving_averages\", \"oscillators\"]}}"
    try:
        structured_llm = current_tool_selector_llm.with_structured_output(ToolSelection)
        response_model = await structured_llm.ainvoke(prompt)
        selected_tools = response_model.tool_names
        valid_selected_tools = [t for t in selected_tools if t in available_technical_indicator_tools]
        if len(valid_selected_tools) != len(selected_tools): print(
            f"Warning: Invalid indicator tools selected. Original: {selected_tools}, Valid: {valid_selected_tools}")
        state["selected_technical_tools"] = valid_selected_tools
        if not valid_selected_tools:
            print("Warning: No valid indicator tools selected by LLM. Defaulting to all indicator tools.")
            state["selected_technical_tools"] = list(available_technical_indicator_tools.keys())
        print(f"LOG: Selected indicator tools: {state['selected_technical_tools']}")
    except Exception as e:
        print(f"ERROR in select_technical_tools_node: {e}. Defaulting to all indicator tools.")
        state["error_message"] += f"Indicator tool selection failed: {str(e)}. Defaulting. "
        state["selected_technical_tools"] = list(available_technical_indicator_tools.keys())
    return state


async def execute_technical_tools_node(state: StockAnalysisState) -> StockAnalysisState:
    print(f"LOG: Execute technical INDICATOR tools node for {state['stock_symbol']}")
    if state["error_message"] or not state["selected_technical_tools"] or not state["raw_stock_data_json"]:
        state["error_message"] += "Skipped technical indicator tool execution. "
        state["executed_technical_indicators"] = {}
        return state
    selected_tools = state["selected_technical_tools"]
    if not selected_tools:
        print("LOG: No indicator tools selected to execute.")
        state["executed_technical_indicators"] = {};
        return state
    raw_data_map = state["raw_stock_data_json"]
    indicator_results = {"daily": {}, "weekly": {}, "monthly": {}}
    tool_invocations = []
    for timeframe in ["daily", "weekly", "monthly"]:
        df_json_str = raw_data_map.get(timeframe)
        if not df_json_str:
            for tool_name in selected_tools:
                indicator_results[timeframe][tool_name] = {"error": f"No data for {timeframe}."}
            continue
        for tool_name in selected_tools:
            tool_func = available_technical_indicator_tools.get(tool_name)
            if tool_func:
                tool_invocations.append((timeframe, tool_name, tool_func.ainvoke({"data_json": df_json_str})))
            else:
                indicator_results[timeframe][tool_name] = {"error": "Indicator tool function not found."}
    gathered_results = await asyncio.gather(*(inv[2] for inv in tool_invocations), return_exceptions=True)
    result_index = 0
    key_levels = {}
    for timeframe, tool_name, _ in tool_invocations:
        current_result = gathered_results[result_index]
        if isinstance(current_result, Exception):
            error_str = str(current_result)
            print(f"    Error executing indicator tool '{tool_name}' for {timeframe}: {error_str}")
            indicator_results[timeframe][tool_name] = {"error": error_str}
        else:
            indicator_results[timeframe][tool_name] = current_result
            if timeframe == 'daily' and isinstance(current_result, dict) and not current_result.get("error"):
                if tool_name == 'moving_averages':
                    key_levels['sma_50'] = current_result.get('sma_50')
                    key_levels['sma_200'] = current_result.get('sma_200')
                    key_levels['last_close'] = current_result.get('last_close')
                elif tool_name == 'volatility':
                    key_levels['bb_low'] = current_result.get('bb_low')
        result_index += 1
    state["executed_technical_indicators"] = indicator_results
    state["key_technical_levels"] = {k: v for k, v in key_levels.items() if v is not None}
    return state


async def generate_chart_node(state: StockAnalysisState) -> StockAnalysisState:
    print(f"LOG: Generate chart node for {state['stock_symbol']} (currently disabled by user request)")
    state["generated_chart_urls"] = ["Chart generation is currently disabled."]
    return state


async def summarize_technical_analysis_node(state: StockAnalysisState) -> StockAnalysisState:
    print(f"LOG: Summarize TA node for {state['stock_symbol']}")
    indicators = state.get("executed_technical_indicators")
    if state["error_message"] and not indicators: state[
        "technical_analysis_summary"] = "TA summary cannot be generated due to critical prior errors."; return state
    if not indicators or all(not tf_data for tf_data in indicators.values()): state[
        "technical_analysis_summary"] = "No technical indicators available to summarize."; return state
    try:
        state["technical_analysis_summary"] = await get_technical_analysis_summary_content(state["stock_symbol"],
                                                                                           indicators)
    except Exception as e:
        print(f"ERROR in summarize_technical_analysis_node: {e}"); state[
            "error_message"] += f"TA summary generation failed: {str(e)}. "; state[
            "technical_analysis_summary"] = "Failed to generate TA summary."
    return state


async def fundamental_analysis_node(state: StockAnalysisState) -> StockAnalysisState:
    print(f"LOG: Fundamental analysis node for {state['stock_symbol']}")
    if state["error_message"] or not state.get("stock_info_json"): state["error_message"] += "Skipped FA. "; state[
        "fundamental_analysis_summary"] = "FA skipped."; return state
    try:
        state["fundamental_analysis_summary"] = await get_fundamental_analysis_summary_content(state["stock_symbol"],
                                                                                               state["stock_info_json"],
                                                                                               state.get(
                                                                                                   "stock_financials_json"),
                                                                                               state.get(
                                                                                                   "stock_balance_sheet_json"),
                                                                                               state.get(
                                                                                                   "stock_cashflow_json"))
    except Exception as e:
        print(f"ERROR in fundamental_analysis_node: {e}"); state["error_message"] += f"FA failed: {str(e)}. "; state[
            "fundamental_analysis_summary"] = "FA failed."
    return state


async def recommendation_node(state: StockAnalysisState) -> StockAnalysisState:
    print(f"LOG: Recommendation node for {state['stock_symbol']}")
    ta_summary = state.get("technical_analysis_summary", "TA info N/A.")
    fa_summary = state.get("fundamental_analysis_summary", "FA info N/A.")
    critical_ta = "cannot be generated" in ta_summary.lower() or "skipped" in ta_summary.lower() or "n/a" in ta_summary.lower() or "failed" in ta_summary.lower()
    critical_fa = "failed" in fa_summary.lower() or "skipped" in fa_summary.lower() or "n/a" in fa_summary.lower()
    if state["error_message"] and (critical_ta or critical_fa): state[
        "final_recommendation"] = "Cannot generate recommendation due to prior errors/missing analysis."; return state
    try:
        state["final_recommendation"] = await generate_recommendation_content(
            state["stock_symbol"],
            ta_summary,
            fa_summary,
            state.get("user_question")
        )
    except Exception as e:
        print(f"ERROR in recommendation_node: {e}")
        state["error_message"] += f"Recommendation generation failed: {str(e)}. "
        state["final_recommendation"] = "Failed to generate recommendation."
    return state


async def position_advice_node(state: StockAnalysisState) -> StockAnalysisState:
    print(f"LOG: Position advice node for {state['stock_symbol']}")
    final_rec = state.get("final_recommendation", "")
    key_levels = state.get("key_technical_levels", {})
    if not final_rec or "cannot generate" in final_rec.lower() or "failed" in final_rec.lower(): state[
        "position_specific_advice"] = "Position advice N/A."; return state
    if not state.get("user_position"): state["position_specific_advice"] = "No user position provided."; return state
    try:
        state["position_specific_advice"] = await generate_position_advice_content(
            state["stock_symbol"], final_rec, state["user_position"], key_levels
        )
        print("LOG: Position-specific advice generated.")
    except Exception as e:
        print(f"ERROR in position_advice_node: {e}")
        state["position_specific_advice"] = "Failed to generate position advice."
    return state


# --- Compile LangGraph App ---
def get_stock_analyzer_app():
    get_llms()
    workflow = StateGraph(StockAnalysisState)
    workflow.add_node("start", start_node)
    workflow.add_node("resolve_symbol", resolve_stock_symbol_node)
    workflow.add_node("fetch_data", fetch_data_node)
    workflow.add_node("select_technical_tools", select_technical_tools_node)
    workflow.add_node("execute_technical_tools", execute_technical_tools_node)
    # workflow.add_node("generate_chart", generate_chart_node) # Chart node commented out
    workflow.add_node("summarize_technical_analysis", summarize_technical_analysis_node)
    workflow.add_node("fundamental_analysis", fundamental_analysis_node)
    workflow.add_node("generate_recommendation", recommendation_node)
    workflow.add_node("generate_position_advice", position_advice_node)

    workflow.set_entry_point("start")
    workflow.add_edge("start", "resolve_symbol")
    workflow.add_edge("resolve_symbol", "fetch_data")
    workflow.add_edge("fetch_data", "select_technical_tools")
    workflow.add_edge("select_technical_tools", "execute_technical_tools")
    # workflow.add_edge("execute_technical_tools", "generate_chart")
    # workflow.add_edge("generate_chart", "summarize_technical_analysis")
    workflow.add_edge("execute_technical_tools", "summarize_technical_analysis")
    workflow.add_edge("summarize_technical_analysis", "fundamental_analysis")
    workflow.add_edge("fundamental_analysis", "generate_recommendation")
    workflow.add_edge("generate_recommendation", "generate_position_advice")
    workflow.add_edge("generate_position_advice", END)
    return workflow.compile()


async def main_test():
    if initialize_analyzer():
        app = get_stock_analyzer_app()
        print("Stock Analyzer App compiled for standalone async testing.")
        inputs = {"user_stock_query": "Infosys", "user_question": "General analysis",
                  "user_position": {"shares": 10, "avg_price": 1400}}
        try:
            result_state = await app.ainvoke(inputs)
            print("\n--- FINAL STATE (for standalone async test) ---")
            print(json.dumps(result_state, indent=2, default=str))
            api_response_data = {
                "user_stock_query": result_state.get("user_stock_query"),
                "stock_symbol": result_state.get("stock_symbol"),
                "generated_chart_urls": result_state.get("generated_chart_urls"),
                "technical_analysis_summary": result_state.get("technical_analysis_summary"),
                "fundamental_analysis_summary": result_state.get("fundamental_analysis_summary"),
                "final_recommendation": result_state.get("final_recommendation"),
                "position_specific_advice": result_state.get("position_specific_advice"),
                "error_message": result_state.get("error_message") if result_state.get("error_message",
                                                                                       "").strip() else None,
            }
            print("\n--- SIMULATED API RESPONSE DATA (for standalone async test) ---")
            print(json.dumps(api_response_data, indent=2, default=str))
        except Exception as e:
            print(f"Error during standalone async test invocation: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Failed to initialize analyzer due to missing OpenAI API key.")


if __name__ == "__main__":
    asyncio.run(main_test())
