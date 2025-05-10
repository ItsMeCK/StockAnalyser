# backend/analyzer_utils.py

import os
import json
from typing import List, Optional, Dict, Any, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
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
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
from datetime import datetime, timedelta

load_dotenv()
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Constants and Configuration ---
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_LANGCHAIN_TRACING_V2 = "LANGCHAIN_TRACING_V2"
ENV_LANGCHAIN_ENDPOINT = "LANGCHAIN_ENDPOINT"
ENV_LANGCHAIN_API_KEY = "LANGCHAIN_API_KEY"
ENV_LANGCHAIN_PROJECT = "LANGCHAIN_PROJECT"
ENV_ALPHAVANTAGE_API_KEY = "ALPHAVANTAGE_API_KEY"

OPENAI_API_KEY = os.getenv(ENV_OPENAI_API_KEY)
LANGCHAIN_TRACING_V2_RAW = os.getenv("LANGSMITH_TRACING", os.getenv(ENV_LANGCHAIN_TRACING_V2, "false"))
LANGCHAIN_TRACING_V2 = LANGCHAIN_TRACING_V2_RAW.lower() == "true"
LANGCHAIN_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", os.getenv(ENV_LANGCHAIN_ENDPOINT))
LANGCHAIN_API_KEY = os.getenv("LANGSMITH_API_KEY", os.getenv(ENV_LANGCHAIN_API_KEY))
LANGCHAIN_PROJECT = os.getenv("LANGSMITH_PROJECT", os.getenv(ENV_LANGCHAIN_PROJECT, "Stock Analyzer Chat App Async"))
ALPHAVANTAGE_API_KEY = os.getenv(ENV_ALPHAVANTAGE_API_KEY, "demo")

LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o")
LLM_SYMBOL_RESOLUTION_MODEL = os.getenv("LLM_SYMBOL_RESOLUTION_MODEL", "gpt-3.5-turbo")
LLM_QUERY_ANALYZER_MODEL = os.getenv("LLM_QUERY_ANALYZER_MODEL", "gpt-3.5-turbo")
LLM_DIRECT_ANSWER_MODEL = os.getenv("LLM_DIRECT_ANSWER_MODEL", "gpt-4o")

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.3))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 300))
POSITION_ADVICE_MAX_TOKENS = int(os.getenv("POSITION_ADVICE_MAX_TOKENS", 500))
TOOL_SELECTOR_LLM_MAX_TOKENS = int(os.getenv("TOOL_SELECTOR_LLM_MAX_TOKENS", 200))
QUESTION_ENHANCER_MAX_TOKENS = int(os.getenv("QUESTION_ENHANCER_MAX_TOKENS", 150))
DIRECT_ANSWER_MAX_TOKENS = int(os.getenv("DIRECT_ANSWER_MAX_TOKENS", 400))

CHARTS_DIR = "/tmp/charts"
try:
    os.makedirs(CHARTS_DIR, exist_ok=True)
except OSError as e:
    print(f"Warning: Could not create CHARTS_DIR ({CHARTS_DIR}) at module level: {e}")
CHARTS_URL_PREFIX = "/charts_data"

tool_descriptions_for_llm = ""


# --- Pydantic Models ---
class ToolSelection(BaseModel):
    tool_names: List[str] = Field(
        description="List of technical indicator tool names to be called (e.g., moving_averages, oscillators).")


class QueryAnalysisOutput(BaseModel):
    focus: str = Field(description="A short phrase describing the primary focus of the user's question.")
    data_type_needed: str = Field(
        description="The primary type of data needed to answer: 'technical', 'fundamental', 'both', or 'general'.")


# --- Global LLM Instances ---
llm: Optional[ChatOpenAI] = None
tool_selector_llm: Optional[ChatOpenAI] = None
llm_pos_advice: Optional[ChatOpenAI] = None
llm_symbol_resolver: Optional[ChatOpenAI] = None
llm_query_analyzer: Optional[ChatOpenAI] = None
llm_direct_answerer: Optional[ChatOpenAI] = None


def initialize_analyzer():
    """
    Initializes global settings, checks API keys, and constructs tool descriptions.
    Should be called once at application startup.
    """
    global tool_descriptions_for_llm # Ensure this global is being assigned to
    print("INFO: Initializing analyzer utilities...") # Added for clarity

    if not OPENAI_API_KEY:
        print(f"CRITICAL ERROR: The environment variable {ENV_OPENAI_API_KEY} is not set.")
        return False # Stop initialization if key is missing

    if not ALPHAVANTAGE_API_KEY:
        print(
            "WARNING: ALPHAVANTAGE_API_KEY is not set or is 'demo'. Fallback API calls to Alpha Vantage will be very limited.")

    if LANGCHAIN_TRACING_V2 and LANGCHAIN_API_KEY and LANGCHAIN_ENDPOINT:
        os.environ[ENV_LANGCHAIN_TRACING_V2] = "true"
        os.environ[ENV_LANGCHAIN_ENDPOINT] = LANGCHAIN_ENDPOINT
        os.environ[ENV_LANGCHAIN_API_KEY] = LANGCHAIN_API_KEY
        if LANGCHAIN_PROJECT:
            os.environ[ENV_LANGCHAIN_PROJECT] = LANGCHAIN_PROJECT
        print(f"LangSmith tracing enabled. Project: {LANGCHAIN_PROJECT}")
    else:
        print("LangSmith tracing is not configured or missing some required environment variables.")

    # Construct tool descriptions
    try:
        print("DEBUG: Attempting to construct tool_descriptions_for_llm...") # Debug print
        descriptions = []
        # Ensure available_technical_indicator_tools is populated correctly before this loop
        if not available_technical_indicator_tools:
            print("CRITICAL_ERROR: available_technical_indicator_tools is empty! Cannot build descriptions.")
            tool_descriptions_for_llm = "Error: Tool definitions are missing."
            return False # Indicate failure

        for name, tool_func_obj in available_technical_indicator_tools.items():
            if tool_func_obj is None:
                print(f"Warning: Tool function object for '{name}' is None. Skipping.")
                descriptions.append(f"- {name}: Error - Tool function not found.")
                continue

            desc = getattr(tool_func_obj, 'description', None) or getattr(tool_func_obj, '__doc__', None)
            if desc:
                # Get the first non-empty line of the description/docstring
                first_line_doc = next((line for line in desc.strip().splitlines() if line.strip()), "")
                if not first_line_doc: # Fallback if all lines are empty (should not happen with good descriptions)
                    first_line_doc = f"Tool to calculate {name.replace('_', ' ')}."
            else:
                first_line_doc = f"Tool to calculate {name.replace('_', ' ')}."
                print(f"Warning: Tool '{name}' has no description or docstring. Using default: '{first_line_doc}'")
            descriptions.append(f"- {name}: {first_line_doc}")

        tool_descriptions_for_llm = "\n".join(descriptions)

        if not tool_descriptions_for_llm.strip() or "Error:" in tool_descriptions_for_llm : # Check if it's empty or contains errors
            print(f"CRITICAL_ERROR: tool_descriptions_for_llm is empty or contains errors after construction attempts. Content: '{tool_descriptions_for_llm}'")
            # Fallback to a generic error if something went wrong but no exception was raised
            if not "Error:" in tool_descriptions_for_llm:
                 tool_descriptions_for_llm = "Error: Could not load indicator tool descriptions due to an unknown issue during construction."
            # return False # Consider returning False to indicate a critical setup failure
        else:
            print("Indicator tool descriptions for LLM constructed successfully.")
            print(f"DEBUG: Constructed tool_descriptions_for_llm:\n{tool_descriptions_for_llm}")


    except Exception as e:
        # ***** MODIFIED PART: Print the actual exception *****
        print(f"CRITICAL_ERROR: Failed to construct tool_descriptions_for_llm due to an exception: {e}")
        import traceback
        traceback.print_exc() # Print the full traceback for detailed debugging
        tool_descriptions_for_llm = "Error: Could not load indicator tool descriptions due to an exception."
        # return False # Indicate failure

    # Initialize LLMs
    get_llms()
    print("INFO: Analyzer utilities initialization complete.")
    return True


def get_llms():
    global llm, tool_selector_llm, llm_pos_advice, llm_symbol_resolver, llm_query_analyzer, llm_direct_answerer
    # print("DEBUG: get_llms() called.") # Optional: for tracing calls
    if llm is None: llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=LLM_TEMPERATURE, max_tokens=LLM_MAX_TOKENS)
    if tool_selector_llm is None: tool_selector_llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0.2,
                                                                 max_tokens=TOOL_SELECTOR_LLM_MAX_TOKENS)
    if llm_pos_advice is None: llm_pos_advice = ChatOpenAI(model=LLM_MODEL_NAME, temperature=LLM_TEMPERATURE,
                                                           max_tokens=POSITION_ADVICE_MAX_TOKENS)
    if llm_symbol_resolver is None:
        llm_symbol_resolver = ChatOpenAI(model=LLM_SYMBOL_RESOLUTION_MODEL, temperature=0.0, max_tokens=50)
    if llm_query_analyzer is None:
        llm_query_analyzer = ChatOpenAI(model=LLM_QUERY_ANALYZER_MODEL, temperature=0.1,
                                        max_tokens=QUESTION_ENHANCER_MAX_TOKENS)
    if llm_direct_answerer is None:
        llm_direct_answerer = ChatOpenAI(model=LLM_DIRECT_ANSWER_MODEL, temperature=0.5,
                                         max_tokens=DIRECT_ANSWER_MAX_TOKENS)
    return llm, tool_selector_llm, llm_pos_advice, llm_symbol_resolver, llm_query_analyzer, llm_direct_answerer


def _sanitize_indicator_value(value: Any) -> Any:
    if isinstance(value, pd.Series):
        if value.empty or value.isna().all(): return None
        if isinstance(value.index, pd.DatetimeIndex):
            value.index = value.index.map(lambda ts: ts.isoformat())
        return value.to_json(orient='split')
    elif pd.isna(value):
        return None
    elif isinstance(value, (np.generic, int, float)):
        if np.isnan(value) or np.isinf(value): # Handle numpy NaN/inf
            return None
        return float(value)
    elif isinstance(value, str):
        return value
    return value

def sanitize_value(value: Any) -> Any:
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    elif isinstance(value, (np.integer, np.floating)):
        item = value.item()
        if np.isnan(item) or np.isinf(item):
            return None
        return item
    elif isinstance(value, np.ndarray):
        return [sanitize_value(v) for v in value.tolist()] # Recursively sanitize
    elif isinstance(value, pd.DataFrame):
         # Sanitize each column
        sanitized_df = pd.DataFrame()
        for col in value.columns:
            sanitized_df[col] = value[col].apply(sanitize_value)
        return sanitized_df.to_dict(orient='records')
    elif isinstance(value, pd.Series):
        return [sanitize_value(v) for v in value.to_list()] # Recursively sanitize
    elif isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    elif isinstance(value, dict):
        return {k: sanitize_value(v) for k, v in value.items()} # Recursively sanitize dicts
    elif isinstance(value, list):
        return [sanitize_value(v) for v in value] # Recursively sanitize lists
    return value


@tool(description="Calculates Simple Moving Averages (SMA 20, 50, 200) and Exponential Moving Averages (EMA 12, 26).")
async def calculate_moving_averages(data_json: str) -> Dict[str, Optional[Any]]:
    try:
        df = pd.read_json(StringIO(data_json), orient='split')
        if df.empty or 'Close' not in df.columns: return {"error": "Invalid or empty data for moving averages"}
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df.dropna(subset=['Close'], inplace=True)
        if df.empty : return {"error": "No valid 'Close' prices after coercing to numeric for moving averages."}
        close = df["Close"]

        indicators = {
            "sma_20": SMAIndicator(close, window=20, fillna=False).sma_indicator().iloc[-1] if len(close) >= 20 else None,
            "sma_50": SMAIndicator(close, window=50, fillna=False).sma_indicator().iloc[-1] if len(close) >= 50 else None,
            "sma_200": SMAIndicator(close, window=200, fillna=False).sma_indicator().iloc[-1] if len(close) >= 200 else None,
            "ema_12": EMAIndicator(close, window=12, fillna=False).ema_indicator().iloc[-1] if len(close) >= 12 else None,
            "ema_26": EMAIndicator(close, window=26, fillna=False).ema_indicator().iloc[-1] if len(close) >= 26 else None,
            "last_close": close.iloc[-1] if not close.empty else None,
            "sma_20_series": SMAIndicator(close, window=20, fillna=False).sma_indicator() if len(close) >= 20 else pd.Series(dtype=float),
            "sma_50_series": SMAIndicator(close, window=50, fillna=False).sma_indicator() if len(close) >= 50 else pd.Series(dtype=float),
            "sma_200_series": SMAIndicator(close, window=200, fillna=False).sma_indicator() if len(close) >= 200 else pd.Series(dtype=float),
        }
        return {k: _sanitize_indicator_value(v) for k, v in indicators.items()}
    except Exception as e:
        print(f"ERROR in calculate_moving_averages: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Failed to calculate moving averages: {str(e)}"}


@tool(description="Calculates MACD (Moving Average Convergence Divergence) and RSI (Relative Strength Index).")
async def calculate_oscillators(data_json: str) -> Dict[str, Optional[Any]]:
    try:
        df = pd.read_json(StringIO(data_json), orient='split')
        if df.empty or 'Close' not in df.columns: return {"error": "Invalid data for oscillators"}
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df.dropna(subset=['Close'], inplace=True)
        if df.empty : return {"error": "No valid 'Close' prices after coercing to numeric for oscillators."}
        close = df["Close"]

        indicators = {}
        if len(close) >= 26:
            macd_indicator = MACD(close, fillna=False)
            indicators["macd"] = macd_indicator.macd().iloc[-1]
            indicators["macd_signal"] = macd_indicator.macd_signal().iloc[-1]
            indicators["macd_diff"] = macd_indicator.macd_diff().iloc[-1]
            indicators["macd_series"] = macd_indicator.macd()
            indicators["macd_signal_series"] = macd_indicator.macd_signal()
            indicators["macd_hist_series"] = macd_indicator.macd_diff()
        else:
            indicators["macd"], indicators["macd_signal"], indicators["macd_diff"] = None, None, None
            indicators["macd_series"], indicators["macd_signal_series"], indicators["macd_hist_series"] = pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

        if len(close) >= 14:
            rsi_series = RSIIndicator(close, fillna=False).rsi()
            indicators["rsi"] = rsi_series.iloc[-1]
            indicators["rsi_series"] = rsi_series
        else:
            indicators["rsi"] = None
            indicators["rsi_series"] = pd.Series(dtype=float)
        return {k: _sanitize_indicator_value(v) for k, v in indicators.items()}
    except Exception as e:
        print(f"ERROR in calculate_oscillators: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Failed to calculate oscillators: {str(e)}"}


@tool(description="Calculates Bollinger Bands (High, Low, Moving Average).")
async def calculate_volatility_indicators(data_json: str) -> Dict[str, Optional[Any]]:
    try:
        df = pd.read_json(StringIO(data_json), orient='split')
        if df.empty or 'Close' not in df.columns: return {"error": "Invalid data for volatility"}
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df.dropna(subset=['Close'], inplace=True)
        if df.empty : return {"error": "No valid 'Close' prices after coercing to numeric for volatility."}
        close = df["Close"]

        indicators = {}
        if len(close) >= 20:
            bb_indicator = BollingerBands(close, fillna=False)
            indicators["bb_high"] = bb_indicator.bollinger_hband().iloc[-1]
            indicators["bb_low"] = bb_indicator.bollinger_lband().iloc[-1]
            indicators["bb_ma"] = bb_indicator.bollinger_mavg().iloc[-1]
            indicators["bb_high_series"] = bb_indicator.bollinger_hband()
            indicators["bb_low_series"] = bb_indicator.bollinger_lband()
            indicators["bb_ma_series"] = bb_indicator.bollinger_mavg()
        else:
            indicators["bb_high"], indicators["bb_low"], indicators["bb_ma"] = None, None, None
            indicators["bb_high_series"], indicators["bb_low_series"], indicators["bb_ma_series"] = pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
        return {k: _sanitize_indicator_value(v) for k, v in indicators.items()}
    except Exception as e:
        print(f"ERROR in calculate_volatility_indicators: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Failed to calculate volatility: {str(e)}"}


available_technical_indicator_tools = {
    "moving_averages": calculate_moving_averages,
    "oscillators": calculate_oscillators,
    "volatility": calculate_volatility_indicators,
}


# --- Data Fetching Logic ---
async def _is_valid_ticker_async(ticker_symbol: str) -> bool:
    try:
        loop = asyncio.get_event_loop()
        stock = yf.Ticker(ticker_symbol)
        df_hist = await loop.run_in_executor(None, lambda s: s.history(period="1d"), stock)
        if not df_hist.empty:
            return True
        stock_info = await loop.run_in_executor(None, lambda s: s.info, stock)
        return bool(stock_info and (stock_info.get('regularMarketPrice') is not None or stock_info.get('shortName')))
    except Exception as e:
        print(f"LOG: yfinance validation check failed for {ticker_symbol}: {e}")
        if "401" in str(e).lower() or "unauthorized" in str(e).lower():
            print(f"CRITICAL_YFINANCE_AUTH_ERROR: Received 401 Unauthorized for {ticker_symbol} from yfinance.")
        return False


def _parse_alphavantage_daily_data(data: Dict, symbol: str) -> pd.DataFrame:
    if "Time Series (Daily)" not in data:
        print(f"LOG: Alpha Vantage - 'Time Series (Daily)' not found for {symbol}")
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
    df.index = pd.to_datetime(df.index)
    df.rename(columns={'1. open': 'Open', '2. high': 'High', '3. low': 'Low', '4. close': 'Close',
                       '5. adjusted close': 'Adj Close', '6. volume': 'Volume'}, inplace=True)
    for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.sort_index(inplace=True)
    return df


def _parse_alphavantage_overview(data: Dict, symbol: str) -> Dict:
    if not data or 'Symbol' not in data:
        print(f"LOG: Alpha Vantage - 'OVERVIEW' data is empty or invalid for {symbol}")
        return {}
    def safe_float(value_str):
        if value_str is None or str(value_str).lower() == 'none': return None
        try: return float(value_str)
        except (ValueError, TypeError): return None

    info = {
        'symbol': data.get('Symbol'), 'shortName': data.get('Name'), 'longName': data.get('Name'),
        'sector': data.get('Sector'), 'industry': data.get('Industry'),
        'longBusinessSummary': data.get('Description'),
        'marketCap': safe_float(data.get('MarketCapitalization')),
        'trailingPE': safe_float(data.get('PERatio')), 'forwardPE': safe_float(data.get('ForwardPE')),
        'trailingEps': safe_float(data.get('EPS')), 'dividendYield': safe_float(data.get('DividendYield')),
        'beta': safe_float(data.get('Beta')), '52WeekHigh': safe_float(data.get('52WeekHigh')),
        '52WeekLow': safe_float(data.get('52WeekLow')),}
    return info


async def fetch_stock_data_alphavantage_async(ticker_symbol: str):
    if not ALPHAVANTAGE_API_KEY:
        print("LOG: Alpha Vantage API key is 'demo' or not set. Skipping Alpha Vantage fetch.")
        raise ValueError("Alpha Vantage API key not configured for fallback.")
    loop = asyncio.get_event_loop()
    def fetch_sync_av():
        print(f"LOG: Attempting Alpha Vantage fetch for {ticker_symbol}")
        ts = TimeSeries(key=ALPHAVANTAGE_API_KEY, output_format='json')
        fd = FundamentalData(key=ALPHAVANTAGE_API_KEY, output_format='json')
        daily_df = pd.DataFrame(); stock_info = {}
        try:
            data_daily, _ = ts.get_daily_adjusted(symbol=ticker_symbol, outputsize='full')
            daily_df = _parse_alphavantage_daily_data(data_daily, ticker_symbol)
            print(f"LOG: Alpha Vantage daily data fetched for {ticker_symbol}. Shape: {daily_df.shape}")
            if daily_df.empty: raise ValueError("Alpha Vantage returned no daily data.")
        except Exception as e:
            print(f"ERROR fetching daily Alpha Vantage data for {ticker_symbol}: {e}"); raise
        try:
            overview_data, _ = fd.get_company_overview(symbol=ticker_symbol)
            stock_info = _parse_alphavantage_overview(overview_data, ticker_symbol)
            print(f"LOG: Alpha Vantage overview data fetched for {ticker_symbol}")
        except Exception as e:
            print(f"ERROR fetching Alpha Vantage company overview for {ticker_symbol}: {e}")
        weekly_df = pd.DataFrame(); monthly_df = pd.DataFrame()
        if not daily_df.empty:
            try:
                weekly_df = daily_df.resample('W-FRI').agg(
                    {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Adj Close': 'last', 'Volume': 'sum'}).dropna()
            except Exception as e: print(f"ERROR resampling weekly AV data: {e}")
            try:
                monthly_df = daily_df.resample('M').agg(
                    {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Adj Close': 'last', 'Volume': 'sum'}).dropna()
            except Exception as e: print(f"ERROR resampling monthly AV data: {e}")
        return {"resolved_symbol": ticker_symbol, "info": stock_info, "daily": daily_df, "weekly": weekly_df,
                "monthly": monthly_df, "financials": pd.DataFrame(), "balance_sheet": pd.DataFrame(), "cashflow": pd.DataFrame()}
    return await loop.run_in_executor(None, fetch_sync_av)


async def fetch_stock_data_yf_async(ticker_symbol: str):
    loop = asyncio.get_event_loop()
    def fetch_sync():
        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        if not info: print(f"Warning: No info dictionary returned by yfinance for ticker '{ticker_symbol}'.")
        daily_data = stock.history(period="1y")
        if daily_data.empty: print(f"Warning: No daily historical data found for ticker '{ticker_symbol}'.")
        return {"resolved_symbol": ticker_symbol, "info": info or {}, "daily": daily_data,
                "weekly": stock.history(period="5y", interval="1wk"),
                "monthly": stock.history(period="max", interval="1mo"),
                "financials": stock.financials if hasattr(stock, 'financials') else pd.DataFrame(),
                "balance_sheet": stock.balance_sheet if hasattr(stock, 'balance_sheet') else pd.DataFrame(),
                "cashflow": stock.cashflow if hasattr(stock, 'cashflow') else pd.DataFrame()}
    return await loop.run_in_executor(None, fetch_sync)


# --- Content Generation Functions ---
async def get_technical_analysis_summary_content(stock_symbol: str, executed_indicators: Dict[str, Dict[str, Dict]]):
    current_llm, _, _, _, _, _ = get_llms()
    prompt_parts = [f"Analyze the technical outlook for {stock_symbol} based ONLY on the following calculated indicator values."]
    prompt_parts.append(f"Provide a VERY BRIEF summary (2-3 bullet points MAX). Focus ONLY on the main trend and key signals (e.g., RSI overbought/oversold, MACD cross). STRICTLY ADHERE to the {LLM_MAX_TOKENS} token limit.")
    has_data = False
    for timeframe in ["daily", "weekly", "monthly"]:
        indicators_by_tool = executed_indicators.get(timeframe, {})
        if not indicators_by_tool: continue
        timeframe_data_exists = False
        timeframe_prompt_parts = [f"\n{timeframe.capitalize()} Indicators:"]
        for tool_name, values in indicators_by_tool.items():
            if values.get("error"):
                timeframe_prompt_parts.append(f"  - {tool_name.replace('_', ' ').capitalize()}: Error ({values['error']})")
                continue
            filtered_values = {k: v for k, v in values.items() if not k.endswith('_series') and k != 'last_close' and v is not None and not isinstance(v,str)}
            if filtered_values:
                values_str = ", ".join([f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}" for k, v in filtered_values.items()])
                timeframe_prompt_parts.append(f"  - {tool_name.replace('_', ' ').capitalize()}: {values_str}")
                timeframe_data_exists = True
        if timeframe_data_exists:
            prompt_parts.extend(timeframe_prompt_parts); has_data = True
    if not has_data: return "No valid technical indicator data points were available to generate a summary."
    prompt_parts.append("\nSummary:")
    prompt = "\n".join(prompt_parts)
    response = await current_llm.ainvoke(prompt)
    return response.content


async def get_fundamental_analysis_summary_content(stock_symbol: str, stock_info_json: str,
                                                   financials_json: Optional[str], balance_sheet_json: Optional[str],
                                                   cashflow_json: Optional[str]):
    current_llm, _, _, _, _, _ = get_llms()
    stock_info = json.loads(stock_info_json) if stock_info_json else {}
    financials_df = pd.read_json(StringIO(financials_json), orient='split') if financials_json else pd.DataFrame()
    balance_sheet_df = pd.read_json(StringIO(balance_sheet_json), orient='split') if balance_sheet_json else pd.DataFrame()

    market_cap = stock_info.get('marketCap'); pe_ratio = stock_info.get('trailingPE') or stock_info.get('forwardPE')
    eps = stock_info.get('trailingEps'); dividend_yield = stock_info.get('dividendYield')
    sector = stock_info.get('sector'); industry = stock_info.get('industry')
    summary = stock_info.get('longBusinessSummary', ""); summary_preview = summary[:500] + "..." if summary and len(summary) > 500 else summary

    latest_annual_revenue = 'N/A'
    if not financials_df.empty and 'Total Revenue' in financials_df.index and not financials_df.loc['Total Revenue'].empty:
        latest_annual_revenue = financials_df.loc['Total Revenue'].iloc[0]
    latest_annual_net_income = 'N/A'
    if not financials_df.empty and 'Net Income' in financials_df.index and not financials_df.loc['Net Income'].empty:
        latest_annual_net_income = financials_df.loc['Net Income'].iloc[0]
    latest_total_assets = 'N/A'
    if not balance_sheet_df.empty and 'Total Assets' in balance_sheet_df.index and not balance_sheet_df.loc['Total Assets'].empty:
        latest_total_assets = balance_sheet_df.loc['Total Assets'].iloc[0]
    latest_total_liabilities = 'N/A'
    liab_col_name = 'Total Liabilities Net Minority Interest'
    if not balance_sheet_df.empty and liab_col_name in balance_sheet_df.index and not balance_sheet_df.loc[liab_col_name].empty:
        latest_total_liabilities = balance_sheet_df.loc[liab_col_name].iloc[0]
    elif not balance_sheet_df.empty and 'Total Liabilities' in balance_sheet_df.index and not balance_sheet_df.loc['Total Liabilities'].empty:
        latest_total_liabilities = balance_sheet_df.loc['Total Liabilities'].iloc[0]

    prompt = (f"Analyze the fundamental data for {stock_symbol}. Provide a VERY BRIEF summary (2-3 bullet points MAX) "
        f"highlighting ONLY the most critical strengths and weaknesses based on the provided metrics. STRICTLY ADHERE to the {LLM_MAX_TOKENS} token limit.\n\n"
        f"Company Information:\nSector: {sector or 'N/A'}\nIndustry: {industry or 'N/A'}\nMarket Cap: {market_cap or 'N/A'}\nBusiness Summary Preview: {summary_preview or 'N/A'}\n\n"
        f"Key Metrics:\nP/E Ratio: {pe_ratio or 'N/A'}\nEPS (Trailing): {eps or 'N/A'}\nDividend Yield: {dividend_yield or 'N/A'}\n\n"
        f"Financial Highlights (Latest Annual):\nTotal Revenue: {latest_annual_revenue}\nNet Income: {latest_annual_net_income}\n\n"
        f"Balance Sheet Highlights (Latest Annual):\nTotal Assets: {latest_total_assets}\nTotal Liabilities: {latest_total_liabilities}\n\nBrief Summary:")
    response = await current_llm.ainvoke(prompt)
    return response.content


async def generate_direct_answer_to_question_content(
    stock_symbol: str, original_user_question: str, analyzed_query_focus: Optional[str],
    ta_summary: Optional[str], fa_summary: Optional[str],
    executed_technical_indicators: Optional[Dict[str, Dict[str, Dict]]],
    stock_info_json: Optional[str], question_data_type_needed: Optional[str]):
    _, _, _, _, _, current_llm_direct_answerer = get_llms()
    prompt_parts = [f"You are a financial analyst. The user is asking about {stock_symbol}."]
    prompt_parts.append(f"User's Original Question: \"{original_user_question}\"")
    if analyzed_query_focus and analyzed_query_focus.lower() not in ["general analysis and recommendation", "general analysis"]:
        prompt_parts.append(f"AI Analyzed Focus of Question: {analyzed_query_focus}")
    prompt_parts.append("\nUse the following context to answer the user's question directly and concisely. Prioritize using the detailed data if relevant to the question's focus, otherwise use the summaries.")
    data_type_needed_for_prompt = question_data_type_needed or "general"
    if data_type_needed_for_prompt in ["technical", "both"] and executed_technical_indicators:
        daily_indicators = executed_technical_indicators.get('daily', {})
        if daily_indicators:
            tech_details = "Key Daily Technical Indicators:\n"
            for tool_name, values in daily_indicators.items():
                if values.get("error"): continue
                filtered = {k: v for k, v in values.items() if not k.endswith("_series") and v is not None and not isinstance(v, str)}
                if filtered: tech_details += f"  - {tool_name.replace('_', ' ').capitalize()}: {json.dumps(filtered)}\n"
            if tech_details != "Key Daily Technical Indicators:\n": prompt_parts.append(tech_details)
    if data_type_needed_for_prompt in ["fundamental", "both"] and stock_info_json:
        stock_info = json.loads(stock_info_json)
        fund_details = "Key Fundamental Data:\n"
        key_fund_fields = ['sector', 'industry', 'marketCap', 'trailingPE', 'forwardPE', 'trailingEps', 'dividendYield', 'beta', '52WeekChange', 'bookValue', 'priceToBook', 'longBusinessSummary']
        added_fields = 0
        for kf in key_fund_fields:
            val = stock_info.get(kf)
            if val is not None:
                if kf == 'longBusinessSummary': fund_details += f"  - Business Summary Snippet: {str(val)[:200]}...\n"
                else: fund_details += f"  - {kf}: {val}\n"
                added_fields +=1
        if added_fields > 0: prompt_parts.append(fund_details)
    prompt_parts.append(f"Technical Analysis Summary (if available):\n{ta_summary or 'Not available.'}")
    prompt_parts.append(f"Fundamental Analysis Summary (if available):\n{fa_summary or 'Not available.'}")
    prompt_parts.append(f"\nInstruction: Based ONLY on the provided context and summaries above, provide a direct and concise answer (2-5 sentences) to the User's Original Question. "
        f"If the provided context does not contain enough information to directly answer the question, clearly state that the available analysis does not specifically address it, and briefly explain why (e.g., 'The analysis does not cover future price predictions.'). "
        f"Focus on addressing the core of their query. STRICTLY ADHERE to the {DIRECT_ANSWER_MAX_TOKENS} token limit.")
    prompt_parts.append("\nDirect Answer to User's Question:")
    prompt = "\n".join(prompt_parts)
    response = await current_llm_direct_answerer.ainvoke(prompt)
    return response.content


async def generate_recommendation_content(stock_symbol: str, ta_summary: str, fa_summary: str):
    current_llm, _, _, _, _, _ = get_llms()
    prompt_parts = [f"Stock: {stock_symbol}"]
    prompt_parts.append(f"Technical Analysis Summary:\n{ta_summary}")
    prompt_parts.append(f"Fundamental Analysis Summary:\n{fa_summary}")
    prompt_parts.append("\nInstructions:")
    prompt_parts.append("1. Provide the overall market recommendation (Buy/Sell/Hold).")
    prompt_parts.append("2. List ONLY 2-3 main bullet points briefly justifying the recommendation.")
    prompt_parts.append(f"Be extremely concise. STRICTLY ADHERE to the {LLM_MAX_TOKENS} token limit for this recommendation section.");
    prompt_parts.append("\nOverall Recommendation and Justification:")
    prompt = "\n".join(prompt_parts)
    response = await current_llm.ainvoke(prompt)
    return response.content


async def generate_position_advice_content(stock_symbol: str, recommendation: str, user_position: dict, key_levels: Dict):
    _, _, current_llm_pos_advice, _, _, _ = get_llms()
    if not user_position or not user_position.get("shares") or not user_position.get("avg_price"):
        return "No position information provided or incomplete information."
    shares = user_position["shares"]; avg_price = user_position["avg_price"]
    levels_str = "Key technical levels (daily): "
    levels_found = []
    if key_levels.get('sma_50') is not None: levels_found.append(f"SMA 50: {float(key_levels['sma_50']):.2f}")
    if key_levels.get('sma_200') is not None: levels_found.append(f"SMA 200: {float(key_levels['sma_200']):.2f}")
    if key_levels.get('bb_low') is not None: levels_found.append(f"Bollinger Low: {float(key_levels['bb_low']):.2f}")
    if key_levels.get('last_close') is not None: levels_found.append(f"Last Close: {float(key_levels['last_close']):.2f}")
    if not levels_found: levels_str += "N/A"
    else: levels_str += ", ".join(levels_found)
    prompt = (f"\nStock: {stock_symbol}\nUser Position: {shares} shares @ avg price ${float(avg_price):.2f}\n"
        f"Overall Recommendation: {recommendation}\n{levels_str}\n\nInstructions:\n"
        f"Provide CONCISE advice for the user's existing position based on the recommendation and key levels.\n"
        f"1.  Identify 1-2 key support levels based on the provided technical levels (e.g., nearest SMA, Bollinger Low). State the level clearly.\n"
        f"2.  Suggest a potential stop-loss level based on the support levels (e.g., slightly below a key support). State the level clearly and mention this is a suggestion for risk management, not financial advice.\n"
        f"3.  Briefly outline Conservative Next Steps (1-2 sentences).\n"
        f"4.  Briefly outline Aggressive Next Steps (1-2 sentences), highlighting risks.\n\n"
        f"Keep the entire response concise and actionable. STRICTLY ADHERE to the {POSITION_ADVICE_MAX_TOKENS} token limit.\n")
    response = await current_llm_pos_advice.ainvoke(prompt)
    return response.content
