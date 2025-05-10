# backend/stock_analyzer.py

import os
import json
from typing import TypedDict, List, Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
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
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
from datetime import datetime, timedelta
import httpx  # For making API calls to Brave Search API

load_dotenv()

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Constants and Configuration ---
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_LANGCHAIN_TRACING_V2 = "LANGCHAIN_TRACING_V2"
ENV_LANGCHAIN_ENDPOINT = "LANGCHAIN_ENDPOINT"
ENV_LANGCHAIN_API_KEY = "LANGCHAIN_API_KEY"
ENV_LANGCHAIN_PROJECT = "LANGCHAIN_PROJECT"
ENV_ALPHAVANTAGE_API_KEY = "ALPHAVANTAGE_API_KEY"
ENV_BRAVE_SEARCH_API_KEY = "BRAVE_SEARCH_API_KEY"

OPENAI_API_KEY = os.getenv(ENV_OPENAI_API_KEY)
LANGCHAIN_TRACING_V2_RAW = os.getenv("LANGSMITH_TRACING", os.getenv(ENV_LANGCHAIN_TRACING_V2, "false"))
LANGCHAIN_TRACING_V2 = LANGCHAIN_TRACING_V2_RAW.lower() == "true"
LANGCHAIN_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", os.getenv(ENV_LANGCHAIN_ENDPOINT))
LANGCHAIN_API_KEY = os.getenv("LANGSMITH_API_KEY", os.getenv(ENV_LANGCHAIN_API_KEY))
LANGCHAIN_PROJECT = os.getenv("LANGSMITH_PROJECT", os.getenv(ENV_LANGCHAIN_PROJECT, "Stock Analyzer Chat App Async"))
ALPHAVANTAGE_API_KEY = os.getenv(ENV_ALPHAVANTAGE_API_KEY, "demo")
BRAVE_API_KEY = os.getenv(ENV_BRAVE_SEARCH_API_KEY)

LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o")
LLM_SYMBOL_RESOLUTION_MODEL = os.getenv("LLM_SYMBOL_RESOLUTION_MODEL", "gpt-3.5-turbo")
LLM_QUERY_ANALYZER_MODEL = os.getenv("LLM_QUERY_ANALYZER_MODEL", "gpt-3.5-turbo")
LLM_DIRECT_ANSWER_MODEL = os.getenv("LLM_DIRECT_ANSWER_MODEL", "gpt-4o")
LLM_NEWS_SUMMARIZER_MODEL = os.getenv("LLM_NEWS_SUMMARIZER_MODEL", "gpt-3.5-turbo")

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.3))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 300))
POSITION_ADVICE_MAX_TOKENS = int(os.getenv("POSITION_ADVICE_MAX_TOKENS", 500))
TOOL_SELECTOR_LLM_MAX_TOKENS = int(os.getenv("TOOL_SELECTOR_LLM_MAX_TOKENS", 200))
QUESTION_ENHANCER_MAX_TOKENS = int(os.getenv("QUESTION_ENHANCER_MAX_TOKENS", 150))
DIRECT_ANSWER_MAX_TOKENS = int(os.getenv("DIRECT_ANSWER_MAX_TOKENS", 400))
NEWS_SUMMARY_MAX_TOKENS = int(os.getenv("NEWS_SUMMARY_MAX_TOKENS", 350))

CHARTS_DIR = "/tmp/charts"  # Standard temporary directory for GAE, or local /tmp
try:
    os.makedirs(CHARTS_DIR, exist_ok=True)
except OSError as e:
    print(f"Warning: Could not create CHARTS_DIR ({CHARTS_DIR}) at module level: {e}")
CHARTS_URL_PREFIX = "/charts_data"  # Matches main.py endpoint

tool_descriptions_for_llm = ""


class ToolSelection(BaseModel):
    tool_names: List[str] = Field(
        description="List of technical indicator tool names to be called (e.g., moving_averages, oscillators).")


class QueryAnalysisOutput(BaseModel):
    focus: str = Field(description="A short phrase describing the primary focus of the user's question.")
    data_type_needed: str = Field(
        description="The primary type of data needed to answer: 'technical', 'fundamental', 'news', 'both', or 'general'.")


def initialize_analyzer():
    global tool_descriptions_for_llm
    if not OPENAI_API_KEY:
        print(f"CRITICAL ERROR: The environment variable {ENV_OPENAI_API_KEY} is not set.")
        return False
    if not ALPHAVANTAGE_API_KEY or ALPHAVANTAGE_API_KEY == "demo":
        print(
            "WARNING: ALPHAVANTAGE_API_KEY is not set or is 'demo'. Fallback API calls to Alpha Vantage will be very limited.")
    if not BRAVE_API_KEY:
        print(
            f"WARNING: The environment variable {ENV_BRAVE_SEARCH_API_KEY} is not set. News fetching will be disabled.")

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
                print(f"Warning: Tool '{name}' has no description or docstring.")  # Indented correctly
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


# Initialize LLM variables to None for lazy loading
llm, tool_selector_llm, llm_pos_advice, llm_symbol_resolver, llm_query_analyzer, llm_direct_answerer, llm_news_summarizer = (None,) * 7


def get_llms():
    """Initializes and returns all LLM instances, creating them if they don't exist."""
    global llm, tool_selector_llm, llm_pos_advice, llm_symbol_resolver, llm_query_analyzer, llm_direct_answerer, llm_news_summarizer
    if llm is None:
        llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=LLM_TEMPERATURE, max_tokens=LLM_MAX_TOKENS)
    if tool_selector_llm is None:
        tool_selector_llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0.2, max_tokens=TOOL_SELECTOR_LLM_MAX_TOKENS)
    if llm_pos_advice is None:
        llm_pos_advice = ChatOpenAI(model=LLM_MODEL_NAME, temperature=LLM_TEMPERATURE,
                                    max_tokens=POSITION_ADVICE_MAX_TOKENS)
    if llm_symbol_resolver is None:
        llm_symbol_resolver = ChatOpenAI(model=LLM_SYMBOL_RESOLUTION_MODEL, temperature=0.0, max_tokens=50)
    if llm_query_analyzer is None:
        llm_query_analyzer = ChatOpenAI(model=LLM_QUERY_ANALYZER_MODEL, temperature=0.1,
                                        max_tokens=QUESTION_ENHANCER_MAX_TOKENS)
    if llm_direct_answerer is None:
        llm_direct_answerer = ChatOpenAI(model=LLM_DIRECT_ANSWER_MODEL, temperature=0.5,
                                         max_tokens=DIRECT_ANSWER_MAX_TOKENS)
    if llm_news_summarizer is None:
        llm_news_summarizer = ChatOpenAI(model=LLM_NEWS_SUMMARIZER_MODEL, temperature=0.2,
                                         max_tokens=NEWS_SUMMARY_MAX_TOKENS)
    return llm, tool_selector_llm, llm_pos_advice, llm_symbol_resolver, llm_query_analyzer, llm_direct_answerer, llm_news_summarizer


def _sanitize_indicator_value(value: Any) -> Any:
    """Sanitizes indicator values for JSON serialization, handling pandas Series and numpy types."""
    if isinstance(value, pd.Series):
        if value.empty or value.isna().all():
            return None
        # Convert Series to JSON string (orient='split' is often good for timeseries)
        return value.to_json(orient='split', date_format='iso')
    elif pd.isna(value):  # Check for pandas/numpy NaN
        return None
    elif isinstance(value, (np.generic, int, float)):  # Handles numpy float, int types
        return float(value)  # Convert to standard Python float
    elif isinstance(value, str):
        return value
    return value  # Return as is if not a special type


@tool(description="Calculates Simple Moving Averages (SMA 20, 50, 200) and Exponential Moving Averages (EMA 12, 26).")
async def calculate_moving_averages(data_json: str) -> Dict[str, Optional[Any]]:
    """Calculates various moving averages from stock price data."""
    try:
        df = pd.read_json(StringIO(data_json), orient='split')
        if df.empty or 'Close' not in df.columns:
            return {"error": "Invalid or empty data for moving averages"}
        close = df["Close"]
        indicators = {
            "sma_20": SMAIndicator(close, window=20).sma_indicator().iloc[-1] if len(close) >= 20 else None,
            "sma_50": SMAIndicator(close, window=50).sma_indicator().iloc[-1] if len(close) >= 50 else None,
            "sma_200": SMAIndicator(close, window=200).sma_indicator().iloc[-1] if len(close) >= 200 else None,
            "ema_12": EMAIndicator(close, window=12).ema_indicator().iloc[-1] if len(close) >= 12 else None,
            "ema_26": EMAIndicator(close, window=26).ema_indicator().iloc[-1] if len(close) >= 26 else None,
            "last_close": close.iloc[-1] if not close.empty else None,
            # Include series for potential charting (will be sanitized, e.g. to JSON string)
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


@tool(description="Calculates MACD (Moving Average Convergence Divergence) and RSI (Relative Strength Index).")
async def calculate_oscillators(data_json: str) -> Dict[str, Optional[Any]]:
    """Calculates MACD and RSI indicators."""
    try:
        df = pd.read_json(StringIO(data_json), orient='split')
        if df.empty or 'Close' not in df.columns:
            return {"error": "Invalid data for oscillators"}
        close = df["Close"]
        indicators = {}
        if len(close) >= 26:  # Common period for MACD
            macd_indicator = MACD(close)
            indicators["macd"] = macd_indicator.macd().iloc[-1]
            indicators["macd_signal"] = macd_indicator.macd_signal().iloc[-1]
            indicators["macd_diff"] = macd_indicator.macd_diff().iloc[-1]
            indicators["macd_series"] = macd_indicator.macd()
            indicators["macd_signal_series"] = macd_indicator.macd_signal()
            indicators["macd_hist_series"] = macd_indicator.macd_diff()
        else:  # Not enough data
            indicators["macd"], indicators["macd_signal"], indicators["macd_diff"] = None, None, None
            indicators["macd_series"], indicators["macd_signal_series"], indicators["macd_hist_series"] = pd.Series(
                dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

        if len(close) >= 14:  # Common period for RSI
            rsi_series = RSIIndicator(close).rsi()
            indicators["rsi"] = rsi_series.iloc[-1]
            indicators["rsi_series"] = rsi_series
        else:  # Not enough data
            indicators["rsi"] = None
            indicators["rsi_series"] = pd.Series(dtype=float)
        return {k: _sanitize_indicator_value(v) for k, v in indicators.items()}
    except Exception as e:
        return {"error": f"Failed to calculate oscillators: {str(e)}"}


@tool(description="Calculates Bollinger Bands (High, Low, Moving Average).")
async def calculate_volatility_indicators(data_json: str) -> Dict[str, Optional[Any]]:
    """Calculates Bollinger Bands."""
    try:
        df = pd.read_json(StringIO(data_json), orient='split')
        if df.empty or 'Close' not in df.columns:
            return {"error": "Invalid data for volatility"}
        close = df["Close"]
        indicators = {}
        if len(close) >= 20:  # Common period for Bollinger Bands
            bb_indicator = BollingerBands(close)
            indicators["bb_high"] = bb_indicator.bollinger_hband().iloc[-1]
            indicators["bb_low"] = bb_indicator.bollinger_lband().iloc[-1]
            indicators["bb_ma"] = bb_indicator.bollinger_mavg().iloc[-1]
            indicators["bb_high_series"] = bb_indicator.bollinger_hband()
            indicators["bb_low_series"] = bb_indicator.bollinger_lband()
            indicators["bb_ma_series"] = bb_indicator.bollinger_mavg()
        else:  # Not enough data
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


def sanitize_value(value: Any) -> Any:
    """Sanitizes values for JSON serialization, handling various special types."""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.datetime64, np.timedelta64)):
        return str(value)  # Convert numpy datetime/timedelta to string
    if isinstance(value, np.generic):
        return value.item()  # Convert numpy scalar types to Python equivalents
    if isinstance(value, pd.Series):
        return [sanitize_value(item) for item in value.tolist()]  # Convert Series to list of sanitized values
    if isinstance(value, dict):
        return {str(k): sanitize_value(v) for k, v in value.items()}  # Recursively sanitize dicts
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]  # Recursively sanitize lists
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None  # Convert NaN/inf to None
    return value


async def _is_valid_ticker_async(ticker_symbol: str) -> bool:
    """Asynchronously checks if a ticker symbol is valid using yfinance."""
    try:
        loop = asyncio.get_event_loop()
        stock = yf.Ticker(ticker_symbol)
        df_hist = await loop.run_in_executor(None, lambda s: s.history(period="1d"), stock)
        if not df_hist.empty:
            return True
        else:
            stock_info = await loop.run_in_executor(None, lambda s: s.info, stock)
            return bool(
                stock_info and (stock_info.get('regularMarketPrice') is not None or stock_info.get('shortName')))
    except Exception as e:
        print(f"LOG: yfinance validation check failed for {ticker_symbol}: {e}")
        if "401" in str(e).lower() or "unauthorized" in str(e).lower():
            print(f"CRITICAL_YFINANCE_AUTH_ERROR: Received 401 Unauthorized for {ticker_symbol} from yfinance.")
        return False


def _parse_alphavantage_daily_data(data: Dict, symbol: str) -> pd.DataFrame:
    """Parses Alpha Vantage TIME_SERIES_DAILY_ADJUSTED into a yfinance-like DataFrame."""
    if "Time Series (Daily)" not in data:
        print(f"LOG: Alpha Vantage - 'Time Series (Daily)' not found for {symbol}")
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
    df.index = pd.to_datetime(df.index)
    df.rename(columns={
        '1. open': 'Open', '2. high': 'High', '3. low': 'Low', '4. close': 'Close',
        '5. adjusted close': 'Adj Close', '6. volume': 'Volume'
    }, inplace=True)
    for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.sort_index(inplace=True)
    return df


def _parse_alphavantage_overview(data: Dict, symbol: str) -> Dict:
    """Parses Alpha Vantage COMPANY_OVERVIEW into a yfinance-like info dictionary."""
    if not data or 'Symbol' not in data:
        print(f"LOG: Alpha Vantage - 'OVERVIEW' data is empty or invalid for {symbol}")
        return {}
    info = {
        'symbol': data.get('Symbol'),
        'shortName': data.get('Name'),
        'longName': data.get('Name'),
        'sector': data.get('Sector'),
        'industry': data.get('Industry'),
        'longBusinessSummary': data.get('Description'),
        'marketCap': float(data.get('MarketCapitalization', 0)) if data.get('MarketCapitalization') not in ['None',
                                                                                                            None,
                                                                                                            '0'] else None,
        'trailingPE': float(data.get('PERatio', 0)) if data.get('PERatio') not in ['None', None, '0'] else None,
        'forwardPE': float(data.get('ForwardPE', 0)) if data.get('ForwardPE') not in ['None', None, '0'] else None,
        'trailingEps': float(data.get('EPS', 0)) if data.get('EPS') not in ['None', None, '0'] else None,
        'dividendYield': float(data.get('DividendYield', 0)) if data.get('DividendYield') not in ['None', None,
                                                                                                  '0'] else None,
        'beta': float(data.get('Beta', 0)) if data.get('Beta') not in ['None', None] else None,  # Beta can be 0
        '52WeekHigh': float(data.get('52WeekHigh', 0)) if data.get('52WeekHigh') not in ['None', None, '0'] else None,
        '52WeekLow': float(data.get('52WeekLow', 0)) if data.get('52WeekLow') not in ['None', None, '0'] else None,
    }
    return info


async def fetch_stock_data_alphavantage_async(ticker_symbol: str):
    """Asynchronously fetches stock data using Alpha Vantage via thread pool."""
    if not ALPHAVANTAGE_API_KEY or ALPHAVANTAGE_API_KEY == "demo":
        print("LOG: Alpha Vantage API key is 'demo' or not set. Skipping Alpha Vantage fetch.")
        raise ValueError("Alpha Vantage API key not configured for fallback.")
    loop = asyncio.get_event_loop()

    def fetch_sync_av():
        print(f"LOG: Attempting Alpha Vantage fetch for {ticker_symbol}")
        ts = TimeSeries(key=ALPHAVANTAGE_API_KEY, output_format='json')
        fd = FundamentalData(key=ALPHAVANTAGE_API_KEY, output_format='json')
        daily_df = pd.DataFrame()
        stock_info = {}
        try:
            data_daily, _ = ts.get_daily_adjusted(symbol=ticker_symbol, outputsize='full')
            daily_df = _parse_alphavantage_daily_data(data_daily, ticker_symbol)
            print(f"LOG: Alpha Vantage daily data fetched for {ticker_symbol}. Shape: {daily_df.shape}")
            if daily_df.empty:
                raise ValueError("Alpha Vantage returned no daily data.")
        except Exception as e:
            print(f"ERROR fetching daily Alpha Vantage data for {ticker_symbol}: {e}")
            raise
        try:
            overview_data, _ = fd.get_company_overview(symbol=ticker_symbol)
            stock_info = _parse_alphavantage_overview(overview_data, ticker_symbol)
            print(f"LOG: Alpha Vantage overview data fetched for {ticker_symbol}. Is empty: {not stock_info}")
        except Exception as e:
            print(
                f"ERROR fetching Alpha Vantage company overview for {ticker_symbol}: {e}")  # Log but don't raise if daily data was fine

        weekly_df, monthly_df = pd.DataFrame(), pd.DataFrame()
        if not daily_df.empty:
            try:
                weekly_df = daily_df.resample('W-FRI').agg(
                    {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Adj Close': 'last',
                     'Volume': 'sum'}).dropna()
            except Exception as e:
                print(f"ERROR resampling weekly AV data for {ticker_symbol}: {e}")
            try:
                monthly_df = daily_df.resample('M').agg(
                    {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Adj Close': 'last',
                     'Volume': 'sum'}).dropna()
            except Exception as e:
                print(f"ERROR resampling monthly AV data for {ticker_symbol}: {e}")
        return {
            "resolved_symbol": ticker_symbol,
            "info": stock_info,
            "daily": daily_df,
            "weekly": weekly_df,
            "monthly": monthly_df,
            "financials": pd.DataFrame(),
            "balance_sheet": pd.DataFrame(),
            "cashflow": pd.DataFrame()
        }

    return await loop.run_in_executor(None, fetch_sync_av)


async def fetch_stock_data_yf_async(ticker_symbol: str):
    """Asynchronously fetches stock data using yfinance via thread pool."""
    loop = asyncio.get_event_loop()

    def fetch_sync():
        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        if not info:
            print(f"Warning: No info dictionary returned by yfinance for ticker '{ticker_symbol}'.")
        daily_data = stock.history(period="1y")  # 1 year of daily data
        if daily_data.empty:
            print(f"Warning: No daily historical data found for ticker '{ticker_symbol}'.")
        return {
            "resolved_symbol": ticker_symbol,
            "info": info or {},
            "daily": daily_data,
            "weekly": stock.history(period="5y", interval="1wk"),
            "monthly": stock.history(period="max", interval="1mo"),
            "financials": stock.financials if hasattr(stock, 'financials') else pd.DataFrame(),
            "balance_sheet": stock.balance_sheet if hasattr(stock, 'balance_sheet') else pd.DataFrame(),
            "cashflow": stock.cashflow if hasattr(stock, 'cashflow') else pd.DataFrame()
        }

    return await loop.run_in_executor(None, fetch_sync)


async def get_technical_analysis_summary_content(stock_symbol: str, executed_indicators: Dict[str, Dict[str, Dict]]):
    """Generates a technical analysis summary using an LLM."""
    current_llm, _, _, _, _, _, _ = get_llms()
    prompt_parts = [
        f"Analyze the technical outlook for {stock_symbol} based ONLY on the following calculated indicator values.",
        f"Provide a VERY BRIEF summary (2-3 bullet points MAX). Focus ONLY on the main trend and key signals (e.g., RSI overbought/oversold, MACD cross). STRICTLY ADHERE to the {LLM_MAX_TOKENS} token limit."
    ]
    has_data = False
    for timeframe in ["daily", "weekly", "monthly"]:
        indicators_by_tool = executed_indicators.get(timeframe, {})
        if not indicators_by_tool:
            continue
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
    """Generates a fundamental analysis summary using an LLM."""
    current_llm, _, _, _, _, _, _ = get_llms()
    stock_info = json.loads(stock_info_json) if stock_info_json else {}
    financials = pd.read_json(StringIO(financials_json), orient='split') if financials_json else pd.DataFrame()
    balance_sheet = pd.read_json(StringIO(balance_sheet_json), orient='split') if balance_sheet_json else pd.DataFrame()

    market_cap = stock_info.get('marketCap')
    pe_ratio = stock_info.get('trailingPE') or stock_info.get('forwardPE')
    eps = stock_info.get('trailingEps')
    dividend_yield = stock_info.get('dividendYield')
    sector = stock_info.get('sector')
    industry = stock_info.get('industry')
    summary = stock_info.get('longBusinessSummary', "")
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


async def summarize_news_content(stock_symbol: str, company_name: Optional[str],
                                 raw_news_articles: List[Dict[str, Any]]) -> str:
    """Summarizes news articles, focusing on sentiment and impact."""
    _, _, _, _, _, _, current_llm_news_summarizer = get_llms()
    if not raw_news_articles:
        return "No news articles were found to summarize."

    news_context = "\n\nRecent News Articles (Snippets & Titles):\n"
    for i, article in enumerate(raw_news_articles[:15]):  # Limit to first 15 articles for prompt size
        news_context += f"{i + 1}. Source: {article.get('source_title', 'N/A')}\n"
        news_context += f"   Title: {article.get('title', 'N/A')}\n"
        news_context += f"   Snippet: {article.get('snippet', 'N/A')}\n"
        if article.get('publication_time'):
            news_context += f"   Published: {article.get('publication_time')}\n"
        news_context += "\n"

    prompt = f"""
    You are a financial news analyst. Based on the following news snippets regarding {company_name or stock_symbol} and the broader market, provide a concise summary.
    Instructions:
    1.  Identify 2-4 key news items that are most likely to impact {company_name or stock_symbol} or the overall market.
    2.  For each key item, briefly state its nature (e.g., earnings report, product launch, economic data, regulatory news).
    3.  Indicate the potential sentiment or impact (Positive, Negative, Neutral/Mixed, or Uncertain) for {company_name or stock_symbol} and/or the broader market.
    4.  Keep the summary to 3-5 bullet points in total.
    5.  Be factual and avoid speculation beyond what's implied in the snippets.
    6.  If news is very generic or seems unimpactful, state that.
    Strictly adhere to a {NEWS_SUMMARY_MAX_TOKENS} token limit for your entire response.
    {news_context}
    Concise News Summary (Impact & Sentiment):
    """
    response = await current_llm_news_summarizer.ainvoke(prompt)
    return response.content


async def generate_direct_answer_to_question_content(
        stock_symbol: str,
        original_user_question: str,
        analyzed_query_focus: Optional[str],
        ta_summary: Optional[str],
        fa_summary: Optional[str],
        news_summary: Optional[str],
        executed_technical_indicators: Optional[Dict[str, Dict[str, Dict]]],
        stock_info_json: Optional[str],
        question_data_type_needed: Optional[str]
):
    """Generates a direct answer to the user's question using all available context."""
    _, _, _, _, _, current_llm_direct_answerer, _ = get_llms()
    prompt_parts = [
        f"You are a financial analyst. The user is asking about {stock_symbol}.",
        f"User's Original Question: \"{original_user_question}\""
    ]
    if analyzed_query_focus and analyzed_query_focus.lower() not in ["general analysis and recommendation",
                                                                     "general analysis"]:
        prompt_parts.append(f"AI Analyzed Focus of Question: {analyzed_query_focus}")

    prompt_parts.append(
        "\nUse the following context to answer the user's question directly and concisely. Prioritize using the detailed data if relevant to the question's focus, otherwise use the summaries.")
    data_type_needed_for_prompt = question_data_type_needed or "general"

    # Include detailed technical indicators if relevant
    if data_type_needed_for_prompt in ["technical", "both", "general"] and executed_technical_indicators:
        daily_indicators = executed_technical_indicators.get('daily', {})
        if daily_indicators:
            tech_details = "Key Daily Technical Indicators:\n"
            added_tech = False
            for tool_name, values in daily_indicators.items():
                if values.get("error"):
                    continue
                filtered = {k: v for k, v in values.items() if
                            not k.endswith("_series") and v is not None and not isinstance(v, str)}
                if filtered:
                    tech_details += f"  - {tool_name.replace('_', ' ').capitalize()}: {json.dumps(filtered)}\n"
                    added_tech = True
            if added_tech:
                prompt_parts.append(tech_details)

    # Include detailed fundamental data if relevant
    if data_type_needed_for_prompt in ["fundamental", "both", "general"] and stock_info_json:
        stock_info = json.loads(stock_info_json)
        fund_details = "Key Fundamental Data:\n"
        added_fund = False
        key_fund_fields = ['sector', 'industry', 'marketCap', 'trailingPE', 'forwardPE', 'trailingEps', 'dividendYield',
                           'beta', 'longBusinessSummary']
        for kf in key_fund_fields:
            val = stock_info.get(kf)
            if val is not None:
                if kf == 'longBusinessSummary':
                    fund_details += f"  - Business Summary Snippet: {str(val)[:200]}...\n"
                else:
                    fund_details += f"  - {kf}: {val}\n"
                added_fund = True
        if added_fund:
            prompt_parts.append(fund_details)

    prompt_parts.append(f"Technical Analysis Summary (if available):\n{ta_summary or 'Not available.'}")
    prompt_parts.append(f"Fundamental Analysis Summary (if available):\n{fa_summary or 'Not available.'}")
    prompt_parts.append(f"Recent News Summary (Sentiment & Impact):\n{news_summary or 'Not available.'}")
    prompt_parts.append(
        f"\nInstruction: Based ONLY on the provided context and summaries above (Technical, Fundamental, and News), provide a direct and concise answer (2-5 sentences) to the User's Original Question. If the provided context does not contain enough information to directly answer the question, clearly state that the available analysis does not specifically address it, and briefly explain why (e.g., 'The analysis does not cover future price predictions.' or 'The news summary does not provide specific details on X.'). Focus on addressing the core of their query. STRICTLY ADHERE to the {DIRECT_ANSWER_MAX_TOKENS} token limit.")
    prompt_parts.append("\nDirect Answer to User's Question:")

    prompt = "\n".join(prompt_parts)
    response = await current_llm_direct_answerer.ainvoke(prompt)
    return response.content


async def generate_recommendation_content(stock_symbol: str, ta_summary: str, fa_summary: str,
                                          news_summary: Optional[str]):
    """Generates an overall recommendation considering TA, FA, and News."""
    current_llm, _, _, _, _, _, _ = get_llms()
    prompt_parts = [
        f"Stock: {stock_symbol}",
        f"Technical Analysis Summary:\n{ta_summary}",
        f"Fundamental Analysis Summary:\n{fa_summary}",
        f"Recent News Summary (Sentiment & Impact):\n{news_summary or 'Not available.'}"
    ]
    prompt_parts.append(
        "\nInstructions:\n1. Provide the overall market recommendation (e.g., Strong Buy, Buy, Hold, Sell, Strong Sell).\n2. List ONLY 2-3 main bullet points briefly justifying the recommendation, considering technicals, fundamentals, AND RECENT NEWS.\n3. Be extremely concise. STRICTLY ADHERE to the {LLM_MAX_TOKENS} token limit for this recommendation section.")
    prompt_parts.append("\nOverall Recommendation and Justification:")

    prompt = "\n".join(prompt_parts)
    response = await current_llm.ainvoke(prompt)
    return response.content


async def generate_position_advice_content(stock_symbol: str, recommendation: str, user_position: dict,
                                           key_levels: Dict):
    """Generates advice for an existing user position."""
    _, _, current_llm_pos_advice, _, _, _, _ = get_llms()
    if not user_position or not user_position.get("shares") or not user_position.get("avg_price"):
        return "No position information provided or incomplete information."

    shares = user_position["shares"]
    avg_price = user_position["avg_price"]
    levels_str = "Key technical levels (daily): "
    levels_found = []
    if key_levels.get('sma_50') is not None:
        levels_found.append(f"SMA 50: {key_levels['sma_50']:.2f}")
    if key_levels.get('sma_200') is not None:
        levels_found.append(f"SMA 200: {key_levels['sma_200']:.2f}")
    if key_levels.get('bb_low') is not None:
        levels_found.append(f"Bollinger Low: {key_levels['bb_low']:.2f}")
    if key_levels.get('last_close') is not None:
        levels_found.append(f"Last Close: {key_levels['last_close']:.2f}")

    if not levels_found:
        levels_str += "N/A"
    else:
        levels_str += ", ".join(levels_found)

    prompt = f"\nStock: {stock_symbol}\nUser Position: {shares} shares @ avg price ${avg_price:.2f}\nOverall Recommendation: {recommendation}\n{levels_str}\n\nInstructions:\nProvide CONCISE advice for the user's existing position based on the recommendation and key levels.\n1.  Identify 1-2 key support levels based on the provided technical levels (e.g., nearest SMA, Bollinger Low). State the level clearly.\n2.  Suggest a potential stop-loss level based on the support levels (e.g., slightly below a key support). State the level clearly and mention this is a suggestion for risk management, not financial advice.\n3.  Briefly outline Conservative Next Steps (1-2 sentences).\n4.  Briefly outline Aggressive Next Steps (1-2 sentences), highlighting risks.\n\nKeep the entire response concise and actionable. STRICTLY ADHERE to the {POSITION_ADVICE_MAX_TOKENS} token limit.\n"
    response = await current_llm_pos_advice.ainvoke(prompt)
    return response.content


# --- LangGraph State Definition ---
class StockAnalysisState(TypedDict):
    user_stock_query: str
    user_question: Optional[str]
    original_user_question: Optional[str]
    analyzed_user_query_focus: Optional[str]
    question_data_type_needed: Optional[str]
    user_position: Optional[dict]
    stock_symbol: Optional[str]
    company_name: Optional[str]  # Name of the company for better news search
    error_message: Optional[str]
    raw_stock_data_json: Optional[Dict[str, Optional[str]]]  # daily, weekly, monthly
    stock_info_json: Optional[str]  # from yfinance info or AV overview
    stock_financials_json: Optional[str]
    stock_balance_sheet_json: Optional[str]
    stock_cashflow_json: Optional[str]
    selected_technical_tools: Optional[List[str]]
    executed_technical_indicators: Optional[Dict[str, Dict[str, Dict]]]  # timeframe -> tool -> results
    generated_chart_urls: Optional[List[str]]  # List of URLs or error messages for charts
    key_technical_levels: Optional[Dict[str, Optional[float]]]  # e.g. sma_50, bb_low
    technical_analysis_summary: Optional[str]
    fundamental_analysis_summary: Optional[str]
    raw_news_articles: Optional[List[Dict[str, Any]]]  # List of news search results
    news_analysis_summary: Optional[str]  # LLM generated summary of news
    direct_answer_to_user_question: Optional[str]
    final_recommendation: Optional[str]
    position_specific_advice: Optional[str]


# --- LangGraph Nodes (Async) ---
async def start_node(state: StockAnalysisState) -> StockAnalysisState:
    """Initializes the state for a new analysis request."""
    print(f"LOG (workflow): Start node for query: {state['user_stock_query']}")
    state["error_message"] = ""  # Clear any previous errors
    state["stock_symbol"] = None
    state["company_name"] = None
    state["original_user_question"] = state.get("user_question")  # Preserve original question
    state["analyzed_user_query_focus"] = None
    state["question_data_type_needed"] = None
    state["selected_technical_tools"] = []
    state["executed_technical_indicators"] = {}
    state["generated_chart_urls"] = []  # Initialize as empty list
    state["key_technical_levels"] = {}
    state["raw_stock_data_json"] = {"daily": None, "weekly": None, "monthly": None}
    state["stock_info_json"] = None
    state["stock_financials_json"] = None
    state["stock_balance_sheet_json"] = None
    state["stock_cashflow_json"] = None
    state["technical_analysis_summary"] = None
    state["fundamental_analysis_summary"] = None
    state["raw_news_articles"] = None  # Initialize news fields
    state["news_analysis_summary"] = None
    state["direct_answer_to_user_question"] = None
    state["final_recommendation"] = None
    state["position_specific_advice"] = None
    return state


async def analyze_user_query_focus_node(state: StockAnalysisState) -> StockAnalysisState:
    """Analyzes the user's query to determine focus and data needs."""
    print(f"LOG (workflow): Analyze user query focus node for: '{state.get('original_user_question')}'")
    _, _, _, _, current_llm_analyzer, _, _ = get_llms()
    original_question = state.get("original_user_question")
    stock_query_context = state.get("user_stock_query", "the specified stock")

    if not original_question or original_question.strip() == "":
        print("LOG (workflow): No user question provided. Defaulting focus and data type.")
        state["analyzed_user_query_focus"] = "general analysis and recommendation"
        state["question_data_type_needed"] = "general"  # Default covers all data types
        return state
    try:
        structured_llm = current_llm_analyzer.with_structured_output(QueryAnalysisOutput)
        prompt_template_str = (
            "You are an AI assistant. Analyze the user's question about a stock. "
            "Identify the primary focus (e.g., 'buy/sell decision', 'long-term outlook', 'impact of recent news', 'dividend info', 'risk assessment', 'specific indicator like RSI'). "
            "Also, determine the primary type of data needed to answer it: 'technical', 'fundamental', 'news', 'both' (for TA/FA), or 'general' (if it's a broad question for overall analysis/recommendation including news). "
            "If the question explicitly mentions news, events, or current affairs, data_type_needed should be 'news' or include 'news'. "
            "If the question is very generic like 'general analysis' or 'what do you think?', set focus to 'general analysis and recommendation' and data_type_needed to 'general'.\n"
            "User's question about {stock_context}: '{question}'"
        )
        analysis_output: QueryAnalysisOutput = await structured_llm.ainvoke(
            prompt_template_str.format(stock_context=state.get("stock_symbol") or stock_query_context,
                                       question=original_question)
        )
        state["analyzed_user_query_focus"] = analysis_output.focus.strip()
        state["question_data_type_needed"] = analysis_output.data_type_needed.lower().strip()
        print(
            f"LOG (workflow): Original question: '{original_question}', Analyzed focus: '{state['analyzed_user_query_focus']}', Data needed: '{state['question_data_type_needed']}'")
    except Exception as e:
        print(f"ERROR (workflow) in analyze_user_query_focus_node: {e}")
        state["error_message"] = (
                    state.get("error_message", "") + f"Failed to analyze user query focus: {str(e)}. ").strip()
        state["analyzed_user_query_focus"] = "general analysis and recommendation"  # Fallback
        state["question_data_type_needed"] = "general"
    return state


async def resolve_stock_symbol_node(state: StockAnalysisState) -> StockAnalysisState:
    """Resolves the user's stock query to an official ticker symbol."""
    print(f"LOG (workflow): Resolve symbol node for query: '{state['user_stock_query']}'")
    user_query_original = state["user_stock_query"]
    _, _, _, current_llm_resolver, _, _, _ = get_llms()
    resolved_symbol = None
    error_accumulator = []
    try:  # LLM-based resolution attempt
        print(
            f"LOG (workflow): Attempting LLM based symbol resolution for '{user_query_original}' (yfinance context)...")
        prompt_template = ChatPromptTemplate.from_messages([("system",
                                                             "You are an expert financial assistant. Your task is to identify the most likely official stock ticker symbol based on the user's query, suitable for yfinance. If the company sounds Indian (e.g., Reliance, Infosys, Tata Steel), suggest the symbol with '.NS' suffix (e.g., RELIANCE.NS, INFY.NS). For well-known international companies (e.g., Apple, Microsoft, Google), provide their common US exchange ticker (e.g., AAPL, MSFT, GOOGL). If the query is already a valid-looking ticker, return it as is. If highly ambiguous or unclear, or if the query is nonsensical as a company name, return 'UNKNOWN'. Respond with ONLY the ticker symbol or 'UNKNOWN'."),
                                                            ("human", "User query: {query}")])
        resolve_chain = prompt_template | current_llm_resolver | StrOutputParser()
        llm_suggested_ticker = await resolve_chain.ainvoke({"query": user_query_original})
        llm_suggested_ticker = llm_suggested_ticker.strip().upper().replace("'", "").replace("\"", "")
        print(f"LOG (workflow): LLM suggested ticker for yfinance: '{llm_suggested_ticker}'")
        if llm_suggested_ticker and llm_suggested_ticker not in ["UNKNOWN", "N/A", ""]:
            if await _is_valid_ticker_async(llm_suggested_ticker):
                resolved_symbol = llm_suggested_ticker
                print(f"LOG (workflow): LLM suggested ticker '{resolved_symbol}' validated by yfinance.")
            else:
                msg = f"LLM suggestion '{llm_suggested_ticker}' was not validated by yfinance."
                print(f"LOG (workflow): {msg}")
                error_accumulator.append(msg)
        else:
            msg = "LLM could not confidently suggest a ticker or returned UNKNOWN/N/A."
            print(f"LOG (workflow): {msg}")
            error_accumulator.append(msg)
    except Exception as e:
        msg = f"Exception during LLM symbol resolution: {e}"
        print(f"ERROR (workflow): {msg}")
        error_accumulator.append(msg)

    if not resolved_symbol:  # Fallback to yfinance heuristics
        print(
            f"LOG (workflow): LLM resolution failed or invalid. Falling back to yfinance heuristics for '{user_query_original}'...")
        user_query_upper = user_query_original.strip().upper()
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
        print(f"LOG (workflow): yfinance Heuristic: Potential symbols to try: {unique_symbols_to_try}")
        for symbol_attempt in unique_symbols_to_try:
            if await _is_valid_ticker_async(symbol_attempt):
                resolved_symbol = symbol_attempt
                print(f"LOG (workflow): yfinance Heuristic resolved '{user_query_original}' to '{resolved_symbol}'")
                error_accumulator = []  # Clear previous errors if heuristic works
                break
            else:
                print(f"LOG (workflow): yfinance Heuristic: '{symbol_attempt}' is not valid.")

    if resolved_symbol:
        state["stock_symbol"] = resolved_symbol
    else:
        final_error_msg = " ".join(
            error_accumulator) if error_accumulator else f"Could not resolve '{user_query_original}' to a known stock symbol after multiple attempts."
        state["error_message"] = (state.get("error_message", "") + final_error_msg).strip()
        print(
            f"LOG (workflow): Failed to resolve symbol for '{user_query_original}'. Final error: {state['error_message']}")
    return state


async def fetch_data_node(state: StockAnalysisState) -> StockAnalysisState:
    """Fetches stock data (prices, info, financials) using yfinance with Alpha Vantage fallback."""
    resolved_symbol = state.get("stock_symbol")
    if state["error_message"] or not resolved_symbol:
        state["error_message"] = (state.get("error_message",
                                            "") + " Skipping data fetch due to unresolved symbol or prior error.").strip()
        return state

    print(f"LOG (workflow): Attempting yfinance fetch for: {resolved_symbol}")
    try:  # yfinance attempt
        fetched_data = await fetch_stock_data_yf_async(resolved_symbol)
        state["stock_symbol"] = fetched_data.get("resolved_symbol", resolved_symbol)
        info_data = fetched_data.get("info", {})
        if not isinstance(info_data, dict):
            info_data = {}
        state["company_name"] = info_data.get("shortName") or info_data.get("longName")
        sanitized_stock_info = {str(k): sanitize_value(v) for k, v in info_data.items()}
        state["stock_info_json"] = json.dumps(sanitized_stock_info)

        state["raw_stock_data_json"] = {
            tf: (data.to_json(orient='split', date_format='iso') if data is not None and not data.empty else None)
            for tf, data in [("daily", fetched_data.get("daily")), ("weekly", fetched_data.get("weekly")),
                             ("monthly", fetched_data.get("monthly"))]
        }
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
                f"Warning: Daily historical price data is empty for {state['stock_symbol']} from yfinance. Some features might be affected.")
            if not state["raw_stock_data_json"]["weekly"] and not state["raw_stock_data_json"]["monthly"]:
                state["error_message"] = (state.get("error_message",
                                                    "") + f" All historical price data is missing for {state['stock_symbol']} from yfinance. Analysis will be limited. ").strip()
        print(
            f"LOG (workflow): Successfully fetched and serialized data for {state['stock_symbol']} using yfinance. Company name: {state['company_name']}")
    except Exception as yf_error:  # yfinance failed, try Alpha Vantage
        print(f"ERROR (workflow) in yfinance fetch_data_node for {resolved_symbol}: {yf_error}")
        state["error_message"] = (state.get("error_message",
                                            "") + f" Primary data fetch (yfinance) failed for {resolved_symbol}: {str(yf_error)}. ").strip()
        if ALPHAVANTAGE_API_KEY and ALPHAVANTAGE_API_KEY != "demo":
            print(f"LOG (workflow): yfinance failed. Attempting Alpha Vantage fallback for {resolved_symbol}...")
            try:
                av_symbol = resolved_symbol.replace(".NS", "")
                fetched_av_data = await fetch_stock_data_alphavantage_async(av_symbol)
                state["stock_symbol"] = fetched_av_data.get("resolved_symbol", av_symbol)
                info_data_av = fetched_av_data.get("info", {})
                if not isinstance(info_data_av, dict):
                    info_data_av = {}
                state["company_name"] = info_data_av.get("shortName") or info_data_av.get("Name")
                sanitized_stock_info_av = {str(k): sanitize_value(v) for k, v in info_data_av.items()}
                state["stock_info_json"] = json.dumps(sanitized_stock_info_av)

                state["raw_stock_data_json"] = {
                    tf: (data.to_json(orient='split',
                                      date_format='iso') if data is not None and not data.empty else None)
                    for tf, data in [("daily", fetched_av_data.get("daily")), ("weekly", fetched_av_data.get("weekly")),
                                     ("monthly", fetched_av_data.get("monthly"))]
                }
                state["stock_financials_json"] = None  # AV fallback doesn't provide these easily
                state["stock_balance_sheet_json"] = None
                state["stock_cashflow_json"] = None
                if not state["raw_stock_data_json"]["daily"]:
                    raise ValueError(f"Alpha Vantage also returned no daily data for {av_symbol}.")
                print(
                    f"LOG (workflow): Successfully fetched data for {state['stock_symbol']} using Alpha Vantage fallback. Company name: {state['company_name']}")
                state["error_message"] = (state.get("error_message",
                                                    "") + f" Used Alpha Vantage as fallback for {resolved_symbol}. ").strip()  # Note fallback usage
            except Exception as av_error:
                print(f"ERROR (workflow) in Alpha Vantage fallback for {resolved_symbol}: {av_error}")
                state["error_message"] = (state.get("error_message",
                                                    "") + f" Alpha Vantage fallback also failed for {resolved_symbol}: {str(av_error)}. ").strip()
        else:
            print(
                f"LOG (workflow): Alpha Vantage API key not available or is 'demo', skipping fallback for {resolved_symbol}.")
    return state


async def select_technical_tools_node(state: StockAnalysisState) -> StockAnalysisState:
    """Selects which technical indicator tools to run based on query focus."""
    global tool_descriptions_for_llm
    print(f"LOG (workflow): Select technical INDICATOR tools node for {state['stock_symbol']}")
    _, current_tool_selector_llm, _, _, _, _, _ = get_llms()

    if not tool_descriptions_for_llm or "Error:" in tool_descriptions_for_llm:
        state["error_message"] = (state.get("error_message",
                                            "") + "Critical error: Indicator tool descriptions not available. Defaulting to all. ").strip()
        state["selected_technical_tools"] = list(available_technical_indicator_tools.keys())
        return state

    if state["error_message"] or not state["raw_stock_data_json"] or not any(state["raw_stock_data_json"].values()):
        state["selected_technical_tools"] = []
        state["error_message"] = (state.get("error_message",
                                            "") + "Skipped indicator tool selection due to prior errors or no price data. ").strip()
        return state

    query_context = state.get("analyzed_user_query_focus") or state.get(
        "original_user_question") or "Provide a general technical analysis."
    print(f"LOG (workflow): Using query context for tool selection: '{query_context}'")

    prompt = f"Based on the user's query focus: \"{query_context}\" for the stock {state['stock_symbol']}, which of the following technical INDICATOR tools should be used?\nAvailable indicator tools and their descriptions:\n{tool_descriptions_for_llm}\n\nRespond with a JSON object containing a single key \"tool_names\" with a list of selected indicator tool names. For a general analysis, select all indicator tools: [\"moving_averages\", \"oscillators\", \"volatility\"]. Example response: {{\"tool_names\": [\"moving_averages\", \"oscillators\"]}}"
    try:
        structured_llm = current_tool_selector_llm.with_structured_output(ToolSelection)
        response_model = await structured_llm.ainvoke(prompt)
        selected_tools = response_model.tool_names

        valid_selected_tools = [t for t in selected_tools if t in available_technical_indicator_tools]
        if len(valid_selected_tools) != len(selected_tools):
            print(
                f"Warning: LLM selected some invalid indicator tools. Original: {selected_tools}, Validated: {valid_selected_tools}")

        state["selected_technical_tools"] = valid_selected_tools
        if not valid_selected_tools:
            print(
                "Warning: No valid indicator tools selected by LLM. Defaulting to all indicator tools for general analysis.")
            state["selected_technical_tools"] = list(available_technical_indicator_tools.keys())
        print(f"LOG (workflow): Selected indicator tools: {state['selected_technical_tools']}")
    except Exception as e:
        print(f"ERROR (workflow) in select_technical_tools_node: {e}. Defaulting to all indicator tools.")
        state["error_message"] = (state.get("error_message",
                                            "") + f"Indicator tool selection failed: {str(e)}. Defaulting to all. ").strip()
        state["selected_technical_tools"] = list(available_technical_indicator_tools.keys())
    return state


async def execute_technical_tools_node(state: StockAnalysisState) -> StockAnalysisState:
    """Executes the selected technical indicator calculation tools."""
    print(f"LOG (workflow): Execute technical INDICATOR tools node for {state['stock_symbol']}")
    if state["error_message"] or not state["selected_technical_tools"] or not state["raw_stock_data_json"]:
        state["error_message"] = (state.get("error_message",
                                            "") + "Skipped technical indicator tool execution due to prior errors or no selected tools/data. ").strip()
        state["executed_technical_indicators"] = {}
        return state

    selected_tools = state["selected_technical_tools"]
    if not selected_tools:
        print("LOG (workflow): No indicator tools selected to execute.")
        state["executed_technical_indicators"] = {}
        return state

    raw_data_map = state["raw_stock_data_json"]
    indicator_results = {"daily": {}, "weekly": {}, "monthly": {}}
    tool_invocations = []

    for timeframe in ["daily", "weekly", "monthly"]:
        df_json_str = raw_data_map.get(timeframe)
        if not df_json_str:  # No data for this timeframe
            for tool_name in selected_tools:
                indicator_results[timeframe][tool_name] = {"error": f"No data available for {timeframe} timeframe."}
            continue
        for tool_name in selected_tools:
            tool_func = available_technical_indicator_tools.get(tool_name)
            if tool_func:
                tool_invocations.append((timeframe, tool_name, tool_func.ainvoke({"data_json": df_json_str})))
            else:
                indicator_results[timeframe][tool_name] = {"error": "Indicator tool function not found."}

    gathered_results = await asyncio.gather(*(inv[2] for inv in tool_invocations),
                                            return_exceptions=True)  # Execute all tool calls concurrently
    result_index = 0
    key_levels = {}

    for timeframe, tool_name, _ in tool_invocations:  # Process results
        current_result = gathered_results[result_index]
        if isinstance(current_result, Exception):
            error_str = str(current_result)
            print(f"    Error executing indicator tool '{tool_name}' for {timeframe}: {error_str}")
            indicator_results[timeframe][tool_name] = {"error": error_str}
        else:
            indicator_results[timeframe][tool_name] = current_result
            if timeframe == 'daily' and isinstance(current_result, dict) and not current_result.get(
                    "error"):  # Extract key levels from daily results
                if tool_name == 'moving_averages':
                    key_levels['sma_50'] = current_result.get('sma_50')
                    key_levels['sma_200'] = current_result.get('sma_200')
                    key_levels['last_close'] = current_result.get('last_close')
                elif tool_name == 'volatility':
                    key_levels['bb_low'] = current_result.get('bb_low')
        result_index += 1

    state["executed_technical_indicators"] = indicator_results
    state["key_technical_levels"] = {k: v for k, v in key_levels.items() if v is not None}  # Store non-None key levels
    return state


async def generate_chart_node(state: StockAnalysisState) -> StockAnalysisState:
    """Placeholder for chart generation (currently disabled)."""
    print(f"LOG (workflow): Generate chart node for {state['stock_symbol']} (currently disabled)")
    state["generated_chart_urls"] = ["Chart generation is currently disabled."]  # Keep as list
    return state


async def summarize_technical_analysis_node(state: StockAnalysisState) -> StockAnalysisState:
    """Generates a summary of the technical analysis."""
    print(f"LOG (workflow): Summarize TA node for {state['stock_symbol']}")
    indicators = state.get("executed_technical_indicators")
    if state["error_message"] and not indicators:
        state[
            "technical_analysis_summary"] = "Technical analysis summary cannot be generated due to critical prior errors."
        return state
    if not indicators or all(not tf_data for tf_data in indicators.values()):
        state["technical_analysis_summary"] = "No technical indicators were calculated or available to summarize."
        return state
    try:
        state["technical_analysis_summary"] = await get_technical_analysis_summary_content(state["stock_symbol"],
                                                                                           indicators)
    except Exception as e:
        print(f"ERROR (workflow) in summarize_technical_analysis_node: {e}")
        state["error_message"] = (state.get("error_message", "") + f"TA summary generation failed: {str(e)}. ").strip()
        state[
            "technical_analysis_summary"] = "Failed to generate technical analysis summary due to an unexpected error."
    return state


async def fundamental_analysis_node(state: StockAnalysisState) -> StockAnalysisState:
    """Generates a summary of the fundamental analysis."""
    print(f"LOG (workflow): Fundamental analysis node for {state['stock_symbol']}")
    if state["error_message"] or not state.get("stock_info_json"):
        state["error_message"] = (state.get("error_message",
                                            "") + "Skipped fundamental analysis due to prior errors or missing stock info. ").strip()
        state["fundamental_analysis_summary"] = "Fundamental analysis skipped due to missing essential data."
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
        state[
            "fundamental_analysis_summary"] = "Failed to generate fundamental analysis summary due to an unexpected error."
    return state


async def _fetch_brave_search_news_async(queries: List[str], count_per_query: int = 3) -> List[Dict[str, Any]]:
    """
    Asynchronously fetches news articles using the Brave Search API.
    """
    if not BRAVE_API_KEY:
        print("LOG (Brave Search): BRAVE_SEARCH_API_KEY not set. Skipping news fetch.")
        return []

    all_fetched_articles: List[Dict[str, Any]] = []
    brave_news_api_url = "https://api.search.brave.com/res/v1/news/search"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": BRAVE_API_KEY
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        for query_str in queries:
            params = {"q": query_str, "count": count_per_query, "safesearch": "moderate"}
            response_text_for_error = ""  # Initialize for potential use in error messages
            try:
                print(f"LOG (Brave Search): Fetching news for query: '{query_str}'")
                response = await client.get(brave_news_api_url, headers=headers, params=params)
                response_text_for_error = response.text  # Store text in case of JSON decode error
                response.raise_for_status()  # Raise HTTPStatusError for bad responses (4xx or 5xx)

                api_response_data = response.json()

                if "results" in api_response_data and isinstance(api_response_data["results"], list):
                    for item in api_response_data["results"]:
                        article_date_str = item.get("date_published") or item.get("page_age")
                        parsed_date_iso = None
                        if article_date_str:
                            if re.match(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', article_date_str):  # Check if ISO-like
                                try:
                                    parsed_date_iso = datetime.fromisoformat(
                                        article_date_str.replace("Z", "+00:00")).isoformat()
                                except ValueError:
                                    pass  # Keep original if ISO parse fails
                            elif isinstance(article_date_str,
                                            str):  # Attempt to parse relative dates like "2 hours ago"
                                try:
                                    num_match = re.search(r'\d+', article_date_str)
                                    if num_match:
                                        num = int(num_match.group(0))
                                        if "hour" in article_date_str:
                                            parsed_date_iso = (datetime.now() - timedelta(hours=num)).isoformat()
                                        elif "day" in article_date_str:
                                            parsed_date_iso = (datetime.now() - timedelta(days=num)).isoformat()
                                        elif "minute" in article_date_str:
                                            parsed_date_iso = (datetime.now() - timedelta(minutes=num)).isoformat()
                                        # Add week, month if needed, though Brave usually gives more precise for news
                                except Exception:  # pylint: disable=broad-except
                                    pass  # Keep original if relative parse fails

                        article = {
                            "query": query_str,
                            "title": item.get("title"),
                            "snippet": item.get("description") or item.get("snippet"),
                            "source_title": item.get("source", {}).get("name") or item.get("meta_url", {}).get(
                                "hostname"),  # Prefer source.name
                            "publication_time": parsed_date_iso or article_date_str,  # Use parsed ISO if available
                            "url": item.get("url")
                        }
                        all_fetched_articles.append(article)
                else:
                    print(
                        f"LOG (Brave Search): No 'results' field or not a list in API response for query '{query_str}'. Response: {api_response_data}")

            except httpx.HTTPStatusError as e:
                print(
                    f"ERROR (Brave Search): HTTP error for query '{query_str}': {e.response.status_code} - {e.response.text}")
            except httpx.RequestError as e:  # Covers network errors, DNS failures, etc.
                print(f"ERROR (Brave Search): Request error for query '{query_str}': {e}")
            except json.JSONDecodeError as e:
                print(
                    f"ERROR (Brave Search): JSON decode error for query '{query_str}': {e}. Response text: {response_text_for_error[:500]}...")  # Show partial response
            except Exception as e:  # Catch-all for other unexpected errors within the try block
                print(f"ERROR (Brave Search): Unexpected error processing query '{query_str}': {e}")
                import traceback  # Import here to avoid top-level import if not needed
                traceback.print_exc()  # Print full traceback for unexpected errors

    print(f"LOG (Brave Search): Total articles fetched from Brave API: {len(all_fetched_articles)}")
    return all_fetched_articles


async def fetch_news_node(state: StockAnalysisState) -> StockAnalysisState:
    """Fetches news articles using the Brave Search API."""
    stock_symbol = state.get("stock_symbol")
    company_name = state.get("company_name")

    if not BRAVE_API_KEY:
        state["error_message"] = (state.get("error_message",
                                            "") + "Brave Search API key not configured. Skipping news fetch. ").strip()
        state["raw_news_articles"] = []
        state["news_analysis_summary"] = "News fetching disabled due to missing API key."
        return state

    if state["error_message"] or not stock_symbol:
        state["error_message"] = (state.get("error_message",
                                            "") + "Skipping news fetch due to unresolved symbol or prior error. ").strip()
        state["raw_news_articles"] = []
        state["news_analysis_summary"] = "News fetching skipped due to prior errors."
        return state

    print(
        f"LOG (workflow): Fetching news for {stock_symbol} (Company: {company_name}) and general market using Brave API.")
    search_queries = []
    if company_name:
        search_queries.append(f"latest news {company_name} stock {stock_symbol}")
        search_queries.append(f"financial news impacting {company_name}")
    else:  # Fallback if company name is not available
        search_queries.append(f"latest stock news {stock_symbol}")
    search_queries.append("latest stock market news today")
    search_queries.append("major economic news affecting stock markets")
    search_queries = search_queries[:5]  # Limit number of queries to Brave API

    try:
        fetched_articles = await _fetch_brave_search_news_async(queries=search_queries, count_per_query=3)

        # Sort articles by publication time if available (best effort)
        def get_sortable_date(article_dict):
            time_val = article_dict.get("publication_time")
            if isinstance(time_val, str):
                try:
                    # Handle timezone 'Z' if present, otherwise assume naive or already offset
                    return datetime.fromisoformat(time_val.replace("Z", "+00:00"))
                except ValueError:
                    return datetime.min  # Fallback for unparseable strings
            elif isinstance(time_val, datetime):
                return time_val
            return datetime.min  # Default for None or other types

        fetched_articles.sort(key=get_sortable_date, reverse=True)

        state["raw_news_articles"] = fetched_articles
        print(
            f"LOG (workflow): Fetched {len(fetched_articles)} news items for {stock_symbol} and market via Brave API.")
        if not fetched_articles:
            state["news_analysis_summary"] = "No relevant news articles were found via Brave Search."

    except Exception as e:
        print(f"ERROR (workflow) in fetch_news_node (Brave API call): {e}")
        state["error_message"] = (
                    state.get("error_message", "") + f"News fetching via Brave API failed: {str(e)}. ").strip()
        state["raw_news_articles"] = []  # Ensure it's an empty list on failure
        state["news_analysis_summary"] = "Failed to fetch news due to an API error."
    return state


async def summarize_news_node(state: StockAnalysisState) -> StockAnalysisState:
    """Summarizes the fetched news articles."""
    stock_symbol = state.get("stock_symbol")
    company_name = state.get("company_name")
    raw_articles = state.get("raw_news_articles")

    # If there was a critical error message already set, or no symbol, or news fetching itself set a summary indicating failure
    if state.get("error_message") or not stock_symbol or \
            (state.get("news_analysis_summary") and "failed" in state.get("news_analysis_summary", "").lower()) or \
            (state.get("news_analysis_summary") and "disabled" in state.get("news_analysis_summary", "").lower()):

        current_error = state.get("error_message", "")
        if "Skipping news summarization" not in current_error:  # Avoid redundant messages
            state["error_message"] = (
                        current_error + "Skipping news summarization due to prior errors or news fetch failure. ").strip()

        if not state.get("news_analysis_summary"):  # If not already set by fetch_news_node
            state["news_analysis_summary"] = "News summarization skipped."
        return state

    if not raw_articles:  # No articles fetched, but no major error from previous steps
        print(f"LOG (workflow): No raw news articles to summarize for {stock_symbol}.")
        # Preserve message from fetch_news if it indicated no articles were found
        state["news_analysis_summary"] = state.get("news_analysis_summary", "No news articles were found to summarize.")
        return state

    print(f"LOG (workflow): Summarizing news for {stock_symbol} (Company: {company_name}).")
    try:
        summary = await summarize_news_content(stock_symbol, company_name, raw_articles)
        state["news_analysis_summary"] = summary
        print(f"LOG (workflow): News summary generated for {stock_symbol}.")
    except Exception as e:
        print(f"ERROR (workflow) in summarize_news_node: {e}")
        state["error_message"] = (state.get("error_message", "") + f"News summarization failed: {str(e)}. ").strip()
        state["news_analysis_summary"] = "Failed to summarize news due to an unexpected error."
    return state


async def generate_direct_answer_node(state: StockAnalysisState) -> StockAnalysisState:
    """Generates a direct answer to the user's question, if provided."""
    print(f"LOG (workflow): Generate direct answer node for {state['stock_symbol']}")
    original_question = state.get("original_user_question")
    if not original_question or original_question.strip() == "":
        state["direct_answer_to_user_question"] = None
        return state  # No question, no direct answer

    ta_summary = state.get("technical_analysis_summary")
    fa_summary = state.get("fundamental_analysis_summary")
    news_summary = state.get("news_analysis_summary")

    # Check if all key analysis components are missing or failed
    missing_ta = not ta_summary or any(err_msg in ta_summary.lower() for err_msg in
                                       ["cannot be generated", "skipped", "failed", "no technical indicators"])
    missing_fa = not fa_summary or any(
        err_msg in fa_summary.lower() for err_msg in ["skipped", "failed", "missing essential data"])
    missing_news = not news_summary or any(err_msg in news_summary.lower() for err_msg in
                                           ["failed", "skipped", "disabled", "no relevant news articles were found",
                                            "no news articles were found"])

    # If there's a general error message AND all summaries are effectively missing/failed, then skip.
    if state.get("error_message") and (missing_ta and missing_fa and missing_news):
        state[
            "direct_answer_to_user_question"] = "Could not answer your specific question due to missing analysis data or prior critical errors."
        current_error = state.get("error_message", "")
        if "Skipped direct question answering" not in current_error:
            state["error_message"] = (
                        current_error + "Skipped direct question answering due to missing all summaries and prior error. ").strip()
        return state

    try:
        state["direct_answer_to_user_question"] = await generate_direct_answer_to_question_content(
            state["stock_symbol"],
            original_question,
            state.get("analyzed_user_query_focus"),
            ta_summary,
            fa_summary,
            news_summary,
            state.get("executed_technical_indicators"),
            state.get("stock_info_json"),
            state.get("question_data_type_needed")
        )
        print(f"LOG (workflow): Direct answer generated for: '{original_question}'")
    except Exception as e:
        print(f"ERROR (workflow) in generate_direct_answer_node: {e}")
        state["error_message"] = (
                    state.get("error_message", "") + f"Failed to generate direct answer: {str(e)}. ").strip()
        state[
            "direct_answer_to_user_question"] = "Sorry, I encountered an issue trying to answer your specific question."
    return state


async def recommendation_node(state: StockAnalysisState) -> StockAnalysisState:
    """Generates an overall stock recommendation."""
    print(f"LOG (workflow): Recommendation node for {state['stock_symbol']}")
    ta_summary = state.get("technical_analysis_summary", "Technical analysis information not available.")
    fa_summary = state.get("fundamental_analysis_summary", "Fundamental analysis information not available.")
    news_summary = state.get("news_analysis_summary", "News summary not available.")

    # Check if summaries are critically flawed
    critical_ta = any(err_msg in ta_summary.lower() for err_msg in ["cannot be generated", "skipped", "failed"])
    critical_fa = any(err_msg in fa_summary.lower() for err_msg in ["skipped", "failed", "missing essential data"])
    critical_news = any(err_msg in news_summary.lower() for err_msg in ["failed", "skipped", "disabled"])

    if state.get("error_message") and (critical_ta and critical_fa and critical_news):
        state[
            "final_recommendation"] = "Cannot generate overall recommendation due to prior errors and missing all analysis components (TA, FA, News)."
        return state

    try:
        state["final_recommendation"] = await generate_recommendation_content(state["stock_symbol"], ta_summary,
                                                                              fa_summary, news_summary)
    except Exception as e:
        print(f"ERROR (workflow) in recommendation_node: {e}")
        state["error_message"] = (
                    state.get("error_message", "") + f"Overall recommendation generation failed: {str(e)}. ").strip()
        state["final_recommendation"] = "Failed to generate overall recommendation due to an unexpected error."
    return state


async def position_advice_node(state: StockAnalysisState) -> StockAnalysisState:
    """Generates advice for an existing user stock position."""
    print(f"LOG (workflow): Position advice node for {state['stock_symbol']}")
    final_rec = state.get("final_recommendation", "")
    key_levels = state.get("key_technical_levels", {})

    if not state.get("user_position"):
        state["position_specific_advice"] = "No user position information was provided for advice."
        return state  # No position, no advice

    if not final_rec or any(err_msg in final_rec.lower() for err_msg in ["cannot generate", "failed"]):
        state[
            "position_specific_advice"] = "Position advice cannot be generated because the overall recommendation is unavailable or failed."
        return state

    try:
        state["position_specific_advice"] = await generate_position_advice_content(state["stock_symbol"], final_rec,
                                                                                   state["user_position"], key_levels)
    except Exception as e:
        print(f"ERROR (workflow) in position_advice_node: {e}")
        state["error_message"] = (
                    state.get("error_message", "") + f"Position advice generation failed: {str(e)}. ").strip()
        state["position_specific_advice"] = "Failed to generate position-specific advice due to an unexpected error."
    return state


# --- Compile LangGraph App ---
def get_stock_analyzer_app():
    """Compiles and returns the LangGraph application."""
    get_llms()  # Ensure LLMs are initialized
    workflow = StateGraph(StockAnalysisState)

    # Add all nodes to the graph
    workflow.add_node("start", start_node)
    workflow.add_node("resolve_symbol", resolve_stock_symbol_node)
    workflow.add_node("analyze_query_focus", analyze_user_query_focus_node)
    workflow.add_node("fetch_data", fetch_data_node)
    workflow.add_node("select_technical_tools", select_technical_tools_node)
    workflow.add_node("execute_technical_tools", execute_technical_tools_node)
    # workflow.add_node("generate_chart", generate_chart_node) # Charting disabled
    workflow.add_node("summarize_technical_analysis", summarize_technical_analysis_node)
    workflow.add_node("fundamental_analysis", fundamental_analysis_node)
    workflow.add_node("fetch_news", fetch_news_node)  # Uses Brave API
    workflow.add_node("summarize_news", summarize_news_node)
    workflow.add_node("generate_direct_answer", generate_direct_answer_node)
    workflow.add_node("generate_recommendation", recommendation_node)
    workflow.add_node("generate_position_advice", position_advice_node)

    # Define the workflow edges
    workflow.set_entry_point("start")
    workflow.add_edge("start", "resolve_symbol")
    workflow.add_edge("resolve_symbol", "analyze_query_focus")
    workflow.add_edge("analyze_query_focus", "fetch_data")
    workflow.add_edge("fetch_data", "select_technical_tools")
    workflow.add_edge("select_technical_tools", "execute_technical_tools")
    workflow.add_edge("execute_technical_tools", "summarize_technical_analysis")
    workflow.add_edge("summarize_technical_analysis", "fundamental_analysis")
    workflow.add_edge("fundamental_analysis", "fetch_news")  # News fetching after FA
    workflow.add_edge("fetch_news", "summarize_news")
    workflow.add_edge("summarize_news", "generate_direct_answer")  # Direct answer uses news
    workflow.add_edge("generate_direct_answer", "generate_recommendation")
    workflow.add_edge("generate_recommendation", "generate_position_advice")
    workflow.add_edge("generate_position_advice", END)  # End of the graph

    return workflow.compile()


async def main_test():
    """Main function for standalone testing of the stock analyzer."""
    if initialize_analyzer():  # This also checks for API keys now
        app = get_stock_analyzer_app()
        print("Stock Analyzer App compiled for standalone async testing (with Brave API if key is set).")

        inputs_list = [
            {"user_stock_query": "Infosys",
             "user_question": "What do you think about INFY? How is recent news affecting it?",
             "user_position": {"shares": 10, "avg_price": 1400}},
            {"user_stock_query": "AAPL", "user_question": "Latest market news impact on Apple?"},
            {"user_stock_query": "MSFT", "user_question": "General analysis of Microsoft including news."},
            {"user_stock_query": "A non existent ticker XYZ123", "user_question": "Any news?"},
            {"user_stock_query": "GOOG", "user_question": None}  # Test with no specific question
        ]

        for i, inputs in enumerate(inputs_list):
            print(f"\n--- RUNNING TEST CASE {i + 1} ---")
            print(f"Input: {inputs}")
            try:
                result_state: StockAnalysisState = await app.ainvoke(inputs)  # type: ignore

                print(f"\n--- FINAL STATE (Test Case {i + 1}) ---")
                # Pretty print the final state
                print(json.dumps(result_state, indent=2,
                                 default=str))  # default=str handles non-serializable like datetime

                # Simulate the API response data structure
                api_response_data = {
                    "user_stock_query": result_state.get("user_stock_query"),
                    "original_user_question": result_state.get("original_user_question"),
                    "analyzed_user_query_focus": result_state.get("analyzed_user_query_focus"),
                    "stock_symbol": result_state.get("stock_symbol"),
                    "company_name": result_state.get("company_name"),
                    "news_analysis_summary": result_state.get("news_analysis_summary"),
                    "technical_analysis_summary": result_state.get("technical_analysis_summary"),
                    "fundamental_analysis_summary": result_state.get("fundamental_analysis_summary"),
                    "direct_answer_to_user_question": result_state.get("direct_answer_to_user_question"),
                    "final_recommendation": result_state.get("final_recommendation"),
                    "position_specific_advice": result_state.get("position_specific_advice"),
                    "error_message": result_state.get("error_message") if result_state.get("error_message",
                                                                                           "").strip() else None,
                }
                print(f"\n--- SIMULATED API RESPONSE DATA (Test Case {i + 1}) ---")
                print(json.dumps(api_response_data, indent=2, default=str))

            except Exception as e:
                print(f"Error during standalone async test invocation (Test Case {i + 1}): {e}")
                import traceback  # Import here to avoid top-level import if not needed
                traceback.print_exc()  # Print full traceback for debugging
    else:
        print(
            "Failed to initialize analyzer. Check API keys (OpenAI, Brave Search, Alpha Vantage) and other configurations.")


if __name__ == "__main__":
    asyncio.run(main_test())
