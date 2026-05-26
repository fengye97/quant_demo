#!/usr/bin/env python3
"""
A-share stock data fetching and processing tool.

Modes:
  python3 get_stock_info.py                  # Default: real-time demo report
  python3 get_stock_info.py --mode supplement # Supplement stock_data.csv with latest monthly data

Data sources:
  - Sina Finance API (quotes.sina.cn) for daily K-line data
  - Tencent Finance API (qt.gtimg.cn) for real-time quotes (market cap, PE, PB, turnover, etc.)

Technical indicators computed from daily data:
  - bias_5/10/20, 振幅_5/10/20, 涨跌幅std_5/10/20, 成交额std_5/10/20
  - KDJ, MACD (DIF/DEA/MACD bar)
  - 涨跌幅, 涨跌幅_10/20
  - 市盈率倒数, 市净率倒数
"""
import argparse
import csv
import json
import math
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
USER_AGENT = "Mozilla/5.0"
MAX_RETRIES = 3
RETRY_DELAY = 2.0
MAX_WORKERS = 20
RATE_LIMIT = 0.02  # seconds between requests per worker

SYMBOLS = {
    "688256": {"name": "寒武纪", "market": "a", "tencent": "sh688256", "sina": "sh688256"},
    "09988": {"name": "阿里巴巴-W", "market": "hk", "tencent": "hk09988", "sina": "hk09988"},
}


# ──────────────────────────────────────────────
# Exceptions
# ──────────────────────────────────────────────
class StockDataError(RuntimeError):
    pass


class StockFetchError(StockDataError):
    """Non-fatal per-stock fetch error."""
    pass


# ──────────────────────────────────────────────
# HTTP helpers
# ──────────────────────────────────────────────
def http_get(
    url: str,
    *,
    referer: Optional[str] = None,
    timeout: int = 20,
    encoding: str = "gbk",
) -> str:
    """HTTP GET with User-Agent and optional Referer."""
    headers = {"User-Agent": USER_AGENT}
    if referer:
        headers["Referer"] = referer
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read().decode(encoding, errors="ignore")


def http_get_with_retry(
    url: str,
    *,
    referer: Optional[str] = None,
    timeout: int = 20,
    encoding: str = "gbk",
    max_retries: int = MAX_RETRIES,
) -> str:
    """HTTP GET with retry logic."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return http_get(url, referer=referer, timeout=timeout, encoding=encoding)
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    raise StockDataError(f"HTTP GET failed after {max_retries} attempts: {last_error}")


# ──────────────────────────────────────────────
# Stock code helpers
# ──────────────────────────────────────────────
def is_a_share(code: str) -> bool:
    """Check if a stock code is an A-share (Shanghai/Shenzhen/Beijing)."""
    return code.startswith(("sh", "sz", "bj"))


def code_to_tencent_symbol(code: str) -> str:
    """Convert stock code to Tencent API symbol."""
    return code  # Tencent uses the same sh/sz/bj prefix format


def code_to_sina_symbol(code: str) -> str:
    """Convert stock code to Sina API symbol."""
    return code  # Sina uses the same sh/sz/bj prefix format


def code_to_eastmoney_secid(code: str) -> str:
    """Convert stock code to Eastmoney secid format (market.code)."""
    prefix = code[:2]
    num = code[2:]
    market_map = {"sh": "1", "sz": "0", "bj": "0"}
    market = market_map.get(prefix, "1")
    return f"{market}.{num}"


# ──────────────────────────────────────────────
# Tencent real-time quote parsing (existing + batch)
# ──────────────────────────────────────────────
def parse_tencent_quote_line(line: str) -> Dict[str, object]:
    """Parse a single Tencent real-time quote line."""
    payload = line.split('="', 1)[1].rsplit('";', 1)[0]
    parts = payload.split("~")
    symbol = parts[2]
    is_hk = line.startswith("v_hk")

    if is_hk:
        return {
            "name": parts[1],
            "symbol": symbol,
            "market": "HK",
            "previous_close": float(parts[4]),
            "open": float(parts[5]),
            "latest": float(parts[3]),
            "high": float(parts[33]),
            "low": float(parts[34]),
            "volume": float(parts[36]),
            "amount": float(parts[37]),
            "trade_time": parts[30],
            "change": float(parts[31]),
            "pct_change": float(parts[32]),
            "pe_ttm": float(parts[57]),
            "pb_ratio": float(parts[58]),
            "turnover_rate": float(parts[59]),
            "currency": "HKD",
            "source": "Tencent",
        }

    return {
        "name": parts[1],
        "symbol": symbol,
        "market": "A",
        "previous_close": float(parts[4]),
        "open": float(parts[5]),
        "latest": float(parts[3]),
        "high": float(parts[33]),
        "low": float(parts[34]),
        "volume": int(float(parts[6])),
        "amount": float(parts[37]),
        "trade_time": parts[30],
        "change": float(parts[31]),
        "pct_change": float(parts[32]),
        "pe_ttm": float(parts[39]),
        "pe_dynamic": float(parts[52]) if parts[52] else None,
        "pe_static": float(parts[53]) if parts[53] else None,
        "pb_ratio": float(parts[46]),
        "turnover_rate": float(parts[38]),
        "currency": parts[45],
        "source": "Tencent",
    }


def fetch_realtime_quotes(symbol_codes: List[str]) -> List[Dict[str, object]]:
    """Fetch real-time quotes for a list of stock codes."""
    quote_ids = []
    for code in symbol_codes:
        config = SYMBOLS[code]
        quote_ids.append(config["tencent"])
    url = "https://qt.gtimg.cn/q=" + ",".join(quote_ids)
    text = http_get(url)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return [parse_tencent_quote_line(line) for line in lines]


def fetch_realtime_quotes_batch(
    stock_codes: List[str], batch_size: int = 100
) -> Dict[str, Dict[str, object]]:
    """
    Fetch real-time quotes for A-share stocks in batches.
    Returns dict keyed by stock code.
    """
    results = {}
    for i in range(0, len(stock_codes), batch_size):
        batch = stock_codes[i : i + batch_size]
        symbols = [s for s in batch]
        url = "https://qt.gtimg.cn/q=" + ",".join(symbols)
        try:
            text = http_get_with_retry(url)
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            for line in lines:
                try:
                    quote = parse_tencent_quote_line(line)
                    results[quote["symbol"]] = quote
                except Exception:
                    continue
        except Exception as e:
            print(f"  Warning: Tencent batch {i}-{i+len(batch)} failed: {e}", file=sys.stderr)
        time.sleep(0.5)  # Rate limit between batches
    return results


# ──────────────────────────────────────────────
# Sina real-time quote parsing (existing)
# ──────────────────────────────────────────────
def parse_sina_quote_line(line: str) -> Dict[str, object]:
    """Parse a single Sina real-time quote line."""
    payload = line.split('="', 1)[1].rsplit('";', 1)[0]
    parts = payload.split(",")
    is_hk = line.startswith("var hq_str_hk")

    if is_hk:
        return {
            "name": parts[1],
            "previous_close": float(parts[3]),
            "open": float(parts[2]),
            "high": float(parts[4]),
            "low": float(parts[5]),
            "latest": float(parts[6]),
            "change": float(parts[7]),
            "pct_change": float(parts[8]),
            "amount": float(parts[11]),
            "volume": float(parts[12]),
            "trade_date": parts[17],
            "trade_time": parts[18],
            "source": "Sina",
        }

    return {
        "name": parts[0],
        "open": float(parts[1]),
        "previous_close": float(parts[2]),
        "latest": float(parts[3]),
        "high": float(parts[4]),
        "low": float(parts[5]),
        "volume": int(float(parts[8])),
        "amount": float(parts[9]),
        "trade_date": parts[30],
        "trade_time": parts[31],
        "source": "Sina",
    }


def fetch_sina_realtime(symbol_codes: List[str]) -> List[Dict[str, object]]:
    """Fetch real-time quotes from Sina for a list of stock codes."""
    quote_ids = []
    for code in symbol_codes:
        config = SYMBOLS[code]
        quote_ids.append(config["sina"])
    url = "https://hq.sinajs.cn/list=" + ",".join(quote_ids)
    text = http_get(url, referer="https://finance.sina.com.cn")
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip().startswith("var hq_str_")
    ]
    return [parse_sina_quote_line(line) for line in lines]


# ──────────────────────────────────────────────
# Sina daily K-line (existing + improved batch)
# ──────────────────────────────────────────────
def fetch_a_share_daily_from_sina(
    code: str, datalen: int = 50
) -> List[Dict[str, object]]:
    """
    Fetch daily K-line data for a single A-share stock from Sina.
    Returns list of dicts with keys: date, open, high, low, close, volume.
    """
    config = SYMBOLS.get(code)
    if config and config["market"] != "a":
        raise StockDataError(f"{code} is not an A-share. Only A-shares are supported.")
    if not config and not is_a_share(code):
        raise StockDataError(f"{code} is not an A-share. Only A-shares are supported.")

    symbol = code_to_sina_symbol(code) if not config else config["sina"]

    query = urllib.parse.urlencode(
        {
            "symbol": symbol,
            "scale": "240",
            "ma": "no",
            "datalen": str(datalen),
        }
    )
    url = f"https://quotes.sina.cn/cn/api/jsonp_v2.php/var%20data=/CN_MarketDataService.getKLineData?{query}"
    text = http_get(url, referer="https://finance.sina.com.cn")
    match = re.search(r"var data=\((.*)\);", text, re.S)
    if not match:
        raise StockDataError(f"Cannot parse Sina daily response for {code}: {text[:200]}")
    rows = json.loads(match.group(1))
    normalized = []
    for row in rows:
        normalized.append(
            {
                "date": row["day"],
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(float(row["volume"])),
                "source": "Sina",
            }
        )
    return normalized


def fetch_single_stock_daily(
    code: str, datalen: int = 50
) -> Tuple[str, Optional[List[Dict[str, object]]], Optional[str]]:
    """
    Fetch daily K-line with retry for a single stock.
    Returns (code, data_list, error_string).
    """
    for attempt in range(MAX_RETRIES):
        try:
            data = fetch_a_share_daily_from_sina(code, datalen=datalen)
            return (code, data, None)
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                return (code, None, str(e))
    return (code, None, "Unknown error")


def fetch_daily_batch(
    stock_codes: List[str],
    datalen: int = 50,
    max_workers: int = MAX_WORKERS,
    verbose: bool = True,
) -> Dict[str, List[Dict[str, object]]]:
    """
    Fetch daily K-line data for multiple A-share stocks concurrently.

    Args:
        stock_codes: List of stock codes (e.g., ['sh600000', 'sz000001'])
        datalen: Number of daily bars to fetch per stock
        max_workers: Number of concurrent workers
        verbose: Print progress

    Returns:
        Dict mapping stock_code -> list of daily bar dicts
    """
    results: Dict[str, List[Dict[str, object]]] = {}
    errors: List[str] = []
    total = len(stock_codes)
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_single_stock_daily, code, datalen): code
            for code in stock_codes
        }
        for future in as_completed(futures):
            code, data, error = future.result()
            completed += 1
            if data is not None:
                results[code] = data
            else:
                errors.append(f"{code}: {error}")
            if verbose and completed % 200 == 0:
                print(
                    f"  Progress: {completed}/{total} stocks "
                    f"({len(results)} ok, {len(errors)} errors)",
                    file=sys.stderr,
                )
            # Rate limiting between submissions
            time.sleep(RATE_LIMIT)

    if verbose and errors:
        print(f"\n  Warning: {len(errors)} stocks failed to fetch:", file=sys.stderr)
        for err in errors[:10]:
            print(f"    {err}", file=sys.stderr)
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more", file=sys.stderr)

    return results


# ──────────────────────────────────────────────
# Cross-check and formatting (existing)
# ──────────────────────────────────────────────
def cross_check_realtime(
    code: str, tencent: Dict[str, object], sina: Dict[str, object]
) -> None:
    """Cross-validate real-time quotes from two sources."""
    latest_diff = abs(float(tencent["latest"]) - float(sina["latest"]))
    high_diff = abs(float(tencent["high"]) - float(sina["high"]))
    low_diff = abs(float(tencent["low"]) - float(sina["low"]))
    if code == "688256":
        volume_diff = abs(int(tencent["volume"]) - int(sina["volume"]))
    else:
        volume_diff = abs(float(tencent["volume"]) - float(sina["volume"]))

    if latest_diff >= 0.01 or high_diff >= 0.01 or low_diff >= 0.01 or volume_diff >= 1:
        raise StockDataError(f"{code} multi-source cross-check failed")


def format_realtime_result(
    code: str, tencent: Dict[str, object], sina: Dict[str, object]
) -> Dict[str, object]:
    """Format real-time quote into display dict."""
    config = SYMBOLS[code]
    result = {
        "股票代码": code,
        "股票名称": config["name"],
        "市场": "A股" if config["market"] == "a" else "港股",
        "最新价": tencent["latest"],
        "今开": tencent["open"],
        "昨收": tencent["previous_close"],
        "最高价": tencent["high"],
        "最低价": tencent["low"],
        "成交量": tencent["volume"],
        "成交额": sina["amount"],
        "涨跌额": tencent["change"],
        "涨跌幅(%)": tencent["pct_change"],
        "换手率(%)": tencent["turnover_rate"],
    }
    if tencent.get("pe_dynamic") is not None:
        result["动态市盈率"] = tencent["pe_dynamic"]
        result["静态市盈率"] = tencent["pe_static"]
        result["TTM市盈率"] = tencent["pe_ttm"]
    else:
        result["TTM市盈率"] = tencent["pe_ttm"]
    result["市净率"] = tencent["pb_ratio"]
    result["最新交易时间"] = tencent["trade_time"]
    return result


def format_daily_result(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Format daily K-line data for display."""
    result = []
    for row in rows:
        result.append(
            {
                "交易日期": row["date"],
                "开盘价": row["open"],
                "最高价": row["high"],
                "最低价": row["low"],
                "收盘价": row["close"],
                "成交量": row["volume"],
            }
        )
    return result


# ──────────────────────────────────────────────
# Technical indicator computation
# ──────────────────────────────────────────────

def compute_sma(values: List[float], period: int) -> List[float]:
    """Simple Moving Average. Returns list same length as input (NaN for insufficient data)."""
    result = []
    for i in range(len(values)):
        if i < period - 1:
            result.append(float("nan"))
        else:
            result.append(sum(values[i - period + 1 : i + 1]) / period)
    return result


def compute_ema(values: List[float], period: int) -> List[float]:
    """Exponential Moving Average. Seeds with SMA of first <period> non-NaN values."""
    if len(values) < period:
        return [float("nan")] * len(values)

    result = [float("nan")] * len(values)

    # Find first <period> consecutive valid values for SMA seed
    valid_idxs = []
    for i, v in enumerate(values):
        if not math.isnan(v):
            valid_idxs.append(i)
            if len(valid_idxs) == period:
                break

    if len(valid_idxs) < period:
        return result  # Not enough valid data

    # Seed with SMA of first period valid values
    seed_idx = valid_idxs[-1]
    seed_vals = [values[j] for j in valid_idxs]
    prev_ema = sum(seed_vals) / period
    result[seed_idx] = prev_ema

    multiplier = 2.0 / (period + 1)
    for i in range(seed_idx + 1, len(values)):
        if math.isnan(values[i]):
            result[i] = prev_ema  # Carry forward
        else:
            prev_ema = (values[i] - prev_ema) * multiplier + prev_ema
            result[i] = prev_ema
    return result


def compute_max_high(highs: List[float], period: int) -> List[float]:
    """Rolling max of highs over period."""
    result = []
    for i in range(len(highs)):
        if i < period - 1:
            result.append(float("nan"))
        else:
            result.append(max(highs[i - period + 1 : i + 1]))
    return result


def compute_min_low(lows: List[float], period: int) -> List[float]:
    """Rolling min of lows over period."""
    result = []
    for i in range(len(lows)):
        if i < period - 1:
            result.append(float("nan"))
        else:
            result.append(min(lows[i - period + 1 : i + 1]))
    return result


def compute_returns(closes: List[float]) -> List[float]:
    """Compute period-over-period returns."""
    result = [float("nan")]
    for i in range(1, len(closes)):
        if closes[i - 1] == 0:
            result.append(float("nan"))
        else:
            result.append((closes[i] - closes[i - 1]) / closes[i - 1])
    return result


def compute_rolling_std(values: List[float], period: int) -> List[float]:
    """Rolling standard deviation over period."""
    result = []
    for i in range(len(values)):
        if i < period - 1:
            result.append(float("nan"))
        else:
            window = [v for v in values[i - period + 1 : i + 1] if not math.isnan(v)]
            if len(window) < 2:
                result.append(float("nan"))
            else:
                mean = sum(window) / len(window)
                variance = sum((v - mean) ** 2 for v in window) / len(window)
                result.append(math.sqrt(variance))
    return result


def compute_all_indicators(
    dates: List[str],
    opens: List[float],
    highs: List[float],
    lows: List[float],
    closes: List[float],
    volumes: List[float],
) -> Dict[str, List[float]]:
    """
    Compute all technical indicators from daily OHLCV data.

    Returns dict with keys matching CSV column names, values are lists
    aligned with input (index 0 = oldest, index -1 = latest).
    """
    n = len(closes)
    indicators: Dict[str, List[float]] = {}

    # Moving averages
    sma5 = compute_sma(closes, 5)
    sma10 = compute_sma(closes, 10)
    sma20 = compute_sma(closes, 20)

    # bias_n = (close - MA_n) / MA_n
    indicators["bias_5"] = [
        (closes[i] - sma5[i]) / sma5[i] if not math.isnan(sma5[i]) else float("nan")
        for i in range(n)
    ]
    indicators["bias_10"] = [
        (closes[i] - sma10[i]) / sma10[i] if not math.isnan(sma10[i]) else float("nan")
        for i in range(n)
    ]
    indicators["bias_20"] = [
        (closes[i] - sma20[i]) / sma20[i] if not math.isnan(sma20[i]) else float("nan")
        for i in range(n)
    ]

    # 振幅_n = (max_high - min_low) / close (over n periods)
    max_high5 = compute_max_high(highs, 5)
    min_low5 = compute_min_low(lows, 5)
    max_high10 = compute_max_high(highs, 10)
    min_low10 = compute_min_low(lows, 10)
    max_high20 = compute_max_high(highs, 20)
    min_low20 = compute_min_low(lows, 20)

    indicators["振幅_5"] = [
        (max_high5[i] - min_low5[i]) / closes[i]
        if not (math.isnan(max_high5[i]) or math.isnan(min_low5[i]))
        else float("nan")
        for i in range(n)
    ]
    indicators["振幅_10"] = [
        (max_high10[i] - min_low10[i]) / closes[i]
        if not (math.isnan(max_high10[i]) or math.isnan(min_low10[i]))
        else float("nan")
        for i in range(n)
    ]
    indicators["振幅_20"] = [
        (max_high20[i] - min_low20[i]) / closes[i]
        if not (math.isnan(max_high20[i]) or math.isnan(min_low20[i]))
        else float("nan")
        for i in range(n)
    ]

    # Daily returns
    daily_returns = compute_returns(closes)

    # 涨跌幅std_n = rolling std of daily returns over n periods
    indicators["涨跌幅std_5"] = compute_rolling_std(daily_returns, 5)
    indicators["涨跌幅std_10"] = compute_rolling_std(daily_returns, 10)
    indicators["涨跌幅std_20"] = compute_rolling_std(daily_returns, 20)

    # 成交额std_n = rolling std of volume (proxy for amount) over n periods
    # Convert volumes to float for std computation
    vol_floats = [float(v) for v in volumes]
    indicators["成交额std_5"] = compute_rolling_std(vol_floats, 5)
    indicators["成交额std_10"] = compute_rolling_std(vol_floats, 10)
    indicators["成交额std_20"] = compute_rolling_std(vol_floats, 20)

    # KDJ (9, 3, 3)
    kdj = compute_kdj(highs, lows, closes, n_period=9)
    indicators["K"] = kdj["K"]
    indicators["D"] = kdj["D"]
    indicators["J"] = kdj["J"]

    # MACD (12, 26, 9)
    macd = compute_macd(closes, fast=12, slow=26, signal=9)
    indicators["DIF"] = macd["DIF"]
    indicators["DEA"] = macd["DEA"]
    indicators["MACD"] = macd["MACD"]

    # 涨跌幅 (daily return)
    indicators["涨跌幅_daily"] = daily_returns

    # 涨跌幅_n: n-day return
    indicators["涨跌幅_10"] = compute_n_period_return(closes, 10)
    indicators["涨跌幅_20"] = compute_n_period_return(closes, 20)

    return indicators


def compute_kdj(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    n_period: int = 9,
) -> Dict[str, List[float]]:
    """
    Compute KDJ indicator.

    RSV = (close - low_n) / (high_n - low_n) * 100
    K = 2/3 * prev_K + 1/3 * RSV
    D = 2/3 * prev_D + 1/3 * K
    J = 3 * K - 2 * D
    """
    n = len(closes)
    K_vals = [float("nan")] * n
    D_vals = [float("nan")] * n
    J_vals = [float("nan")] * n

    K_prev = 50.0
    D_prev = 50.0

    for i in range(n_period - 1, n):
        high_n = max(highs[i - n_period + 1 : i + 1])
        low_n = min(lows[i - n_period + 1 : i + 1])
        if high_n == low_n:
            rsv = 50.0
        else:
            rsv = (closes[i] - low_n) / (high_n - low_n) * 100

        if math.isnan(K_prev):
            K_prev = 50.0
            D_prev = 50.0

        K_val = 2.0 / 3.0 * K_prev + 1.0 / 3.0 * rsv
        D_val = 2.0 / 3.0 * D_prev + 1.0 / 3.0 * K_val
        J_val = 3.0 * K_val - 2.0 * D_val

        K_vals[i] = K_val
        D_vals[i] = D_val
        J_vals[i] = J_val
        K_prev = K_val
        D_prev = D_val

    return {"K": K_vals, "D": D_vals, "J": J_vals}


def compute_macd(
    closes: List[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Dict[str, List[float]]:
    """
    Compute MACD indicator.

    EMA_fast = EMA(close, fast)
    EMA_slow = EMA(close, slow)
    DIF = EMA_fast - EMA_slow
    DEA = EMA(DIF, signal)
    MACD_bar = 2 * (DIF - DEA)
    """
    n = len(closes)
    ema_fast = compute_ema(closes, fast)
    ema_slow = compute_ema(closes, slow)

    DIF_vals = [float("nan")] * n
    for i in range(n):
        if not math.isnan(ema_fast[i]) and not math.isnan(ema_slow[i]):
            DIF_vals[i] = ema_fast[i] - ema_slow[i]

    DEA_vals = compute_ema(DIF_vals, signal)
    MACD_vals = [float("nan")] * n

    for i in range(n):
        if not math.isnan(DIF_vals[i]) and not math.isnan(DEA_vals[i]):
            MACD_vals[i] = 2.0 * (DIF_vals[i] - DEA_vals[i])

    return {"DIF": DIF_vals, "DEA": DEA_vals, "MACD": MACD_vals}


def compute_n_period_return(closes: List[float], period: int) -> List[float]:
    """Compute n-period return: (close_t - close_{t-n}) / close_{t-n}."""
    result = []
    for i in range(len(closes)):
        if i < period:
            result.append(float("nan"))
        elif closes[i - period] == 0:
            result.append(float("nan"))
        else:
            result.append((closes[i] - closes[i - period]) / closes[i - period])
    return result


# ──────────────────────────────────────────────
# Monthly aggregation
# ──────────────────────────────────────────────

def aggregate_daily_to_monthly(
    daily_data: List[Dict[str, object]],
    year: int,
    month: int,
) -> Optional[Dict[str, object]]:
    """
    Aggregate daily bars into a single monthly bar.

    Args:
        daily_data: List of daily bar dicts with date, open, high, low, close, volume
        year, month: Target year/month

    Returns:
        Monthly bar dict with keys: date, open, high, low, close, volume,
        amount_est (estimated amount), trading_days, daily_returns
    """
    month_rows = []
    for row in daily_data:
        d = row["date"]
        # Date format: "2026-05-15"
        try:
            parts = d.split("-")
            if len(parts) == 3:
                y, m, _ = int(parts[0]), int(parts[1]), int(parts[2])
                if y == year and m == month:
                    month_rows.append(row)
        except (ValueError, IndexError):
            continue

    if not month_rows:
        return None

    month_rows.sort(key=lambda r: r["date"])

    monthly_open = month_rows[0]["open"]
    monthly_close = month_rows[-1]["close"]
    monthly_high = max(r["high"] for r in month_rows)
    monthly_low = min(r["low"] for r in month_rows)
    monthly_volume = sum(r["volume"] for r in month_rows)

    # Estimate amount: volume * typical_price for each day
    estimated_amount = 0.0
    for r in month_rows:
        typical_price = (r["high"] + r["low"] + r["close"]) / 3.0
        estimated_amount += r["volume"] * typical_price

    # VWAP estimate
    if monthly_volume > 0:
        vwap = estimated_amount / monthly_volume
    else:
        vwap = monthly_close

    # Daily returns within the month (for 下周期每天涨跌幅 of PREVIOUS month)
    daily_returns = []
    for i in range(1, len(month_rows)):
        prev_close = month_rows[i - 1]["close"]
        curr_close = month_rows[i]["close"]
        if prev_close != 0:
            daily_returns.append((curr_close - prev_close) / prev_close)

    return {
        "date": month_rows[-1]["date"],  # Last trading day of the month
        "open": monthly_open,
        "high": monthly_high,
        "low": monthly_low,
        "close": monthly_close,
        "volume": monthly_volume,
        "amount_est": estimated_amount,
        "vwap_est": vwap,
        "trading_days": len(month_rows),
        "daily_returns": daily_returns,
    }


# ──────────────────────────────────────────────
# CSV supplement mode
# ──────────────────────────────────────────────

CSV_HEADERS = [
    "交易日期", "股票代码", "股票名称", "是否交易",
    "开盘价", "最高价", "最低价", "收盘价", "VWAP",
    "成交额", "流通市值", "总市值", "上市至今交易天数",
    "财报季度", "财报年份",
    "归母净利润", "归母净利润_ttm", "归母净利润_ttm同比",
    "归母净利润_单季", "归母净利润_单季同比", "归母净利润_单季环比",
    "经营活动产生的现金流量净额", "经营活动产生的现金流量净额_ttm",
    "经营活动产生的现金流量净额_ttm同比",
    "经营活动产生的现金流量净额_单季",
    "经营活动产生的现金流量净额_单季同比",
    "经营活动产生的现金流量净额_单季环比",
    "净资产",
    "涨跌幅_10", "涨跌幅_20",
    "bias_5", "bias_10", "bias_20",
    "振幅_5", "振幅_10", "振幅_20",
    "涨跌幅std_5", "涨跌幅std_10", "涨跌幅std_20",
    "成交额std_5", "成交额std_10", "成交额std_20",
    "K", "D", "J",
    "DIF", "DEA", "MACD",
    "市盈率倒数", "市净率倒数",
    "新版申万一级行业名称", "新版申万二级行业名称", "新版申万三级行业名称",
    "涨跌幅", "下周期每天涨跌幅",
]

# Columns that carry forward from the previous month (financial data, industry)
CARRY_FORWARD_COLS = {
    13: "财报季度",
    14: "财报年份",
    15: "归母净利润",
    16: "归母净利润_ttm",
    17: "归母净利润_ttm同比",
    18: "归母净利润_单季",
    19: "归母净利润_单季同比",
    20: "归母净利润_单季环比",
    21: "经营活动产生的现金流量净额",
    22: "经营活动产生的现金流量净额_ttm",
    23: "经营活动产生的现金流量净额_ttm同比",
    24: "经营活动产生的现金流量净额_单季",
    25: "经营活动产生的现金流量净额_单季同比",
    26: "经营活动产生的现金流量净额_单季环比",
    27: "净资产",
    50: "新版申万一级行业名称",
    51: "新版申万二级行业名称",
    52: "新版申万三级行业名称",
}


def read_csv_data(csv_path: str) -> Tuple[List[str], Dict[str, List[List[str]]]]:
    """
    Read stock_data.csv and group rows by stock code.

    Returns:
        (headers, stock_data) where stock_data is dict: stock_code -> list of rows
    """
    stock_data: Dict[str, List[List[str]]] = defaultdict(list)
    headers = []
    with open(csv_path, "r", encoding="gbk") as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            stock_data[row[1]].append(row)
    # Sort each stock's rows by date
    for code in stock_data:
        stock_data[code].sort(key=lambda r: r[0])
    return headers, stock_data


def get_last_row(stock_rows: List[List[str]]) -> Optional[List[str]]:
    """Get the most recent row for a stock."""
    if not stock_rows:
        return None
    return stock_rows[-1]


def supplement_csv(
    csv_path: str,
    target_year: int = 2026,
    target_month: int = 5,
    max_stocks: Optional[int] = None,
    cache_dir: Optional[str] = None,
    datalen: int = 120,
) -> int:
    """
    Supplement stock_data.csv with data for the target month.

    Workflow:
    1. Read CSV and identify all unique stocks
    2. Get last row for each stock (for carry-forward financial data)
    3. Batch-fetch daily K-line data for all stocks (or load from cache)
    4. Aggregate daily data into monthly bars
    5. Compute technical indicators from daily data
    6. Get real-time quotes for market cap / PE / PB (or load from cache)
    7. Append new rows to CSV

    Args:
        csv_path: Path to stock_data.csv
        target_year, target_month: Target period
        max_stocks: Limit stocks for testing
        cache_dir: If provided, save/load fetched data to/from this directory
        datalen: Daily bars to fetch per stock. Default 120 (~5 months) provides
                 adequate warmup for MACD (needs 26+9=35 bars before target month),
                 bias_20, 振幅_20, std_20 (needs 20 bars). Using datalen=50 is
                 insufficient — MACD values become unreliable when most of the
                 50 bars are the target month itself.

    Returns:
        Number of new rows appended.
    """
    import pickle as _pickle
    import os as _os

    target_str = f"{target_year}-{target_month:02d}"
    daily_cache_file = None
    rt_cache_file = None
    if cache_dir:
        _os.makedirs(cache_dir, exist_ok=True)
        daily_cache_file = _os.path.join(cache_dir, f"daily_{target_str}.pkl")
        rt_cache_file = _os.path.join(cache_dir, f"rtquotes_{target_str}.pkl")
    print(f"Reading {csv_path} ...", file=sys.stderr)
    headers, stock_data = read_csv_data(csv_path)
    print(f"  Found {len(stock_data)} unique stocks", file=sys.stderr)

    # Get stock codes that have data in the previous month (for carry-forward)
    prev_month = target_month - 1
    prev_year = target_year
    if prev_month == 0:
        prev_month = 12
        prev_year = target_year - 1
    prev_month_str = f"{prev_year}-{prev_month:02d}"

    stocks_to_update = []
    for code, rows in stock_data.items():
        last_row = rows[-1]
        last_date = last_row[0]
        # Only update stocks with data in the previous month
        if last_date.startswith(prev_month_str) or last_date >= f"{prev_year}-{prev_month:02d}-01":
            stocks_to_update.append(code)

    # Also check if target month data already exists
    target_str = f"{target_year}-{target_month:02d}"
    stocks_to_update = [
        code for code in stocks_to_update
        if not any(r[0].startswith(target_str) for r in stock_data[code])
    ]

    if not stocks_to_update:
        # If no stocks match prev_month filter (edge case), update all stocks
        stocks_to_update = list(stock_data.keys())
        stocks_to_update = [
            code for code in stocks_to_update
            if not any(r[0].startswith(target_str) for r in stock_data[code])
        ]

    print(f"  {len(stocks_to_update)} stocks need {target_str} data", file=sys.stderr)

    if max_stocks and max_stocks < len(stocks_to_update):
        stocks_to_update = stocks_to_update[:max_stocks]
        print(f"  Limited to {max_stocks} stocks for testing", file=sys.stderr)

    # ── Step 1: Fetch daily K-line data ──
    daily_data = None
    if daily_cache_file and _os.path.exists(daily_cache_file):
        print(f"  Loading cached daily data from {daily_cache_file}", file=sys.stderr)
        with open(daily_cache_file, "rb") as f:
            daily_data = _pickle.load(f)
        print(f"  Loaded {len(daily_data)} stocks from cache", file=sys.stderr)
    else:
        print(f"\nFetching daily K-line data for {len(stocks_to_update)} stocks ...", file=sys.stderr)
        daily_data = fetch_daily_batch(stocks_to_update, datalen=datalen)
        print(f"  Successfully fetched {len(daily_data)} stocks", file=sys.stderr)
        if daily_cache_file:
            with open(daily_cache_file, "wb") as f:
                _pickle.dump(daily_data, f)
            print(f"  Saved daily data cache to {daily_cache_file}", file=sys.stderr)

    # ── Step 2: Fetch real-time quotes for market cap, PE, PB ──
    rt_quotes = None
    if rt_cache_file and _os.path.exists(rt_cache_file):
        print(f"  Loading cached real-time quotes from {rt_cache_file}", file=sys.stderr)
        with open(rt_cache_file, "rb") as f:
            rt_quotes = _pickle.load(f)
        print(f"  Loaded {len(rt_quotes)} quotes from cache", file=sys.stderr)
    else:
        print(f"\nFetching real-time quotes for market cap data ...", file=sys.stderr)
        rt_quotes = fetch_realtime_quotes_batch(stocks_to_update)
        print(f"  Successfully fetched {len(rt_quotes)} real-time quotes", file=sys.stderr)
        if rt_cache_file:
            with open(rt_cache_file, "wb") as f:
                _pickle.dump(rt_quotes, f)
            print(f"  Saved real-time quotes cache to {rt_cache_file}", file=sys.stderr)

    # ── Step 3: Build new rows ──
    print(f"\nBuilding {target_str} monthly rows ...", file=sys.stderr)
    new_rows = []
    skipped = 0

    for code in stocks_to_update:
        # Get daily data
        daily = daily_data.get(code)
        if not daily:
            skipped += 1
            continue

        # Aggregate May 2026 daily data into monthly bar
        monthly = aggregate_daily_to_monthly(daily, target_year, target_month)
        if not monthly:
            skipped += 1
            continue

        # Get last month's row for carry-forward
        last_row = get_last_row(stock_data[code])
        if not last_row:
            skipped += 1
            continue

        # Compute indicators from daily data
        daily_sorted = sorted(daily, key=lambda r: r["date"])
        d_dates = [r["date"] for r in daily_sorted]
        d_opens = [r["open"] for r in daily_sorted]
        d_highs = [r["high"] for r in daily_sorted]
        d_lows = [r["low"] for r in daily_sorted]
        d_closes = [r["close"] for r in daily_sorted]
        d_volumes = [r["volume"] for r in daily_sorted]

        indicators = compute_all_indicators(d_dates, d_opens, d_highs, d_lows, d_closes, d_volumes)

        # Use last valid indicator values (for the most recent trading day)
        last_idx = len(d_closes) - 1

        def last_valid(values: List[float]) -> float:
            """Get the last non-NaN value from indicator list."""
            for v in reversed(values):
                if not math.isnan(v):
                    return v
            return float("nan")

        # ── Real-time market data ──
        rt = rt_quotes.get(code, {})

        # Helper: safe float conversion
        def safe_float(val, default=0.0):
            if val is None or val == '':
                return default
            try:
                return float(val)
            except (ValueError, TypeError):
                return default

        # 流通市值 and 总市值: carry forward from previous month
        prev_float_mktcap = last_row[10] if last_row[10] else "0"  # 流通市值
        prev_total_mktcap = last_row[11] if last_row[11] else "0"  # 总市值

        # 市盈率倒数 = 1 / PE_ttm
        # Try real-time PE first, fall back to previous month's PE inverse
        pe_ttm = safe_float(rt.get("pe_ttm", 0)) if rt else 0.0
        if pe_ttm and pe_ttm != 0:
            pe_inverse = 1.0 / pe_ttm
        else:
            pe_inverse = safe_float(last_row[48], 0.0)

        # 市净率倒数 = 1 / PB
        pb_ratio = safe_float(rt.get("pb_ratio", 0)) if rt else 0.0
        if pb_ratio and pb_ratio != 0:
            pb_inverse = 1.0 / pb_ratio
        else:
            pb_inverse = safe_float(last_row[49], 0.0)

        # 上市至今交易天数
        prev_trading_days = safe_float(last_row[12], 0.0)
        trading_days = prev_trading_days + monthly["trading_days"]

        # 涨跌幅 (monthly return)
        prev_close = safe_float(last_row[7], monthly["open"])
        month_return = (monthly["close"] - prev_close) / prev_close if prev_close != 0 else 0

        # 下周期每天涨跌幅: leave empty for current month (needs next month's data)
        next_daily_returns = "[]"

        # ── Build the row ──
        new_row = [""] * len(CSV_HEADERS)

        # Column 0: 交易日期
        new_row[0] = monthly["date"]
        # Column 1: 股票代码
        new_row[1] = code
        # Column 2: 股票名称
        new_row[2] = last_row[2]
        # Column 3: 是否交易
        new_row[3] = "1"
        # Column 4-7: OHLC
        new_row[4] = str(monthly["open"])
        new_row[5] = str(monthly["high"])
        new_row[6] = str(monthly["low"])
        new_row[7] = str(monthly["close"])
        # Column 8: VWAP
        new_row[8] = str(monthly["vwap_est"])
        # Column 9: 成交额
        new_row[9] = str(monthly["amount_est"])
        # Column 10: 流通市值
        new_row[10] = prev_float_mktcap
        # Column 11: 总市值
        new_row[11] = prev_total_mktcap
        # Column 12: 上市至今交易天数
        new_row[12] = str(trading_days)
        # Column 13-27: Carry forward financial data
        for col_idx in CARRY_FORWARD_COLS:
            if col_idx < len(last_row):
                new_row[col_idx] = last_row[col_idx]
        # Column 28: 涨跌幅_10
        v_10 = last_valid(indicators["涨跌幅_10"])
        new_row[28] = str(v_10) if not math.isnan(v_10) else "0"
        # Column 29: 涨跌幅_20
        v_20 = last_valid(indicators["涨跌幅_20"])
        new_row[29] = str(v_20) if not math.isnan(v_20) else "0"
        # Column 30-32: bias
        for col_idx, key in [(30, "bias_5"), (31, "bias_10"), (32, "bias_20")]:
            v = last_valid(indicators[key])
            new_row[col_idx] = str(v) if not math.isnan(v) else "0"
        # Column 33-35: 振幅
        for col_idx, key in [(33, "振幅_5"), (34, "振幅_10"), (35, "振幅_20")]:
            v = last_valid(indicators[key])
            new_row[col_idx] = str(v) if not math.isnan(v) else "0"
        # Column 36-38: 涨跌幅std
        for col_idx, key in [(36, "涨跌幅std_5"), (37, "涨跌幅std_10"), (38, "涨跌幅std_20")]:
            v = last_valid(indicators[key])
            new_row[col_idx] = str(v) if not math.isnan(v) else "0"
        # Column 39-41: 成交额std
        for col_idx, key in [(39, "成交额std_5"), (40, "成交额std_10"), (41, "成交额std_20")]:
            v = last_valid(indicators[key])
            new_row[col_idx] = str(v) if not math.isnan(v) else "0"
        # Column 42-44: KDJ
        for col_idx, key in [(42, "K"), (43, "D"), (44, "J")]:
            v = last_valid(indicators[key])
            new_row[col_idx] = str(v) if not math.isnan(v) else "0"
        # Column 45-47: MACD
        for col_idx, key in [(45, "DIF"), (46, "DEA"), (47, "MACD")]:
            v = last_valid(indicators[key])
            new_row[col_idx] = str(v) if not math.isnan(v) else "0"
        # Column 48: 市盈率倒数
        new_row[48] = str(pe_inverse)
        # Column 49: 市净率倒数
        new_row[49] = str(pb_inverse)
        # Column 50-52: industry (carried forward)
        for col_idx in [50, 51, 52]:
            if col_idx < len(last_row):
                new_row[col_idx] = last_row[col_idx]
        # Column 53: 涨跌幅
        new_row[53] = str(month_return)
        # Column 54: 下周期每天涨跌幅
        new_row[54] = next_daily_returns

        # Also update the previous month's 下周期每天涨跌幅
        # Find the last row for this stock and update its column 54
        if monthly["daily_returns"] and len(stock_data[code]) > 0:
            prev_last = stock_data[code][-1]
            # Only update if previous month is the one we expect
            if prev_last[0].startswith(prev_month_str):
                prev_last[54] = str(monthly["daily_returns"])

        new_rows.append(new_row)

    print(f"  Built {len(new_rows)} new rows ({skipped} skipped)", file=sys.stderr)

    if not new_rows:
        print("  No new rows to append. Data may already be up to date.", file=sys.stderr)
        return 0

    # ── Step 4: Update previous month's 下周期每天涨跌幅 in memory ──
    # (already done above by mutating stock_data[code][-1][54])

    # ── Step 5: Write back to CSV ──
    # Sort only the new rows; the existing CSV is already sorted so we just append.
    new_rows.sort(key=lambda r: (r[0], r[1]))

    print(f"\nWriting updated CSV ...", file=sys.stderr)

    # Build backfill map from the in-memory stock_data (already loaded).
    backfill_map = {}
    for code, rows in stock_data.items():
        if rows:
            last_row = rows[-1]
            if last_row[0].startswith(prev_month_str) and last_row[54] != "[]":
                backfill_map[(code, last_row[0])] = last_row[54]

    # Stream existing file → temp file applying backfill, then append new rows.
    # Writing to a temp file first makes the operation atomic: if the process is
    # interrupted mid-write the original CSV is untouched; the temp is abandoned.
    tmp_path = csv_path + ".tmp"
    total_written = 0
    with open(csv_path, "r", encoding="gbk") as fin, \
         open(tmp_path, "w", encoding="gbk", newline="") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        existing_headers = next(reader)
        writer.writerow(existing_headers)
        for row in reader:
            key = (row[1], row[0])  # (code, date)
            if key in backfill_map:
                row[54] = backfill_map[key]
            writer.writerow(row)
            total_written += 1
        writer.writerows(new_rows)
        total_written += len(new_rows)

    # Atomic rename: replaces the original only after the write is fully complete.
    os.replace(tmp_path, csv_path)

    print(f"  Done. Total rows now: {total_written}", file=sys.stderr)
    print(f"  Appended {len(new_rows)} new rows for {target_str}", file=sys.stderr)

    return len(new_rows)


# ──────────────────────────────────────────────
# Incremental per-stock daily cache (restart-safe)
# ──────────────────────────────────────────────
#
# Layout: <cache_dir>/daily_stocks/<code>.csv (UTF-8)
#   date,open,high,low,close,volume   (sorted by date ascending)
#
# Each stock has its own tiny CSV. Writes use a temp file + os.replace so a
# crash mid-update leaves the original cache intact, and the next run resumes
# from each stock's last cached day.

DAILY_CACHE_DIRNAME = "daily_stocks"


def _daily_cache_path(cache_dir: str, code: str) -> str:
    return os.path.join(cache_dir, DAILY_CACHE_DIRNAME, f"{code}.csv")


def _load_daily_cache_one(cache_dir: str, code: str) -> List[Dict[str, object]]:
    """Return cached daily rows sorted by date ascending, or [] if none."""
    path = _daily_cache_path(cache_dir, code)
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append({
                    "date": r["date"],
                    "open": float(r["open"]),
                    "high": float(r["high"]),
                    "low": float(r["low"]),
                    "close": float(r["close"]),
                    "volume": int(float(r["volume"])),
                })
            except (KeyError, ValueError):
                continue
    rows.sort(key=lambda x: x["date"])
    return rows


def _save_daily_cache_one(cache_dir: str, code: str, rows: List[Dict[str, object]]) -> None:
    """Atomically write per-stock daily cache (temp file + os.replace)."""
    path = _daily_cache_path(cache_dir, code)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "open", "high", "low", "close", "volume"])
        for r in sorted(rows, key=lambda x: x["date"]):
            writer.writerow([
                r["date"], r["open"], r["high"], r["low"], r["close"], r["volume"]
            ])
    os.replace(tmp, path)


def _merge_daily(cached: List[Dict[str, object]],
                 fetched: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Merge cached + fetched daily rows, dedup by date (fetched wins on overlap)."""
    by_date: Dict[str, Dict[str, object]] = {r["date"]: r for r in cached}
    for r in fetched:
        by_date[r["date"]] = r
    return sorted(by_date.values(), key=lambda x: x["date"])


def _fetch_incremental_single(
    code: str,
    cached: List[Dict[str, object]],
    today_str: str,
    min_history_days: int = 120,
    fresh_gap_days: int = 3,
) -> Tuple[str, Optional[List[Dict[str, object]]], Optional[str], bool]:
    """
    Fetch only the missing daily bars for one stock and merge with cache.

    Returns (code, merged_rows or None, error or None, did_network_fetch).
    If cache is already up-to-date (last_cached_date within `fresh_gap_days`
    of today_str), no network fetch is performed and merged_rows == cached.
    `fresh_gap_days=3` covers weekend gaps: a Friday-end cache survives Sat/Sun/Mon
    re-runs without unnecessary re-fetch (next real bar isn't until Mon close).
    """
    from datetime import datetime as _dt

    if cached:
        last_date = cached[-1]["date"]
        if last_date >= today_str:
            return (code, cached, None, False)
        try:
            d_last = _dt.strptime(last_date, "%Y-%m-%d")
            d_today = _dt.strptime(today_str, "%Y-%m-%d")
            cal_gap = (d_today - d_last).days
        except ValueError:
            cal_gap = min_history_days
        # Cache freshness: skip network fetch if cached data is within fresh_gap_days
        if cal_gap <= fresh_gap_days:
            return (code, cached, None, False)
        # Small buffer for weekends/holidays; cap at min_history_days
        datalen = max(5, min(cal_gap + 10, min_history_days))
    else:
        # First seed: full warmup window
        datalen = min_history_days

    for attempt in range(MAX_RETRIES):
        try:
            fetched = fetch_a_share_daily_from_sina(code, datalen=datalen)
            merged = _merge_daily(cached, fetched)
            return (code, merged, None, True)
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                return (code, None, str(e), True)
    return (code, None, "Unknown error", True)


def fetch_daily_batch_incremental(
    stock_codes: List[str],
    cache_dir: str,
    today_str: Optional[str] = None,
    min_history_days: int = 120,
    max_workers: int = 8,  # Sina rate-limits ~3-5k bursts; 8 workers stays under
    verbose: bool = True,
) -> Dict[str, List[Dict[str, object]]]:
    """
    Incremental concurrent fetch.

    For each stock: load its daily cache → fetch only the missing days → merge
    → atomic-save its cache immediately. The result dict maps code → full
    merged daily rows (cached + new) for downstream aggregation.

    Restart safety: each completed stock has its updated cache on disk before
    the next stock starts processing. An interrupted run resumes naturally on
    the next invocation (each stock skips already-cached days).
    """
    if today_str is None:
        today_str = time.strftime("%Y-%m-%d")

    os.makedirs(os.path.join(cache_dir, DAILY_CACHE_DIRNAME), exist_ok=True)

    results: Dict[str, List[Dict[str, object]]] = {}
    errors: List[str] = []
    no_op = 0
    fetched_n = 0
    total = len(stock_codes)
    completed = 0

    def _process(code: str):
        cached = _load_daily_cache_one(cache_dir, code)
        return _fetch_incremental_single(code, cached, today_str, min_history_days)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process, code): code for code in stock_codes}
        for future in as_completed(futures):
            code, merged, error, did_fetch = future.result()
            completed += 1
            if merged is not None:
                results[code] = merged
                if did_fetch:
                    fetched_n += 1
                    # Only rewrite cache when we actually pulled new data
                    _save_daily_cache_one(cache_dir, code, merged)
                else:
                    no_op += 1
            else:
                errors.append(f"{code}: {error}")
            if verbose and completed % 200 == 0:
                print(
                    f"  Progress: {completed}/{total} stocks "
                    f"({fetched_n} fetched, {no_op} cache-hit, {len(errors)} errors)",
                    file=sys.stderr,
                    flush=True,
                )
            time.sleep(RATE_LIMIT)

    if verbose:
        print(
            f"  Total: {completed}/{total} processed "
            f"({fetched_n} fetched from network, {no_op} cache-hit, {len(errors)} errors)",
            file=sys.stderr,
            flush=True,
        )
        for err in errors[:10]:
            print(f"    {err}", file=sys.stderr, flush=True)
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more", file=sys.stderr, flush=True)

    return results


def _build_target_month_row(
    code: str,
    daily: List[Dict[str, object]],
    rt: Dict[str, object],
    stock_rows: List[List[str]],
    target_year: int,
    target_month: int,
) -> Optional[List[str]]:
    """
    Build a single stock_data.csv row for (code, target_year-target_month).

    Returns None if there's no daily bar for this month or no previous row to
    carry forward from.
    """
    monthly = aggregate_daily_to_monthly(daily, target_year, target_month)
    if not monthly:
        return None
    last_row = get_last_row(stock_rows)
    if not last_row:
        return None

    daily_sorted = sorted(daily, key=lambda r: r["date"])
    d_dates = [r["date"] for r in daily_sorted]
    d_opens = [r["open"] for r in daily_sorted]
    d_highs = [r["high"] for r in daily_sorted]
    d_lows = [r["low"] for r in daily_sorted]
    d_closes = [r["close"] for r in daily_sorted]
    d_volumes = [r["volume"] for r in daily_sorted]
    indicators = compute_all_indicators(d_dates, d_opens, d_highs, d_lows, d_closes, d_volumes)

    def last_valid(values: List[float]) -> float:
        for v in reversed(values):
            if not math.isnan(v):
                return v
        return float("nan")

    def safe_float(val, default=0.0):
        if val is None or val == "":
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    prev_float_mktcap = last_row[10] if last_row[10] else "0"
    prev_total_mktcap = last_row[11] if last_row[11] else "0"

    pe_ttm = safe_float(rt.get("pe_ttm", 0)) if rt else 0.0
    pe_inverse = 1.0 / pe_ttm if pe_ttm else safe_float(last_row[48], 0.0)
    pb_ratio = safe_float(rt.get("pb_ratio", 0)) if rt else 0.0
    pb_inverse = 1.0 / pb_ratio if pb_ratio else safe_float(last_row[49], 0.0)

    prev_trading_days = safe_float(last_row[12], 0.0)
    trading_days = prev_trading_days + monthly["trading_days"]

    prev_close = safe_float(last_row[7], monthly["open"])
    month_return = (monthly["close"] - prev_close) / prev_close if prev_close != 0 else 0

    row = [""] * len(CSV_HEADERS)
    row[0] = monthly["date"]
    row[1] = code
    row[2] = last_row[2]
    row[3] = "1"
    row[4] = str(monthly["open"])
    row[5] = str(monthly["high"])
    row[6] = str(monthly["low"])
    row[7] = str(monthly["close"])
    row[8] = str(monthly["vwap_est"])
    row[9] = str(monthly["amount_est"])
    row[10] = prev_float_mktcap
    row[11] = prev_total_mktcap
    row[12] = str(trading_days)
    for col_idx in CARRY_FORWARD_COLS:
        if col_idx < len(last_row):
            row[col_idx] = last_row[col_idx]
    v = last_valid(indicators["涨跌幅_10"]); row[28] = str(v) if not math.isnan(v) else "0"
    v = last_valid(indicators["涨跌幅_20"]); row[29] = str(v) if not math.isnan(v) else "0"
    for col_idx, key in [(30, "bias_5"), (31, "bias_10"), (32, "bias_20"),
                         (33, "振幅_5"), (34, "振幅_10"), (35, "振幅_20"),
                         (36, "涨跌幅std_5"), (37, "涨跌幅std_10"), (38, "涨跌幅std_20"),
                         (39, "成交额std_5"), (40, "成交额std_10"), (41, "成交额std_20"),
                         (42, "K"), (43, "D"), (44, "J"),
                         (45, "DIF"), (46, "DEA"), (47, "MACD")]:
        v = last_valid(indicators[key])
        row[col_idx] = str(v) if not math.isnan(v) else "0"
    row[48] = str(pe_inverse)
    row[49] = str(pb_inverse)
    for col_idx in [50, 51, 52]:
        if col_idx < len(last_row):
            row[col_idx] = last_row[col_idx]
    row[53] = str(month_return)
    row[54] = "[]"  # next-period daily returns: filled in once next month exists
    # daily_returns of THIS month — used to backfill prev month's column 54
    row.append(monthly["daily_returns"])  # sentinel field, stripped before write
    return row


def supplement_csv_incremental(
    csv_path: str,
    target_year: int,
    target_month: int,
    cache_dir: str,
    max_stocks: Optional[int] = None,
    min_history_days: int = 120,
    today_str: Optional[str] = None,
) -> int:
    """
    Restart-safe incremental supplement.

    Differs from supplement_csv:
      * No destructive "delete the target month then refetch" stage.
      * Per-stock daily cache: each stock only fetches days since last cache.
      * stock_data.csv update is an in-place upsert (replace target-month row
        if present, append otherwise) via stream + os.replace.

    Returns the number of stock_data.csv rows written for target month
    (replaced + newly appended).
    """
    if today_str is None:
        today_str = time.strftime("%Y-%m-%d")
    target_str = f"{target_year}-{target_month:02d}"
    prev_month = target_month - 1
    prev_year = target_year
    if prev_month == 0:
        prev_month = 12
        prev_year -= 1
    prev_month_str = f"{prev_year}-{prev_month:02d}"

    print(f"Reading {csv_path} ...", file=sys.stderr, flush=True)
    _, stock_data = read_csv_data(csv_path)
    print(f"  Found {len(stock_data)} unique stocks", file=sys.stderr, flush=True)

    # Eligible stocks: those with a row in prev month OR target month (handles
    # the case where target month was partially populated by a previous run).
    eligible = []
    for code, rows in stock_data.items():
        if not rows:
            continue
        last_date = rows[-1][0]
        if last_date >= f"{prev_year}-{prev_month:02d}-01":
            eligible.append(code)
    if not eligible:
        eligible = list(stock_data.keys())
    if max_stocks and max_stocks < len(eligible):
        eligible = eligible[:max_stocks]
        print(f"  Limited to {max_stocks} stocks for testing", file=sys.stderr, flush=True)
    print(f"  {len(eligible)} stocks eligible for {target_str} update", file=sys.stderr, flush=True)

    # Step 1: incremental daily fetch (each stock's cache persisted on completion)
    print(f"\nIncremental daily fetch (per-stock cache, restart-safe) ...", file=sys.stderr, flush=True)
    daily_data = fetch_daily_batch_incremental(
        eligible, cache_dir, today_str=today_str, min_history_days=min_history_days
    )

    # Step 2: real-time quotes (still bulk; small response, fast)
    os.makedirs(cache_dir, exist_ok=True)
    rt_cache_file = os.path.join(cache_dir, f"rtquotes_{target_str}.pkl")
    if os.path.exists(rt_cache_file):
        print(f"  Loading cached real-time quotes from {rt_cache_file}",
              file=sys.stderr, flush=True)
        with open(rt_cache_file, "rb") as f:
            import pickle as _pickle
            rt_quotes = _pickle.load(f)
    else:
        print(f"\nFetching real-time quotes ...", file=sys.stderr, flush=True)
        rt_quotes = fetch_realtime_quotes_batch(eligible)
        with open(rt_cache_file, "wb") as f:
            import pickle as _pickle
            _pickle.dump(rt_quotes, f)
        print(f"  Fetched {len(rt_quotes)} real-time quotes", file=sys.stderr, flush=True)

    # Step 3: build monthly rows from updated daily cache
    print(f"\nBuilding {target_str} monthly rows ...", file=sys.stderr, flush=True)
    new_rows: List[List[str]] = []
    backfill_prev_col54: Dict[Tuple[str, str], str] = {}
    skipped = 0
    for code in eligible:
        daily = daily_data.get(code)
        if not daily:
            skipped += 1
            continue
        row_with_sentinel = _build_target_month_row(
            code, daily, rt_quotes.get(code, {}) if rt_quotes else {},
            stock_data[code], target_year, target_month,
        )
        if row_with_sentinel is None:
            skipped += 1
            continue
        # Pop sentinel daily_returns (used for prev-month col 54 backfill)
        daily_returns = row_with_sentinel.pop()
        if daily_returns and stock_data[code]:
            # 向后搜索找上月行：当月已有行时 [-1] 指向当月行而非上月行，
            # 导致回填条件永远不成立，上月的「下周期每天涨跌幅」无法更新。
            prev_month_row = next(
                (r for r in reversed(stock_data[code]) if r[0].startswith(prev_month_str)),
                None,
            )
            if prev_month_row:
                backfill_prev_col54[(code, prev_month_row[0])] = str(daily_returns)
        new_rows.append(row_with_sentinel)
    print(f"  Built {len(new_rows)} {target_str} rows ({skipped} skipped)",
          file=sys.stderr, flush=True)

    if not new_rows:
        print("  Nothing to write; CSV unchanged.", file=sys.stderr, flush=True)
        return 0

    # Step 4: upsert into CSV (replace existing target-month rows, append rest)
    print(f"\nUpserting {target_str} rows into CSV ...", file=sys.stderr, flush=True)
    new_keys = {(r[0][:7], r[1]) for r in new_rows}
    tmp_path = csv_path + ".tmp"
    kept = replaced = 0
    with open(csv_path, "r", encoding="gbk") as fin, \
         open(tmp_path, "w", encoding="gbk", newline="") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        existing_headers = next(reader)
        writer.writerow(existing_headers)
        for row in reader:
            key = (row[0][:7], row[1])
            if key in new_keys:
                # Drop old target-month row; new one will be appended below.
                replaced += 1
                continue
            bk_key = (row[1], row[0])
            if bk_key in backfill_prev_col54:
                row[54] = backfill_prev_col54[bk_key]
            writer.writerow(row)
            kept += 1
        new_rows.sort(key=lambda r: (r[0], r[1]))
        writer.writerows(new_rows)
    os.replace(tmp_path, csv_path)
    print(
        f"  Done. Kept {kept}, replaced {replaced}, appended {len(new_rows)} "
        f"(total now {kept + len(new_rows)})",
        file=sys.stderr,
        flush=True,
    )
    return len(new_rows)


# ──────────────────────────────────────────────
# Original demo report (kept for backward compat)
# ──────────────────────────────────────────────
def build_demo_report() -> Dict[str, object]:
    """Build the original demo report with real-time quotes and daily K-line."""
    codes = ["688256", "09988"]
    tencent_quotes = {item["symbol"]: item for item in fetch_realtime_quotes(codes)}
    sina_quotes = {
        SYMBOLS[code]["name"]: item
        for code, item in zip(codes, fetch_sina_realtime(codes))
    }

    report: Dict[str, object] = {"示例结果": {}}

    for code in codes:
        config = SYMBOLS[code]
        tencent = tencent_quotes[code]
        sina = sina_quotes[config["name"]]
        cross_check_realtime(code, tencent, sina)
        report["示例结果"][code] = {
            "实时行情": format_realtime_result(code, tencent, sina),
        }

    report["示例结果"]["688256"]["收盘后日线"] = format_daily_result(
        fetch_a_share_daily_from_sina("688256", datalen=2)
    )
    return report


# ──────────────────────────────────────────────
# Main CLI
# ──────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="A-share stock data fetching tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 get_stock_info.py                           # Real-time demo report
  python3 get_stock_info.py --mode supplement         # Supplement CSV with latest data
  python3 get_stock_info.py --mode supplement --max-stocks 10  # Test with 10 stocks
  python3 get_stock_info.py --mode supplement --year 2026 --month 5
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["realtime", "supplement"],
        default="realtime",
        help="Operation mode (default: realtime)",
    )
    parser.add_argument(
        "--csv",
        default="stock_trade_demo/stock_data.csv",
        help="Path to stock_data.csv (for supplement mode)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2026,
        help="Target year for supplement (default: 2026)",
    )
    parser.add_argument(
        "--month",
        type=int,
        default=5,
        help="Target month for supplement (default: 5)",
    )
    parser.add_argument(
        "--max-stocks",
        type=int,
        default=None,
        help="Max stocks to process (for testing)",
    )
    parser.add_argument(
        "--cache-dir",
        default=".cache",
        help="Directory for caching fetched data (default: .cache). Set to empty string to disable.",
    )
    parser.add_argument(
        "--datalen",
        type=int,
        default=120,
        help=(
            "Number of daily bars to fetch per stock (default: 120, ~5 months). "
            "Must be at least 80 for reliable MACD/bias_20/std_20 computation. "
            "Used as the warmup window for the FIRST seed of per-stock daily cache; "
            "subsequent runs only fetch days since the last cached date."
        ),
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help=(
            "Use the legacy monthly supplement_csv (full ~120-day fetch per stock, "
            "no per-stock cache, not restart-safe). Default is the incremental path "
            "that maintains per-stock daily cache and only fetches the delta."
        ),
    )

    args = parser.parse_args()

    if args.mode == "realtime":
        report = build_demo_report()
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.mode == "supplement":
        import os as _os
        csv_full = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), args.csv)
        if not _os.path.exists(csv_full):
            # Try relative to cwd
            csv_full = args.csv
        if not _os.path.exists(csv_full):
            raise SystemExit(f"ERROR: CSV file not found: {csv_full}")

        cache_dir_val = args.cache_dir if args.cache_dir else None

        if args.legacy:
            count = supplement_csv(
                csv_full,
                target_year=args.year,
                target_month=args.month,
                max_stocks=args.max_stocks,
                cache_dir=cache_dir_val,
                datalen=args.datalen,
            )
            print(f"\nLegacy supplement complete. {count} rows added.")
        else:
            if not cache_dir_val:
                raise SystemExit("ERROR: --cache-dir is required for incremental mode (default '.cache' is fine)")
            count = supplement_csv_incremental(
                csv_full,
                target_year=args.year,
                target_month=args.month,
                cache_dir=cache_dir_val,
                max_stocks=args.max_stocks,
                min_history_days=args.datalen,
            )
            print(f"\nIncremental supplement complete. {count} {args.year}-{args.month:02d} rows upserted.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}")
