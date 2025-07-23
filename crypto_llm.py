from flask import Flask, request, jsonify, send_file
import uuid
import random
from flask_cors import CORS
import json
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
import yfinance as yf
import matplotlib.pyplot as plt
from prophet import Prophet
import matplotlib.pyplot as plt
import io
import base64
import os
import config

app = Flask(__name__)
CORS(app)

class ChatAssistant:
    def __init__(
        self,
        api_key: str,
        prompt: str,
        provider: str = "openai_chat_completion",
        base_url: str = "https://api.metisai.ir",
        model: str = "gpt-4o-mini-2024-07-18",
        max_tokens: int = 150
    ):
        self.prompt = prompt
        self.endpoint = f"{base_url}/api/v1/wrapper/{provider}/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.model = model
        self.max_tokens = max_tokens
        self.messages = [{"role": "system", "content": "You are a helpful assistant."}]

    def start(self):
        """
        Begin the interactive chat loop.
        Type 'stop' or 'exit' to end the session.
        """
        print("🟢 Chat session started. (type 'stop' or 'exit' to end)\n")
        while True:
            user_input = self.prompt
            if user_input.lower() in {"stop", "exit"}:
                print("🔴 Chat session ended.")
                break

            self.messages.append({"role": "user", "content": user_input})
            payload = {
                "model": self.model,
                "messages": self.messages,
                "max_tokens": self.max_tokens
            }

            resp = requests.post(self.endpoint, json=payload, headers=self.headers)
            print("tamam shod")
            if resp.status_code != 200:
                print(f"⚠️ Error {resp.status_code}: {resp.text}")
                continue

            data = resp.json()
            reply = data["choices"][0]["message"]["content"]
            self.messages.append({"role": "assistant", "content": reply})
            print(reply)
            return reply

np.NaN = np.nan

CC_BASE = "https://min-api.cryptocompare.com/data"

def cc_get(path, params=None, tries=5, backoff=1):
    url = f"{CC_BASE}/{path}"
    for i in range(tries):
        r = requests.get(url, params=params or {})
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            time.sleep(backoff * (i + 1))
            continue
        r.raise_for_status()
    raise RuntimeError(f"CryptoCompare error on {path}")

def get_top_coins_cc(limit=30, tsym="USD"):
    j = cc_get("top/mktcap", {"tsym": tsym, "limit": limit})
    return [coin["CoinInfo"]["Name"] for coin in j["Data"]]

def fetch_close_prices_cc(symbol, days=30, tsym="USD"):
    j = cc_get("v2/histoday", {"fsym": symbol, "tsym": tsym, "limit": days})
    data = j["Data"]["Data"]
    df = pd.DataFrame(data)[["time", "close"]]
    df["date"] = pd.to_datetime(df["time"], unit="s").dt.date
    return df.set_index("date")[["close"]].rename(columns={"close": "Close"})

FX_BASE = "https://api.frankfurter.dev/v1"

def fetch_fx_rates(base="EUR", quote="USD", days=30):
    end = datetime.utcnow().date()
    start = end - timedelta(days=days)
    url = f"{FX_BASE}/{start.isoformat()}..{end.isoformat()}"
    params = {"from": base, "to": quote}
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json().get("rates", {})
    rows = []
    for day, info in sorted(data.items()):
        rate = info.get(quote)
        if rate is not None:
            rows.append({"date": pd.to_datetime(day).date(), "Close": float(rate)})
    if not rows:
        raise ValueError(f"No valid rates for {base}/{quote}")
    return pd.DataFrame(rows).set_index("date")

def annotate_currency_df(fx_df):
    df = fx_df.copy().sort_index()
    df['MA20']  = df['Close'].rolling(20).mean()
    df['MA50']  = df['Close'].rolling(50).mean()
    df['MA200'] = df['Close'].rolling(200).mean()
    sma20 = df['MA20']
    std20 = df['Close'].rolling(20).std()
    df['BB_upper'] = sma20 + 2 * std20
    df['BB_lower'] = sma20 - 2 * std20
    df['ROC10'] = df['Close'].pct_change(10) * 100

    delta = df['Close'].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI14'] = 100 - (100 / (1 + rs))

    if {'High', 'Low'}.issubset(df.columns):
        tr1 = df['High'] - df['Low']
        tr2 = (df['High'] - df['Close'].shift()).abs()
        tr3 = (df['Low']  - df['Close'].shift()).abs()
        df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR14'] = df['TR'].rolling(14).mean()

    df['Volatility'] = df['Close'].pct_change().rolling(30).std() * np.sqrt(252)
    cummax = df['Close'].cummax()
    df['Drawdown'] = (df['Close'] - cummax) / cummax

    return df

def advanced_currency_analysis(fx_df):
    df = fx_df.copy().sort_index().tail(30)
    L = df.iloc[-1]
    result = {
        'price':     round(L['Close'], 4),
        'ma20':      round(L['MA20'], 4),
        'ma50':      round(L['MA50'], 4),
        'ma200':     round(L['MA200'], 4),
        'bb_upper':  round(L['BB_upper'], 4),
        'bb_lower':  round(L['BB_lower'], 4),
        'roc10':     round(L['ROC10'], 2),
        'rsi14':     round(L['RSI14'], 2),
        'volatility':round(L['Volatility'], 4),
        'drawdown':  round(L['Drawdown'], 4)
    }
    if 'ATR14' in df.columns:
        result['atr14'] = round(L['ATR14'], 4)
    return result 

def forecast_next_7_days(df):
    df = df.copy()

    if 'Close' in df.columns:
        close_ser = df['Close']
    else:
        try:
            close_ser = df.xs('Close', axis=1, level=-1)
        except Exception:
            raise ValueError("Couldn't find a 'Close' column at any level")

    ts = close_ser.reset_index()
    ts.columns = ['ds', 'y']
    ts['ds'] = pd.to_datetime(ts['ds'])
    ts['y']  = pd.to_numeric(ts['y'], errors='coerce')
    ts = ts.dropna(subset=['y'])

    if len(ts) < 30:
        return None

    m = Prophet(daily_seasonality=False, weekly_seasonality=True)
    m.fit(ts)

    future = m.make_future_dataframe(periods=7)
    fc     = m.predict(future).tail(7)

    mean   = fc['yhat'].mean()
    lower  = fc['yhat_lower'].min()
    upper  = fc['yhat_upper'].max()
    trend  = '↑' if mean > ts['y'].iloc[-1] else '↓'

    return {
        'forecast_mean':  round(mean,  4),
        'forecast_lower': round(lower, 4),
        'forecast_upper': round(upper, 4),
        'trend':          trend
    }


def engineer_features(dfs):
    rows = []
    for sym, df in dfs.items():
        d = df.copy()
        d['SMA20'] = d['Close'].rolling(20).mean()

        delta = d['Close'].diff()
        gain  = delta.clip(lower=0)
        loss  = -delta.clip(upper=0)
        rs    = gain.rolling(14).mean() / loss.rolling(14).mean()
        d['RSI14'] = 100 - (100 / (1 + rs))

        ema12 = d['Close'].ewm(span=12, adjust=False).mean()
        ema26 = d['Close'].ewm(span=26, adjust=False).mean()
        macd  = ema12 - ema26
        d['MACD']        = macd
        d['MACD_signal'] = macd.ewm(span=9, adjust=False).mean()
        d['Vol30']       = d['Close'].pct_change().rolling(30).std() * np.sqrt(252)

        forecast = forecast_next_7_days(df)
        if not forecast:
            continue

        L = d.iloc[-1]
        rows.append({
            'symbol':      sym,
            'price':       round(L['Close'], 4),
            'sma20':       round(L['SMA20'], 4),
            'rsi14':       round(L['RSI14'], 2),
            'macd':        round(L['MACD'], 4),
            'macd_signal': round(L['MACD_signal'], 4),
            'vol30':       round(L['Vol30'], 4),
            'forecast':    forecast
        })

    return pd.DataFrame(rows).set_index('symbol')

def make_compact_blobs(dfs):
    series = []
    for sym, df in dfs.items():
        last4 = df['Close'].tail(4)
        if last4.dropna().shape[0] < 4:
            continue
        s = last4.copy()
        s.name = sym
        series.append(s)

    price_df   = pd.concat(series, axis=1)
    price_json = price_df.to_json(orient='split')
    rets       = price_df.pct_change().dropna(how='any')
    corr       = rets.corr().abs()
    np.fill_diagonal(corr.values, np.nan)
    top40      = corr.stack().nlargest(40)
    corr_dict  = top40.to_dict()

    return price_json, corr_dict

PROMPT_TEMPLATE = """
⚡ **You are an expert robo-advisor.** ⚡
✅ **Provide exactly five** technical trading or investment strategies based solely on the data below.
🚫 **Avoid trivial picks; use only quantitative and technical insights.**
⏱️ **Answer in under 300 tokens.**
give each advice in one seperate line and do not use markdown syntax.

USER PROMPT:
{user_prompt}

FX ANALYSIS:
{fx_analysis}

MARKET SNAPSHOT (as of {date_iso}, forecasted values included):
{snapshot}

LAST-4-DAYS CLOSES JSON:
{price_json}

TOP 40 ABS CORRELATIONS:
{corr_dict}
"""

def build_prompt(user_prompt, feats, price_json, corr_dict, fx_analysis):
    date_iso = datetime.utcnow().strftime('%Y-%m-%d')
    snap = [
        f"{s}: price {r.price}, SMA20 {r.sma20}, RSI14 {r.rsi14}, "
        f"MACD {r.macd}/{r.macd_signal}, vol30 {r.vol30}, "
        f"🔮 7d forecast → {r.forecast['forecast_mean']} "
        f"[{r.forecast['forecast_lower']}, {r.forecast['forecast_upper']}] {r.forecast['trend']} (predicted)"
        for s, r in feats.iterrows()
    ]

    return PROMPT_TEMPLATE.format(
        user_prompt=user_prompt,
        fx_analysis=fx_analysis,
        date_iso=date_iso,
        snapshot='\n'.join(snap),
        price_json=price_json,
        corr_dict=corr_dict
    )

def collect_data(period_days=30):
    data = {}
    stocks = [
        'AAPL', 'MSFT', 'GOOG', 'AMZN',
        'TSLA', 'NVDA', 'JPM', 'V', 'JNJ',
        'WMT', 'BAC', 'PG', 'DIS', 'MA',
        'HD', 'UNH', 'XOM', 'CVX', 'PFE'
    ]
    cnt1 = 1
    for sym in stocks:
        df = yf.download(sym, period=f"{period_days}d", interval="1d",
                         auto_adjust=True, progress=False)
        print(cnt1)
        cnt1 += 1
        if not df.empty:
            data[sym] = df[['Close']].copy()
            data[sym].index = data[sym].index.date

    data['EURUSD'] = fetch_fx_rates('EUR', 'USD', days=period_days + 200)
    cnt = 1
    for sym in get_top_coins_cc(limit=30):
        print(cnt)
        cnt += 1
        df = fetch_close_prices_cc(sym, days=period_days)
        if not df.empty:
            data[sym] = df

    return data

def plot_asset(df, symbol=None):
    df = df.sort_index()
    plt.close('all')
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 12))

    ax = axes[0]
    ax.plot(df.index, df['Close'], label='Close')
    for col in ['MA20', 'MA50', 'MA200']:
        if col in df:
            ax.plot(df.index, df[col], label=col)
    if 'BB_upper' in df and 'BB_lower' in df:
        ax.plot(df.index, df['BB_upper'], linestyle='--', label='BB upper')
        ax.plot(df.index, df['BB_lower'], linestyle='--', label='BB lower')
    ax.set_ylabel('Price')
    ax.legend(loc='best')
    ax.grid(True)

    ax = axes[1]
    ax.plot(df.index, df['RSI14'])
    ax.axhline(70, linestyle='--')
    ax.axhline(30, linestyle='--')
    ax.set_ylabel('RSI14')
    ax.grid(True)

    ax = axes[2]
    ax.plot(df.index, df['Volatility'])
    ax.set_ylabel('Volatility (ann.)')
    ax.grid(True)

    ax = axes[3]
    ax.plot(df.index, df['Drawdown'])
    ax.axhline(0, linewidth=0.5)
    ax.set_ylabel('Drawdown')
    ax.grid(True)

    fig.suptitle(f"Technical overview: {symbol or ''}", y=0.92)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)

    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def plot_correlation(price_json):
    df = pd.read_json(price_json, orient='split')
    rets = df.pct_change().dropna(how='any')
    corr = rets.corr()

    plt.close('all')
    fig, ax = plt.subplots(figsize=(8, 8))
    cax = ax.matshow(corr, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = range(len(corr.columns))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)
    ax.set_title(f"Correlation matrix (last {df.shape[0]} days)")
    plt.tight_layout()

    # Save to in-memory buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def api_crypto(user_text):

    data = collect_data(30)
    fx_full = annotate_currency_df(data['EURUSD']).tail(30)
    fx_analysis = advanced_currency_analysis(fx_full)
    print("FX Analysis:", fx_analysis)

    img_asset = plot_asset(fx_full, symbol='EURUSD')
    
    feats = engineer_features(data)
    price_json, corr_dict = make_compact_blobs(data)
    
    img_corr = plot_correlation(price_json)

    full_prompt = build_prompt(
        user_text,
        feats,
        price_json,
        corr_dict,
        fx_analysis
    ) 
    response_text = ChatAssistant(
        api_key=config.api_key,
        prompt=full_prompt, 
        provider=config.provider,
        base_url=config.base_url,
        model=config.model,
        max_tokens=config.max_tokens
    ).start() + "\nBelow are technical charts that provide further analytical context."

    return response_text, img_asset, img_corr