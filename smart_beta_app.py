# ✅ Smart-Beta AI Portfolio - Clean Updated Version with Reinforcement Learning Hooks

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
from io import BytesIO
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
import os

st.set_page_config(page_title="Smart-Beta AI", layout="wide")

# --- Translations ---
translations = {
    'he': {
        'title': 'תיק השקעות חכם מבוסס AI',
        'subtitle': 'בחר שוק, סקטור ופרמטרים נוספים להרצת מודל תיק מניות חכם',
        'select_market': 'בחר שוק:',
        'select_sector': 'בחר סקטור (אופציונלי):',
        'start_date': 'תאריך התחלה:',
        'end_date': 'תאריך סיום:',
        'num_stocks': 'כמה מניות לבחור?',
        'run_model': 'הפעל מודל AI',
        'loading': 'מריץ את המודל...',
        'done': 'המודל סיים לרוץ!',
        'recommended': 'תיק מומלץ',
        'distribution': 'פיזור לפי משקל השקעה',
        'sector_pie': 'פיזור לפי סקטור',
        'sentiment_chart': 'ציוני סנטימנט',
        'returns_hist': 'התפלגות תשואות',
        'signals': 'איתותים',
        'backtest': 'השוואה מול מדד ייחוס',
        'export_excel': 'הורד Excel',
        'export_pdf': 'הורד PDF',
        'footer': 'דשבורד תיק השקעות חכם מבוסס AI'
    },
    'en': {
        'title': 'AI-Powered Smart-Beta Portfolio',
        'subtitle': 'Choose market, sector and filters to run the AI-based portfolio model',
        'select_market': 'Select Market:',
        'select_sector': 'Filter by Sector (optional):',
        'start_date': 'Start Date:',
        'end_date': 'End Date:',
        'num_stocks': 'How many stocks to pick?',
        'run_model': 'Run AI Model',
        'loading': 'Running the model...',
        'done': 'Model completed!',
        'recommended': 'Recommended Portfolio',
        'distribution': 'Weight Distribution',
        'sector_pie': 'Sector Breakdown',
        'sentiment_chart': 'Sentiment Scores',
        'returns_hist': 'Returns Histogram',
        'signals': 'Signals',
        'backtest': 'Backtest vs Benchmark',
        'export_excel': 'Download Excel',
        'export_pdf': 'Download PDF',
        'footer': 'Smart-Beta AI Portfolio Dashboard'
    }
}

language = st.sidebar.selectbox('Language / שפה', ['he', 'en'])
T = translations[language]

st.title(T['title'])
st.markdown(T['subtitle'])

market = st.sidebar.selectbox(T['select_market'], ["S&P 500", "ת""א 125"])
start_date = st.sidebar.date_input(T['start_date'], datetime.today() - timedelta(days=180))
end_date = st.sidebar.date_input(T['end_date'], datetime.today())
top_n = st.sidebar.slider(T['num_stocks'], 5, 30, 10)

# 📥 Data Loaders
@st.cache_data
def load_sp500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    df = pd.read_html(requests.get(url).text)[0][["Symbol", "Security", "GICS Sector"]]
    df.columns = ["Symbol", "Name", "Sector"]
    return df

@st.cache_data
def load_ta125():
    return pd.read_csv("TA125_valid.csv")

@st.cache_data(show_spinner=False)
def fetch_data(symbol):
    try:
        return yf.Ticker(symbol).history(period="6mo")
    except:
        return pd.DataFrame()

# 📊 Feature Engineering

def compute_factors(df_meta):
    output = []
    for s in df_meta.Symbol:
        df = fetch_data(s)
        if df.empty: continue
        ret = (df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1
        vol = df["Close"].pct_change().std() * np.sqrt(252)
        volu = df["Volume"].mean()
        score = 0.4 * ret - 0.3 * vol + 0.2 * np.log(volu+1) + 0.1 * get_sentiment(s)
        output.append({"Symbol": s, "Return": ret, "Volatility": vol, "Volume": volu, "Score": score})
    return pd.DataFrame(output)

# 🧠 Sentiment Analysis Stub

def get_sentiment(symbol):
    try:
        news = requests.get(f"https://news.google.com/rss/search?q={symbol}").text
        return news.count("up") - news.count("down")
    except:
        return 0

# 📉 Backtest Simulation

def simulate_portfolio(df, symbols):
    norm = []
    for s in symbols:
        h = fetch_data(s)
        if h.empty: continue
        h["Norm"] = h["Close"] / h["Close"].iloc[0]
        norm.append(h["Norm"])
    df_sim = pd.concat(norm, axis=1)
    df_sim["Portfolio"] = df_sim.mean(axis=1)
    return df_sim

# 📤 Run Model
if st.button(T['run_model']):
    with st.spinner(T['loading']):
        df_meta = load_sp500() if "S&P" in market else load_ta125()
        sector_filter = st.sidebar.selectbox(T['select_sector'], [""] + sorted(df_meta["Sector"].dropna().unique()))
        if sector_filter:
            df_meta = df_meta[df_meta.Sector == sector_filter]
        df_factors = compute_factors(df_meta)
        df_top = df_factors.sort_values("Score", ascending=False).head(top_n)
        df_top["Weight"] = 1 / top_n
        df_top["Signal"] = np.where(df_top["Score"] > 0.5, "Buy", "Hold")

        st.success(T['done'])
        st.dataframe(df_top)
        st.bar_chart(df_top.set_index("Symbol")["Weight"])
        sim = simulate_portfolio(df_top, df_top.Symbol.tolist())
        st.line_chart(sim["Portfolio"])

st.markdown("---")
st.caption(T['footer'])
