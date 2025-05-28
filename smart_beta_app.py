# ✅ Smart-Beta AI Full App (עם חיזוק, פרופיל אישי ומקורות Live) - 2025-05-28

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from io import BytesIO
import os
import json
import random

st.set_page_config(page_title="Smart-Beta AI", layout="wide")

# --- תרגום דו-לשוני ---
translations = {
    'he': {
        'title': 'תיק השקעות חכם מבוסס AI',
        'subtitle': 'בחר שוק, סקטור, פרופיל אישי והרץ מודל AI',
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
        'risk_level': 'רמת סיכון:',
        'preferred_sectors': 'סקטורים מועדפים (רב-בחירה):',
        'recommend_now': 'בנה לי תיק עכשיו'
    },
    'en': {
        'title': 'AI-Powered Smart-Beta Portfolio',
        'subtitle': 'Choose market, sector, personal profile and run the AI model',
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
        'risk_level': 'Risk Level:',
        'preferred_sectors': 'Preferred Sectors (multi-select):',
        'recommend_now': 'Build My Portfolio'
    }
}

language = st.sidebar.selectbox('בחר שפה / Select Language', ['he', 'en'])
T = translations[language]

st.title(T['title'])
st.markdown(T['subtitle'])

# --- פרופיל משתמש ---
risk_level = st.sidebar.select_slider(T['risk_level'], options=["Low", "Medium", "High"])
preferred_sectors = st.sidebar.multiselect(T['preferred_sectors'], ["Technology", "Health Care", "Financials", "Energy", "Industrials"])

# --- שוק, תאריכים ומספר מניות ---
market = st.sidebar.selectbox(T['select_market'], ["S&P 500", "ת\"א 125"])
start_date = st.sidebar.date_input(T['start_date'], datetime.today() - timedelta(days=365))
end_date = st.sidebar.date_input(T['end_date'], datetime.today())
top_n = st.sidebar.slider(T['num_stocks'], 5, 30, 10)

# --- פונקציות עזר ---
@st.cache_data
def load_sp500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url).text
    df = pd.read_html(html)[0][["Symbol", "Security", "GICS Sector"]]
    df.columns = ["Symbol", "Name", "Sector"]
    return df

@st.cache_data
def load_ta125():
    return pd.read_csv("TA125_valid.csv")

@st.cache_data(show_spinner=False)
def get_ticker_data(symbol):
    try:
        return yf.Ticker(symbol).history(period="6mo")
    except:
        return pd.DataFrame()

def get_sentiment_score(name):
    pos = ["עלייה", "חיובי", "המלצה"]
    neg = ["ירידה", "הפסד", "שלילי"]
    try:
        txt = requests.get(f"https://news.google.com/rss/search?q={name}").text
        return sum(txt.count(k) for k in pos) - sum(txt.count(n) for n in neg)
    except:
        return 0

def calculate_score(row):
    base = 0.4 * row["Return"] - 0.3 * row["Volatility"] + 0.2 * np.log(row["Volume"] + 1) + 0.1 * row["Sentiment"]
    if risk_level == "Low": base -= 0.2 * row["Volatility"]
    if risk_level == "High": base += 0.1 * row["Return"]
    return base

# --- לחצן הפעלת המודל ---
if st.button(T['recommend_now']):
    with st.spinner(T['loading']):
        df_meta = load_sp500() if market == "S&P 500" else load_ta125()
        if preferred_sectors:
            df_meta = df_meta[df_meta["Sector"].isin(preferred_sectors)]
        stocks = df_meta["Symbol"].tolist()
        data = []
        for s in stocks:
            hist = get_ticker_data(s)
            if hist.empty or len(hist) < 20: continue
            r = (hist["Close"].iloc[-1] / hist["Close"].iloc[0]) - 1
            v = hist["Close"].pct_change().std() * np.sqrt(252)
            vol = hist["Volume"].mean()
            name = df_meta[df_meta["Symbol"] == s]["Name"].values[0]
            sec = df_meta[df_meta["Symbol"] == s]["Sector"].values[0]
            sent = get_sentiment_score(name)
            data.append({"Ticker": s, "Name": name, "Return": r, "Volatility": v, "Volume": vol, "Sector": sec, "Sentiment": sent})
        df = pd.DataFrame(data)
        df["Score"] = df.apply(calculate_score, axis=1)
        df_top = df.sort_values("Score", ascending=False).head(top_n)
        df_top["Weight"] = round(1 / top_n, 3)
        df_top["Signal"] = np.where(df_top["Score"] > 0.5, "Buy", "Hold")
        st.success(T['done'])
        st.dataframe(df_top)
        st.subheader(T['distribution'])
        st.bar_chart(df_top.set_index("Ticker")["Weight"])
        st.subheader(T['sector_pie'])
        fig1, ax1 = plt.subplots()
        df_top["Sector"].value_counts().plot.pie(autopct='%1.1f%%', ax=ax1)
        st.pyplot(fig1)
        st.subheader(T['sentiment_chart'])
        st.bar_chart(df_top.set_index("Ticker")["Sentiment"])
        st.subheader(T['returns_hist'])
        fig2, ax2 = plt.subplots()
        ax2.hist(df_top["Return"], bins=8)
        st.pyplot(fig2)
        st.subheader(T['signals'])
        st.table(df_top[["Ticker", "Signal"]])
