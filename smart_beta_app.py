# Smart-Beta AI Portfolio App - כולל מודול חיזוי חכם עם XGBoost ו־דשבורד מלא

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
from io import BytesIO
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
import tempfile
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

st.set_page_config(page_title="Smart-Beta AI Portfolio", layout="wide")

st.image("banner.png", use_container_width=True)

# --- תרגום דו-לשוני ---
translations = {
    'he': {
        'title': 'תיק השקעות חכם מבוסס AI',
        'subtitle': 'בחר שוק, סקטור ופרמטרים נוספים להרצת מודל תיק מניות חכם',
        'select_market': 'בחר שוק:',
        'select_sector': 'בחר סקטור (אופציונלי):',
        'risk_level': 'בחר רמת סיכון:',
        'start_date': 'תאריך התחלה:',
        'end_date': 'תאריך סיום:',
        'num_stocks': 'כמה מניות לבחור?',
        'run_model': 'הפעל מודל AI',
        'run_predictive': 'הפעל מודל חיזוי חכם',
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
        'footer': 'דשבורד תיק השקעות חכם מבוסס AI - גרסה מלאה'
    },
    'en': {
        'title': 'AI-Powered Smart-Beta Portfolio',
        'subtitle': 'Choose market, sector and filters to run the AI-based portfolio model',
        'select_market': 'Select Market:',
        'select_sector': 'Filter by Sector (optional):',
        'risk_level': 'Select Risk Level:',
        'start_date': 'Start Date:',
        'end_date': 'End Date:',
        'num_stocks': 'How many stocks to pick?',
        'run_model': 'Run AI Model',
        'run_predictive': 'Run Predictive Model',
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
        'footer': 'Smart-Beta AI Portfolio Dashboard - Full Version'
    }
}

language = st.sidebar.selectbox('בחר שפה / Select Language', ['he', 'en'])
T = translations[language]

st.title(T['title'])
st.markdown(T['subtitle'])

market = st.sidebar.selectbox(T['select_market'], ["S&P 500", "ת\"א 125"])
risk_level = st.sidebar.selectbox(T['risk_level'], ["Low", "Medium", "High"])
start_date = st.sidebar.date_input(T['start_date'], datetime.today() - timedelta(days=365))
end_date = st.sidebar.date_input(T['end_date'], datetime.today())
top_n = st.sidebar.slider(T['num_stocks'], 5, 30, 10)

@st.cache_data
def load_ta125_static():
    return pd.read_csv("TA125_valid.csv")

@st.cache_data
def load_sp500_online(limit=100):
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url).text
    df = pd.read_html(html)[0][["Symbol", "Security", "GICS Sector"]]
    df.columns = ["Symbol", "Name", "Sector"]
    return df.head(limit)

@st.cache_data(show_spinner=False)
def get_ticker_data(symbol):
    try:
        return yf.Ticker(symbol).history(period="6mo")
    except:
        return pd.DataFrame()

def fetch_factors(symbols, df_meta):
    data = []
    for symbol in symbols:
        try:
            hist = get_ticker_data(symbol)
            if hist.empty: continue
            returns = (hist["Close"].iloc[-1] / hist["Close"].iloc[0]) - 1
            vol = hist["Close"].pct_change().std() * np.sqrt(252)
            volume = hist["Volume"].mean()
            name = yf.Ticker(symbol).info.get("shortName", symbol)
            sector = df_meta[df_meta["Symbol"] == symbol]["Sector"].values[0]
            data.append({
                "Ticker": symbol,
                "Name": name,
                "Return": round(returns, 3),
                "Volatility": round(vol, 3),
                "Volume": int(volume),
                "Sector": sector
            })
        except:
            continue
    df = pd.DataFrame(data)
    if risk_level == "Low":
        df = df[df["Volatility"] < 0.25]
    elif risk_level == "Medium":
        df = df[(df["Volatility"] >= 0.25) & (df["Volatility"] <= 0.5)]
    else:
        df = df[df["Volatility"] > 0.5]
    return df

if st.button(T['run_model']):
    with st.spinner(T['loading']):
        df_meta = load_ta125_static() if market == "ת\"א 125" else load_sp500_online()
        df = fetch_factors(df_meta["Symbol"].tolist(), df_meta)
        if df.empty:
            st.warning("⚠️ לא נמצאו נתונים. נסה לבחור שוק או סקטור אחר.")
            st.stop()
        df_top = df.sort_values("Return", ascending=False).head(top_n)
        df_top["Weight"] = round(1 / top_n, 3)
        st.success(T['done'])
        st.subheader(T['recommended'])
        st.dataframe(df_top, use_container_width=True)

        st.subheader(T['distribution'])
        st.bar_chart(df_top.set_index("Ticker")["Weight"])

        st.subheader(T['sector_pie'])
        fig1, ax1 = plt.subplots()
        df_top["Sector"].value_counts().plot.pie(autopct='%1.1f%%', ax=ax1)
        ax1.set_ylabel("")
        st.pyplot(fig1)

        st.subheader(T['returns_hist'])
        fig2, ax2 = plt.subplots()
        ax2.hist(df_top["Return"], bins=8)
        ax2.set_xlabel("Return")
        st.pyplot(fig2)

        st.subheader(T['signals'])
        df_top["Signal"] = np.where(df_top["Return"] > 0.05, "Buy", "Hold")
        st.table(df_top[["Ticker", "Signal"]])
