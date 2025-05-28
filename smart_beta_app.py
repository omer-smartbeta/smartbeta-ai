# Smart-Beta AI Portfolio App - גרסה מעודכנת עם טעינת Google Trends

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
from pytrends.request import TrendReq
import tempfile
import os

st.set_page_config(page_title="Smart-Beta AI Portfolio", layout="wide")

st.image("banner.png", use_container_width=True)

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
        'footer': 'דשבורד תיק השקעות חכם מבוסס AI - גרסה מלאה'
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
        'footer': 'Smart-Beta AI Portfolio Dashboard - Full Version'
    }
}

language = st.sidebar.selectbox('בחר שפה / Select Language', ['he', 'en'])
T = translations[language]

st.title(T['title'])
st.markdown(T['subtitle'])

market = st.sidebar.selectbox(T['select_market'], ["S&P 500", "ת\"א 125"])
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

def validate_symbols(symbols):
    valid = []
    for s in symbols:
        try:
            data = get_ticker_data(s)
            if not data.empty:
                valid.append(s)
        except:
            continue
    return valid

def get_sentiment_score(name):
    pos = ["עלייה", "חיובי", "המלצה"]
    neg = ["ירידה", "הפסד", "שלילי"]
    try:
        txt = requests.get(f"https://news.google.com/rss/search?q={name}").text
        return sum(txt.count(k) for k in pos) - sum(txt.count(n) for n in neg)
    except:
        return 0

def get_trend_score(keyword):
    try:
        pytrends = TrendReq(hl='en-US', tz=360)
        pytrends.build_payload([keyword], cat=0, timeframe='now 7-d')
        data = pytrends.interest_over_time()
        return int(data[keyword].mean()) if not data.empty else 0
    except:
        return 0

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
            sentiment = get_sentiment_score(name)
            trend = get_trend_score(name)
            score = 0.35 * returns - 0.25 * vol + 0.15 * np.log(volume + 1) + 0.1 * sentiment + 0.15 * trend
            data.append({
                "Ticker": symbol,
                "Name": name,
                "Return": round(returns, 3),
                "Volatility": round(vol, 3),
                "Volume": int(volume),
                "Sector": sector,
                "Sentiment": sentiment,
                "Trend": trend,
                "Score": score
            })
        except:
            continue
    return pd.DataFrame(data)

# שאר הפונקציות (convert_df_to_excel, simulate_backtest, create_pdf_report וכו') יישארו זהים
# יש להוסיף בחלק הדשבורד: גרף חדש להצגת Trend

# בתוך בלוק הפעלת המודל אחרי df_top = ...
# הוסף את השורה הבאה:
st.subheader("Google Trends (Last 7 Days)")
st.bar_chart(df_top.set_index("Ticker")["Trend"])

# 👈 כעת המודל תומך גם בנתוני מגמות חיפוש בזמן אמת (Google Trends)
