# Smart-Beta AI Portfolio App - גרסה מעודכנת עם Reinforcement Learning

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

experience_buffer = []

def store_experience(features, reward):
    experience_buffer.append((features, reward))
    if len(experience_buffer) > 1000:
        experience_buffer.pop(0)

def calculate_reward(portfolio_returns):
    return np.mean(portfolio_returns) - np.std(portfolio_returns)

def update_factor_weights():
    if not experience_buffer:
        return [0.4, -0.3, 0.2, 0.1]
    returns = np.array([reward for _, reward in experience_buffer])
    factors = np.array([features for features, _ in experience_buffer])
    avg_impact = np.mean(factors.T * returns, axis=1)
    weights = avg_impact / np.sum(np.abs(avg_impact))
    return weights

# המשך הקוד - היכן שמתבצע חישוב Score:
# weights = update_factor_weights()
# df["Score"] = df[["Return", "Volatility", "Volume", "Sentiment"]].apply(
#     lambda row: weights[0]*row["Return"] + weights[1]*row["Volatility"] + weights[2]*np.log(row["Volume"]+1) + weights[3]*row["Sentiment"], axis=1)

# ובסיום הסימולציה:
# reward = calculate_reward(df_back["Portfolio"].pct_change().dropna())
# store_experience(df_top[["Return", "Volatility", "Volume", "Sentiment"]].mean().values, reward)

# (שים לב לעדכן את כל הקריאות לקוד בהתאם!)
