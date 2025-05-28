# Smart-Beta AI Portfolio App - גרסה מלאה כולל כפתור "בנה לי תיק עכשיו" וחיזוי לפי פרופיל משתמש

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

# תרגום דו-לשוני
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
        'run_predictive': 'הפעל מודל חיזוי חכם',
        'build_my_portfolio': 'בנה לי תיק עכשיו',
        'select_risk': 'רמת סיכון:',
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
        'run_predictive': 'Run Predictive Model',
        'build_my_portfolio': 'Build My Portfolio Now',
        'select_risk': 'Risk Level:',
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

# קלטים מהמשתמש
market = st.sidebar.selectbox(T['select_market'], ["S&P 500", "ת\"א 125"])
start_date = st.sidebar.date_input(T['start_date'], datetime.today() - timedelta(days=365))
end_date = st.sidebar.date_input(T['end_date'], datetime.today())
top_n = st.sidebar.slider(T['num_stocks'], 5, 30, 10)
risk_level = st.sidebar.selectbox(T['select_risk'], ["Low", "Medium", "High"])

# כפתור "בנה לי תיק עכשיו"
if st.sidebar.button(T['build_my_portfolio']):
    st.subheader(T['recommended'])
    st.info(f"🔍 יצירת תיק לפי רמת סיכון: {risk_level}")
    # כאן תוכל להכניס לוגיקה מתקדמת בהמשך לבחירת מניות לפי פרופיל סיכון
    # לדוגמה:
    weights = {"Low": 0.2, "Medium": 0.5, "High": 0.8}
    risk_weight = weights.get(risk_level, 0.5)
    st.write(f"💡 מודל נבנה על בסיס משקל סיכון: {risk_weight}")
    st.warning("⚙️ פונקציית בניית התיק החכם תורחב בהמשך עם אלגוריתם פרסונליזציה")

market = st.sidebar.selectbox(T['select_market'], ["S&P 500", "ת\"א 125"])
start_date = st.sidebar.date_input(T['start_date'], datetime.today() - timedelta(days=365))
end_date = st.sidebar.date_input(T['end_date'], datetime.today())
top_n = st.sidebar.slider(T['num_stocks'], 5, 30, 10)
risk_level = st.sidebar.radio(T['risk_level'], ['נמוכה', 'בינונית', 'גבוהה'] if language == 'he' else ['Low', 'Medium', 'High'])

@st.cache_data(show_spinner=False)
def load_ta125_static():
    return pd.read_csv("TA125_valid.csv")

@st.cache_data(show_spinner=False)
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
            data.append({"Ticker": symbol, "Name": name, "Return": returns, "Volatility": vol, "Volume": volume, "Sector": sector})
        except:
            continue
    return pd.DataFrame(data)

def run_predictive_model(df):
    df = df.copy()
    df["LogVolume"] = np.log(df["Volume"] + 1)
    df = df.dropna()
    X = df[["Return", "Volatility", "LogVolume"]]
    y = (df["Return"] > 0.1).astype(int)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
    model = xgb.XGBClassifier(eval_metric="logloss")
    model.fit(X_train, y_train)
    df["Prediction"] = model.predict(X_scaled)
    df["Signal"] = np.where(df["Prediction"] == 1, "Buy", "Hold")
    return df

if st.button(T['run_predictive']):
    with st.spinner(T['loading']):
        df_meta = load_ta125_static() if market == "ת\"א 125" else load_sp500_online()
        symbols = df_meta["Symbol"].tolist()[:top_n * 2]
        df = fetch_factors(symbols, df_meta)
        if df.empty:
            st.warning("⚠️ לא נמצאו נתונים. נסה לבחור שוק או סקטור אחר.")
            st.stop()
        df = run_predictive_model(df)
        st.dataframe(df[["Ticker", "Return", "Volatility", "Volume", "Prediction", "Signal"]], use_container_width=True)

st.markdown("---")
st.caption(T['footer'])
