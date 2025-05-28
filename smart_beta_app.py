# Smart-Beta AI Portfolio App - גרסה מלאה כולל חיזוי, דשבורד, ניתוח סיכונים ותשואות

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
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Smart-Beta AI Portfolio", layout="wide")

# Banner
st.image("banner.png", use_container_width=True)

# --- Translation ---
translations = {
    'he': {
        'title': 'תיק השקעות חכם מבוסס AI',
        'subtitle': 'בחר שוק, סקטור, רמת סיכון ופרמטרים נוספים להרצת מודל תיק מניות חכם',
        'select_market': 'בחר שוק:',
        'select_sector': 'בחר סקטור (אופציונלי):',
        'select_risk': 'בחר רמת סיכון:',
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
        'risk_return': 'סיכון מול תשואה',
        'export_excel': 'הורד Excel',
        'export_pdf': 'הורד PDF',
        'footer': 'דשבורד תיק השקעות חכם מבוסס AI - גרסה מלאה'
    },
    'en': {
        'title': 'AI-Powered Smart-Beta Portfolio',
        'subtitle': 'Choose market, sector, risk level and filters to run the AI-based portfolio model',
        'select_market': 'Select Market:',
        'select_sector': 'Filter by Sector (optional):',
        'select_risk': 'Select Risk Level:',
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
        'risk_return': 'Risk vs Return',
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
risk_level = st.sidebar.selectbox(T['select_risk'], ["Low", "Medium", "High"])

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
    return pd.DataFrame(data)

def run_predictive_model(df):
    st.subheader("📈 " + T['recommended'])
    df = df.copy()
    df["LogVolume"] = np.log(df["Volume"] + 1)
    df = df.dropna()
    X = df[["Return", "Volatility", "LogVolume"]]
    y = (df["Return"] > 0.1).astype(int)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(eval_metric="logloss")
    model.fit(X_train, y_train)
    df["Prediction"] = model.predict(X_scaled)
    df["Score"] = model.predict_proba(X_scaled)[:, 1]
    df["Signal"] = np.where(df["Prediction"] == 1, "Buy", "Hold")

    # סינון לפי רמת סיכון
    if risk_level == "Low":
        df = df[df["Volatility"] < 0.2]
    elif risk_level == "High":
        df = df[df["Volatility"] > 0.4]

    df = df.sort_values("Score", ascending=False).head(top_n)
    df["Weight"] = round(1 / top_n, 3)

    st.dataframe(df, use_container_width=True)

    st.subheader(T['distribution'])
    st.bar_chart(df.set_index("Ticker")["Weight"])

    st.subheader(T['sector_pie'])
    fig1, ax1 = plt.subplots()
    df["Sector"].value_counts().plot.pie(autopct='%1.1f%%', ax=ax1)
    ax1.set_ylabel("")
    st.pyplot(fig1)

    st.subheader(T['risk_return'])
    fig2, ax2 = plt.subplots()
    ax2.scatter(df["Volatility"], df["Return"], c=df["Score"], cmap='viridis')
    ax2.set_xlabel("Volatility")
    ax2.set_ylabel("Return")
    st.pyplot(fig2)

if st.button(T['run_predictive']):
    with st.spinner(T['loading']):
        df_meta = load_ta125_static() if market == "ת\"א 125" else load_sp500_online()
        sector_filter = st.sidebar.selectbox(T['select_sector'], [""] + sorted(df_meta["Sector"].dropna().unique()))
        if sector_filter:
            df_meta = df_meta[df_meta["Sector"] == sector_filter]
        symbols = df_meta["Symbol"].tolist()[:top_n * 2]
        df = fetch_factors(symbols, df_meta)
        if df.empty:
            st.warning("⚠️ לא נמצאו נתונים. נסה לבחור שוק או סקטור אחר.")
            st.stop()
        else:
            run_predictive_model(df)

st.markdown("---")
st.caption(T['footer'])
