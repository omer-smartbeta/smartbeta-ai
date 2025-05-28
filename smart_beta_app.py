# Smart-Beta AI Portfolio App - גרסה מתקדמת עם ניהול היסטוריית תיקים והשוואת ביצועים

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
import uuid
import json

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
        'footer': 'דשבורד תיק השקעות חכם מבוסס AI - גרסה מלאה',
        'history': 'היסטוריית תיקים והשוואות',
        'compare': 'השווה שני תיקים',
        'select_portfolio_1': 'בחר תיק ראשון',
        'select_portfolio_2': 'בחר תיק שני'
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
        'footer': 'Smart-Beta AI Portfolio Dashboard - Full Version',
        'history': 'Portfolio History & Comparisons',
        'compare': 'Compare Two Portfolios',
        'select_portfolio_1': 'Select First Portfolio',
        'select_portfolio_2': 'Select Second Portfolio'
    }
}

language = st.sidebar.selectbox('בחר שפה / Select Language', ['he', 'en'])
T = translations[language]

st.title(T['title'])
st.markdown(T['subtitle'])

# כאן תוכל להכניס את קוד המודל הרגיל שלך כפי שהיה קודם...

# ------------------- פונקציות חדשות -------------------

def save_portfolio_to_file(df_top, stats, user_profile):
    portfolio_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data = {
        "id": portfolio_id,
        "timestamp": timestamp,
        "profile": user_profile,
        "stats": stats,
        "portfolio": df_top.to_dict(orient="records")
    }
    with open(f"history_{portfolio_id}.json", "w", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return portfolio_id

def load_saved_portfolios():
    files = [f for f in os.listdir('.') if f.startswith("history_") and f.endswith(".json")]
    portfolios = []
    for file in files:
        with open(file, encoding='utf-8') as f:
            portfolios.append(json.load(f))
    return portfolios

def compare_two_portfolios(p1, p2):
    df1 = pd.DataFrame(p1['portfolio'])
    df2 = pd.DataFrame(p2['portfolio'])
    st.subheader(f"{T['compare']}")
    st.write(f"📅 {p1['timestamp']} vs {p2['timestamp']}")
    st.dataframe(pd.concat([df1["Ticker"], df1[["Score", "Weight"]].add_suffix("_1"), df2[["Score", "Weight"]].add_suffix("_2")], axis=1))

# ------------------- ממשק היסטוריה -------------------

if "💼 " + T['history'] in st.sidebar.radio("Menu", ["📊 " + T['run_model'], "💼 " + T['history']]):
    portfolios = load_saved_portfolios()
    if len(portfolios) >= 2:
        p1 = st.selectbox(T['select_portfolio_1'], portfolios, format_func=lambda p: p['timestamp'])
        p2 = st.selectbox(T['select_portfolio_2'], portfolios, format_func=lambda p: p['timestamp'])
        compare_two_portfolios(p1, p2)
    elif portfolios:
        st.info("נשמר תיק אחד בלבד. השוואה תתאפשר לאחר שמירת תיק נוסף.")
    else:
        st.warning("אין תיקים שמורים. הרץ את המודל ושמור תיק.")
