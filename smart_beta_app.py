# Smart-Beta AI Portfolio App - גרסה מעודכנת עם פרופיל משתמש אישי

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

# ⬇️ הוספת באנר גרפי בראש הדשבורד
st.image("banner.png", use_container_width=True)

# --- תרגום דו-לשוני ---
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
        'profile_title': 'הגדר את פרופיל ההשקעה שלך',
        'risk_level': 'רמת סיכון מועדפת',
        'preferred_market': 'שוק מועדף',
        'preferred_sector': 'סקטור מועדף',
        'recommend_me': 'המלץ לי תיק עכשיו',
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
        'profile_title': 'Set Your Investment Profile',
        'risk_level': 'Preferred Risk Level',
        'preferred_market': 'Preferred Market',
        'preferred_sector': 'Preferred Sector',
        'recommend_me': 'Recommend Portfolio Now',
        'footer': 'Smart-Beta AI Portfolio Dashboard - Full Version'
    }
}

language = st.sidebar.selectbox('בחר שפה / Select Language', ['he', 'en'])
T = translations[language]

st.title(T['title'])
st.markdown(T['subtitle'])

# --- הגדרת פרופיל משתמש אישי ---
st.sidebar.subheader(T['profile_title'])
if 'profile' not in st.session_state:
    st.session_state.profile = {
        'risk': 'בינונית',
        'market': 'S&P 500',
        'sector': ''
    }

st.session_state.profile['risk'] = st.sidebar.selectbox(T['risk_level'], ['נמוכה', 'בינונית', 'גבוהה'])
st.session_state.profile['market'] = st.sidebar.selectbox(T['preferred_market'], ['S&P 500', 'ת"א 125'])
st.session_state.profile['sector'] = st.sidebar.text_input(T['preferred_sector'], '')

if st.sidebar.button(T['recommend_me']):
    st.success(f"פרופיל נשמר: {st.session_state.profile}")

# המשך הקוד כאן... (ללא שינוי לקוד המודל)

st.markdown("---")
st.caption(T['footer'])
