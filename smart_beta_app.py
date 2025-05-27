# Smart-Beta AI Portfolio App - גרסה מעודכנת עם טעינה מ-Google Sheets וטיקרים תקינים בלבד

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

# --- טוען קובץ מקומי שנבדק מראש (טיקרים תקינים בלבד) ---
@st.cache_data
def load_ta125_static():
    return pd.read_csv("TA125_valid.csv")

# --- טוען S&P 500 מתוך ויקיפדיה ---
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

def get_sentiment_score(name):
    pos = ["עלייה", "חיובי", "המלצה"]
    neg = ["ירידה", "הפסד", "שלילי"]
    try:
        txt = requests.get(f"https://news.google.com/rss/search?q={name}").text
        return sum(txt.count(k) for k in pos) - sum(txt.count(n) for n in neg)
    except:
        return 0

def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

def simulate_backtest(tickers, benchmark_symbol="SPY"):
    portfolio = []
    for t in tickers:
        try:
            hist = yf.Ticker(t).history(period="6mo")
            if hist.empty or len(hist) < 10:
                continue
            hist["Norm"] = hist["Close"] / hist["Close"].iloc[0]
            portfolio.append(hist["Norm"])
        except:
            continue
    if not portfolio: return None
    df_back = pd.concat(portfolio, axis=1)
    df_back.columns = tickers
    df_back["Portfolio"] = df_back.mean(axis=1)
    df_back["Daily Return"] = df_back["Portfolio"].pct_change()
    try:
        bench = yf.Ticker(benchmark_symbol).history(period="6mo")
        bench["Benchmark"] = bench["Close"] / bench["Close"].iloc[0]
        df_back["Benchmark"] = bench["Benchmark"]
    except:
        df_back["Benchmark"] = np.nan
    return df_back

def create_pdf_report(df_top, metrics):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, height - 50, "Smart-Beta Portfolio Report")
    c.setFont("Helvetica", 12)
    c.drawString(40, height - 80, f"Sharpe Ratio: {metrics['sharpe']:.2f}")
    c.drawString(40, height - 100, f"Max Drawdown: {metrics['drawdown']:.2%}")
    c.drawString(40, height - 120, f"Cumulative Return: {metrics['return']:.2%}")
    data = [["Ticker", "Weight", "Sector", "Signal"]]
    for _, row in df_top.iterrows():
        data.append([row["Ticker"], f"{row['Weight']:.2f}", row["Sector"], row["Signal"]])
    t = Table(data, colWidths=[100, 80, 150, 80])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("GRID", (0, 0), (-1, -1), 1, colors.black)]))
    t.wrapOn(c, width, height)
    t.drawOn(c, 40, height - 320)
    c.save()
    return buffer.getvalue()

# --- הפעלת המודל ---
if st.button(T['run_model']):
    with st.spinner(T['loading']):
        df_meta = load_ta125_static() if market == "ת\"א 125" else load_sp500_online()
        sector_filter = st.sidebar.selectbox(T['select_sector'], [""] + sorted(df_meta["Sector"].dropna().unique()))
        if sector_filter:
            df_meta = df_meta[df_meta["Sector"] == sector_filter]
        symbols = validate_symbols(df_meta["Symbol"].tolist())
        df = fetch_factors(symbols, df_meta)
        if df.empty:
            st.error("לא נמצאו נתונים")
        else:
            df["Sentiment"] = df["Name"].apply(get_sentiment_score)
            df["Score"] = df[["Return", "Volatility", "Volume", "Sentiment"]].apply(
                lambda row: 0.4 * row["Return"] - 0.3 * row["Volatility"] + 0.2 * np.log(row["Volume"] + 1) + 0.1 * row["Sentiment"], axis=1)
            df_top = df.sort_values("Score", ascending=False).head(top_n)
            df_top["Weight"] = round(1 / top_n, 3)
            df_top["Signal"] = np.where(df_top["Score"] > 0.5, "Buy" if language == "en" else "קניה", "Hold" if language == "en" else "החזק")
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
            st.subheader(T['sentiment_chart'])
            st.bar_chart(df_top.set_index("Ticker")["Sentiment"])
            st.subheader(T['returns_hist'])
            fig2, ax2 = plt.subplots()
            ax2.hist(df_top["Return"], bins=8)
            ax2.set_xlabel("Return")
            st.pyplot(fig2)
            st.subheader(T['signals'])
            st.table(df_top[["Ticker", "Signal"]].set_index("Ticker"))
            benchmark_symbol = "SPY" if market == "S&P 500" else "ICL.TA"
            df_back = simulate_backtest(df_top["Ticker"].tolist(), benchmark_symbol)
            if df_back is not None:
                st.line_chart(df_back[["Portfolio", "Benchmark"]])
                sharpe = df_back["Daily Return"].mean() / df_back["Daily Return"].std() * np.sqrt(252)
                drawdown = (df_back["Portfolio"].cummax() - df_back["Portfolio"]).max()
                cum_return = df_back["Portfolio"].iloc[-1] - 1
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                st.metric("Max Drawdown", f"{drawdown:.2%}")
                st.metric("Cumulative Return", f"{cum_return:.2%}")
                pdf = create_pdf_report(df_top, {
                    "sharpe": sharpe,
                    "drawdown": drawdown,
                    "return": cum_return
                })
                st.download_button(T['export_pdf'], data=pdf, file_name="smart_beta_report.pdf", mime="application/pdf")
            excel = convert_df_to_excel(df_top)
            st.download_button(T['export_excel'], data=excel, file_name="portfolio.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("---")
st.caption(T['footer'])
