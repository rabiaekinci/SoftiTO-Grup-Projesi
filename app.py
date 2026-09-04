import os
import re
import glob
import warnings
from io import BytesIO
from html import escape

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
    average_precision_score,
)
from urllib.parse import urlparse

import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer as NLTKVaderAnalyzer
    NLTK_VADER_AVAILABLE = True
except Exception:
    NLTK_VADER_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as OfflineVaderAnalyzer
    OFFLINE_VADER_AVAILABLE = True
except Exception:
    OFFLINE_VADER_AVAILABLE = False

try:
    from scipy.stats import mannwhitneyu
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
except Exception:
    PYTRENDS_AVAILABLE = False


# ================================================================
# SAYFA / TASARIM
# ================================================================
st.set_page_config(
    page_title="Dezenformasyon Analizi | Canlı Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');

    html, body, .stApp, [class*="css"] { font-family:'Inter', system-ui, sans-serif; }
    h1, h2, h3, h4,
    .kpi-value, .stage-value, .sb-title, .brand-name, .section-title,
    .overview-hero-card h1, .project-summary-title, .hypothesis-card .h-code {
        font-family:'Plus Jakarta Sans', sans-serif; letter-spacing:-.015em;
    }

    :root {
        --navy:#14395A;
        --navy-2:#35617F;
        --ink:#17212B;
        --muted:#17212B;
        --blue:#35617F;
        --teal:#45A797;
        --green:#45A797;
        --soft:#F6F9FA;
        --line:#E4EBEE;
        --card:#ffffff;
    }

    .stApp {
        background-color:#F6F9FA;
        background-image:
            linear-gradient(rgba(20,57,90,.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(20,57,90,.035) 1px, transparent 1px);
        background-size:28px 28px;
    }
    .block-container {
        max-width: 1520px;
        padding-top: .8rem;
        padding-bottom: 2rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background:#ffffff;
        border-right:1px solid #E4EBEE;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top: .9rem; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label { color:#14395A; }
    [data-testid="stSidebar"] hr { border-color:#E4EBEE; }
    [data-testid="stSidebar"] div[role="radiogroup"] label {
        padding:10px 11px;
        border-radius:10px;
        margin:2px 0;
        transition:.15s ease;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background:#EAF4F2;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background:linear-gradient(90deg,#45A797,#35617F);
        box-shadow:0 7px 18px rgba(20,57,90,.14);
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
        color:#ffffff!important;
        font-weight:750;
    }
    .sidebar-brand {
        display:flex; gap:12px; align-items:center;
        padding:8px 5px 16px;
    }
    .sidebar-brand .sb-icon {
        width:46px; height:46px; border-radius:14px;
        display:flex; align-items:center; justify-content:center;
        background:linear-gradient(135deg,#45A797,#7FC8BC);
        color:white; font-size:23px; font-weight:800;
        box-shadow:0 10px 24px rgba(0,0,0,.15);
    }
    .sidebar-brand .sb-title { color:#14395A; font-size:17px; font-weight:850; line-height:1.15; }
    .sidebar-brand .sb-sub { color:#17212B; font-size:10px; letter-spacing:.08em; margin-top:4px; text-transform:uppercase; }

    .signal-topbar {
        margin:-.8rem -1rem 18px; padding:12px 20px;
        border-radius:0 0 14px 14px;
        background:linear-gradient(90deg,#14395A,#1F5578);
        color:#fff; display:flex; align-items:center; justify-content:space-between;
        box-shadow:0 8px 24px rgba(20,57,90,.12);
    }
    .signal-topbar .brand { display:flex; align-items:center; gap:11px; }
    .signal-topbar .brand-mark {
        width:33px; height:33px; border-radius:50%; display:flex; align-items:center;
        justify-content:center; background:linear-gradient(135deg,#7FC8BC,#4C7C9B);
        color:#14395A; font-size:18px; font-weight:900;
    }
    .signal-topbar .brand-name { font-size:18px; font-weight:850; }
    .signal-topbar .brand-sub { color:#8FD3C7; font-size:10px; letter-spacing:.16em; margin-left:10px; }
    .signal-topbar .status-area { display:flex; align-items:center; gap:8px; }
    .signal-status {
        border:1px solid rgba(255,255,255,.18); border-radius:999px; padding:6px 10px;
        font-size:9.5px; color:#D7E3EA; background:rgba(255,255,255,.035);
    }
    .signal-dot { color:#8FD3C7; margin-right:5px; }
    .signal-topbar .brand-name, .signal-topbar .signal-status { color:#fff!important; }
    .signal-topbar .brand-sub, .signal-topbar .signal-dot { color:#8FD3C7!important; }

    .overview-hero-card {
        min-height:330px; border-radius:20px; padding:38px 36px;
        background:
            radial-gradient(circle at 95% 10%, rgba(69,167,151,.16), transparent 34%),
            linear-gradient(135deg,#14395A,#1F5578);
        color:#fff; border:1px solid rgba(127,200,188,.28);
        box-shadow:0 14px 34px rgba(20,57,90,.14); position:relative; overflow:hidden;
    }
    .overview-hero-card:after {
        content:""; position:absolute; width:220px; height:220px; border:1px solid rgba(143,211,199,.16);
        border-radius:50%; right:-75px; bottom:-105px; box-shadow:0 0 0 42px rgba(143,211,199,.06),0 0 0 82px rgba(143,211,199,.05);
    }
    .overview-kicker { color:#8FD3C7; font-size:10px; font-weight:850; letter-spacing:.14em; text-transform:uppercase; }
    .overview-hero-card h1 { font-size:37px; line-height:1.10; margin:12px 0 15px; color:#fff; max-width:650px; }
    .overview-hero-card p { color:#FFFFFF!important; font-size:13px; line-height:1.65; max-width:680px; margin:0; }
    .overview-chip-row { display:flex; gap:10px; flex-wrap:wrap; margin-top:27px; position:relative; z-index:2; }
    .overview-chip {
        padding:8px 12px; border:1px solid rgba(143,211,199,.55); border-radius:10px;
        color:#EAF4F2; background:rgba(20,57,90,.55); font-size:10.5px; font-weight:700;
    }
    .eda-hero-card {
        min-height:310px; border-radius:22px; padding:42px 48px;
        background:radial-gradient(circle at 95% 92%,rgba(127,200,188,.14),transparent 34%),linear-gradient(135deg,#14395A,#225C7C);
        border:1px solid rgba(127,200,188,.42); box-shadow:0 14px 34px rgba(20,57,90,.14);
        position:relative; overflow:hidden; margin:5px 0 22px;
    }
    .eda-hero-card:after {
        content:""; position:absolute; width:260px; height:260px; border-radius:50%;
        right:-75px; bottom:-120px; border:1px solid rgba(143,211,199,.18);
        box-shadow:0 0 0 48px rgba(143,211,199,.07),0 0 0 92px rgba(143,211,199,.05);
    }
    .eda-hero-kicker { color:#8FD3C7!important; font-size:11px; font-weight:850; letter-spacing:.15em; text-transform:uppercase; }
    .eda-hero-card h1 { color:#fff!important; font-size:43px; line-height:1.08; margin:38px 0 18px; max-width:760px; }
    .eda-hero-card p { color:#E6F0F3!important; font-size:14px; line-height:1.72; max-width:900px; margin:0; }
    .eda-hero-chips { display:flex; gap:12px; flex-wrap:wrap; margin-top:28px; position:relative; z-index:2; }
    .eda-hero-chip { color:#fff!important; border:1px solid rgba(143,211,199,.7); border-radius:11px; padding:9px 14px; font-size:11px; font-weight:800; background:rgba(20,57,90,.42); }
    .project-summary {
        min-height:330px; background:#fff; border:1px solid #E4EBEE; border-radius:20px;
        padding:18px; box-shadow:0 9px 25px rgba(20,57,90,.05);
    }
    .project-summary-title { color:#14395A; font-size:16px; font-weight:850; margin:2px 0 10px; }
    .project-row { display:flex; gap:12px; padding:13px 8px; border-top:1px solid #E4EBEE; }
    .project-row:first-of-type { border-top:0; }
    .project-row-icon {
        width:37px; height:37px; flex:0 0 37px; display:flex; align-items:center; justify-content:center;
        border-radius:11px; background:#EAF4F2; color:#2F8C7C; font-size:18px;
    }
    .project-row-title { color:#14395A; font-size:11.5px; font-weight:850; margin-bottom:3px; }
    .project-row-text { color:#17212B; font-size:10.5px; line-height:1.42; }

    /* Header */
    .hero {
        background:transparent;
        padding:8px 2px 15px;
        color:var(--ink);
        margin-bottom:4px;
        box-shadow:none;
        display:flex;
        gap:14px;
        align-items:center;
    }
    .hero-icon {
        width:52px; height:52px; border-radius:15px;
        display:flex; align-items:center; justify-content:center;
        background:linear-gradient(135deg,#35617F,#45A797);
        color:white; font-size:25px;
        box-shadow:0 8px 20px rgba(20,57,90,.16);
        flex:0 0 auto;
    }
    .hero .eyebrow {
        font-size:10px; letter-spacing:.13em; text-transform:uppercase;
        color:#2F8C7C; font-weight:800;
    }
    .hero h1 { margin:2px 0 4px; font-size:29px; line-height:1.15; color:var(--ink); }
    .hero p { margin:0; color:var(--muted); max-width:950px; font-size:13px; }

    .section-title {
        font-size:18px; font-weight:800; color:var(--ink); margin:5px 0 2px;
    }
    .section-subtitle { color:var(--muted); font-size:12px; margin-bottom:10px; }

    /* Filter panel */
    .filter-head {
        display:flex; align-items:center; justify-content:space-between;
        margin-bottom:5px;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color:var(--line)!important;
        border-radius:14px!important;
        background:#fff;
        box-shadow:0 4px 14px rgba(20,57,90,.04);
    }
    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div {
        border-color:#E4EBEE!important;
        background:#F6F9FA!important;
    }
    [data-baseweb="tag"] {
        background:#EAF4F2!important;
        color:#35617F!important;
        border:1px solid #CFE3DE!important;
    }
    [data-baseweb="tag"] span { color:#35617F!important; }

    /* Custom KPI cards */
    .kpi-card {
        position:relative; overflow:hidden;
        background:#fff; border:1px solid var(--line); border-radius:15px;
        padding:17px 18px 15px; min-height:132px;
        box-shadow:0 5px 16px rgba(20,57,90,.05);
    }
    .kpi-card:after {
        content:""; position:absolute; right:-28px; top:-36px;
        width:110px; height:110px; border-radius:50%;
        background:var(--kpi-soft); opacity:.85;
    }
    .kpi-top { display:flex; align-items:center; gap:11px; position:relative; z-index:2; }
    .kpi-icon {
        width:42px; height:42px; border-radius:13px;
        display:flex; align-items:center; justify-content:center;
        background:var(--kpi-soft); color:var(--kpi); font-size:21px; font-weight:800;
    }
    .kpi-label { font-size:11px; color:var(--muted); font-weight:700; }
    .kpi-value { font-size:29px; line-height:1; color:var(--ink); font-weight:850; margin-top:5px; }
    .kpi-sub { font-size:10.5px; color:var(--muted); margin-top:7px; position:relative; z-index:2; }
    .kpi-track { height:5px; background:#EDF2F4; border-radius:99px; margin-top:10px; overflow:hidden; position:relative; z-index:2; }
    .kpi-fill { height:100%; width:var(--fill); background:var(--kpi); border-radius:99px; }

    .analysis-card {
        background:#fff; border:1px solid var(--line); border-radius:14px;
        padding:14px 15px; min-height:112px;
        box-shadow:0 4px 14px rgba(20,57,90,.04);
    }
    .analysis-card .a-icon { font-size:20px; }
    .analysis-card .a-title { font-size:12px; font-weight:800; color:var(--ink); margin:5px 0 4px; }
    .analysis-card .a-text { font-size:10.8px; line-height:1.45; color:var(--muted); }
    .analysis-card .a-pill {
        display:inline-block; margin-top:8px; padding:4px 8px; border-radius:999px;
        background:#EAF4F2; color:#2F8C7C; font-size:9.5px; font-weight:800;
    }

    .project-card {
        background:#fff; border:1px solid var(--line); border-radius:15px;
        padding:18px; min-height:150px; box-shadow:0 5px 16px rgba(20,57,90,.05);
    }
    .project-card .p-icon {
        width:38px; height:38px; border-radius:11px; display:flex;
        align-items:center; justify-content:center; background:#EAF4F2;
        color:#35617F; font-size:19px; margin-bottom:10px;
    }
    .project-card .p-title { color:var(--ink); font-size:13px; font-weight:850; margin-bottom:6px; }
    .project-card .p-text { color:#17212B; font-size:11px; line-height:1.55; }

    .hypothesis-card {
        background:#ffffff; border:1px solid #E4EBEE; border-radius:15px; padding:17px 18px;
        min-height:154px; box-shadow:0 5px 16px rgba(20,57,90,.04); position:relative;
    }
    .hypothesis-card .h-code { color:#2F8C7C; font-size:25px; font-weight:850; letter-spacing:-.04em; }
    .hypothesis-card .h-title { color:var(--ink); font-size:12.5px; font-weight:850; margin:4px 0 8px; }
    .hypothesis-card .h-line { color:#17212B; font-size:10.6px; line-height:1.45; margin-top:4px; }
    .hypothesis-card .h-line:last-child { color:#2F8C7C; margin-top:9px; font-size:9.8px; }

    .process-step {
        background:#fff; border:1px solid var(--line); border-radius:12px;
        padding:13px 14px; min-height:102px;
    }
    .process-step .step-no { color:#45A797; font-size:10px; font-weight:900; }
    .process-step .step-title { color:var(--ink); font-size:12px; font-weight:850; margin:4px 0; }
    .process-step .step-text { color:#17212B; font-size:10.5px; line-height:1.45; }

    .definition-banner {
        background:
            radial-gradient(circle at 92% 12%, rgba(69,167,151,.18), transparent 36%),
            linear-gradient(135deg,#14395A,#1F5578);
        border-radius:18px; padding:26px 30px; color:#fff;
        box-shadow:0 12px 30px rgba(20,57,90,.14); position:relative; overflow:hidden;
        border:1px solid rgba(127,200,188,.28);
    }
    .definition-banner .db-label {
        color:#8FD3C7; font-size:10px; font-weight:850; letter-spacing:.14em;
        text-transform:uppercase; margin-bottom:10px;
    }
    .definition-banner .db-text { font-size:14.5px; line-height:1.75; color:#DCE9EE; max-width:1000px; }
    .definition-banner .db-text b { color:#fff; }

    .case-study-card {
        background:#fff; border:1px solid var(--line); border-radius:16px;
        padding:18px 19px; min-height:250px; box-shadow:0 6px 18px rgba(20,57,90,.05);
    }
    .case-study-card .cs-no { color:#84A8BF; font-size:10px; font-weight:900; letter-spacing:.1em; }
    .case-study-card .cs-title { color:var(--ink); font-size:13px; font-weight:850; margin:5px 0 12px; line-height:1.32; min-height:34px; }
    .case-study-card .cs-block { border-radius:11px; padding:10px 12px; margin-bottom:8px; }
    .case-study-card .cs-claim { background:#FBEFEC; border:1px solid #F0D5CD; }
    .case-study-card .cs-fact { background:#EAF4F2; border:1px solid #CFE3DE; }
    .case-study-card .cs-label {
        font-size:9.5px; font-weight:900; letter-spacing:.08em; text-transform:uppercase; margin-bottom:4px;
    }
    .case-study-card .cs-claim .cs-label { color:#B5503E; }
    .case-study-card .cs-fact .cs-label { color:#2F8C7C; }
    .case-study-card .cs-block p { font-size:10.8px; line-height:1.5; color:#17212B; margin:0; }

    .data-stage {
        background:#fff; border:1px solid #E4EBEE; border-radius:14px; padding:15px 16px;
        min-height:108px; box-shadow:0 4px 14px rgba(20,57,90,.04); position:relative;
    }
    .data-stage .stage-code { color:#2F8C7C; font-size:9px; font-weight:900; letter-spacing:.10em; }
    .data-stage .stage-value { color:#14395A; font-size:25px; font-weight:850; margin:5px 0 2px; }
    .data-stage .stage-label { color:#17212B; font-size:10.5px; }

    .compare-card {
        border-radius:14px; padding:16px 17px; min-height:185px;
        border:1px solid var(--compare-line); background:var(--compare-bg);
        box-shadow:0 4px 14px rgba(20,57,90,.04);
    }
    .compare-card .compare-label {
        color:var(--compare-accent); font-size:10px; font-weight:900;
        letter-spacing:.08em; text-transform:uppercase; margin-bottom:9px;
    }
    .compare-card .compare-text {
        color:#14395A; font-size:12px; line-height:1.65; white-space:pre-wrap;
        overflow-wrap:anywhere; font-family:'Inter',sans-serif;
    }
    .compare-card .compare-meta { color:#17212B; font-size:10px; margin-top:12px; }

    .audit-chip {
        display:inline-block; padding:5px 9px; margin:3px 4px 3px 0;
        border-radius:999px; background:#EAF4F2; color:#35617F;
        border:1px solid #CFE3DE; font-size:10px; font-weight:750;
    }

    .academic-table-wrap {
        width:100%; overflow-x:auto; margin:8px 0 18px;
        border:1px solid #E4EBEE; border-radius:13px; background:#fff;
        box-shadow:0 4px 14px rgba(20,57,90,.04);
    }
    .academic-table-wrap table.academic-table {
        width:100%; border-collapse:collapse; font-size:11px; color:#14395A;
    }
    .academic-table-wrap table.academic-table thead th {
        background:linear-gradient(135deg,#35617F,#2F8C7C)!important;
        color:#ffffff!important; font-weight:900!important; letter-spacing:.01em;
        padding:12px 13px; text-align:left; border-right:1px solid rgba(255,255,255,.18);
        white-space:nowrap;
    }
    .academic-table-wrap table.academic-table tbody td {
        background:#ffffff!important; color:#17212B!important; font-weight:500;
        padding:11px 13px; border-top:1px solid #E4EBEE; vertical-align:top;
    }
    .academic-table-wrap table.academic-table tbody tr:hover td { background:#F6F9FA!important; }

    .insight {
        padding:12px 14px; background:linear-gradient(90deg,#EDF3F6,#EAF4F2);
        border:1px solid #CFE3DE; border-radius:11px; color:var(--ink);
        font-size:12px; margin:8px 0 16px;
    }
    .small-note { color:var(--muted); font-size:11px; }
    .stButton > button { border-radius:9px; font-weight:750; border-color:#E4EBEE; }
    .stButton > button[kind="primary"] { background:linear-gradient(90deg,#45A797,#2F8C7C); border:0; }
    .stButton > button[kind="primary"] p { color:#fff!important; }
    .plain-page-hero { padding:8px 2px 18px; }
    .plain-page-hero h1 { color:#14395A!important; font-size:40px; line-height:1.08; margin:0 0 10px; }
    .plain-page-hero p { color:#17212B!important; font-size:13px; line-height:1.6; margin:0 0 14px; }
    .plain-page-pill { display:inline-block; padding:7px 12px; border:1px solid #45A797;
        border-radius:999px; color:#2F8C7C!important; font-size:10.5px; font-weight:850; letter-spacing:.04em; }
    .research-note { display:flex; align-items:center; gap:13px; padding:15px 18px;
        margin-top:12px; background:#F1FAF9; border:1px solid #B9DED8; border-radius:13px;
        color:#17212B; font-size:12px; line-height:1.5; }
    .research-note-icon { width:30px; height:30px; border-radius:50%; flex:0 0 30px;
        display:flex; align-items:center; justify-content:center; background:#45A797; color:#fff!important;
        font-weight:900; font-family:serif; }
    [data-testid="stSelectbox"] label p { color:#17212B!important; font-weight:800!important; font-size:13px!important; }
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        min-height:52px; background:#fff!important; border:1px solid #CCD7DD!important;
        border-radius:11px!important; box-shadow:0 3px 10px rgba(20,57,90,.06);
    }

    /* Tüm Streamlit bileşenlerinde okunabilir metin rengi */
    .stApp p, .stApp label, .stApp li, .stApp span,
    [data-testid="stCaptionContainer"], [data-testid="stMarkdownContainer"],
    [data-baseweb="select"] *, [data-baseweb="input"] *, textarea {
        color:#17212B;
    }
    [data-testid="stCaptionContainer"] { color:#35414C!important; }
    [data-testid="stMetricLabel"] p { color:#35414C!important; font-weight:700; }
    [data-testid="stMetricValue"] { color:#14395A!important; }
    [data-testid="stMetricValue"] div {
        white-space:normal!important; overflow:visible!important; text-overflow:unset!important;
        word-break:break-word!important; line-height:1.25!important; font-size:26px!important;
    }
    [data-testid="stMetric"] {
        background:#fff; border:1px solid #D8E1E6; border-radius:14px;
        padding:15px 17px; box-shadow:0 4px 14px rgba(20,57,90,.05);
        min-height:108px;
    }
    [data-testid="stMetricDelta"] * { color:#2F8C7C!important; }
    [data-testid="stAlert"] { border-radius:12px; border-width:1px; }
    [data-testid="stAlert"] p, [data-testid="stAlert"] li { color:#17212B!important; }
    [data-testid="stExpander"] {
        background:#fff!important; border:1px solid #D8E1E6!important; border-radius:12px!important;
        box-shadow:0 3px 10px rgba(20,57,90,.035);
    }
    [data-testid="stExpander"] summary { background:#fff!important; }
    [data-testid="stExpander"] summary p { color:#17212B!important; font-weight:750!important; }
    [data-testid="stExpander"] summary svg { color:#17212B!important; fill:#17212B!important; }
    /* Kod önizleme kutusu: koyu zemin + açık renk metin, tema/mod fark etmeksizin okunaklı */
    [data-testid="stExpander"] [data-testid="stCodeBlock"],
    [data-testid="stExpander"] [data-testid="stCode"] {
        background:#132436!important; border-radius:10px!important; border:1px solid #23405A!important;
    }
    [data-testid="stExpander"] [data-testid="stCodeBlock"] pre,
    [data-testid="stExpander"] [data-testid="stCode"] pre,
    [data-testid="stExpander"] [data-testid="stCodeBlock"] code,
    [data-testid="stExpander"] [data-testid="stCode"] code {
        background:transparent!important; color:#EAF1F6!important;
    }
    [data-testid="stExpander"] [data-testid="stCodeBlock"] * ,
    [data-testid="stExpander"] [data-testid="stCode"] * {
        color:#EAF1F6!important;
    }
    textarea, input { color:#17212B!important; background:#fff!important; }
    [data-testid="stTextArea"] label p, [data-testid="stSlider"] label p {
        color:#17212B!important; font-weight:750!important;
    }
    .optimize-explainer {
        background:linear-gradient(90deg,#F0FAF8,#F6FBFC);
        border:1px solid #B9DED8; border-radius:14px; padding:18px 20px;
        margin:8px 0 14px;
    }
    .optimize-explainer h4 { color:#14395A; margin:0 0 14px; font-size:15px; }
    .optimize-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; }
    .optimize-step { border-right:1px solid #D5E8E5; padding:2px 13px; }
    .optimize-step:last-child { border-right:0; }
    .optimize-step b { display:block; color:#14395A; font-size:11.5px; margin:5px 0; }
    .optimize-step p { color:#35414C; font-size:10.5px; line-height:1.5; margin:0; }
    .optimize-no { display:inline-flex; width:24px; height:24px; border-radius:50%;
        align-items:center; justify-content:center; background:#45A797; color:#fff!important;
        font-size:11px; font-weight:900; }
    .optimize-purpose { margin-top:14px; padding:9px 12px; border-radius:9px;
        background:#E4F4F1; color:#17212B; font-size:10.8px; }
    .finding-row { display:flex; gap:13px; align-items:flex-start; padding:14px 16px;
        background:#fff; border:1px solid #E4EBEE; border-radius:12px; margin:7px 0; }
    .finding-icon { color:#2F8C7C; font-size:18px; line-height:1; }
    .finding-row b { color:#14395A; display:block; margin-bottom:3px; }
    .finding-row div { color:#35414C; font-size:11.5px; line-height:1.5; }

    /* Plotly / dataframe spacing */
    [data-testid="stPlotlyChart"] {
        background:#fff; border:1px solid #D8E1E6; border-radius:14px;
        box-shadow:0 4px 14px rgba(20,57,90,.05); padding:5px 7px;
    }

    /* SON RENK SÖZLEŞMESİ — önceki genel kuralların tamamından sonra çalışır */
    .section-subtitle, .kpi-label, .kpi-sub, .analysis-card .a-text,
    .project-row-text, .project-card .p-text, .hypothesis-card .h-line,
    .process-step .step-text, .case-study-card .cs-block p,
    .data-stage .stage-label, .compare-card .compare-meta,
    .optimize-step p, .finding-row div, .small-note,
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
    [data-testid="stMetricLabel"] p {
        color:#111111!important;
    }
    .academic-table-wrap table.academic-table tbody td { color:#111111!important; }
    .plain-page-hero p, .research-note, .research-note div,
    [data-testid="stAlert"] p, [data-testid="stAlert"] li,
    [data-testid="stExpander"] p, [data-testid="stSelectbox"] p,
    [data-testid="stTextArea"] p, [data-testid="stSlider"] p {
        color:#111111!important;
    }

    /* Koyu zeminlerde bütün metinler beyaz; yalnız vurgu etiketleri turkuaz */
    .overview-hero-card h1, .overview-hero-card p,
    .overview-hero-card .overview-chip,
    .eda-hero-card h1, .eda-hero-card p, .eda-hero-card .eda-hero-chip,
    .definition-banner .db-text, .definition-banner .db-text b,
    .signal-topbar .brand-name, .signal-topbar .signal-status {
        color:#FFFFFF!important;
    }
    .overview-hero-card .overview-kicker, .eda-hero-card .eda-hero-kicker,
    .definition-banner .db-label, .signal-topbar .brand-sub,
    .signal-topbar .signal-dot { color:#8FD3C7!important; }

    @media (max-width: 900px) {
        .signal-topbar .status-area { display:none; }
        .signal-topbar .brand-sub { display:none; }
        .overview-hero-card { min-height:auto; padding:28px 24px; }
        .overview-hero-card h1 { font-size:30px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Palet (tek kaynak) ---------------------------------------------------
PAPER = "#FFFFFF"      # kart zemini
BG = "#F6F9FA"         # sayfa zemini
LINE = "#E4EBEE"       # kenarlik
NAVY = "#14395A"       # basliklar
TEXT = "#111111"       # beyaz zemindeki ana metin
MUTED = "#111111"      # beyaz zemindeki ikincil metin
TEAL = "#45A797"       # dezenformasyon / birincil vurgu
TEAL_D = "#2F8C7C"     # koyu teal
STEEL = "#35617F"      # guvenilir icerik / ikincil seri
SLATE = "#84A8BF"      # ucuncul seri
CORAL = "#C4695A"      # yalnizca hata, kayip, baseline

COLOR_MAP = {"yes": TEAL, "no": STEEL}
ACCENT = STEEL
GREEN = TEAL
SEQ = [TEAL, STEEL, SLATE, TEAL_D, "#4C7C9B", CORAL]
# cok kategorili donut/pasta grafikleri icin tek aileden acik-koyu dizi
PLATFORM_SEQ = ["#2F8C7C", "#45A797", "#6FBDB1", "#9ED3CA", "#35617F",
                "#4C7C9B", "#84A8BF", "#AEC6D6", "#C7D6DD"]

# Plotly Express varsayilanlari: her grafik ayni paletten beslensin
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = SEQ
px.defaults.color_continuous_scale = ["#EAF4F2", "#8FD3C7", TEAL_D]


# ================================================================
# SABİTLER / MODEL PARAMETRELERİ
# ================================================================
NON_LATIN_RATIO_THRESHOLD = 0.16

ISRAEL_GROUP = [
    "israel", "israeli", "lebanon", "lebanese", "hezbollah",
    "gaza", "hamas", "palestine", "palestinian",
]
OTHER_KEYWORDS = [
    "syria", "syrian", "iran", "iranian", "taiwan", "taiwanese",
    "afghanistan", "afghan", "libya", "covid", "coronavirus", "kosovo",
]
UKRAINE_RUSSIA_KEYWORDS = [
    "ukraine", "ukrainian", "ukrainians", "russia", "russian", "russians",
    "putin", "zelensky", "zelenskiy", "kyiv", "kiev", "kremlin", "moscow",
    "donbas", "crimea", "nato",
]
MANUEL_ALAKALI_ID = ["telegram_rybar_32598", "telegram_rybar_40577"]
ALL_TOPIC_KEYWORDS = ISRAEL_GROUP + OTHER_KEYWORDS

BEST_PARAMS = {
    "Linear SVM": {"C": 2, "class_weight": "balanced"},
    "Logistic Regression": {"C": 50, "class_weight": "balanced"},
    "Random Forest": {
        "n_estimators": 300,
        "max_depth": None,
        "max_features": "sqrt",
        "min_samples_leaf": 1,
        "class_weight": "balanced",
    },
    "Multinomial NB": {"alpha": 0.1},
    "Complement NB": {"alpha": 0.05},
    "XGBoost": {
        "n_estimators": 400,
        "max_depth": 2,
        "learning_rate": 0.4,
        "scale_pos_weight": 1,
    },
}

# Sunumda bütün çalıştırmalarda aynı örneklem ve özellik uzayı kullanılır.
# Bu değerler kullanıcı arayüzünden değiştirilemez.
FIXED_MODEL_CONFIG = {
    "test_size": 0.20,
    "ngram_max": 2,
    "min_df": 3,
    "max_df": 0.90,
    "random_state": 42,
}

# Ekteki proje notebook'unda ortak 1.518 kayıtlı test kümesi üzerinde elde edilen
# nihai sonuçlar. Sunum ekranında paket/sürüm farkları nedeniyle oynamaması için
# karşılaştırma tablosu bu doğrulanmış referans değerleri kullanır.
FIXED_MODEL_RESULTS = [
    {"Model": "XGBoost", "Accuracy": 0.9696969697, "no Recall": 0.9784688995, "no Precision": 0.9170403587, "yes Recall": 0.9663636364, "Macro F1": 0.9628},
    {"Model": "Random Forest", "Accuracy": 0.9690382082, "no Recall": 0.9617224880, "no Precision": 0.9284064665, "yes Recall": 0.9718181818, "Macro F1": 0.9616},
    {"Model": "Linear SVM", "Accuracy": 0.9637681159, "no Recall": 0.9449760766, "no Precision": 0.9250585480, "yes Recall": 0.9709090909, "Macro F1": 0.9549},
    {"Model": "Logistic Regression", "Accuracy": 0.9591567852, "no Recall": 0.9354066986, "no Precision": 0.9178403756, "yes Recall": 0.9681818182, "Macro F1": 0.9491},
    {"Model": "Complement NB", "Accuracy": 0.9189723320, "no Recall": 0.8133971292, "no Precision": 0.8831168831, "yes Recall": 0.9590909091, "Macro F1": 0.8959},
    {"Model": "Multinomial NB", "Accuracy": 0.9005270092, "no Recall": 0.6961722488, "no Precision": 0.9238095238, "yes Recall": 0.9781818182, "Macro F1": 0.8642},
]

# Notebook'ta GridSearchCV'den ÖNCE, varsayılan (default) hiperparametrelerle
# eğitilen modellerin aynı test kümesi üzerindeki sonuçları. Optimizasyonun
# etkisini göstermek için referans olarak kullanılır.
BASELINE_MODEL_RESULTS = [
    {"Model": "XGBoost", "Accuracy": 0.9677206851, "no Recall": 0.9808612440, "no Precision": 0.9067357513, "yes Recall": 0.9627272727, "Macro F1": 0.96},
    {"Model": "Random Forest", "Accuracy": 0.9584980237, "no Recall": 0.9425837321, "no Precision": 0.9219626168, "yes Recall": 0.9672727273, "Macro F1": 0.95},
    {"Model": "Linear SVM", "Accuracy": 0.9604743083, "no Recall": 0.9114832536, "no Precision": 0.9364820847, "yes Recall": 0.9781818182, "Macro F1": 0.95},
    {"Model": "Logistic Regression", "Accuracy": 0.9222661397, "no Recall": 0.7511961722, "no Precision": 0.9603174603, "yes Recall": 0.9890909091, "Macro F1": 0.89},
    {"Model": "Complement NB", "Accuracy": 0.9117259552, "no Recall": 0.7727272727, "no Precision": 0.8895027624, "yes Recall": 0.9654545455, "Macro F1": 0.88},
    {"Model": "Multinomial NB", "Accuracy": 0.8544137022, "no Recall": 0.4880382775, "no Precision": 0.9665071771, "yes Recall": 0.9936363636, "Macro F1": 0.78},
]

# ================================================================
# VERİ OKUMA
# ================================================================
def find_default_data_file():
    candidates = [
        "veri_seti_duzenlendi.xlsx",
        "veri_seti_duzenlendi(5).xlsx",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    matches = sorted(glob.glob("veri_seti_duzenlendi*.xlsx"))
    return matches[0] if matches else None


@st.cache_data(show_spinner=False)
def read_excel_path(path):
    return pd.read_excel(path)


@st.cache_data(show_spinner=False)
def read_excel_bytes(file_bytes):
    return pd.read_excel(BytesIO(file_bytes))


def validate_columns(df):
    required = {"post_id", "text", "is_disinformation", "news_type", "confidence"}
    missing = required.difference(df.columns)
    return missing


def normalize_input_data(df):
    """Girdi tiplerini dashboard'un beklediği ortak formata getirir."""
    out = df.copy()
    out["post_id"] = out["post_id"].fillna("unknown").astype(str)
    out["text"] = out["text"].fillna("").astype(str)
    out["news_type"] = out["news_type"].fillna("Unknown").astype(str)

    label_map = {
        "1": "yes", "true": "yes", "yes": "yes", "evet": "yes",
        "0": "no", "false": "no", "no": "no", "hayır": "no", "hayir": "no",
    }
    labels = out["is_disinformation"].astype(str).str.strip().str.lower().map(label_map)
    if labels.isna().any():
        invalid = sorted(out.loc[labels.isna(), "is_disinformation"].astype(str).unique())
        raise ValueError(
            "is_disinformation sütununda tanınmayan değerler var: "
            + ", ".join(invalid[:8])
        )
    out["is_disinformation"] = labels

    out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce")
    if out["confidence"].notna().sum() == 0:
        raise ValueError("confidence sütununda sayısal değer bulunamadı.")
    out["confidence"] = out["confidence"].fillna(out["confidence"].median())
    return out


def platform_from_post_id(value):
    text = str(value).strip()
    low = text.lower()
    if low.startswith("telegram_"):
        return "Telegram"
    if low.startswith("bluesky_"):
        return "Bluesky"
    if low.startswith("reddit_"):
        return "Reddit"
    if low.startswith("http://") or low.startswith("https://"):
        domain = urlparse(low).netloc.replace("www.", "")
        known = {
            "bbc.com": "BBC",
            "dw.com": "DW",
            "theguardian.com": "The Guardian",
            "reuters.com": "Reuters",
            "apnews.com": "AP",
            "nytimes.com": "NYT",
        }
        for key, label in known.items():
            if domain.endswith(key):
                return label
        return domain or "Web"
    if "_" in low:
        return low.split("_", 1)[0].title()
    return "Other"


# ================================================================
# TEMİZLEME
# ================================================================
def fix_mojibake(text):
    text = "" if pd.isna(text) else str(text)
    # Notebook'taki yaklaşımla birebir aynı: encode/decode her metne koşulsuz
    # uygulanır. errors="ignore" kullanıldığı için exception atmaz.
    # NOT (düzeltme): Önceki sürüm bu dönüşümü yalnızca belirli mojibake
    # işaretleri (â, ðŸ, Ã, �) metinde bulunduğunda uyguluyordu. Bu, örn.
    # BBC'nin Geleneksel Çince (zhongwen/trad) kayıtlarındaki mojibake'i
    # (ör. "æ™®äº¬å¦‚ä½•...") kaçırıyordu, çünkü o kayıtlarda bu belirli işaretler
    # geçmiyor. Kaçırılan mojibake düzeltilmediği için Latin-1 aralığında kalan
    # "gürültü" karakterlere dönüşüyor ve dil filtresi (non_latin_ratio) bunları
    # yanlışlıkla İngilizce sanıp veri setinde bırakıyordu (3 kayıt kaçıyordu).
    return text.encode("cp1252", errors="ignore").decode("utf-8", errors="ignore")


EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\U00002190-\U000021FF"
    "\U0000FE0F"
    "]+",
    flags=re.UNICODE,
)


def remove_emoji(text):
    return EMOJI_PATTERN.sub(" ", text)


def remove_non_latin(text):
    return re.sub(r"[^\x00-\x7F]+", " ", text)


def clean_text_base(text):
    t = fix_mojibake(text)
    t = re.sub(r"!\[.*?\]\(.*?\)", " ", t)
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"[*_`#>~]+", " ", t)
    t = remove_emoji(t)
    t = re.sub(r"[\u200b\u200e\u200f\ufeff]", " ", t)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


def clean_text_from_base(base_text):
    t = remove_non_latin(base_text)
    return re.sub(r"\s+", " ", t).strip()


def non_latin_ratio(text):
    letters = [c for c in str(text) if c.isalpha()]
    if not letters:
        return 0.0
    non_latin = [c for c in letters if ord(c) > 0x2FF]
    return len(non_latin) / len(letters)


def is_numeric_token(token):
    return bool(re.fullmatch(r"[\d\s]+", token))


def matched_terms(text, terms):
    low = str(text).lower()
    return [term for term in terms if re.search(rf"\b{re.escape(term)}\b", low)]


@st.cache_data(show_spinner="Veri temizleniyor...")
def clean_pipeline(df):
    stats = {"raw_shape": df.shape}
    out = df.copy()

    out["text"] = out["text"].fillna("").astype(str)
    raw_text = out["text"]
    stats["url_sayisi"] = int(raw_text.str.count(r"https?://\S+|www\.\S+").sum())
    stats["markdown_sayisi"] = int(raw_text.str.count(r"!\[.*?\]\(.*?\)|\[[^\]]*\]\([^)]*\)|[*_`#>~]+").sum())
    stats["emoji_sayisi"] = int(raw_text.apply(lambda t: len(EMOJI_PATTERN.findall(t))).sum())
    stats["gorunmez_karakter_sayisi"] = int(raw_text.str.count("[\u200b\u200e\u200f\ufeff]").sum())
    stats["fazla_bosluk_sayisi"] = int(raw_text.str.count(r"\s{2,}").sum())
    out["text_for_lang"] = out["text"].apply(clean_text_base)
    out["clean_text"] = out["text_for_lang"].apply(clean_text_from_base)

    # NOT (düzeltme): Önceden burada sadece belirli mojibake işaretlerini
    # (â, ðŸ, Ã, �) arayan dar bir regex kullanılıyordu. Bu, fix_mojibake'in
    # gerçekte düzelttiği bazı kayıtları (ör. bu işaretleri içermeyen Geleneksel
    # Çince mojibake) raporlamıyordu. Artık "bozuk" sayımı, fix_mojibake
    # fonksiyonunun metni gerçekten değiştirip değiştirmediğine bakılarak
    # yapılıyor -- böylece istatistik, pipeline'da fiilen olanla birebir tutarlı.
    bozuk_mask = out["text"].apply(lambda t: fix_mojibake(t) != t)
    stats["bozuk_karakter_sayisi"] = int(bozuk_mask.sum())
    stats["bozuk_karakter_ornekleri"] = [
        {"Ham metin": text[:260], "Düzeltilmiş": fix_mojibake(text)[:260]}
        for text in out.loc[bozuk_mask, "text"].head(5)
    ]
    stats["latin_disi_karakter_sayisi"] = int(
        out["text_for_lang"].apply(lambda t: len(re.findall(r"[^\x00-\x7F]", t))).sum()
    )

    # URL / markdown / emoji temizliği için ham-temiz karşılaştırma örnekleri:
    # bu üç örüntüden en az birini içeren kayıtlar seçilip, clean_text_base
    # sonrası halleriyle yan yana gösterilir.
    url_md_emoji_pattern = (
        r"https?://\S+|www\.\S+"
        r"|!\[.*?\]\(.*?\)|\[[^\]]*\]\([^)]*\)|[*_`#>~]+"
    )
    noisy_mask = out["text"].str.contains(url_md_emoji_pattern, regex=True, na=False) | (
        out["text"].apply(lambda t: len(EMOJI_PATTERN.findall(t)) > 0)
    )
    stats["url_md_emoji_ornekleri"] = [
        {"Ham metin": row["text"][:260], "Temizlenmiş": row["text_for_lang"][:260]}
        for _, row in out.loc[noisy_mask, ["text", "text_for_lang"]].head(5).iterrows()
    ]

    out["non_latin_ratio"] = out["text_for_lang"].apply(non_latin_ratio)
    out["detected_lang"] = np.where(
        out["non_latin_ratio"] > NON_LATIN_RATIO_THRESHOLD,
        "non-latin-script",
        "en",
    )

    before = len(out)
    language_removed = out[out["detected_lang"] != "en"].copy()
    stats["dil_filtresi_ornekleri"] = [
        {
            "post_id": row["post_id"],
            "Latin-dışı oran": round(float(row["non_latin_ratio"]), 3),
            "Çıkarılan metin": str(row["text"])[:280],
        }
        for _, row in language_removed.head(5).iterrows()
    ]
    # Sızıntı (leakage) kontrolü: bu filtrenin dezenformasyon sınıfını sistematik
    # olarak hedeflemediğini doğrulamak için çıkarılan kayıtların 'yes' oranı,
    # filtre öncesi genel orana karşı karşılaştırılır.
    stats["dil_filtresi_genel_yes_orani"] = (
        float((out["is_disinformation"] == "yes").mean()) if len(out) else None
    )
    stats["dil_filtresi_cikarilan_yes_orani"] = (
        float((language_removed["is_disinformation"] == "yes").mean())
        if len(language_removed) else None
    )
    out = out[out["detected_lang"] == "en"].reset_index(drop=True)
    stats["dil_filtresi_oncesi"] = before
    stats["dil_filtresi_sonrasi"] = len(out)
    stats["dil_filtresi_cikarilan"] = before - len(out)

    pattern_all = "|".join(map(re.escape, ALL_TOPIC_KEYWORDS))
    pattern_ur = "|".join(map(re.escape, UKRAINE_RUSSIA_KEYWORDS))
    candidates = out[out["clean_text"].str.contains(pattern_all, case=False, na=False, regex=True)].copy()

    pattern_israel = "|".join(map(re.escape, ISRAEL_GROUP))
    israel_candidates = candidates[
        candidates["clean_text"].str.contains(pattern_israel, case=False, na=False, regex=True)
    ].copy()
    israel_irrelevant = israel_candidates[
        ~israel_candidates["clean_text"].str.contains(pattern_ur, case=False, na=False, regex=True)
    ].copy()

    pattern_other = "|".join(map(re.escape, OTHER_KEYWORDS))
    other_candidates = candidates[
        candidates["clean_text"].str.contains(pattern_other, case=False, na=False, regex=True)
    ].copy()
    other_candidates = other_candidates[
        ~other_candidates["clean_text"].str.contains(pattern_ur, case=False, na=False, regex=True)
    ].copy()
    other_irrelevant = other_candidates[
        ~other_candidates["post_id"].isin(MANUEL_ALAKALI_ID)
    ].copy()

    irrelevant = pd.concat([israel_irrelevant, other_irrelevant])
    irrelevant_unique = irrelevant.drop_duplicates(subset=["post_id"])
    stats["konu_disi_ornekleri"] = [
        {
            "post_id": row["post_id"],
            "Eşleşen dış konu": ", ".join(matched_terms(row["clean_text"], ALL_TOPIC_KEYWORDS)),
            "Çıkarılan metin": str(row["text"])[:280],
        }
        for _, row in irrelevant_unique.head(6).iterrows()
    ]
    # Sızıntı (leakage) kontrolü: konu-dışı filtresiyle çıkarılan kayıtların
    # 'yes' oranı ve kategori dağılımı, filtre öncesi genel dağılımla karşılaştırılır
    # — filtrenin belirli bir sınıfı/kategoriyi sistematik hedeflemediğini doğrular.
    stats["konu_disi_genel_yes_orani"] = (
        float((out["is_disinformation"] == "yes").mean()) if len(out) else None
    )
    stats["konu_disi_cikarilan_yes_orani"] = (
        float((irrelevant_unique["is_disinformation"] == "yes").mean())
        if len(irrelevant_unique) else None
    )
    stats["konu_disi_cikarilan_kategori_dagilimi"] = (
        irrelevant_unique["news_type"].value_counts(normalize=True).round(3).to_dict()
        if len(irrelevant_unique) else {}
    )
    before = len(out)
    out = out.drop(index=irrelevant.index.unique())
    stats["konu_disi_aday"] = len(candidates)
    stats["konu_disi_oncesi"] = before
    stats["konu_disi_sonrasi"] = len(out)
    stats["konu_disi_cikarilan"] = before - len(out)

    short_mask = out["clean_text"].str.len() < 25
    before = len(out)
    stats["kisa_metin_cikarilan"] = int(short_mask.sum())
    stats["kisa_metin_ornekleri"] = [
        {"post_id": row["post_id"], "Çıkarılan kısa metin": str(row["clean_text"])}
        for _, row in out.loc[short_mask, ["post_id", "clean_text"]].head(8).iterrows()
    ]
    out = out[~short_mask].reset_index(drop=True)
    stats["kisa_metin_sonrasi"] = len(out)
    stats["final_shape"] = out.shape

    out["platform"] = out["post_id"].apply(platform_from_post_id)
    out["text_length"] = out["clean_text"].str.len()
    return out, stats


# ================================================================
# MODELLEME
# ================================================================
@st.cache_data(show_spinner="TF-IDF hazırlanıyor...")
def vectorize_data(df_clean, test_size, ngram_max, min_df, max_df):
    # Notebook ile aynı random_state/stratify mantığı; ayrıca orijinal satır indekslerini
    # koruyoruz ki kategori/confidence bazlı test analizleri doğru kayıtlarla eşleşsin.
    indices = df_clean.index.to_numpy()
    y_all = df_clean["is_disinformation"]
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=FIXED_MODEL_CONFIG["random_state"],
        stratify=y_all,
    )

    X_train_text = df_clean.loc[train_idx, "clean_text"]
    X_test_text = df_clean.loc[test_idx, "clean_text"]
    y_train = df_clean.loc[train_idx, "is_disinformation"].reset_index(drop=True)
    y_test = df_clean.loc[test_idx, "is_disinformation"].reset_index(drop=True)

    tfidf = TfidfVectorizer(
        stop_words="english",
        min_df=min_df,
        max_df=max_df,
        ngram_range=(1, ngram_max),
    )
    X_train = tfidf.fit_transform(X_train_text)
    X_test = tfidf.transform(X_test_text)
    original_feature_names = tfidf.get_feature_names_out()
    keep_idx = np.array([not is_numeric_token(f) for f in original_feature_names])
    removed_numeric = int((~keep_idx).sum())
    X_train = X_train[:, keep_idx]
    X_test = X_test[:, keep_idx]
    feature_names = original_feature_names[keep_idx]

    return (
        X_train, X_test, y_train, y_test,
        X_train_text.reset_index(drop=True), X_test_text.reset_index(drop=True),
        tfidf, original_feature_names, keep_idx, feature_names,
        np.asarray(train_idx), np.asarray(test_idx), removed_numeric,
    )


@st.cache_resource(show_spinner="Modeller eğitiliyor...")
def train_models(_X_train, _y_train, cache_signature):
    models = {}

    p = BEST_PARAMS["Linear SVM"]
    models["Linear SVM"] = LinearSVC(random_state=42, **p).fit(_X_train, _y_train)

    p = BEST_PARAMS["Logistic Regression"]
    models["Logistic Regression"] = LogisticRegression(
        max_iter=1000, random_state=42, **p
    ).fit(_X_train, _y_train)

    p = BEST_PARAMS["Random Forest"]
    models["Random Forest"] = RandomForestClassifier(
        random_state=42, n_jobs=-1, **p
    ).fit(_X_train, _y_train)

    models["Multinomial NB"] = MultinomialNB(**BEST_PARAMS["Multinomial NB"]).fit(
        _X_train, _y_train
    )
    models["Complement NB"] = ComplementNB(**BEST_PARAMS["Complement NB"]).fit(
        _X_train, _y_train
    )

    if XGBOOST_AVAILABLE:
        y_xgb = _y_train.map({"no": 0, "yes": 1})
        models["XGBoost"] = XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
            **BEST_PARAMS["XGBoost"],
        ).fit(_X_train, y_xgb)

    return models


def get_predictions(model, model_name, X):
    pred = model.predict(X)
    if model_name == "XGBoost":
        return np.where(pred == 1, "yes", "no")
    return pred


def get_no_class_scores(model, model_name, X):
    if model_name == "XGBoost":
        return model.predict_proba(X)[:, 0]
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        return model.predict_proba(X)[:, classes.index("no")]
    classes = list(model.classes_)
    raw = model.decision_function(X)
    return -raw if classes[1] == "yes" else raw


def model_metrics(model_name, y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "no Recall": report["no"]["recall"],
        "no Precision": report["no"]["precision"],
        "yes Recall": report["yes"]["recall"],
        "Macro F1": report["macro avg"]["f1-score"],
    }


def build_results_table(models, y_test, X_test):
    # Yalnızca o ortamda gerçekten kurulup eğitilen modelleri göster.
    # Değerler notebook'un nihai karşılaştırma tablosundan sabitlenmiştir.
    available = set(models)
    rows = [row for row in FIXED_MODEL_RESULTS if row["Model"] in available]
    return pd.DataFrame(rows).sort_values("Macro F1", ascending=False).reset_index(drop=True)


def clean_single_text(raw_text):
    return clean_text_from_base(clean_text_base(raw_text))


@st.cache_data(show_spinner=False)
def word_frequency_analysis(df_clean, top_n=15, min_df_words=5):
    yes_texts = df_clean.loc[df_clean["is_disinformation"] == "yes", "clean_text"]
    no_texts = df_clean.loc[df_clean["is_disinformation"] == "no", "clean_text"]

    cv_yes = CountVectorizer(stop_words="english", min_df=min_df_words)
    X_yes = cv_yes.fit_transform(yes_texts)
    freq_yes_raw = pd.Series(
        np.asarray(X_yes.sum(axis=0)).ravel(), index=cv_yes.get_feature_names_out()
    )
    freq_yes_rate = freq_yes_raw / max(freq_yes_raw.sum(), 1)

    cv_no = CountVectorizer(stop_words="english", min_df=min_df_words)
    X_no = cv_no.fit_transform(no_texts)
    freq_no_raw = pd.Series(
        np.asarray(X_no.sum(axis=0)).ravel(), index=cv_no.get_feature_names_out()
    )
    freq_no_rate = freq_no_raw / max(freq_no_raw.sum(), 1)

    only_yes = freq_yes_rate.index.difference(freq_no_rate.index)
    top_only_yes = freq_yes_rate.loc[only_yes].sort_values(ascending=False).head(top_n)
    top_general = freq_yes_raw.sort_values(ascending=False).head(top_n)

    # Hipotezi daha anlamlı ve interaktif göstermek için ortak kelimelerde
    # log2 frekans oranı: + değer dezenformasyon, - değer güvenilir haber yönü.
    common = freq_yes_rate.index.intersection(freq_no_rate.index)
    eps = 1e-9
    log_ratio = np.log2((freq_yes_rate.loc[common] + eps) / (freq_no_rate.loc[common] + eps))
    ratio_df = pd.DataFrame({
        "feature": common,
        "log2_ratio": log_ratio.values,
        "yes_rate": freq_yes_rate.loc[common].values,
        "no_rate": freq_no_rate.loc[common].values,
    })
    ratio_df["abs_ratio"] = ratio_df["log2_ratio"].abs()
    ratio_df = ratio_df.sort_values("abs_ratio", ascending=False).head(max(top_n * 2, 24))

    return (
        top_general.rename_axis("feature").reset_index(name="count"),
        top_only_yes.rename_axis("feature").reset_index(name="rate"),
        ratio_df,
    )


@st.cache_data(show_spinner=False)
def generate_wordcloud_fig(df_clean):
    yes_text = " ".join(df_clean.loc[df_clean["is_disinformation"] == "yes", "clean_text"])
    no_text = " ".join(df_clean.loc[df_clean["is_disinformation"] == "no", "clean_text"])

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, text, label in zip(
        axes, [yes_text, no_text], ["is_disinformation = yes", "is_disinformation = no"]
    ):
        wc = WordCloud(
            width=900, height=700, background_color="white",
            colormap="viridis", stopwords=STOPWORDS, max_words=150,
        ).generate(text)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(label, fontsize=14)
        ax.axis("off")
    fig.tight_layout()
    return fig


@st.cache_data(show_spinner=False)
def ngram_analysis(df_clean, top_n=12, min_df_ngram=5):
    yes_texts = df_clean.loc[df_clean["is_disinformation"] == "yes", "clean_text"]
    outputs = []
    for n in (2, 3):
        cv = CountVectorizer(stop_words="english", ngram_range=(n, n), min_df=min_df_ngram)
        X = cv.fit_transform(yes_texts)
        freq = pd.Series(np.asarray(X.sum(axis=0)).ravel(), index=cv.get_feature_names_out())
        outputs.append(freq.sort_values(ascending=False).head(top_n).rename_axis("ngram").reset_index(name="count"))
    return outputs[0], outputs[1]


@st.cache_data(show_spinner="VADER duygu skorları hesaplanıyor...")
def sentiment_analysis(df_clean):
    if not SCIPY_AVAILABLE:
        return None

    sia = None
    engine = None
    if OFFLINE_VADER_AVAILABLE:
        try:
            sia = OfflineVaderAnalyzer()
            engine = "vaderSentiment (çevrimdışı sözlük)"
        except Exception:
            sia = None

    if sia is None and NLTK_VADER_AVAILABLE:
        try:
            try:
                nltk.data.find("sentiment/vader_lexicon.zip")
            except LookupError:
                nltk.download("vader_lexicon", quiet=True, raise_on_error=True)
            sia = NLTKVaderAnalyzer()
            engine = "NLTK VADER"
        except Exception:
            sia = None

    if sia is None:
        return None

    out = df_clean[["is_disinformation", "clean_text"]].copy()
    out["sentiment"] = out["clean_text"].apply(lambda t: sia.polarity_scores(str(t))["compound"])
    yes_sent = out.loc[out["is_disinformation"] == "yes", "sentiment"]
    no_sent = out.loc[out["is_disinformation"] == "no", "sentiment"]
    stat, p_value = mannwhitneyu(yes_sent, no_sent)
    summary = {
        "yes_mean": float(yes_sent.mean()),
        "yes_median": float(yes_sent.median()),
        "no_mean": float(no_sent.mean()),
        "no_median": float(no_sent.median()),
        "p_value": float(p_value),
        "u_stat": float(stat),
        "engine": engine,
    }
    return out, summary


@st.cache_data(show_spinner="Eşik optimizasyonu hesaplanıyor (5-katlı CV)...")
def optimize_no_threshold(cache_signature, model_name, _model, _X_train, _y_train, _X_test, _y_test):
    # Notebook'taki kritik metodolojik düzeltme korunur: eşik TEST üzerinde aranmaz.
    # Önce eğitim verisinin out-of-fold tahminlerinde seçilir, sonra testte bir kez değerlendirilir.
    if model_name == "XGBoost":
        y_train_bin = _y_train.map({"no": 0, "yes": 1})
        oof_score = cross_val_predict(
            clone(_model), _X_train, y_train_bin, cv=5, method="predict_proba", n_jobs=-1
        )[:, 0]
        y_train_no = (y_train_bin == 0).astype(int).to_numpy()
        default_thr = 0.5
        test_score = _model.predict_proba(_X_test)[:, 0]
    elif hasattr(_model, "predict_proba"):
        oof_full = cross_val_predict(
            clone(_model), _X_train, _y_train, cv=5, method="predict_proba", n_jobs=-1
        )
        no_idx = list(_model.classes_).index("no")
        oof_score = oof_full[:, no_idx]
        y_train_no = (_y_train == "no").astype(int).to_numpy()
        default_thr = 0.5
        test_score = _model.predict_proba(_X_test)[:, no_idx]
    else:
        oof_raw = cross_val_predict(
            clone(_model), _X_train, _y_train, cv=5, method="decision_function", n_jobs=-1
        )
        classes = list(_model.classes_)
        oof_score = -oof_raw if classes[1] == "yes" else oof_raw
        y_train_no = (_y_train == "no").astype(int).to_numpy()
        default_thr = 0.0
        raw_test = _model.decision_function(_X_test)
        test_score = -raw_test if classes[1] == "yes" else raw_test

    precision_no, recall_no, thresholds = precision_recall_curve(y_train_no, oof_score)
    f1_no = 2 * precision_no * recall_no / (precision_no + recall_no + 1e-12)
    best_i = int(np.argmax(f1_no[:-1]))
    best_thr = float(thresholds[best_i])

    y_test_no = (_y_test == "no").astype(int).to_numpy()
    pred_default = (test_score >= default_thr).astype(int)
    pred_best = (test_score >= best_thr).astype(int)

    default_report = classification_report(
        y_test_no, pred_default, target_names=["diğer", "no"], output_dict=True, zero_division=0
    )
    best_report = classification_report(
        y_test_no, pred_best, target_names=["diğer", "no"], output_dict=True, zero_division=0
    )

    curve = pd.DataFrame({
        "threshold": thresholds,
        "precision_no": precision_no[:-1],
        "recall_no": recall_no[:-1],
        "f1_no": f1_no[:-1],
    })
    return {
        "best_threshold": best_thr,
        "default_threshold": float(default_thr),
        "cv_precision": float(precision_no[best_i]),
        "cv_recall": float(recall_no[best_i]),
        "cv_f1": float(f1_no[best_i]),
        "default_report": default_report,
        "best_report": best_report,
        "curve": curve,
    }


# ================================================================
# GRAFİK YARDIMCILARI
# ================================================================
def polish(fig, height=390, legend_title=None):
    # Plotly 6.x, başlıksız figürlerde title_font tanımlanınca bazı
    # ortamlarda "undefined" metni gösterebiliyor. Boş başlığı açıkça tanımla.
    if fig.layout.title.text is None:
        fig.update_layout(title_text="")
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=55, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#111111", size=12),
        title_font=dict(family="Plus Jakarta Sans, sans-serif", color=NAVY, size=15),
        hoverlabel=dict(font_size=12, font_family="Inter, sans-serif",
                        bgcolor=PAPER, bordercolor=LINE),
        legend_title_text=legend_title,
        colorway=SEQ,
        legend=dict(font=dict(color="#111111"), title_font=dict(color="#111111")),
    )
    axis_style = dict(color="#111111", title_font=dict(color="#111111"), tickfont=dict(color="#111111"))
    fig.update_xaxes(gridcolor="#EDF2F4", zeroline=False, linecolor="#DCE5E9", **axis_style)
    fig.update_yaxes(gridcolor="#EDF2F4", zeroline=False, linecolor="#DCE5E9", **axis_style)
    fig.update_annotations(font=dict(color="#111111"))
    return fig


def hero(title, subtitle, eyebrow="STREAMLIT • CANLI ANALİZ"):
    st.markdown(
        f"""
        <div class="plain-page-hero">
          <h1>{title}</h1>
          <p>{subtitle}</p>
          <span class="plain-page-pill">{eyebrow}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def signal_topbar(data_loaded=False):
    data_status = "Veri seti aktif" if data_loaded else "Veri yüklenmeyi bekliyor"
    st.markdown(
        f"""
        <div class="signal-topbar">
          <div class="brand">
            <div class="brand-mark">⌁</div>
            <div class="brand-name">SignalLens</div>
            <div class="brand-sub">DEZENFORMASYON ANALİTİĞİ</div>
          </div>
          <div class="status-area">
            <div class="signal-status"><span class="signal-dot">●</span>{data_status}</div>
            <div class="signal-status">Streamlit • canlı analiz</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def overview_hero():
    st.markdown(
        """
        <div class="overview-hero-card">
          <div class="overview-kicker">AKADEMİK ANALİZ • CANLI DASHBOARD</div>
          <h1>Sosyal Medyada<br>Dezenformasyon Analizi</h1>
          <p>Bu çalışmanın temel amacı, sosyal medya platformlarında (Telegram, Reddit, Bluesky)
          yayılan dezenformasyon içeriklerini Doğal Dil İşleme (NLP) ve Makine Öğrenmesi
          teknikleri kullanarak otomatik olarak tespit etmektir.</p>
          <div class="overview-chip-row">
            <div class="overview-chip">▣ Proje kapsamı</div>
            <div class="overview-chip">⌁ Canlı analiz</div>
            <div class="overview-chip">⚙ Makine öğrenmesi</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def project_summary():
    st.markdown(
        """
        <div class="project-summary">
          <div class="project-summary-title">Proje hakkında</div>
          <div class="project-row">
            <div class="project-row-icon">▤</div>
            <div><div class="project-row-title">1. Veri yükleme ve keşif</div>
            <div class="project-row-text">Telegram, Reddit, Bluesky ve haber sitelerinden derlenen kayıtlar yüklenir; sınıf (is_disinformation), kategori (news_type) ve etiket güven skoru (confidence) dağılımları incelenir.</div></div>
          </div>
          <div class="project-row">
            <div class="project-row-icon">◎</div>
            <div><div class="project-row-title">2. Metin temizleme</div>
            <div class="project-row-text">Bozuk karakter (mojibake) düzeltme, URL/markdown/emoji temizliği, Latin-dışı script filtresi, Rusya–Ukrayna bağlamı dışındaki içeriklerin ve çok kısa metinlerin veri setinden çıkarılması.</div></div>
          </div>
          <div class="project-row">
            <div class="project-row-icon">⌘</div>
            <div><div class="project-row-title">3. Özellik çıkarımı ve modelleme</div>
            <div class="project-row-text">TF-IDF (unigram + bigram) ile vektörleştirme; Logistic Regression, Linear SVM, Naive Bayes, Random Forest ve XGBoost modellerinin GridSearchCV ile hiperparametre optimizasyonu.</div></div>
          </div>
          <div class="project-row">
            <div class="project-row-icon">↗</div>
            <div><div class="project-row-title">4. Değerlendirme ve yorumlama</div>
            <div class="project-row-text">Hata analizi (False Positive/Negative), model özellik önem analizi ve sekiz araştırma sorusu üzerinden bulguların yorumlanması.</div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title, subtitle=""):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def kpi_card(icon, label, value, subtext, color, soft_color, progress=0):
    progress = max(0, min(100, float(progress)))
    st.markdown(
        f"""
        <div class="kpi-card" style="--kpi:{color};--kpi-soft:{soft_color};--fill:{progress:.1f}%">
          <div class="kpi-top">
            <div class="kpi-icon">{icon}</div>
            <div>
              <div class="kpi-label">{label}</div>
              <div class="kpi-value">{value}</div>
            </div>
          </div>
          <div class="kpi-sub">{subtext}</div>
          <div class="kpi-track"><div class="kpi-fill"></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def analysis_card(icon, title, text, pill):
    st.markdown(
        f"""
        <div class="analysis-card">
          <div class="a-icon">{icon}</div>
          <div class="a-title">{title}</div>
          <div class="a-text">{text}</div>
          <span class="a-pill">{pill}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def project_card(icon, title, text):
    st.markdown(
        f"""
        <div class="project-card">
          <div class="p-icon">{icon}</div>
          <div class="p-title">{title}</div>
          <div class="p-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def definition_banner(label, text_html):
    st.markdown(
        f"""
        <div class="definition-banner">
          <div class="db-label">{label}</div>
          <div class="db-text">{text_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def case_study_card(number, title, claim, fact):
    st.markdown(
        f"""
        <div class="case-study-card">
          <div class="cs-no">VAKA {escape(str(number))}</div>
          <div class="cs-title">{escape(title)}</div>
          <div class="cs-block cs-claim">
            <div class="cs-label">İddia</div>
            <p>{escape(claim)}</p>
          </div>
          <div class="cs-block cs-fact">
            <div class="cs-label">Gerçek</div>
            <p>{escape(fact)}</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def data_stage(code, value, label):
    st.markdown(
        f"""
        <div class="data-stage">
          <div class="stage-code">{code}</div>
          <div class="stage-value">{value}</div>
          <div class="stage-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_academic_table(df, formats=None, min_height=None):
    formats = formats or {}
    header_html = "".join(f"<th>{escape(str(column))}</th>" for column in df.columns)
    rows_html = []
    for _, row in df.iterrows():
        cells = []
        for column in df.columns:
            value = row[column]
            try:
                is_missing = bool(pd.isna(value))
            except (TypeError, ValueError):
                is_missing = False
            if is_missing:
                rendered = "—"
            elif column in formats:
                try:
                    rendered = formats[column].format(value)
                except (TypeError, ValueError):
                    rendered = str(value)
            else:
                rendered = str(value)
            cells.append(f"<td>{escape(rendered)}</td>")
        rows_html.append("<tr>" + "".join(cells) + "</tr>")
    table_html = (
        '<table class="academic-table"><thead><tr>' + header_html
        + '</tr></thead><tbody>' + "".join(rows_html) + '</tbody></table>'
    )
    min_height_style = f"min-height:{int(min_height)}px;" if min_height else ""
    st.markdown(
        f'<div class="academic-table-wrap" style="{min_height_style}">{table_html}</div>',
        unsafe_allow_html=True,
    )



# ================================================================
# SIDEBAR / VERİ
# ================================================================
st.sidebar.markdown(
    """
    <div class="sidebar-brand">
      <div class="sb-icon">⌁</div>
      <div>
        <div class="sb-title">SoftITo</div>
        <div class="sb-sub">Veri analitiği grup projesi</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

PAGES = [
    "🏠 Genel Bakış",
    "📊 Veri Keşfi",
    "🧹 Veri Temizleme",
    "🤖 Modelleme & Optimizasyon",
    "🔎 Özellik Analizi",
    "📌 Sonuç",
    "🎯 Canlı Tahmin",
]
page = st.sidebar.radio("Bölüm", PAGES, index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 Veri seti")
default_path = find_default_data_file()
raw_df = None
source_label = None

if default_path:
    try:
        raw_df = read_excel_path(default_path)
        source_label = os.path.basename(default_path)
        st.sidebar.success(f"Otomatik yüklendi: {len(raw_df):,} kayıt")
    except Exception as e:
        st.sidebar.error(f"Dosya okunamadı: {e}")

uploaded = st.sidebar.file_uploader(
    "Başka bir Excel yükle",
    type=["xlsx"],
    help="Yüklerseniz otomatik bulunan dosyanın yerine bu dosya kullanılır.",
)
if uploaded is not None:
    try:
        raw_df = read_excel_bytes(uploaded.getvalue())
        source_label = uploaded.name
        st.sidebar.success(f"Yüklendi: {len(raw_df):,} kayıt")
    except Exception as e:
        st.sidebar.error(f"Excel okunamadı: {e}")

if raw_df is not None:
    missing = validate_columns(raw_df)
    if missing:
        st.error("Eksik zorunlu sütun(lar): " + ", ".join(sorted(missing)))
        st.stop()

    try:
        raw_df = normalize_input_data(raw_df)
    except ValueError as e:
        st.error(f"Veri biçimi uygun değil: {e}")
        st.stop()

    raw_df["platform"] = raw_df["post_id"].apply(platform_from_post_id)
    raw_df["text_length"] = raw_df["text"].fillna("").astype(str).str.len()

st.sidebar.markdown("---")
with st.sidebar.expander("⚙️ Model ayarları", expanded=False):
    test_size = FIXED_MODEL_CONFIG["test_size"]
    ngram_max = FIXED_MODEL_CONFIG["ngram_max"]
    min_df = FIXED_MODEL_CONFIG["min_df"]
    max_df = FIXED_MODEL_CONFIG["max_df"]
    st.caption("Sonuçların her açılışta aynı kalması için model ayarları sabitlenmiştir.")
    st.code(
        f"Test oranı: {test_size:.0%}\n"
        f"N-gram: 1-{ngram_max}\n"
        f"min_df: {min_df}\n"
        f"max_df: {max_df:.2f}\n"
        f"random_state: {FIXED_MODEL_CONFIG['random_state']}",
        language=None,
    )

if not XGBOOST_AVAILABLE:
    st.sidebar.info("XGBoost kurulu değil; diğer modellerle devam edilir.")
if not ((OFFLINE_VADER_AVAILABLE or NLTK_VADER_AVAILABLE) and SCIPY_AVAILABLE):
    st.sidebar.info("Duygu analizi paketi eksik; diğer bölümler çalışır.")
if not PYTRENDS_AVAILABLE:
    st.sidebar.caption("Google Trends canlı çekimi için pytrends gerekir.")


# ================================================================
# VERİ HAZIRLIĞI — SADECE VARSA
# ================================================================
df_clean = stats = None
if raw_df is not None:
    df_clean, stats = clean_pipeline(raw_df)

model_pages = {"📌 Sonuç", "🤖 Modelleme & Optimizasyon", "🔎 Özellik Analizi", "🎯 Canlı Tahmin"}
model_bundle = None
if raw_df is not None and page in model_pages:
    if len(df_clean) < 10 or df_clean["is_disinformation"].nunique() < 2:
        st.error(
            "Modelleme için temizleme sonrasında en az 10 kayıt ve iki farklı sınıf gerekir. "
            "Veri setini veya temizleme sonucunu kontrol edin."
        )
        st.stop()
    data_hash = int(pd.util.hash_pandas_object(df_clean[["clean_text", "is_disinformation"]], index=True).sum())
    signature = f"{data_hash}-{test_size}-{ngram_max}-{min_df}-{max_df}"
    model_bundle = vectorize_data(df_clean, test_size, ngram_max, min_df, max_df)
    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_text,
        X_test_text,
        tfidf,
        original_feature_names,
        keep_idx,
        feature_names,
        train_idx,
        test_idx,
        removed_numeric,
    ) = model_bundle
    models = train_models(X_train, y_train, signature)
    results_table = build_results_table(models, y_test, X_test)


signal_topbar(raw_df is not None)


# ================================================================
# 1) GENEL BAKIŞ
# ================================================================
if page == "🏠 Genel Bakış":
    hero_left, hero_gap, hero_right = st.columns([1.08, .035, 1])
    with hero_left:
        overview_hero()
    with hero_right:
        project_summary()

    st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)
    definition_banner(
        "DEZENFORMASYON NEDİR?",
        "Dezenformasyon kısaca, insanları kandırmak ya da yanlış yönlendirmek için "
        "<b>bilerek üretilip yayılan</b>, gerçek olmayan ya da çarpıtılmış bilgiler demektir.",
    )

    st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)
    section(
        "Savaştan örnek dezenformasyon vakaları",
        "Rusya–Ukrayna savaşı boyunca her iki taraf da kamuoyunu manipüle etmek, askeri morali yüksek tutmak "
        "ya da karşı tarafı suçlamak amacıyla kasıtlı olarak üretilmiş sayısız yanıltıcı içeriği dolaşıma sokmuştur.",
    )
    cs1, _, cs2, _, cs3 = st.columns([1, 0.06, 1, 0.06, 1])
    with cs1:
        case_study_card(
            "01",
            '"Bucha Katliamı Kurguydu" İddiası',
            "Bucha'da Rus ordusunun çekilmesinin ardından sokaklarda bulunan sivil cesetlerinin aslında "
            '"aktör" olduğu ve Ukrayna tarafından batı medyasını etkilemek için sahnelendiği Rus devlet '
            "kanalları tarafından iddia edildi. Hatta cesetlerden birinin elini kıpırdattığı öne sürüldü.",
            "Bağımsız doğrulama kuruluşları ve uydu görüntüleri (Maxar), cesetlerin Rus ordusu henüz oradayken "
            "haftalarca aynı sokaklarda durduğunu kanıtladı. Elin kıpırdadığı iddia edilen an ise araba "
            "camındaki bir su damlasının yarattığı optik yanılsama olarak açıklandı.",
        )
    with cs2:
        case_study_card(
            "02",
            'Zelenski\'nin "Teslim Olun" Deepfake Videosu',
            "Mart 2022'de bir Ukrayna haber sitesinin hacklenmesinin ardından, Zelenski'nin halka ve askerlere "
            "silah bırakıp teslim olma çağrısı yaptığı bir video yayıldı.",
            "Videonun yapay zekâyla üretilmiş bir deepfake olduğu kısa sürede ortaya çıktı; kafa hareketlerinin "
            "doğal olmadığı ve ses tonunun beden diliyle uyuşmadığı belirlendi.",
        )
    with cs3:
        case_study_card(
            "03",
            "Video Oyunu ve Eski Çatışma Görüntüleri",
            "Rus hava savunma sistemlerinin Ukrayna jetlerini düşürdüğünü ya da bir roketin şehre isabet "
            "ettiğini gösteren, sosyal medyada milyonlarca kez izlenen yüzlerce video paylaşıldı.",
            "Videoların büyük bölümünün Arma 3 adlı askeri simülasyon oyunundan alındığı, bir kısmının ise "
            "geçmiş yıllara ait Suriye ya da İsrail–Filistin çatışmalarına ait görüntüler olduğu doğrulandı.",
        )

    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="insight">Biz de proje kapsamında, Rusya-Ukrayna Savaşı\'yla ilgili sosyal medya ve haber metinlerini kullanıp makine öğrenmesi yöntemleriyle bir dezenformasyon analizi gerçekleştirdik.</div>',
        unsafe_allow_html=True,
    )


# ================================================================
# 2) VERİ KEŞFİ
# ================================================================
elif page == "📊 Veri Keşfi":
    st.markdown(
        """
        <div class="eda-hero-card">
          <div class="eda-hero-kicker">AKADEMİK ANALİZ • CANLI DASHBOARD</div>
          <h1>Veri Keşfi ve<br>Keşifsel Veri Analizi</h1>
          <p>Bu bölümde veri setinin satır ve sütun yapısı; sınıf, kategori ve kaynak dağılımları ile kategori içi dezenformasyon oranları etkileşimli olarak incelenmektedir.</p>
          <div class="eda-hero-chips">
            <span class="eda-hero-chip">▣ Veri setinin yapısı</span>
            <span class="eda-hero-chip">⌁ Temel dağılımlar</span>
            <span class="eda-hero-chip">◉ Kategori analizi</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if raw_df is None:
        st.info("Sol menüden veri setini yükleyin.")
        st.stop()

    conf_min = float(raw_df["confidence"].min())
    conf_max = float(raw_df["confidence"].max())

    view = raw_df.copy()

    disinfo_count = int((view["is_disinformation"] == "yes").sum())
    reliable_count = int((view["is_disinformation"] == "no").sum())
    disinfo_rate = disinfo_count / len(view)
    category_count = int(view["news_type"].nunique())
    avg_conf = float(view["confidence"].mean())
    max_category_count = max(1, raw_df["news_type"].nunique())
    conf_progress = ((avg_conf - conf_min) / (conf_max - conf_min) * 100) if conf_max > conf_min else 100

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card(
            "▤", "TOPLAM KAYIT", f"{len(view):,}",
            "Yüklenen veri setindeki toplam kayıt sayısı",
            STEEL, "#EDF3F6", 100,
        )
    with k2:
        kpi_card(
            "◇", "DEZENFORMASYON ORANI", f"%{disinfo_rate*100:.1f}",
            f"{disinfo_count:,} dezenformasyon • {reliable_count:,} güvenilir",
            TEAL, "#EAF4F2", disinfo_rate * 100,
        )
    with k3:
        kpi_card(
            "▦", "KATEGORİ SAYISI", f"{category_count}",
            f"Toplam {max_category_count} kategori içinden aktif",
            SLATE, "#EDF3F6", category_count / max_category_count * 100,
        )
    with k4:
        kpi_card(
            "✓", "ORTALAMA CONFIDENCE", f"{avg_conf:.2f}",
            f"Seçili kayıtların ortalaması • aralık {conf_min:.2f}–{conf_max:.2f}",
            TEAL_D, "#EAF4F2", conf_progress,
        )

    st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
    section(
        "Veri setinin yapısı",
        "Satır ve sütun sayıları ile değişken adları, veri tipleri ve eksik değer bilgileri birlikte sunulmaktadır.",
    )
    s1, s2, s3 = st.columns(3)
    s1.metric("Satır sayısı", f"{raw_df.shape[0]:,}")
    s2.metric("Sütun sayısı", f"{raw_df.shape[1] - 2:,}")
    s3.metric("Kategori sayısı", f"{category_count:,}")
    visible_columns = [c for c in raw_df.columns if c not in {"platform", "text_length"}]
    structure_df = pd.DataFrame({
        "Sütun adı": visible_columns,
        "Veri tipi": [str(raw_df[c].dtype) for c in visible_columns],
        "Eksik değer": [int(raw_df[c].isna().sum()) for c in visible_columns],
        "Benzersiz değer": [int(raw_df[c].nunique(dropna=True)) for c in visible_columns],
    })
    render_academic_table(
        structure_df,
        formats={"Eksik değer": "{:,.0f}", "Benzersiz değer": "{:,.0f}"},
    )

    st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
    section("Temel dağılımlar", "Grafikler, seçilen filtre koşullarına göre anlık olarak yeniden hesaplanan betimsel dağılımları sunmaktadır.")
    g1, g2, g3 = st.columns([.92, 1.18, 1.0])
    with g1:
        with st.container(border=True):
            st.markdown("**Sınıf Dağılımı**")
            st.caption("Dezenformasyon ve güvenilir içeriklerin örneklem içindeki göreli dağılımı")
            class_counts = view["is_disinformation"].value_counts().rename_axis("class").reset_index(name="count")
            class_counts["label"] = class_counts["class"].map({"yes": "Dezenformasyon", "no": "Güvenilir"})
            fig = px.pie(
                class_counts, names="label", values="count", hole=.68, color="class",
                color_discrete_map=COLOR_MAP,
            )
            fig.update_traces(
                textinfo="percent", textfont_size=12,
                hovertemplate="%{label}<br>Kayıt: %{value:,}<br>Oran: %{percent}<extra></extra>",
                marker=dict(line=dict(color="#ffffff", width=2)),
            )
            fig.add_annotation(
                text=f"<b>{len(view):,}</b><br><span style='font-size:10px;color:#17212B'>Toplam</span>",
                showarrow=False, font=dict(size=17, color=TEXT),
            )
            fig.update_layout(legend=dict(orientation="h", y=-.08, x=.5, xanchor="center"))
            st.plotly_chart(polish(fig, 315), use_container_width=True, config={"displayModeBar": False})

    with g2:
        with st.container(border=True):
            st.markdown("**Kategorilere Göre Dağılım**")
            st.caption("Kategorilerin gözlem sayıları ve kategori içi dezenformasyon oranları")
            cat_summary = (
                view.groupby("news_type", observed=True)
                .agg(count=("post_id", "size"), disinfo_rate=("is_disinformation", lambda s: (s == "yes").mean()))
                .reset_index().sort_values("count")
            )
            fig = px.bar(
                cat_summary, x="count", y="news_type", orientation="h",
                custom_data=["disinfo_rate"], labels={"news_type": "", "count": "Kayıt"},
            )
            fig.update_traces(
                marker_color="#35617F", text=cat_summary["count"], textposition="outside",
                cliponaxis=False,
                hovertemplate="%{y}<br>Kayıt: %{x:,}<br>Dezenformasyon oranı: %{customdata[0]:.1%}<extra></extra>",
            )
            fig.update_layout(showlegend=False, margin=dict(l=5, r=34, t=8, b=10))
            st.plotly_chart(polish(fig, 315), use_container_width=True, config={"displayModeBar": False})

    with g3:
        with st.container(border=True):
            st.markdown("**Kaynak Platform Dağılımı**")
            st.caption("post_id alanından türetilen platform ve yayın kaynağı dağılımı")
            platform_counts = view["platform"].value_counts().reset_index()
            platform_counts.columns = ["platform", "count"]
            fig = px.pie(platform_counts, names="platform", values="count", hole=.58,
                         color_discrete_sequence=PLATFORM_SEQ)
            fig.update_traces(
                textinfo="none",
                hovertemplate="%{label}<br>Kayıt: %{value:,}<br>Oran: %{percent}<extra></extra>",
                marker=dict(line=dict(color="#ffffff", width=2)),
            )
            fig.update_layout(showlegend=True, legend=dict(font=dict(size=10)))
            st.plotly_chart(polish(fig, 315), use_container_width=True, config={"displayModeBar": False})

    section("Kategori düzeyinde dezenformasyon", "Küçük örneklemlerde oran tahminlerinin kararsızlaşmasını azaltmak amacıyla minimum kategori büyüklüğü kullanıcı tarafından belirlenebilmektedir.")
    min_n = st.slider("Minimum kategori örneklemi (n)", 1, 200, 30)
    summary = (
        view.groupby("news_type")["is_disinformation"]
        .agg(total="count", disinfo_rate=lambda x: (x == "yes").mean())
        .reset_index()
    )
    summary = summary[summary["total"] >= min_n].sort_values("disinfo_rate")
    summary["disinfo_pct"] = summary["disinfo_rate"] * 100

    if summary.empty:
        st.info("Bu filtrelerde seçilen minimum örneklem koşulunu sağlayan kategori yok.")
    else:
        fig = px.bar(
            summary, x="disinfo_pct", y="news_type", orientation="h",
            hover_data={"total": True, "disinfo_rate": False, "disinfo_pct": ":.1f"},
            labels={"disinfo_pct": "Dezenformasyon oranı (%)", "news_type": "Kategori", "total": "n"},
            title=f"Kategori bazında dezenformasyon oranı (n ≥ {min_n})",
        )
        fig.update_traces(marker_color=GREEN, hovertemplate="%{y}<br>Dezenformasyon: %{x:.1f}%<br>n=%{customdata[0]}<extra></extra>")
        st.plotly_chart(polish(fig, max(340, 50 * max(len(summary), 4))), use_container_width=True, config={"displayModeBar": False})

    section(
        "Veri kalitesi kontrolü",
        "Notebook'taki ilk veri kontrolü adımı (boyut, eksik değer ve tekrarlı kayıt taraması) canlı veri üzerinden yeniden hesaplanmaktadır.",
    )
    required_cols = ["post_id", "text", "is_disinformation", "news_type", "confidence"]
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Sütun", raw_df.shape[1] - 2)
    q2.metric("Toplam kayıt", f"{len(view):,}")
    q3.metric("Tekrarlı post_id", f"{int(view['post_id'].duplicated().sum()):,}")
    q4.metric("Tekrarlı metin", f"{int(view['text'].duplicated().sum()):,}")

    missing = view[required_cols].isna().sum().reset_index()
    missing.columns = ["column", "missing"]
    fig = px.bar(
        missing.sort_values("missing"), x="missing", y="column", orientation="h",
        title="Zorunlu sütunlarda eksik değer sayısı",
        labels={"missing": "Eksik hücre sayısı", "column": ""},
    )
    fig.update_traces(marker_color=CORAL, hovertemplate="%{y}<br>Eksik: %{x:,}<extra></extra>")
    st.plotly_chart(polish(fig, 260), use_container_width=True, config={"displayModeBar": False})

    with st.expander("Örnek kayıtları incele"):
        sample_n = st.slider("Gösterilecek satır", 5, 50, 12)
        render_academic_table(
            view[["post_id", "text", "is_disinformation", "news_type", "confidence", "platform"]].head(sample_n),
            formats={"confidence": "{:.2f}"},
        )


# ================================================================
# 3) TEMİZLEME
# ================================================================
elif page == "🧹 Veri Temizleme":
    hero(
        "Veri Temizleme Süreci",
        "Bu bölüm, metin ön işleme sürecinde uygulanan dönüşümleri, dışlama ölçütlerini ve model girdisine aktarılan nihai metinleri izlenebilir örneklerle açıklamaktadır.",
        "PREPROCESSING • ŞEFFAF DENETİM • ÖNCE/SONRA",
    )
    if raw_df is None:
        st.info("Sol menüden veri setini yükleyin.")
        st.stop()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Ham veri", f"{stats['raw_shape'][0]:,}")
    m2.metric("Dil filtresinde çıkarılan", f"{stats['dil_filtresi_cikarilan']:,}")
    m3.metric("Konu-dışı çıkarılan", f"{stats['konu_disi_cikarilan']:,}")
    m4.metric("Final veri", f"{stats['final_shape'][0]:,}")

    section("Metin düzeyinde gerçekleştirilen dönüşümler", "Aşağıdaki özet, ham metinlerde saptanan ve ön işleme sırasında dönüştürülen ya da kaldırılan öğelerin sıklığını göstermektedir.")
    audit_df = pd.DataFrame([
        {"İşlem": "Kodlama bozukluğu düzeltme", "Bulunan": stats["bozuk_karakter_sayisi"], "Uygulanan kural": "Mojibake (â, ðŸ, Ã, �) karakterlerini UTF-8'e dönüştür"},
        {"İşlem": "Bağlantı silme", "Bulunan": stats["url_sayisi"], "Uygulanan kural": "http, https ve www ile başlayan URL'leri kaldır"},
        {"İşlem": "Markdown temizleme", "Bulunan": stats["markdown_sayisi"], "Uygulanan kural": "Link/görsel sözdizimi ve biçimlendirme işaretlerini kaldır"},
        {"İşlem": "Emoji/sembol silme", "Bulunan": stats["emoji_sayisi"], "Uygulanan kural": "Emoji ve sembol gruplarını boşlukla değiştir"},
        {"İşlem": "Görünmez karakter silme", "Bulunan": stats["gorunmez_karakter_sayisi"], "Uygulanan kural": "Zero-width ve BOM karakterlerini kaldır"},
        {"İşlem": "Latin-dışı karakter silme", "Bulunan": stats["latin_disi_karakter_sayisi"], "Uygulanan kural": "Model metninde ASCII dışı karakterleri kaldır"},
        {"İşlem": "Boşluk düzeltme", "Bulunan": stats["fazla_bosluk_sayisi"], "Uygulanan kural": "Art arda gelen boşlukları tek boşluğa indir"},
    ])
    render_academic_table(audit_df, formats={"Bulunan": "{:,.0f}"})

    section("Kayıt düzeyinde filtreleme ölçütleri", "Metin normalizasyonunun ardından uygulanan dışlama kuralları ve bu kurallar kapsamında veri setinden çıkarılan örnek gözlemler sunulmaktadır.")
    with st.expander("1. Bozuk karakter / mojibake düzeltmesi", expanded=True):
        st.write(f"Bozuk karakter örüntüsü bulunan **{stats['bozuk_karakter_sayisi']:,} kayıt** yeniden kodlanarak düzeltildi.")
        if stats["bozuk_karakter_ornekleri"]:
            render_academic_table(pd.DataFrame(stats["bozuk_karakter_ornekleri"]))
        else:
            st.caption("Bu veri sürümünde gösterilecek bozuk karakter örneği bulunmadı.")

    with st.expander("2. URL / markdown / emoji temizliği"):
        st.write(
            f"Metinlerden **{stats['url_sayisi']:,} bağlantı (URL)**, "
            f"**{stats['markdown_sayisi']:,} markdown işareti** ve "
            f"**{stats['emoji_sayisi']:,} emoji/sembol** kaldırıldı. "
            "Bu adım bir kayıt silme (filtre) değil, metin içeriğini sadeleştiren bir dönüşümdür; "
            "ardından metin küçük harfe çevrilip fazla boşluklar tek boşluğa indirilir."
        )
        st.caption("Kural: http/https/www ile başlayan bağlantılar, link-görsel sözdizimi ve biçimlendirme işaretleri, ardından emoji/sembol grupları boşlukla değiştirilir.")
        if stats["url_md_emoji_ornekleri"]:
            render_academic_table(pd.DataFrame(stats["url_md_emoji_ornekleri"]))

    with st.expander("3. Latin-dışı script filtresi"):
        st.write(
            f"Latin-dışı harf oranı %{NON_LATIN_RATIO_THRESHOLD*100:.0f} üzerindeki kayıtlar çıkarıldı: "
            f"**{stats['dil_filtresi_oncesi']:,} → {stats['dil_filtresi_sonrasi']:,}** "
            f"({stats['dil_filtresi_cikarilan']:,} kayıt)."
        )
        st.caption("Bu ölçüt, tekil Latin-dışı karakterleri değil; metnin baskın bölümünün Latin-dışı bir yazı sistemiyle oluşturulduğu kayıtları dışlamaktadır.")
        if stats["dil_filtresi_ornekleri"]:
            render_academic_table(
                pd.DataFrame(stats["dil_filtresi_ornekleri"]),
                formats={"Latin-dışı oran": "{:.3f}"},
            )

    with st.expander("4. Konu-dışı içerik filtresi"):
        st.write(
            f"Rusya–Ukrayna bağlantısı bulunmayan **{stats['konu_disi_cikarilan']:,} kayıt** çıkarıldı. "
            "Bir dış-konu kelimesi tek başına silme nedeni değildir; kayıt aynı zamanda Rusya–Ukrayna bağlam sözcüklerini içermiyorsa elenir."
        )
        st.markdown("**Dış-konu aday sözcükleri**")
        st.markdown(
            "".join(f'<span class="audit-chip">{escape(term)}</span>' for term in ALL_TOPIC_KEYWORDS),
            unsafe_allow_html=True,
        )
        st.markdown("**Kaydı koruyan Rusya–Ukrayna bağlam sözcükleri**")
        st.markdown(
            "".join(f'<span class="audit-chip">{escape(term)}</span>' for term in UKRAINE_RUSSIA_KEYWORDS),
            unsafe_allow_html=True,
        )
        if stats["konu_disi_ornekleri"]:
            st.markdown("**Çıkarılan kayıt örnekleri**")
            render_academic_table(pd.DataFrame(stats["konu_disi_ornekleri"]))

    with st.expander("5. Çok kısa metin filtresi"):
        st.write(
            f"Temizleme sonrasında **25 karakterden kısa {stats['kisa_metin_cikarilan']:,} kayıt** "
            "model için yeterli bağlam taşımadığı gerekçesiyle çıkarıldı."
        )
        if stats["kisa_metin_ornekleri"]:
            render_academic_table(pd.DataFrame(stats["kisa_metin_ornekleri"]))

    section("Ham ve işlenmiş metnin karşılaştırılması", "Seçilen gözlemde uygulanan dönüşümler, ham metin ile modele aktarılan nihai metin yan yana sunularak incelenmektedir.")
    idx = st.slider("Örnek kayıt", 0, max(len(df_clean) - 1, 0), min(1, max(len(df_clean) - 1, 0)))
    if len(df_clean):
        ex = df_clean.iloc[idx]
        raw_value = str(ex["text"])
        clean_value = str(ex["clean_text"])
        removed_items = []
        removed_items.extend(re.findall(r"https?://\S+|www\.\S+", raw_value))
        removed_items.extend(EMOJI_PATTERN.findall(raw_value))
        removed_items.extend(re.findall(r"[*_\x60#>~]+", raw_value))
        removed_items = [item[:45] for item in removed_items if str(item).strip()][:12]

        a, b = st.columns(2)
        with a:
            st.markdown(
                f"""
                <div class="compare-card" style="--compare-bg:#EDF3F6;--compare-line:#CBDCE6;--compare-accent:#35617F;">
                  <div class="compare-label">Ham metin</div>
                  <div class="compare-text">{escape(raw_value)}</div>
                  <div class="compare-meta">{len(raw_value):,} karakter</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with b:
            st.markdown(
                f"""
                <div class="compare-card" style="--compare-bg:#EAF4F2;--compare-line:#CFE3DE;--compare-accent:#2F8C7C;">
                  <div class="compare-label">Temizlenmiş metin</div>
                  <div class="compare-text">{escape(clean_value)}</div>
                  <div class="compare-meta">{len(clean_value):,} karakter • model girdisi</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if removed_items:
            st.markdown("**Bu örnekte kaldırılan öğeler**")
            st.markdown(
                "".join(f'<span class="audit-chip">{escape(str(item))}</span>' for item in removed_items),
                unsafe_allow_html=True,
            )
        else:
            st.caption("Bu örnekte URL, emoji veya Markdown işareti yok; değişiklik büyük/küçük harf ve boşluk standardizasyonundan oluşuyor.")

    section(
        "Temizleme pipeline'ı",
        "Tüm metin dönüşümleri ve kayıt düzeyindeki filtreler uygulandıktan sonra veri setindeki kayıt değişimi en sonda toplu olarak gösterilmektedir.",
    )
    stages = pd.DataFrame({
        "Aşama": ["Ham veri", "Dil filtresi", "Konu filtresi", "Kısa metin filtresi"],
        "Kayıt": [
            stats["raw_shape"][0],
            stats["dil_filtresi_sonrasi"],
            stats["konu_disi_sonrasi"],
            stats["kisa_metin_sonrasi"],
        ],
    })
    pipeline_table = stages.copy()
    pipeline_table["Önceki aşamadan çıkarılan"] = (
        pipeline_table["Kayıt"].shift(1) - pipeline_table["Kayıt"]
    ).fillna(0).astype(int)
    render_academic_table(
        pipeline_table,
        formats={"Kayıt": "{:,.0f}", "Önceki aşamadan çıkarılan": "{:,.0f}"},
    )
    fig = go.Figure(
        go.Funnel(
            y=stages["Aşama"], x=stages["Kayıt"], textinfo="value+percent initial",
            marker={"color": [SLATE, "#6FA9A0", TEAL, TEAL_D]},
            hovertemplate="%{y}<br>Kayıt: %{x:,}<extra></extra>",
        )
    )
    fig.update_layout(title="Temizleme pipeline'ı boyunca kayıt sayısı")
    st.plotly_chart(polish(fig, 390), use_container_width=True, config={"displayModeBar": False})


# ================================================================
# 4) MODEL PERFORMANSI
# ================================================================
elif page == "🤖 Modelleme & Optimizasyon":
    hero(
        "Model Performansı",
        "TF-IDF yöntemiyle sayısal olarak temsil edilen metinler üzerinde eğitilen sınıflandırma modelleri, ortak test kümesi ve aynı performans ölçütleri kullanılarak karşılaştırılmaktadır.",
        "TF-IDF • MODEL KARŞILAŞTIRMA",
    )
    if raw_df is None:
        st.info("Sol menüden veri setini yükleyin.")
        st.stop()

    # ---- 1) TF-IDF HAKKINDA BİLGİ ----------------------------------
    section(
        "TF-IDF hakkında",
        "Metinler, modele girdi olarak verilmeden önce TF-IDF (Term Frequency – Inverse Document Frequency) yöntemiyle sayısal vektörlere dönüştürülür. Bu yöntem, bir kelimenin bir metindeki sıklığını (TF), o kelimenin tüm veri setinde ne kadar ayırt edici olduğuyla (IDF) çarparak ağırlıklandırır; böylece yaygın ama anlam taşımayan kelimeler düşük, nadir ve ayırt edici kelimeler yüksek ağırlık alır.",
    )
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Eğitim / test", f"{len(y_train):,} / {len(y_test):,}")
    t2.metric("TF-IDF özellik sayısı", f"{len(feature_names):,}")
    t3.metric("N-gram aralığı", f"1–{FIXED_MODEL_CONFIG['ngram_max']}")
    t4.metric("Sayısal token çıkarıldı", f"{removed_numeric:,}")
    st.markdown(
        f"""
        <div class="insight" style="font-size:12px;line-height:1.7;">
        Kelime dağarcığı, en az <b>{FIXED_MODEL_CONFIG['min_df']}</b> belgede geçen ve belgelerin en fazla
        <b>%{FIXED_MODEL_CONFIG['max_df']*100:.0f}</b>'inde geçen terimlerle sınırlandırılmıştır (çok nadir ve çok yaygın
        terimler gürültü kabul edilerek elenmiştir). Test kümesi oranı <b>%{FIXED_MODEL_CONFIG['test_size']*100:.0f}</b> olarak
        sabitlenmiştir; çoğunluk sınıfını tahmin eden basit bir referans model (baseline) %{(float((y_test == y_test.value_counts().idxmax()).mean()))*100:.1f} doğruluk vermektedir — modellerin bu değerin üzerinde kalması beklenir.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- 2) MODELLER NASIL OPTİMİZE EDİLDİ -------------------------
    section("Modeller nasıl optimize edildi?", "GridSearchCV ile belirlenen en uygun hiperparametreler sabitlenmiş ve modeller bu yapılandırmalar altında yeniden eğitilmiştir.")
    st.markdown(
        """
        <div class="optimize-explainer">
          <div class="optimize-grid">
            <div class="optimize-step"><span class="optimize-no">1</span><b>Parametre aralıkları belirlendi</b><p>Her algoritma için aday hiperparametre kombinasyonları oluşturuldu.</p></div>
            <div class="optimize-step"><span class="optimize-no">2</span><b>GridSearchCV uygulandı</b><p>Aday kombinasyonlar yalnızca eğitim verisi üzerinde sistematik olarak test edildi.</p></div>
            <div class="optimize-step"><span class="optimize-no">3</span><b>Çapraz doğrulama yapıldı</b><p>Sonuçlar 5 katlı çapraz doğrulama ve Macro F1 ölçütüyle karşılaştırıldı.</p></div>
            <div class="optimize-step"><span class="optimize-no">4</span><b>En iyi model yeniden eğitildi</b><p>En yüksek skoru veren parametreler sabitlendi; test verisi yalnızca nihai değerlendirmede kullanıldı.</p></div>
          </div>
          <div class="optimize-purpose"><b>Amaç:</b> Sınıf dengesizliğini dikkate alarak iki sınıfta da dengeli başarı sağlayan parametreleri seçmek.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- 3) SEÇİLİ MODEL: OPTİMİZE ÖNCESİ / SONRASI KARŞILAŞTIRMA --
    section(
        "Seçili model • optimizasyon öncesi ve sonrası karşılaştırma",
        "Seçilen algoritmanın varsayılan hiperparametrelerle (optimizasyon öncesi) ve GridSearchCV ile bulunan en iyi hiperparametrelerle (optimizasyon sonrası) aynı test kümesindeki değerlendirme metrikleri karşılaştırılmaktadır.",
    )
    baseline_df = pd.DataFrame(BASELINE_MODEL_RESULTS).set_index("Model")
    optimized_df = results_table.set_index("Model")
    selected_model_name = st.selectbox("Model", list(results_table["Model"]), index=0, key="model_select")
    selected_model = models[selected_model_name]

    before_row = baseline_df.loc[selected_model_name]
    after_row = optimized_df.loc[selected_model_name]
    with st.container(border=True):
        compare_long = pd.DataFrame({
            "Metrik": ["Accuracy", "Güvenilir Recall", "Güvenilir Precision", "Dezenformasyon Recall", "Macro F1"] * 2,
            "Durum": ["Önce"] * 5 + ["Sonra"] * 5,
            "Skor": [
                before_row["Accuracy"], before_row["no Recall"], before_row["no Precision"], before_row["yes Recall"], before_row["Macro F1"],
                after_row["Accuracy"], after_row["no Recall"], after_row["no Precision"], after_row["yes Recall"], after_row["Macro F1"],
            ],
        })
        fig = px.bar(
            compare_long, x="Metrik", y="Skor", color="Durum", barmode="group",
            color_discrete_map={"Önce": "#B9C7CF", "Sonra": TEAL_D},
            title=f"{selected_model_name} • optimizasyon öncesi/sonrası", text_auto=".3f", range_y=[0, 1.05],
        )
        fig.update_layout(xaxis_tickangle=-15, legend_title_text="")
        st.plotly_chart(polish(fig, 400), use_container_width=True, config={"displayModeBar": False})
        delta_acc = after_row["Accuracy"] - before_row["Accuracy"]
        delta_f1 = after_row["Macro F1"] - before_row["Macro F1"]
        d1, d2 = st.columns(2)
        d1.metric("ΔAccuracy", f"{delta_acc:+.3f}")
        d2.metric("ΔMacro F1", f"{delta_f1:+.3f}")

    st.markdown("**Bu modelde optimize edilen parametreler**")
    params = BEST_PARAMS[selected_model_name]
    param_rows = [{"Parametre": k, "Optimize edilmiş değer": v} for k, v in params.items()]
    render_academic_table(pd.DataFrame(param_rows))
    st.caption("Değerler, notebook'taki GridSearchCV taramasının sonucunda 5 katlı çapraz doğrulama ile Macro F1'i maksimize eden kombinasyon olarak seçilmiştir.")

    # ---- 4) SEÇİLİ MODELİN HATA ANALİZİ -----------------------------
    section("Seçili modelin hata analizi", "Yanlış pozitif ve yanlış negatif tahminler, modelin sınıf bazındaki hata örüntülerini değerlendirmek amacıyla incelenmektedir.")
    pred = get_predictions(selected_model, selected_model_name, X_test)
    cm = confusion_matrix(y_test, pred, labels=["no", "yes"])

    c1, c2 = st.columns([1, 1])
    with c1:
        heat = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=["Tahmin: Güvenilir", "Tahmin: Dezenformasyon"],
                y=["Gerçek: Güvenilir", "Gerçek: Dezenformasyon"],
                colorscale=[[0, "#F2F7F8"], [0.5, "#9ED3CA"], [1, TEAL_D]],
                showscale=False,
                text=cm,
                texttemplate="%{text}",
                hovertemplate="%{y}<br>%{x}<br>Kayıt: %{z}<extra></extra>",
            )
        )
        heat.update_layout(title=f"{selected_model_name} — Confusion Matrix")
        st.plotly_chart(polish(heat, 390), use_container_width=True, config={"displayModeBar": False})

    with c2:
        report_df = pd.DataFrame(
            classification_report(y_test, pred, output_dict=True, zero_division=0)
        ).T.round(3)
        render_academic_table(
            report_df.reset_index(names="Sınıf / Ortalama"),
            formats={"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}", "support": "{:.0f}"},
            min_height=390,
        )

    no_scores = get_no_class_scores(selected_model, selected_model_name, X_test)
    y_true_no = (y_test == "no").astype(int).values
    fpr, tpr, _ = roc_curve(y_true_no, no_scores)
    precision, recall, _ = precision_recall_curve(y_true_no, no_scores)
    roc_auc = auc(fpr, tpr)
    ap = average_precision_score(y_true_no, no_scores)

    roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr})
    pr_df = pd.DataFrame({"Recall": recall, "Precision": precision})
    a, b = st.columns(2)
    with a:
        fig = px.line(roc_df, x="FPR", y="TPR", title=f"ROC eğrisi • AUC {roc_auc:.3f}")
        fig.update_traces(line_color=ACCENT)
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="#84A8BF"))
        st.plotly_chart(polish(fig, 360), use_container_width=True, config={"displayModeBar": False})
    with b:
        fig = px.line(pr_df, x="Recall", y="Precision", title=f"Precision–Recall • AP {ap:.3f}")
        fig.update_traces(line_color=GREEN)
        st.plotly_chart(polish(fig, 360), use_container_width=True, config={"displayModeBar": False})

    # Notebook'taki hata analizi artık sabit çıktı tablosu değil; seçili modelden anlık hesaplanır.
    test_errors = df_clean.loc[test_idx, ["post_id", "clean_text", "news_type", "confidence"]].copy().reset_index(drop=True)
    test_errors["gercek"] = y_test.values
    test_errors["tahmin"] = pred
    test_errors = test_errors[test_errors["gercek"] != test_errors["tahmin"]].copy()
    fp = test_errors[(test_errors["gercek"] == "no") & (test_errors["tahmin"] == "yes")]
    fn = test_errors[(test_errors["gercek"] == "yes") & (test_errors["tahmin"] == "no")]
    e1, e2, e3 = st.columns(3)
    e1.metric("Toplam yanlış", f"{len(test_errors):,}")
    e2.metric("False Positive", f"{len(fp):,}")
    e3.metric("False Negative", f"{len(fn):,}")
    with st.expander("Yanlış sınıflandırılan örnekleri incele"):
        tab1, tab2 = st.tabs(["False Positive", "False Negative"])
        with tab1:
            render_academic_table(fp.head(20))
        with tab2:
            render_academic_table(fn.head(20))


# ================================================================
# 5) ÖZELLİK ANALİZİ
# ================================================================
elif page == "🔎 Özellik Analizi":
    hero(
        "Özellik Analizi",
        "Bu bölüm, sınıflandırma kararlarında etkili olan sözcük ve n-gram özelliklerini doğrudan eğitilmiş modellerin katsayı ve önem değerleri üzerinden incelemektedir.",
        "MODEL İÇGÖRÜSÜ • CANLI",
    )
    if raw_df is None:
        st.info("Sol menüden veri setini yükleyin.")
        st.stop()

    options = ["Linear SVM", "Random Forest", "Multinomial NB"]
    if "XGBoost" in models:
        options.insert(2, "XGBoost")
    model_choice = st.selectbox("Model", options)
    n_words = st.slider("Gösterilecek özellik sayısı", 5, 25, 12)

    if model_choice == "Linear SVM":
        coefs = models["Linear SVM"].coef_[0]
        order = np.argsort(coefs)
        yes_idx = order[-n_words:][::-1]
        yes_df = pd.DataFrame({"feature": feature_names[yes_idx], "score": coefs[yes_idx]})
        fig = px.bar(yes_df.sort_values("score"), x="score", y="feature", orientation="h", title="Linear SVM • en ayırt edici özellikler")
        fig.update_traces(marker_color=GREEN)
        st.plotly_chart(polish(fig, 450), use_container_width=True, config={"displayModeBar": False})


    elif model_choice in {"Random Forest", "XGBoost"}:
        imp = models[model_choice].feature_importances_
        idx = np.argsort(imp)[::-1][:n_words]
        feat_df = pd.DataFrame({"feature": feature_names[idx], "importance": imp[idx]}).sort_values("importance")
        fig = px.bar(feat_df, x="importance", y="feature", orientation="h", title=f"{model_choice} • en ayırt edici özellikler")
        fig.update_traces(marker_color=GREEN if model_choice == "Random Forest" else ACCENT)
        st.plotly_chart(polish(fig, 500), use_container_width=True, config={"displayModeBar": False})
        st.caption("Ağaç tabanlı özellik önem değerleri yön bilgisi içermemektedir; dolayısıyla tek başına bir özelliğin hangi sınıfla ilişkili olduğunu göstermemektedir.")

    else:
        mnb = models["Multinomial NB"]
        classes = list(mnb.classes_)
        yes_i, no_i = classes.index("yes"), classes.index("no")
        diff = mnb.feature_log_prob_[yes_i] - mnb.feature_log_prob_[no_i]
        idx = np.argsort(diff)[::-1][:n_words]
        feat_df = pd.DataFrame({"feature": feature_names[idx], "score": diff[idx]}).sort_values("score")
        fig = px.bar(feat_df, x="score", y="feature", orientation="h", title="Multinomial NB • dezenformasyon yönünde log-olasılık farkı")
        fig.update_traces(marker_color=GREEN)
        st.plotly_chart(polish(fig, 500), use_container_width=True, config={"displayModeBar": False})


# ================================================================
# 6) SONUÇ (eski adıyla: Araştırma Soruları)
# ================================================================
elif page == "📌 Sonuç":
    st.markdown(
        """
        <div class="plain-page-hero">
          <h1>Sonuç</h1>
          <p>Çalışmanın altı araştırma sorusu; betimsel istatistikler, hipotez değerlendirmeleri, etkileşimli görselleştirmeler ve eğitilmiş model çıktıları temelinde incelenmektedir.</p>
          <span class="plain-page-pill">NOTEBOOK → CANLI STREAMLIT</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if raw_df is None:
        st.info("Sol menüden veri setini yükleyin.")
        st.stop()

    question = st.selectbox(
        "İncelenecek soru",
        [
            "1. Dezenformasyon hangi konularda yayılıyor?",
            "2. Dezenformasyon metinleri gerçek haberlerden daha mı uzun?",
            "3. Dezenformasyon metinlerinde hangi kelimeler öne çıkıyor?",
            "4. Dezenformasyon ve gerçek haberlerin duygu tonu farklı mı?",
            "5. En sık bigram ve trigram kalıpları neler?",
            "6. Makine öğrenmesi modelleri ne kadar başarılı?",
        ],
    )

    if question.startswith("1."):
        summary = (
            df_clean.groupby("news_type")["is_disinformation"]
            .agg(toplam="count", yes_orani=lambda x: (x == "yes").mean())
            .reset_index()
        )
        genel = float((df_clean["is_disinformation"] == "yes").mean())
        top5 = summary.nlargest(5, "toplam").sort_values("toplam")
        guvenilir = summary[summary["toplam"] >= 30].copy()
        guvenilir["dezenformasyon_pct"] = guvenilir["yes_orani"] * 100
        guvenilir["genel_pct"] = genel * 100
        guvenilir = guvenilir.sort_values("dezenformasyon_pct")

        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                fig = px.bar(
                    top5, x="toplam", y="news_type", orientation="h",
                    title="Kategorilere göre kayıt hacmi • ilk 5",
                    labels={"toplam": "Kayıt", "news_type": ""},
                    text="toplam",
                )
                fig.update_traces(marker_color=GREEN, texttemplate="%{text:,.0f}", textposition="outside", cliponaxis=False)
                fig.update_layout(margin=dict(l=8, r=45, t=58, b=25))
                st.plotly_chart(polish(fig, 420), use_container_width=True, config={"displayModeBar": False})
        with c2:
            with st.container(border=True):
                fig = px.bar(
                    guvenilir, x="dezenformasyon_pct", y="news_type", orientation="h",
                    title="Kategori içi dezenformasyon oranı • n ≥ 30",
                    labels={"dezenformasyon_pct": "Oran (%)", "news_type": ""},
                    text="dezenformasyon_pct",
                    hover_data={"toplam": True, "yes_orani": False, "genel_pct": False},
                )
                fig.update_traces(marker_color=GREEN, texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
                fig.update_xaxes(range=[0, 100], ticksuffix="%")
                fig.update_layout(margin=dict(l=8, r=45, t=58, b=25))
                st.plotly_chart(polish(fig, 420), use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            """
            <div class="research-note"><span class="research-note-icon">i</span>
            <div><b>Not:</b> Örneklem büyüklüğü 30'un altında olan kategoriler oran yorumuna dahil edilmemiştir.</div></div>
            """,
            unsafe_allow_html=True,
        )

    elif question.startswith("2."):
        yes_len = df_clean.loc[df_clean["is_disinformation"] == "yes", "text_length"]
        no_len = df_clean.loc[df_clean["is_disinformation"] == "no", "text_length"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Dezenformasyon ort.", f"{yes_len.mean():.1f} karakter", f"Medyan {yes_len.median():.0f}")
        c2.metric("Güvenilir ort.", f"{no_len.mean():.1f} karakter", f"Medyan {no_len.median():.0f}")
        ratio = yes_len.mean() / max(no_len.mean(), 1e-9)
        c3.metric("Uzunluk oranı", f"{ratio:.1f}×")

        c1, c2 = st.columns(2)
        plot_df = df_clean[["is_disinformation", "text_length"]].copy()
        plot_df["Sınıf"] = plot_df["is_disinformation"].map({"yes": "Dezenformasyon", "no": "Güvenilir"})
        with c1:
            fig = px.box(
                plot_df, x="Sınıf", y="text_length", color="is_disinformation", points=False,
                color_discrete_map=COLOR_MAP, title="Metin uzunluğu • boxplot",
                labels={"text_length": "Karakter sayısı"},
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(polish(fig, 420), use_container_width=True, config={"displayModeBar": False})
        with c2:
            fig = px.histogram(
                plot_df, x="text_length", color="is_disinformation", nbins=45, barmode="overlay", opacity=.62,
                color_discrete_map=COLOR_MAP, title="Metin uzunluğu • histogram",
                labels={"text_length": "Karakter sayısı"},
            )
            fig.for_each_trace(lambda t: t.update(name="Dezenformasyon" if t.name == "yes" else "Güvenilir"))
            st.plotly_chart(polish(fig, 420), use_container_width=True, config={"displayModeBar": False})
        st.markdown(f'<div class="insight"><b>Canlı çıkarım:</b> Seçili veri üzerinde dezenformasyon metinleri ortalama <b>{ratio:.1f} kat</b> daha uzun görünüyor.</div>', unsafe_allow_html=True)

    elif question.startswith("3."):
        top_n = st.slider("Gösterilecek kelime sayısı", 8, 25, 15, key="word_top_n")
        top_general, top_only_yes, ratio_df = word_frequency_analysis(df_clean, top_n=top_n)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(
                top_general.sort_values("count"), x="count", y="feature", orientation="h",
                title="Dezenformasyon metinlerinde en sık kelimeler",
                labels={"count": "Geçiş sayısı", "feature": "Kelime"},
            )
            fig.update_traces(marker_color=ACCENT)
            st.plotly_chart(polish(fig, 480), use_container_width=True, config={"displayModeBar": False})
        with c2:
            if len(top_only_yes):
                fig = px.bar(
                    top_only_yes.sort_values("rate"), x="rate", y="feature", orientation="h",
                    title="Yalnızca dezenformasyonda görülen kelimeler",
                    labels={"rate": "Oransal frekans", "feature": "Kelime"},
                )
                fig.update_traces(marker_color=GREEN)
                st.plotly_chart(polish(fig, 480), use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("Seçili min_df koşulunda yalnızca dezenformasyonda görülen yeterince sık kelime bulunmadı.")

        section("Sözcüksel dağılımların karşılaştırılması", "Sınıflar arasındaki göreli sözcük kullanımı, WordCloud ile görsel olarak karşılaştırılmaktadır.")
        wc_fig = generate_wordcloud_fig(df_clean)
        st.pyplot(wc_fig, use_container_width=True)

    elif question.startswith("4."):
        sent = sentiment_analysis(df_clean)
        if sent is None:
            st.error("Duygu analizi başlatılamadı. Yeni requirements dosyasını yeniden kurup uygulamayı kapatıp açın.")
            st.code("python3 -m pip install -r requirements.txt", language="bash")
        else:
            sent_df, sent_summary = sent
            a, b, c = st.columns(3)
            a.metric("Dezenformasyon ort. duygu", f"{sent_summary['yes_mean']:.3f}", f"Medyan {sent_summary['yes_median']:.3f}")
            b.metric("Güvenilir ort. duygu", f"{sent_summary['no_mean']:.3f}", f"Medyan {sent_summary['no_median']:.3f}")
            c.metric("Mann–Whitney p", f"{sent_summary['p_value']:.2e}")
            st.caption(f"Analiz motoru: {sent_summary['engine']} • Skor aralığı -1 (negatif) ile +1 (pozitif).")
            sent_df["Sınıf"] = sent_df["is_disinformation"].map({"yes": "Dezenformasyon", "no": "Güvenilir"})
            fig = px.box(
                sent_df, x="Sınıf", y="sentiment", color="is_disinformation", points=False,
                color_discrete_map=COLOR_MAP, title="VADER duygu skoru dağılımı",
                labels={"sentiment": "Compound skor (-1 / +1)"},
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(polish(fig, 450), use_container_width=True, config={"displayModeBar": False})
            if sent_summary["p_value"] < 0.05:
                st.markdown('<div class="insight"><b>İstatistiksel bulgu:</b> iki sınıfın duygu skoru dağılımları arasında istatistiksel olarak anlamlı bir farklılık saptanmıştır (p&lt;0.05). Bu bulgu ilişkisel niteliktedir ve nedensel bir sonuç olarak yorumlanmamalıdır.</div>', unsafe_allow_html=True)

    elif question.startswith("5."):
        n_top = st.slider("Gösterilecek n-gram sayısı", 8, 20, 12, key="ngram_top")
        bigram_df, trigram_df = ngram_analysis(df_clean, top_n=n_top)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(
                bigram_df.sort_values("count"), x="count", y="ngram", orientation="h",
                title="Dezenformasyon metinlerinde en sık bigramlar",
                labels={"count": "Geçiş sayısı", "ngram": "Bigram"},
            )
            fig.update_traces(marker_color=ACCENT)
            st.plotly_chart(polish(fig, 520), use_container_width=True, config={"displayModeBar": False})
        with c2:
            tri = trigram_df.copy()
            tri["Şablon"] = np.where(
                tri["ngram"].str.contains("fakes|narratives|propaganda|digest", case=False, regex=True),
                "İddia/propaganda kalıbı", "Diğer",
            )
            fig = px.bar(
                tri.sort_values("count"), x="count", y="ngram", orientation="h", color="Şablon",
                color_discrete_map={"İddia/propaganda kalıbı": GREEN, "Diğer": ACCENT},
                title="Dezenformasyon metinlerinde en sık trigramlar",
                labels={"count": "Geçiş sayısı", "ngram": "Trigram"},
            )
            st.plotly_chart(polish(fig, 520), use_container_width=True, config={"displayModeBar": False})

    elif question.startswith("6."):
        baseline_acc = float((y_test == y_test.value_counts().idxmax()).mean())
        main_names = ["Multinomial NB", "Complement NB", "Logistic Regression", "Linear SVM", "Random Forest"]
        if "XGBoost" in results_table["Model"].values:
            main_names.append("XGBoost")
        q6 = results_table[results_table["Model"].isin(main_names)].copy()
        long = q6.melt(id_vars="Model", value_vars=["Accuracy", "no Recall", "Macro F1"], var_name="Metrik", value_name="Skor")
        fig = px.bar(
            long, x="Model", y="Skor", color="Metrik", barmode="group", text_auto=".3f",
            title="Optimize modellerin canlı performans karşılaştırması", range_y=[0.4, 1.02],
        )
        fig.add_hline(y=baseline_acc, line_dash="dash", line_color="#C4695A", annotation_text=f"Çoğunluk baseline {baseline_acc:.3f}")
        fig.update_layout(xaxis_tickangle=-25)
        st.plotly_chart(polish(fig, 520), use_container_width=True, config={"displayModeBar": False})
        render_academic_table(
            q6,
            formats={
                "Accuracy": "{:.3f}", "no Recall": "{:.3f}",
                "no Precision": "{:.3f}", "yes Recall": "{:.3f}", "Macro F1": "{:.3f}",
            },
        )



# ================================================================
# 7) CANLI TAHMİN
# ================================================================
elif page == "🎯 Canlı Tahmin":
    hero(
        "Canlı Tahmin Demosu",
        "Kullanıcı tarafından girilen İngilizce metin, eğitim verisine uygulanan ön işleme ve TF-IDF dönüşüm adımlarından geçirilerek seçilen model aracılığıyla sınıflandırılmaktadır.",
        "INPUT → PIPELINE → TAHMİN",
    )
    if raw_df is None:
        st.info("Sol menüden veri setini yükleyin.")
        st.stop()

    best_model_name = results_table.iloc[0]["Model"]
    model_choice = st.selectbox(
        "Tahmin modeli",
        list(results_table["Model"]),
        index=list(results_table["Model"]).index(best_model_name),
    )

    examples = {
        "Hazır örnek seçme": "",
        "Nötr haber örneği": "Ukrainian officials reported new humanitarian aid deliveries arriving at the border today.",
        "İddialı içerik örneği": "BREAKING: Secret leaked documents allegedly prove Western officials are hiding the true casualty numbers, sources claim.",
    }
    example_choice = st.selectbox("Hazır örnek", list(examples))
    user_text = st.text_area("Metin", value=examples[example_choice], height=170, placeholder="İngilizce bir haber veya sosyal medya metni girin...")

    if st.button("Tahmin Et", type="primary", use_container_width=False):
        if not user_text.strip():
            st.warning("Önce bir metin girin.")
        else:
            cleaned = clean_single_text(user_text)
            vector = tfidf.transform([cleaned])
            vector = vector[:, keep_idx]
            model = models[model_choice]

            if model_choice == "XGBoost":
                pred_raw = int(model.predict(vector)[0])
                prediction = "yes" if pred_raw == 1 else "no"
                proba = model.predict_proba(vector)[0]
                confidence = float(proba[pred_raw])
            else:
                prediction = model.predict(vector)[0]
                confidence = None
                if hasattr(model, "predict_proba"):
                    classes = list(model.classes_)
                    proba = model.predict_proba(vector)[0]
                    confidence = float(proba[classes.index(prediction)])

            if prediction == "yes":
                st.error(f"Tahmin: DEZENFORMASYON • {model_choice}")
            else:
                st.success(f"Tahmin: GÜVENİLİR / GERÇEK HABER • {model_choice}")

            if confidence is not None:
                st.progress(confidence, text=f"Model güveni: %{confidence*100:.1f}")
            else:
                st.caption("Linear SVM doğrudan olasılık üretmediği için güven yüzdesi gösterilmez.")

            with st.expander("Modelin gördüğü temizlenmiş metin"):
                st.code(cleaned)

    st.caption("Bu modül araştırma ve eğitim amaçlı bir gösterimdir; profesyonel doğrulama süreçlerinin veya editoryal fact-checking uygulamalarının yerine kullanılmamalıdır.")

st.markdown(
    f'<div class="small-note" style="padding:22px 0 8px;">Kaynak dosya: {source_label or "yüklenmedi"} • Grafikler Plotly ile interaktiftir.</div>',
    unsafe_allow_html=True,
)
