# =============================================================================
# SPARKS — Stochastic Peatland Adaptive Risk & Kalman System
# Dasbor peringatan dini kekeringan gambut (prabencana), rancangan-final-v3.md.
# SATU BERKAS: semua kode Python ada di sini. Berkas lain yang dibutuhkan hanya
# kalimantan.geojson (batas peta), .env (MAP_KEY dan Project ID, opsional), dan
# .streamlit/config.toml (tema gelap).
#
# Menjalankan dasbor :  streamlit run app.py   (pilih tahun + bulan, tekan "Tampilkan periode ini")
# Data dasbor        :  python app.py proses --tahun 2026 --bulan-mulai 6 --bulan-selesai 10
# Analisis esai      :  python app.py esai   (kotak pilot Sentinel-1, latih 2018 / uji 2019 -> hasil/)
# Memakai fungsinya  :  import app  (dasbor tidak ikut berjalan)
#
# Isi berkas (cari "BAGIAN" untuk melompat):
#   BAGIAN 1  Tema tampilan (dari SIAGA-RESTORE v3.4)
#   BAGIAN 2  Hitungan inti: Kalman, peluang kering, level, knapsack, validasi, FIRMS
#   BAGIAN 3  Pipeline esai (kotak pilot Sentinel-1 100 m): indeks kebasahan relatif, MLE,
#             Drought Code, DeLong, olah sampai hasil esai di hasil/
#   BAGIAN 4  Data dasbor se-Kalimantan per kabupaten: NASA SMAP L4 (muka air gambut) + FIRMS,
#             2016 sampai ±4 hari lalu
#   BAGIAN 5  Dasbor Streamlit 3 tab (peta, level, anggaran)
# =============================================================================

from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
from datetime import date, timedelta
from io import StringIO
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st
from scipy.optimize import Bounds, LinearConstraint, milp, minimize
from scipy.stats import norm

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass


# =============================================================================
# BAGIAN 1. TEMA TAMPILAN
# Disalin dari SIAGA-RESTORE v3.4: tema gelap kontras >= 4,5:1, perbaikan HTML
# st.markdown, topeng peta Kalimantan. Hanya tampilan; tidak ada hitungan model.
# =============================================================================
_st_markdown_asli = st.markdown


def _markdown_html_rapi(body, unsafe_allow_html: bool = False, **kwargs):
    if unsafe_allow_html and isinstance(body, str):
        body = "\n".join(baris.strip() for baris in body.splitlines() if baris.strip())
    return _st_markdown_asli(body, unsafe_allow_html=unsafe_allow_html, **kwargs)


st.markdown = _markdown_html_rapi


C = {
    "bg": "#0a1628", "bg_soft": "#0f1f33", "panel": "#132b45",
    "panel_hover": "#1a3a5c", "panel_active": "#1e4570",
    "border": "#1f3a5c", "border_soft": "#16304d", "border_active": "#2a5a8c",
    "text": "#e8f0f7", "text_muted": "#b3c5d6", "text_dim": "#9fb4c8",  # kontras >= 4,5:1
    "accent": "#00d4ff", "accent_soft": "#0099cc",
    "danger": "#ff5a5a", "warning": "#ffb800", "success": "#00d47e",
    "purple": "#a78bfa", "orange": "#ff7a45",
}

KALIMANTAN_GEOJSON = Path(__file__).resolve().parent / "kalimantan.geojson"


def inject_css():
    st.markdown(f"""
    <style>
    .stApp {{ background: {C['bg']}; color: {C['text']}; }}
    .block-container {{ padding: 0.6rem 1.4rem 3rem; max-width: 100%; }}
    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif;
    }}
    h1, h2, h3, h4, h5, h6 {{ color: {C['text']}; }}
    /* Huruf dasar terang di atas latar gelap. Tombol & input di bawah ini
       mendapat aturan !important sendiri supaya tidak ikut putih di atas putih. */
    p, span, div, label, li {{ color: {C['text']}; }}
    [data-baseweb="tab"] p {{ color: {C['text_muted']} !important; font-weight: 600; }}
    [data-baseweb="tab"][aria-selected="true"] p {{ color: {C['accent']} !important; }}

    /* Huruf di tombol berlatar biru muda: gelap agar terbaca. */
    .stButton button, .stButton button * {{ color: {C['bg']} !important; }}

    /* Widget input, pilihan, angka: latar gelap + huruf terang. */
    [data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="select"] > div,
    [data-baseweb="textarea"], .stNumberInput > div > div {{
        background: {C['bg']} !important; border-color: {C['border_active']} !important;
    }}
    [data-baseweb="input"] input, [data-baseweb="base-input"] input, [data-baseweb="select"] *,
    [data-baseweb="textarea"] textarea {{
        color: {C['text']} !important; -webkit-text-fill-color: {C['text']} !important;
    }}
    input::placeholder, textarea::placeholder {{
        color: {C['text_dim']} !important; -webkit-text-fill-color: {C['text_dim']} !important;
    }}
    .stNumberInput button {{ background: {C['panel']} !important; }}
    [data-baseweb="input"] button {{ background: {C['panel']} !important; }}
    [data-testid="stTextInputRootElement"], [data-testid="stNumberInputContainer"] {{
        background: {C['bg']} !important; border-color: {C['border_active']} !important;
    }}
    [data-testid="stTextInputRootElement"] svg, [data-testid="stTextInputRootElement"] button * {{
        fill: {C['text']} !important; color: {C['text']} !important;
    }}
    [data-baseweb="input"] button svg {{ fill: {C['text']} !important; color: {C['text']} !important; }}
    .stNumberInput button svg {{ fill: {C['text']} !important; color: {C['text']} !important; }}
    [data-baseweb="popover"] li, [data-baseweb="menu"] li {{
        background: {C['bg_soft']} !important; color: {C['text']} !important;
    }}
    /* Kotak centang yang belum dicentang: tepi terlihat, bukan kotak putih polos. */
    [data-testid="stCheckbox"] label > span:first-child {{
        background: {C['bg']} !important; border: 2px solid {C['text_muted']} !important;
    }}
    /* Teks bantu, keterangan, judul expander, blok kode */
    [data-testid="stCaptionContainer"], [data-testid="stWidgetLabel"] * {{ color: {C['text_muted']} !important; }}
    [data-testid="stWidgetLabel"] p {{ color: {C['text']} !important; }}
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {{ color: {C['text']} !important; }}
    pre, code, [data-testid="stCode"] pre {{ background: {C['bg_soft']} !important; color: {C['text']} !important; }}
    [data-testid="stTooltipIcon"] svg {{ stroke: {C['text_muted']} !important; }}

    /* FIX UTAMA: hanya sembunyikan MainMenu & footer — header BIARKAN */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    .stDeployButton {{ display: none; }}

    /* Header jadi transparan tapi tombol tetap muncul */
    header[data-testid="stHeader"] {{
        background: transparent !important;
    }}

    /* Tombol BUKA sidebar (muncul saat sidebar tertutup).
       Streamlit >= 1.4x memakai data-testid="stExpandSidebarButton";
       selector lama dipertahankan untuk versi lama. */
    header[data-testid="stHeader"] {{ z-index: 999990 !important; }}
    [data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="collapsedControl"] button {{
        visibility: visible !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 6px !important;
        opacity: 1 !important;
        background: {C['accent']} !important;
        border: 2px solid {C['accent']} !important;
        border-radius: 10px !important;
        padding: 6px 14px 6px 10px !important;
        margin: 6px 0 0 6px !important;
        box-shadow: 0 4px 16px rgba(0,212,255,0.45) !important;
    }}
    [data-testid="stExpandSidebarButton"] span,
    [data-testid="stSidebarCollapsedControl"] button span,
    [data-testid="collapsedControl"] button span {{
        color: {C['bg']} !important;
        font-size: 1.4rem !important;
    }}
    [data-testid="stExpandSidebarButton"]::after,
    [data-testid="stSidebarCollapsedControl"] button::after,
    [data-testid="collapsedControl"] button::after {{
        content: "Menu";
        color: {C['bg']};
        font-weight: 800;
        font-size: .9rem;
    }}
    [data-testid="stExpandSidebarButton"]:hover {{ filter: brightness(1.15); }}

    /* Tombol TUTUP sidebar: selalu terlihat, bukan hanya saat kursor di atasnya. */
    [data-testid="stSidebarCollapseButton"] {{
        visibility: visible !important;
        display: block !important;
        opacity: 1 !important;
    }}
    [data-testid="stSidebarCollapseButton"] button {{
        visibility: visible !important;
        opacity: 1 !important;
        background: {C['panel']} !important;
        border: 1px solid {C['accent']} !important;
        border-radius: 8px !important;
    }}
    [data-testid="stSidebarCollapseButton"] span {{ color: {C['accent']} !important; }}

    /* Beri ruang agar tombol Menu tidak tertutup topbar. */
    .block-container {{ padding-top: 3.4rem !important; }}

    .topbar {{
        display: flex; align-items: center; justify-content: space-between;
        padding: 14px 24px;
        background: linear-gradient(90deg, {C['bg_soft']} 0%, {C['panel']} 100%);
        border-radius: 14px; border: 1px solid {C['border']};
        margin-bottom: 14px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    }}
    .topbar .brand {{ display: flex; align-items: center; gap: 14px; }}
    .topbar .brand-icon {{
        font-size: 2rem; filter: drop-shadow(0 0 8px {C['danger']});
    }}
    .topbar .brand-title {{
        font-size: 1.55rem; font-weight: 800;
        letter-spacing: 2px; color: {C['text']}; line-height: 1;
    }}
    .topbar .brand-sub {{
        font-size: .7rem; color: {C['text_muted']};
        letter-spacing: 2.5px; text-transform: uppercase; margin-top: 4px;
    }}
    .topbar .status {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}

    .status-pill {{
        display: inline-flex; align-items: center; gap: 6px;
        padding: 6px 13px; border-radius: 20px;
        font-size: .72rem; font-weight: 600; letter-spacing: .5px;
        border: 1px solid; white-space: nowrap;
    }}
    .status-pill.live {{
        background: rgba(0,212,126,.12);
        border-color: rgba(0,212,126,.4); color: {C['success']};
    }}
    .status-pill.offline {{
        background: rgba(139,163,184,.10);
        border-color: rgba(139,163,184,.3); color: {C['text_muted']};
    }}
    .status-pill.info {{
        background: rgba(0,212,255,.10);
        border-color: rgba(0,212,255,.35); color: {C['accent']};
    }}
    .status-pill.warn {{
        background: rgba(255,184,0,.10);
        border-color: rgba(255,184,0,.35); color: {C['warning']};
    }}
    .pulse {{
        width: 8px; height: 8px; border-radius: 50%;
        background: {C['success']}; box-shadow: 0 0 8px {C['success']};
        animation: pulse 1.8s infinite;
    }}
    .pulse.red {{ background: {C['danger']}; box-shadow: 0 0 8px {C['danger']}; }}
    .pulse.amber {{ background: {C['warning']}; box-shadow: 0 0 8px {C['warning']}; }}
    @keyframes pulse {{
        0%,100% {{ opacity:1; transform:scale(1); }}
        50% {{ opacity:.4; transform:scale(1.3); }}
    }}

    .kpi-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 10px; margin-bottom: 16px;
    }}
    .kpi-box {{
        background: linear-gradient(135deg, {C['panel']} 0%, {C['bg_soft']} 100%);
        border: 1px solid {C['border']}; border-radius: 12px;
        padding: 14px 16px; position: relative; overflow: hidden;
        transition: all .25s ease;
    }}
    .kpi-box::before {{
        content: ""; position: absolute; left: 0; top: 0; bottom: 0;
        width: 3px; background: {C['accent']};
    }}
    .kpi-box.danger::before  {{ background: {C['danger']}; }}
    .kpi-box.warning::before {{ background: {C['warning']}; }}
    .kpi-box.success::before {{ background: {C['success']}; }}
    .kpi-box.purple::before  {{ background: {C['purple']}; }}
    .kpi-box.orange::before  {{ background: {C['orange']}; }}
    .kpi-box:hover {{
        border-color: {C['border_active']};
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,212,255,.10);
    }}
    .kpi-box .kpi-icon {{ font-size: 1rem; opacity: .85; margin-bottom: 5px; }}
    .kpi-box .kpi-label {{
        font-size: .66rem; font-weight: 600;
        color: {C['text_muted']};
        letter-spacing: 1.2px; text-transform: uppercase;
    }}
    .kpi-box .kpi-value {{
        font-size: 1.75rem; font-weight: 700;
        color: {C['text']}; letter-spacing: -.5px;
        line-height: 1.1; margin-top: 4px;
        font-variant-numeric: tabular-nums;
    }}
    .kpi-box .kpi-hint {{
        font-size: .7rem; color: {C['text_dim']}; margin-top: 4px;
    }}

    .panel-title {{
        display: flex; align-items: center; justify-content: space-between;
        font-size: .78rem; font-weight: 700; color: {C['text']};
        letter-spacing: 1.5px; text-transform: uppercase;
        padding-bottom: 10px; border-bottom: 1px solid {C['border_soft']};
        margin-bottom: 12px;
    }}
    .panel-title .accent-line {{
        display: inline-block; width: 3px; height: 13px;
        background: {C['accent']}; margin-right: 8px;
        vertical-align: middle; box-shadow: 0 0 8px {C['accent']};
    }}
    .panel-title small {{
        font-size: .66rem; color: {C['text_muted']};
        letter-spacing: .5px; text-transform: none; font-weight: 500;
    }}

    .sec-head {{
        display: flex; align-items: center; gap: 12px; margin: 16px 0 14px;
    }}
    .sec-head .num {{
        width: 36px; height: 36px; border-radius: 10px;
        background: linear-gradient(135deg, {C['accent']}, {C['accent_soft']});
        color: {C['bg']}; font-weight: 800; font-size: 1rem;
        display: flex; align-items: center; justify-content: center;
        box-shadow: 0 0 16px rgba(0,212,255,.3);
    }}
    .sec-head .txt h3 {{ margin: 0; font-size: 1.1rem; font-weight: 700; }}
    .sec-head .txt p {{ margin: 2px 0 0; font-size: .8rem; color: {C['text_muted']}; }}

    section[data-testid="stSidebar"] {{
        background: {C['bg_soft']};
        border-right: 1px solid {C['border']};
    }}
    section[data-testid="stSidebar"] .block-container {{ padding-top: 1rem; }}
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{
        color: {C['text']}; font-size: .82rem;
        letter-spacing: 1.2px; text-transform: uppercase;
        border-bottom: 1px solid {C['border']};
        padding-bottom: 8px; margin-bottom: 12px;
    }}

    .stTextInput input, .stNumberInput input {{
        background: {C['bg']} !important; color: {C['text']} !important;
        border: 1px solid {C['border']} !important; border-radius: 8px !important;
    }}
    .stTextInput input:focus, .stNumberInput input:focus {{
        border-color: {C['accent']} !important;
        box-shadow: 0 0 0 2px rgba(0,212,255,.15) !important;
    }}
    .stCheckbox label, .stRadio label {{ color: {C['text']} !important; }}

    .stButton button {{
        background: linear-gradient(135deg, {C['accent']}, {C['accent_soft']});
        color: {C['bg']} !important; border: none; border-radius: 10px;
        font-weight: 700; letter-spacing: .5px; padding: 8px 16px;
        transition: all .2s; box-shadow: 0 4px 12px rgba(0,212,255,.25);
    }}
    .stButton button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(0,212,255,.4);
    }}
    .stDownloadButton > button {{
        background: {C['panel']}; color: {C['text']} !important;
        border: 1px solid {C['border']}; border-radius: 10px; font-weight: 600;
    }}
    .stDownloadButton > button:hover {{
        border-color: {C['accent']}; color: {C['accent']} !important;
    }}

    div[data-testid="stDataFrame"] {{
        background: {C['bg_soft']};
        border-radius: 10px; border: 1px solid {C['border']};
    }}
    div[data-testid="stDataFrame"] * {{ color: {C['text']} !important; }}

    .legend-row {{
        display: flex; gap: 14px; align-items: center;
        padding: 10px 14px; background: {C['bg']};
        border-radius: 10px; border: 1px solid {C['border_soft']};
        font-size: .75rem; margin-top: 8px; flex-wrap: wrap;
    }}
    .legend-row .li {{
        display: flex; align-items: center; gap: 6px;
        color: {C['text_muted']};
    }}
    .legend-row .dot {{ width: 9px; height: 9px; border-radius: 50%; }}

    .alert-item {{
        display: flex; align-items: center; gap: 12px;
        padding: 10px 12px; background: {C['bg']};
        border-left: 3px solid {C['danger']};
        border-radius: 8px; margin-bottom: 6px;
    }}
    .alert-item.medium {{ border-left-color: {C['warning']}; }}
    .alert-item.low {{ border-left-color: {C['success']}; }}
    .alert-item .ico {{ font-size: 1.05rem; }}
    .alert-item .body {{ flex: 1; min-width: 0; }}
    .alert-item .body .t1 {{
        font-size: .8rem; font-weight: 600; color: {C['text']};
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }}
    .alert-item .body .t2 {{
        font-size: .68rem; color: {C['text_muted']};
        margin-top: 2px; font-variant-numeric: tabular-nums;
    }}
    .alert-item .val {{ font-size: .92rem; font-weight: 700; }}
    .alert-item .val.high {{ color: {C['danger']}; }}
    .alert-item .val.medium {{ color: {C['warning']}; }}
    .alert-item .val.low {{ color: {C['success']}; }}

    .info-strip {{
        display: flex; gap: 10px; align-items: flex-start;
        background: rgba(0,212,255,.06);
        border: 1px solid rgba(0,212,255,.2);
        border-left: 3px solid {C['accent']};
        border-radius: 10px; padding: 11px 15px;
        font-size: .8rem; color: {C['text']}; margin: 8px 0;
    }}
    .info-strip.warn {{
        background: rgba(255,184,0,.06);
        border-color: rgba(255,184,0,.25);
        border-left-color: {C['warning']};
    }}
    .info-strip.err {{
        background: rgba(255,59,59,.06);
        border-color: rgba(255,59,59,.25);
        border-left-color: {C['danger']};
    }}
    .info-strip.ok {{
        background: rgba(0,212,126,.06);
        border-color: rgba(0,212,126,.25);
        border-left-color: {C['success']};
    }}

    .stat-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
        gap: 10px; margin-top: 10px;
    }}
    .stat-item {{
        background: {C['bg']}; border: 1px solid {C['border_soft']};
        border-radius: 10px; padding: 10px 13px;
    }}
    .stat-item .lbl {{
        font-size: .66rem; color: {C['text_muted']};
        text-transform: uppercase; letter-spacing: .6px;
    }}
    .stat-item .val {{
        font-size: 1.05rem; font-weight: 600;
        color: {C['text']}; margin-top: 3px;
        font-variant-numeric: tabular-nums;
    }}

    .footer {{
        text-align: center; color: {C['text_dim']};
        font-size: .7rem; padding: 20px 0 10px;
        margin-top: 30px; border-top: 1px solid {C['border']};
        letter-spacing: .5px; line-height: 1.7;
    }}
    .footer b {{ color: {C['text_muted']}; }}

    .cart-item {{
        display: flex; align-items: center; gap: 10px;
        padding: 9px 11px; background: {C['bg']};
        border: 1px solid {C['border_soft']};
        border-radius: 8px; margin-bottom: 6px;
    }}
    .cart-item .info {{ flex: 1; min-width: 0; }}
    .cart-item .info .id {{
        font-size: .75rem; font-weight: 600; color: {C['text']};
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }}
    .cart-item .info .meta {{
        font-size: .66rem; color: {C['text_muted']}; margin-top: 2px;
    }}
    .cart-item .price {{
        font-size: .82rem; font-weight: 700; color: {C['accent']};
    }}

    .detail-card {{
        background: linear-gradient(135deg, {C['panel']} 0%, {C['bg_soft']} 100%);
        border: 1px solid {C['accent']}; border-radius: 12px;
        padding: 14px 16px; box-shadow: 0 0 24px rgba(0,212,255,.15);
        margin-bottom: 12px;
    }}
    .detail-card .header {{
        display: flex; align-items: center; justify-content: space-between;
        margin-bottom: 10px;
    }}
    .detail-card .header .id {{
        font-size: .9rem; font-weight: 700; color: {C['accent']};
    }}
    .detail-card .header .badge {{
        font-size: .65rem; font-weight: 700;
        padding: 3px 9px; border-radius: 6px;
    }}
    .detail-card .grid {{
        display: grid; grid-template-columns: 1fr 1fr;
        gap: 8px; margin-top: 10px;
    }}
    .detail-card .grid .cell {{
        background: {C['bg']}; border-radius: 8px; padding: 8px 10px;
    }}
    .detail-card .grid .cell .k {{
        font-size: .62rem; color: {C['text_muted']};
        text-transform: uppercase; letter-spacing: .6px;
    }}
    .detail-card .grid .cell .v {{
        font-size: .95rem; font-weight: 700; color: {C['text']};
        margin-top: 2px; font-variant-numeric: tabular-nums;
    }}

    .feature-bar {{
        display: flex; align-items: center; gap: 10px;
        padding: 10px 14px; background: {C['bg']};
        border-radius: 10px; border: 1px solid {C['border_soft']};
        margin-bottom: 6px; font-size: .78rem;
    }}
    .feature-bar .name {{ flex: 1; color: {C['text']}; font-weight: 500; }}
    .feature-bar .bar-wrap {{
        flex: 2; height: 6px; background: {C['bg_soft']};
        border-radius: 3px; overflow: hidden;
    }}
    .feature-bar .bar-fill {{
        height: 100%;
        background: linear-gradient(90deg, {C['accent']}, {C['purple']});
        border-radius: 3px;
    }}
    .feature-bar .val {{
        font-size: .72rem; color: {C['text_muted']};
        min-width: 42px; text-align: right;
    }}

    .progress-wrap {{
        background: {C['bg']}; border: 1px solid {C['border']};
        border-radius: 12px; padding: 14px 18px; margin-top: 10px;
    }}
    .progress-head {{
        display: flex; justify-content: space-between;
        align-items: center; margin-bottom: 8px;
    }}
    .progress-head .label {{
        font-size: .75rem; color: {C['text_muted']};
        letter-spacing: .6px; text-transform: uppercase;
    }}
    .progress-head .pct {{
        font-size: 1.05rem; font-weight: 700; color: {C['accent']};
    }}
    .progress-track {{
        height: 10px; background: {C['bg_soft']};
        border-radius: 6px; overflow: hidden;
    }}
    .progress-fill {{
        height: 100%;
        background: linear-gradient(90deg, {C['accent']}, {C['success']});
        border-radius: 6px;
    }}
    .progress-foot {{
        display: flex; justify-content: space-between;
        margin-top: 6px; font-size: .7rem; color: {C['text_dim']};
    }}

    .hint-chip {{
        display: inline-block; padding: 3px 9px;
        background: rgba(0,212,255,.08);
        border: 1px solid rgba(0,212,255,.25);
        border-radius: 6px; font-size: .68rem; color: {C['accent']};
    }}

    .empty-state {{
        text-align: center; padding: 40px 20px;
        background: {C['bg_soft']}; border: 1px dashed {C['border']};
        border-radius: 12px; color: {C['text_muted']};
    }}
    .empty-state .icon {{ font-size: 3rem; margin-bottom: 12px; opacity: .6; }}
    .empty-state .msg {{ font-size: .9rem; }}

    .hotspot-row {{
        display: flex; align-items: center; gap: 10px;
        padding: 8px 12px; background: {C['bg']};
        border: 1px solid {C['border_soft']};
        border-radius: 8px; margin-bottom: 5px; font-size: .78rem;
    }}
    .hotspot-row.selected {{
        border-color: {C['accent']}; background: {C['panel_active']};
        box-shadow: 0 0 12px rgba(0,212,255,.2);
    }}
    .hotspot-row .num {{
        width: 26px; height: 26px; border-radius: 6px;
        background: {C['bg_soft']}; display: flex; align-items: center;
        justify-content: center; font-size: .7rem; font-weight: 700;
        color: {C['text_muted']}; flex-shrink: 0;
    }}
    .hotspot-row .coords {{
        flex: 1; color: {C['text']};
        font-variant-numeric: tabular-nums; font-size: .75rem;
    }}
    .hotspot-row .cat {{
        font-size: .68rem; font-weight: 700;
        padding: 2px 8px; border-radius: 5px; flex-shrink: 0;
    }}
    .hotspot-row .score {{
        font-weight: 700; min-width: 34px; text-align: right; flex-shrink: 0;
    }}

    section[data-testid="stSidebar"] .stButton button {{
        padding: 6px 12px; font-size: .85rem;
    }}

    /* Tombol periode aktif */
    section[data-testid="stSidebar"] button[kind="primary"] {{
        background: linear-gradient(135deg, {C['success']}, {C['accent']}) !important;
        color: {C['bg']} !important;
        box-shadow: 0 0 20px rgba(0,212,126,.4) !important;
        border: 1px solid {C['success']} !important;
    }}
    </style>
    """, unsafe_allow_html=True)


def section_head(num: str, title: str, desc: str = ""):
    st.markdown(f"""
    <div class="sec-head">
        <div class="num">{num}</div>
        <div class="txt">
            <h3>{title}</h3>
            {f'<p>{desc}</p>' if desc else ''}
        </div>
    </div>
    """, unsafe_allow_html=True)


def kpi_grid(items: list, kolom: int | None = None):
    gaya = f' style="grid-template-columns:repeat({kolom},1fr)"' if kolom else ""
    html = f'<div class="kpi-grid"{gaya}>'
    for it in items:
        html += f"""
        <div class="kpi-box {it.get('accent', '')}">
            
            <div class="kpi-label">{it['label']}</div>
            <div class="kpi-value">{it['value']}</div>
            {f'<div class="kpi-hint">{it["hint"]}</div>' if it.get('hint') else ''}
        </div>
        """
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def info_strip(text: str, kind: str = ""):
    icons = {"": "", "warn": "", "err": "", "ok": ""}
    st.markdown(
        f'<div class="info-strip {kind}"><div>{text}</div></div>',
        unsafe_allow_html=True)


def legend_row(items: list):
    html = '<div class="legend-row">'
    for label, color in items:
        html += f'<div class="li"><span class="dot" style="background:{color};box-shadow:0 0 6px {color}"></span>{label}</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def empty_state(icon: str, msg: str):
    st.markdown(f"""
    <div class="empty-state">
        
        <div class="msg">{msg}</div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================


def _geo_kalimantan():
    """Batas provinsi Kalimantan (Natural Earth 1:10m, domain publik)."""
    if not KALIMANTAN_GEOJSON.exists():
        return None
    return json.loads(KALIMANTAN_GEOJSON.read_text(encoding="utf-8"))


def layer_kalimantan() -> list:
    """Topeng gelap di luar Kalimantan + garis batas provinsi.

    Topeng = kotak seluruh dunia dengan "lubang" berbentuk daratan Kalimantan,
    sehingga yang terlihat di peta hanya wilayah Kalimantan.
    """
    geo = _geo_kalimantan()
    if geo is None:
        st.caption("kalimantan.geojson tidak ditemukan di folder app.py; peta menampilkan semua wilayah.")
        return []
    daratan = next(f for f in geo["features"] if f["properties"]["provinsi"] == "__daratan__")
    provinsi = {"type": "FeatureCollection",
                "features": [f for f in geo["features"] if f["properties"]["provinsi"] != "__daratan__"]}
    dunia = [[-180, -85], [180, -85], [180, 85], [-180, 85], [-180, -85]]
    lubang = [poly[0] for poly in daratan["geometry"]["coordinates"]]
    topeng = {"type": "Feature", "properties": {},
              "geometry": {"type": "Polygon", "coordinates": [dunia] + lubang}}
    bg = [int(C["bg"][i:i + 2], 16) for i in (1, 3, 5)]
    return [
        pdk.Layer("GeoJsonLayer", topeng, id="topeng", filled=True, stroked=False,
                  get_fill_color=bg + [255], pickable=False),
        # Hanya garis: isian di sini menimpa lapisan kabupaten (sama-sama di ketinggian 0).
        pdk.Layer("GeoJsonLayer", provinsi, id="batas-provinsi", filled=False, stroked=True,
                  get_line_color=[180, 205, 225, 200], line_width_min_pixels=1.2, pickable=False,
                  parameters={"depthTest": False}),
    ]

# =============================================================================
# BAGIAN 2. HITUNGAN INTI
# Hitungan inti SPARKS (tanpa Streamlit), sesuai rancangan-final-v3.md.
#
# Lapis 2  : filter Kalman 1-D dan peluang kering p = Pr(h < -0,40 m).
# Lapis 4  : ambang level peringatan dari biaya harapan c_FN*FN + c_FP*FP.
# Lapis 5  : manfaat blok v_k = sum(p_i * A_i), knapsack 0-1 (MILP eksak + greedy),
#            kurva anggaran-manfaat, prioritas kokoh, tangkapan hotspot.
# Validasi : label api dari hotspot VIIRS minggu t+1..t+k (tidak pernah dipakai
#            sebagai masukan), AUC-ROC, AUC-PR, Brier, kurva reliabilitas.
# FIRMS    : pengambilan arsip hotspot NASA FIRMS (API area) dengan cache per potongan.
# =============================================================================
AMBANG_MUKA_AIR_M = -0.40       # PP 57/2016; Taufik dkk. 2021
LUAS_PETAK_HA = 1.0             # petak 100 m x 100 m
KM_PER_DERAJAT = 111.32
LEVEL = ["Aman", "Waspada", "Siaga", "Awas"]


# =============================================================================
# LAPIS 2 — FILTER KALMAN DAN PELUANG KERING
# =============================================================================

def kalman_muka_air(hujan, et, z, R, a, b, c, h_ref, Q, h0, P0):
    """Filter Kalman 1-D per petak, langkah harian (bisa banyak petak sekaligus).

    h_t = h_{t-1} + a*P_t - b*ET_t - c*(h_{t-1} - h_ref) + eta,  eta ~ N(0, Q)
    z_t = h_t + eps,  eps ~ N(0, R_t);  z_t = NaN bila tidak ada citra hari itu.
    hujan, et, z, R berbentuk (T,) atau (T, N); parameter skalar atau (N,).
    Mengembalikan (h_hat, var) dengan bentuk yang sama dengan z.
    """
    hujan, et, z, R = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (hujan, et, z, R)))
    T = z.shape[0]
    h_hat, var = np.empty(z.shape), np.empty(z.shape)
    h = np.broadcast_to(np.asarray(h0, dtype=float), z.shape[1:]).copy()
    P = np.broadcast_to(np.asarray(P0, dtype=float), z.shape[1:]).copy()
    phi = 1.0 - np.asarray(c, dtype=float)
    for t in range(T):
        h = phi * h + c * h_ref + a * hujan[t] - b * et[t]
        P = phi * phi * P + Q
        ada = np.isfinite(z[t])
        K = np.where(ada, P / (P + R[t]), 0.0)
        h = h + K * np.where(ada, z[t] - h, 0.0)
        P = (1.0 - K) * P
        h_hat[t], var[t] = h, P
    return h_hat, var


def peluang_kering(h_hat, sigma, ambang: float = AMBANG_MUKA_AIR_M):
    """p = Pr(h < ambang) = Phi((ambang - h_hat) / sigma)."""
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-9)
    return norm.cdf((ambang - np.asarray(h_hat, dtype=float)) / sigma)


def lengkapi_peluang(df: pd.DataFrame) -> pd.DataFrame:
    """Isi kolom p, p_bawah, p_atas dari h_hat_m dan sigma_m bila belum ada.

    p_bawah/p_atas memakai h_hat -/+ 1,645 sigma (selaras interval 90%).
    Jika ada kolom h_bawah_m/h_atas_m (interval conformal), kolom itu dipakai.
    """
    d = df.copy()
    if "p" not in d:
        d["p"] = peluang_kering(d["h_hat_m"], d["sigma_m"])
    if "p_bawah" not in d or "p_atas" not in d:
        if {"h_bawah_m", "h_atas_m", "sigma_m"}.issubset(d.columns):
            # skenario: estimasi digeser ke batas interval conformal
            d["p_bawah"] = peluang_kering(d["h_atas_m"], d["sigma_m"])
            d["p_atas"] = peluang_kering(d["h_bawah_m"], d["sigma_m"])
        elif {"h_hat_m", "sigma_m"}.issubset(d.columns):
            d["p_bawah"] = peluang_kering(d["h_hat_m"] + 1.645 * d["sigma_m"], d["sigma_m"])
            d["p_atas"] = peluang_kering(d["h_hat_m"] - 1.645 * d["sigma_m"], d["sigma_m"])
        else:
            d["p_bawah"] = d["p_atas"] = d["p"]
    return d


# =============================================================================
# LAPIS 4 — LEVEL PERINGATAN
# =============================================================================

def ambang_biaya_minimum(y, skor, rasio_biaya: float) -> float:
    """Ambang t yang meminimalkan c_FN*FN + c_FP*FP dengan c_FN/c_FP = rasio_biaya.

    Petak diberi peringatan bila skor >= t. Kandidat t = semua nilai skor unik.
    """
    y = np.asarray(y, dtype=bool)
    s = np.asarray(skor, dtype=float)
    if len(s) == 0 or y.all() or (~y).all():
        return float("nan")
    order = np.argsort(-s)
    s_sorted, y_sorted = s[order], y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(~y_sorted)
    fn = y.sum() - tp
    biaya = rasio_biaya * fn + fp
    # hanya titik di akhir blok nilai skor yang sama
    akhir = np.r_[s_sorted[1:] != s_sorted[:-1], True]
    i = np.argmin(np.where(akhir, biaya, np.inf))
    return float(s_sorted[i])


def ambang_level(y, skor, rasio_biaya: float) -> tuple[float, float, float] | None:
    """Ambang (Waspada, Siaga, Awas) [A]: Siaga dari rasio biaya pilihan pengguna,
    Waspada dari rasio 2x (lebih hati-hati), Awas dari rasio 0,5x."""
    t = [ambang_biaya_minimum(y, skor, rasio_biaya * f) for f in (2.0, 1.0, 0.5)]
    if any(np.isnan(t)):
        return None
    t = np.maximum.accumulate(t)  # jaga urutan Waspada <= Siaga <= Awas
    return float(t[0]), float(t[1]), float(t[2])


def beri_level(p, ambang: tuple[float, float, float]) -> np.ndarray:
    idx = np.searchsorted(np.asarray(ambang), np.asarray(p, dtype=float), side="right")
    return np.asarray(LEVEL, dtype=object)[idx]


# =============================================================================
# LAPIS 5 — BLOK PATROLI DAN KNAPSACK
# =============================================================================

def id_blok(lat, lon, ukuran_km: float = 1.0, lat_ref: float | None = None) -> np.ndarray:
    """Kode blok persegi ukuran_km dari koordinat petak."""
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    lat_ref = float(np.nanmean(lat)) if lat_ref is None else lat_ref
    dlat = ukuran_km / KM_PER_DERAJAT
    dlon = ukuran_km / (KM_PER_DERAJAT * np.cos(np.radians(lat_ref)))
    i, j = np.floor(lat / dlat).astype(int), np.floor(lon / dlon).astype(int)
    return np.array([f"B{a:+06d}{b:+07d}" for a, b in zip(i, j)])


def manfaat_blok(petak: pd.DataFrame, kolom_p: str = "p") -> pd.DataFrame:
    """v_k = sum_i p_i * A_i [ha] per blok (petak harus punya kolom blok_id)."""
    d = petak.assign(_v=petak[kolom_p] * petak.get("luas_ha", LUAS_PETAK_HA))
    g = d.groupby("blok_id")
    out = pd.DataFrame({
        "lat": g["lat"].mean(), "lon": g["lon"].mean(),
        "n_petak": g.size(), "luas_ha": g["luas_ha"].sum() if "luas_ha" in d else g.size() * LUAS_PETAK_HA,
        "manfaat_ha": g["_v"].sum(), "p_maks": g[kolom_p].max(),
    }).reset_index()
    return out


def knapsack_milp(manfaat, biaya, anggaran: float) -> tuple[np.ndarray, str]:
    """Knapsack 0-1 eksak: max sum v_k x_k  s.t. sum c_k x_k <= B, x biner (HiGHS)."""
    v, c = np.asarray(manfaat, dtype=float), np.asarray(biaya, dtype=float)
    n = len(v)
    if n == 0 or anggaran <= 0:
        return np.zeros(n, dtype=bool), "kosong"
    gratis = c <= 0  # biaya 0 selalu diambil bila manfaatnya positif
    # Skala ulang manfaat ke maksimum 1 (solusi optimal tidak berubah) supaya nilai p yang
    # sangat kecil tetap membedakan kandidat dan tidak hilang di toleransi solver.
    skala = v.max() if v.max() > 0 else 1.0
    res = milp(-v / skala, constraints=LinearConstraint(c[None, :], -np.inf, anggaran),
               integrality=np.ones(n), bounds=Bounds(0, 1))
    if res.status != 0 or res.x is None:
        return greedy(v, c, anggaran), "greedy (MILP gagal)"
    x = res.x > 0.5
    x |= gratis & (v > 0)
    return x, "optimal"


def greedy(manfaat, biaya, anggaran: float) -> np.ndarray:
    """Pembanding: ambil rasio manfaat/biaya terbesar lebih dulu selama muat."""
    v, c = np.asarray(manfaat, dtype=float), np.asarray(biaya, dtype=float)
    rasio = np.where(c > 0, v / np.where(c > 0, c, 1), np.inf)
    x, sisa = np.zeros(len(v), dtype=bool), float(anggaran)
    for i in np.argsort(-rasio, kind="stable"):
        if c[i] <= sisa and v[i] > 0:
            x[i], sisa = True, sisa - c[i]
    return x


def kurva_anggaran(manfaat, biaya, n_titik: int = 10) -> pd.DataFrame:
    """V*(B) untuk B = 10%, 20%, ..., 100% dari total biaya semua blok."""
    total = float(np.sum(biaya))
    baris = []
    for f in np.linspace(1 / n_titik, 1, n_titik):
        B = f * total
        x, _ = knapsack_milp(manfaat, biaya, B)
        g = greedy(manfaat, biaya, B)
        baris.append({"persen_anggaran": round(100 * f), "anggaran": B,
                      "manfaat_optimal_ha": float(np.sum(np.asarray(manfaat)[x])),
                      "manfaat_greedy_ha": float(np.sum(np.asarray(manfaat)[g]))})
    return pd.DataFrame(baris)


def manfaat_sekat(petak: pd.DataFrame, sekat: pd.DataFrame, radius_km: float = 1.0,
                  kolom_p: str = "p") -> pd.DataFrame:
    """v_k = sum p_i * A_i untuk petak dalam radius_km dari sekat kanal k."""
    from scipy.spatial import cKDTree
    lat0 = float(petak["lat"].mean())
    kx = KM_PER_DERAJAT * np.cos(np.radians(lat0))
    pts = np.c_[petak["lat"].to_numpy() * KM_PER_DERAJAT, petak["lon"].to_numpy() * kx]
    q = np.c_[sekat["lat"].to_numpy() * KM_PER_DERAJAT, sekat["lon"].to_numpy() * kx]
    tetangga = cKDTree(pts).query_ball_point(q, r=radius_km)
    v = (petak[kolom_p] * petak.get("luas_ha", LUAS_PETAK_HA)).to_numpy()
    p = petak[kolom_p].to_numpy()
    return pd.DataFrame({
        "sekat_id": sekat["sekat_id"].to_numpy(), "lat": sekat["lat"].to_numpy(),
        "lon": sekat["lon"].to_numpy(),
        "n_petak": [len(t) for t in tetangga],
        "manfaat_ha": [float(v[t].sum()) for t in tetangga],
        "p_maks": [float(p[t].max()) if t else 0.0 for t in tetangga],
    })


def prioritas_kokoh(blok_bawah, blok_tengah, blok_atas, biaya, anggaran) -> np.ndarray:
    """Jumlah skenario (0-3) tempat blok terpilih: manfaat dari p_bawah, p, p_atas."""
    return sum(knapsack_milp(v, biaya, anggaran)[0].astype(int)
               for v in (blok_bawah, blok_tengah, blok_atas))


# =============================================================================
# VALIDASI DENGAN HOTSPOT (HANYA SEBAGAI ALAT UKUR)
# =============================================================================

def petakan_hotspot(hotspot: pd.DataFrame, petak: pd.DataFrame, radius_km: float = 0.25) -> pd.DataFrame:
    """Pasangkan setiap hotspot ke petak terdekat (dalam radius_km) beserta minggunya."""
    from scipy.spatial import cKDTree
    if hotspot.empty or petak.empty:
        return hotspot.assign(petak_id=pd.Series(dtype=object), minggu=pd.Series(dtype="datetime64[ns]"))
    lat0 = float(petak["lat"].mean())
    kx = KM_PER_DERAJAT * np.cos(np.radians(lat0))
    pts = np.c_[petak["lat"].to_numpy() * KM_PER_DERAJAT, petak["lon"].to_numpy() * kx]
    q = np.c_[hotspot["latitude"].to_numpy() * KM_PER_DERAJAT, hotspot["longitude"].to_numpy() * kx]
    dist, idx = cKDTree(pts).query(q, distance_upper_bound=radius_km)
    ok = np.isfinite(dist)
    h = hotspot.loc[ok].copy()
    h["petak_id"] = petak["petak_id"].to_numpy()[idx[ok]]
    tgl = pd.to_datetime(h["acq_date"])
    h["minggu"] = (tgl - pd.to_timedelta(tgl.dt.weekday, unit="D")).dt.normalize()
    return h


def label_api(petak_mingguan: pd.DataFrame, hotspot_petak: pd.DataFrame, k_minggu: int = 2) -> pd.Series:
    """y_{i,t} = 1 bila ada hotspot di petak i pada minggu t+1..t+k (bukan minggu t).

    Hanya minggu masa depan yang dipakai, sehingga tidak ada kebocoran ke skor minggu t.
    """
    kunci = set()
    for pid, m in zip(hotspot_petak["petak_id"], pd.to_datetime(hotspot_petak["minggu"])):
        for j in range(1, k_minggu + 1):
            kunci.add((pid, m - pd.Timedelta(weeks=j)))
    m = pd.to_datetime(petak_mingguan["minggu"])
    return pd.Series([(a, b) in kunci for a, b in zip(petak_mingguan["petak_id"], m)],
                     index=petak_mingguan.index, name="api")


def metrik_prediksi(y, skor) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    y, s = np.asarray(y, dtype=int), np.asarray(skor, dtype=float)
    if y.min() == y.max():
        return {"auc_roc": float("nan"), "auc_pr": float("nan"), "n": len(y), "n_api": int(y.sum())}
    return {"auc_roc": float(roc_auc_score(y, s)), "auc_pr": float(average_precision_score(y, s)),
            "n": len(y), "n_api": int(y.sum()), "prevalensi": float(y.mean())}


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def kurva_reliabilitas(y, p, n_bin: int = 10) -> pd.DataFrame:
    d = pd.DataFrame({"y": np.asarray(y, dtype=float), "p": np.asarray(p, dtype=float)})
    d["bin"] = np.minimum((d["p"] * n_bin).astype(int), n_bin - 1)
    g = d.groupby("bin").agg(p_rata=("p", "mean"), frekuensi_api=("y", "mean"), n=("y", "size"))
    return g.reset_index()


def kurva_roc(y, skor) -> pd.DataFrame:
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(np.asarray(y, dtype=int), np.asarray(skor, dtype=float))
    return pd.DataFrame({"fpr": fpr, "tpr": tpr})


def tangkapan_hotspot(terpilih_blok: set, hotspot_petak: pd.DataFrame, peta_blok: dict) -> float:
    """Bagian hotspot (minggu-minggu berikutnya) yang jatuh di blok terpilih."""
    if hotspot_petak.empty:
        return float("nan")
    blok = hotspot_petak["petak_id"].map(peta_blok)
    return float(blok.isin(terpilih_blok).mean())


def uji_tangkapan(petak_mingguan: pd.DataFrame, hotspot_petak: pd.DataFrame, biaya_blok: pd.Series,
                  persen_anggaran: float, k_minggu: int = 2, kolom_dc: str | None = "dc",
                  seed: int = 0) -> pd.DataFrame:
    """Tangkapan hotspot per minggu: pilihan SPARKS vs Drought Code vs acak, anggaran sama.

    Anggaran tiap minggu = persen_anggaran% dari total biaya semua blok.
    """
    rng = np.random.default_rng(seed)
    peta_blok = dict(zip(petak_mingguan["petak_id"], petak_mingguan["blok_id"]))
    hp = hotspot_petak.assign(minggu=pd.to_datetime(hotspot_petak["minggu"]))
    hasil = []
    for m, d in petak_mingguan.groupby(pd.to_datetime(petak_mingguan["minggu"])):
        depan = hp[(hp["minggu"] > m) & (hp["minggu"] <= m + pd.Timedelta(weeks=k_minggu))]
        if depan.empty:
            continue
        blok = manfaat_blok(d)
        c = biaya_blok.reindex(blok["blok_id"]).fillna(1.0).to_numpy()
        B = persen_anggaran / 100 * c.sum()
        baris = {"minggu": m, "n_hotspot": len(depan)}
        x, _ = knapsack_milp(blok["manfaat_ha"], c, B)
        baris["SPARKS"] = tangkapan_hotspot(set(blok["blok_id"][x]), depan, peta_blok)
        if kolom_dc and kolom_dc in d:
            bdc = d.groupby("blok_id")[kolom_dc].mean().reindex(blok["blok_id"]).to_numpy()
            xdc, _ = knapsack_milp(bdc - np.nanmin(bdc) + 1e-6, c, B)
            baris["Drought Code"] = tangkapan_hotspot(set(blok["blok_id"][xdc]), depan, peta_blok)
        xr = greedy(rng.random(len(c)), c, B)
        baris["Acak"] = tangkapan_hotspot(set(blok["blok_id"][xr]), depan, peta_blok)
        hasil.append(baris)
    return pd.DataFrame(hasil)


# =============================================================================
# NASA FIRMS (DATA VALIDASI)
# =============================================================================

FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
FIRMS_HARI_MAKS = 5   # batas DAY_RANGE yang dipakai per permintaan
REQUEST_TIMEOUT = 60


class FIRMSError(RuntimeError):
    pass


def firms_url(api_key: str, sumber: str, bbox: tuple, mulai: date, n_hari: int) -> str:
    """Format FIRMS Area API: /{key}/{sumber}/{w,s,e,n}/{DAY_RANGE}/{DATE}."""
    if not 1 <= n_hari <= FIRMS_HARI_MAKS:
        raise FIRMSError(f"DAY_RANGE {n_hari} di luar 1–{FIRMS_HARI_MAKS}")
    w, s, e, n = bbox
    return f"{FIRMS_BASE_URL}/{api_key}/{sumber}/{w},{s},{e},{n}/{n_hari}/{mulai:%Y-%m-%d}"


def _baca_firms(teks: str) -> pd.DataFrame:
    if not teks.strip():
        return pd.DataFrame(columns=["latitude", "longitude", "acq_date"])
    df = pd.read_csv(StringIO(teks))
    if len(df.columns) == 1 and not {"latitude", "longitude"}.issubset(df.columns):
        raise FIRMSError(teks.strip()[:200])
    if not {"latitude", "longitude", "acq_date"}.issubset(df.columns):
        raise FIRMSError("Kolom latitude/longitude/acq_date tidak ditemukan")
    return df


def ambil_firms(api_key: str, bbox: tuple, mulai: date, selesai: date,
                sumber: str = "VIIRS_SNPP_SP", folder_cache: Path | None = None,
                sesi: requests.Session | None = None, paralel: int = 1) -> pd.DataFrame:
    """Ambil arsip hotspot dalam potongan 5 hari; tiap potongan disimpan di cache.
    paralel > 1: potongan yang belum ada di cache diminta beberapa sekaligus."""
    from concurrent.futures import ThreadPoolExecutor
    if not api_key:
        raise FIRMSError("MAP_KEY kosong")
    sesi = sesi or requests.Session()
    if folder_cache is not None:
        folder_cache.mkdir(parents=True, exist_ok=True)
    rencana, t = [], mulai
    while t <= selesai:
        n = min(FIRMS_HARI_MAKS, (selesai - t).days + 1)
        kunci = hashlib.md5(f"{sumber}|{bbox}|{t}|{n}".encode()).hexdigest()[:16]
        rencana.append((t, n, folder_cache / f"firms_{kunci}.csv" if folder_cache is not None else None))
        t += timedelta(days=n)

    def satu(t, n, f):
        if f is not None and f.exists():
            return f.read_text(encoding="utf-8")
        try:
            r = sesi.get(firms_url(api_key, sumber, bbox, t, n), timeout=REQUEST_TIMEOUT)
        except requests.RequestException as e:
            raise FIRMSError(f"Koneksi gagal: {e}") from e
        if r.status_code != 200:
            raise FIRMSError(f"HTTP {r.status_code}: {r.text[:200]}")
        _baca_firms(r.text)  # validasi sebelum disimpan
        if f is not None:
            f.write_text(r.text, encoding="utf-8")
        return r.text

    if paralel > 1:
        with ThreadPoolExecutor(paralel) as ex:
            teks = list(ex.map(lambda x: satu(*x), rencana))
    else:
        teks = [satu(*x) for x in rencana]
    potongan = [_baca_firms(x) for x in teks]
    df = pd.concat(potongan, ignore_index=True) if potongan else pd.DataFrame()
    if df.empty:
        return pd.DataFrame(columns=["latitude", "longitude", "acq_date"])
    df["acq_date"] = pd.to_datetime(df["acq_date"])
    df = df[(df["acq_date"] >= pd.Timestamp(mulai)) & (df["acq_date"] <= pd.Timestamp(selesai))]
    return df.drop_duplicates().reset_index(drop=True)


# =============================================================================
# DATA MASUKAN
# =============================================================================

KOLOM_WAJIB = {"petak_id", "lat", "lon", "minggu", "h_hat_m", "sigma_m"}


def siapkan_petak(df: pd.DataFrame, ukuran_blok_km: float = 1.0) -> pd.DataFrame:
    """Validasi berkas keluaran Lapis 0–2 dan tambahkan p, interval p, blok_id."""
    kurang = KOLOM_WAJIB - set(df.columns)
    if kurang and not ({"p"} <= set(df.columns) and (KOLOM_WAJIB - {"h_hat_m", "sigma_m"}) <= set(df.columns)):
        raise ValueError(f"Kolom wajib tidak ada: {sorted(kurang)}")
    d = df.copy()
    d["minggu"] = pd.to_datetime(d["minggu"])
    if "luas_ha" not in d:
        d["luas_ha"] = LUAS_PETAK_HA
    d = lengkapi_peluang(d)
    lat_ref = float(d["lat"].mean())
    d["blok_id"] = id_blok(d["lat"], d["lon"], ukuran_blok_km, lat_ref)
    return d

# =============================================================================
# BAGIAN 3. PIPELINE DATA ASLI
# Pipeline data asli SPARKS (Lapis 1–2 mode tanpa label SIPALAGA).
#
# Masukan (diambil dari Google Earth Engine oleh fungsi 3b di bawah):
#   s1_<tahun>.csv     satu baris per petak 100 m: lat, lon, mu_vv, sd_vv, n_basah,
#                      dan satu kolom per akuisisi Sentinel-1 bernama vv_YYYYMMDD (dB, rata-rata 100 m)
#   cuaca_<tahun>.csv  satu baris per hari: tanggal, hujan_mm (CHIRPS), et_mm (ERA5-Land),
#                      tmax_c (ERA5-Land), rata-rata wilayah studi
#
# Tanpa data muka air lapangan, Lapis 1 tidak bisa mengubah backscatter menjadi meter.
# Sesuai fallback rancangan v3 (§9), keadaan laten menjadi INDEKS KEBASAHAN RELATIF:
#   z_t = (σ⁰_VV,t − μ_basah) / sd_basah      (satuan: simpangan baku musim basah)
#   h_t = h_{t-1} + a·P_t − b·ET_t − c·(h_{t-1} − h_ref) + η_t,   η ~ N(0, Q)
#   z_t = h_t + ε_t,   ε ~ N(0, R)
# Parameter (a, b, c, Q, R) diestimasi (h_ref = 0) dengan maksimum likelihood (dekomposisi
# galat prediksi) pada musim latih, lalu dipakai apa adanya pada musim uji.
# Peluang kering: p = Pr(h < −1,645) [A] = peluang kebasahan lebih rendah daripada
# 95% kondisi musim basah di petak yang sama.
# =============================================================================
AMBANG_INDEKS = -1.645   # [A] persentil 5% musim basah


# =============================================================================
# DROUGHT CODE (Van Wagner 1987) — baseline domain
# =============================================================================

def drought_code(hujan_mm, tmax_c, dc0: float = 15.0, lf: float = 1.4) -> np.ndarray:
    """Drought Code harian. lf = faktor panjang hari; 1,4 untuk lintang ±10° (Lawson & Armitage 2008).

    Suhu harian maksimum dipakai sebagai pengganti suhu pukul 12 siang [A].
    """
    dc, out = float(dc0), []
    for r, t in zip(np.asarray(hujan_mm, dtype=float), np.asarray(tmax_c, dtype=float)):
        if r > 2.8:
            rd = 0.83 * r - 1.27
            qo = 800.0 * np.exp(-dc / 400.0)
            qr = qo + 3.937 * rd
            dc = max(400.0 * np.log(800.0 / qr), 0.0)
        v = max(0.36 * (max(t, -2.8) + 2.8) + lf, 0.0)
        dc = dc + 0.5 * v
        out.append(dc)
    return np.array(out)


# =============================================================================
# SUSUN DERET HARIAN
# =============================================================================

def deret_observasi(s1: pd.DataFrame, tanggal: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Matriks z (hari × petak): indeks kebasahan pada hari akuisisi, NaN di hari lain."""
    kol = sorted(k for k in s1.columns if k.startswith("vv_"))
    z = np.full((len(tanggal), len(s1)), np.nan)
    sd = np.where(s1["sd_vv"].to_numpy() > 0.05, s1["sd_vv"].to_numpy(), np.nan)
    posisi = {t: i for i, t in enumerate(tanggal.normalize())}
    hari_citra = []
    for k in kol:
        t = pd.Timestamp(k[3:])
        if t in posisi:
            z[posisi[t]] = (s1[k].to_numpy() - s1["mu_vv"].to_numpy()) / sd
            hari_citra.append(posisi[t])
    return z, np.array(hari_citra, dtype=int)


# =============================================================================
# MAKSIMUM LIKELIHOOD
# =============================================================================

def _uraikan(theta):
    a, b, c = np.exp(theta[0]), np.exp(theta[1]), 1 / (1 + np.exp(-theta[2]))
    Q, R = np.exp(theta[3]), np.exp(theta[4])
    # h_ref = 0: indeks dibakukan terhadap musim basah, jadi rata-rata musim basah = 0 [A].
    # Membebaskan h_ref membuat h_ref dan b tidak teridentifikasi (ET hampir konstan).
    return a, b, c, 0.0, Q, R


def log_likelihood(theta, hujan, et, z) -> float:
    """Log-likelihood Gaussian dari galat prediksi satu langkah (semua petak sampel)."""
    a, b, c, h_ref, Q, R = _uraikan(theta)
    T, N = z.shape
    ada0 = np.isfinite(z)
    pertama = np.argmax(ada0, axis=0)
    h = np.where(ada0.any(0), z[pertama, np.arange(N)], h_ref)
    P = np.full(N, 1.0)
    ll, phi = 0.0, 1 - c
    for t in range(T):
        h = phi * h + c * h_ref + a * hujan[t] - b * et[t]
        P = phi * phi * P + Q
        ada = np.isfinite(z[t])
        if ada.any():
            S = P[ada] + R
            e = z[t, ada] - h[ada]
            ll += -0.5 * np.sum(np.log(2 * np.pi * S) + e * e / S)
            K = P[ada] / S
            h[ada] = h[ada] + K * e
            P[ada] = (1 - K) * P[ada]
    return float(ll)


def estimasi_parameter(hujan, et, z, n_sampel: int = 400, seed: int = 0) -> dict:
    """MLE parameter dinamika pada sampel petak (tiap petak dianggap ulangan independen)."""
    rng = np.random.default_rng(seed)
    cukup = np.where(np.isfinite(z).sum(0) >= 4)[0]
    if len(cukup) == 0:
        raise ValueError("Tidak ada petak dengan ≥ 4 citra Sentinel-1")
    idx = rng.choice(cukup, size=min(n_sampel, len(cukup)), replace=False)
    zs = z[:, idx]
    if np.ndim(hujan) == 2:   # cuaca per petak (mis. per kabupaten tempat piksel berada)
        hujan, et = np.asarray(hujan)[:, idx], np.asarray(et)[:, idx]
    skala_p = max(np.nanmean(hujan), 1e-3)
    skala_e = max(np.nanmean(et), 1e-3)
    # Titik awal Q dan R mengikuti ragam data: ±1 untuk indeks kebasahan, ±0,01 m² untuk muka air.
    ragam = float(np.nanvar(zs)) if np.isfinite(zs).sum() > 1 else 1.0
    ragam = ragam if ragam > 0 else 1.0
    theta0 = np.array([np.log(0.02 / skala_p * min(ragam, 1.0) ** 0.5), np.log(0.02 / skala_e * min(ragam, 1.0) ** 0.5),
                       -3.0, np.log(0.01 * ragam), np.log(0.3 * ragam)])
    hasil = minimize(lambda th: -log_likelihood(th, hujan, et, zs), theta0, method="Nelder-Mead",
                     options={"maxiter": 3000, "xatol": 1e-4, "fatol": 1e-3})
    a, b, c, h_ref, Q, R = _uraikan(hasil.x)
    return {"a": a, "b": b, "c": c, "h_ref": h_ref, "Q": Q, "R": R,
            "log_likelihood": -float(hasil.fun), "n_petak_sampel": int(len(idx)),
            "konvergen": bool(hasil.success)}


# =============================================================================
# FILTER SEMUA PETAK DAN SUSUN petak_mingguan.csv
# =============================================================================

def filter_semua(hujan, et, z, par: dict) -> tuple[np.ndarray, np.ndarray]:
    ada = np.isfinite(z)
    pertama = np.where(ada.any(0), np.argmax(ada, axis=0), 0)
    h0 = np.where(ada.any(0), z[pertama, np.arange(z.shape[1])], par["h_ref"])
    return kalman_muka_air(hujan[:, None], et[:, None], z, par["R"], par["a"], par["b"], par["c"],
                           par["h_ref"], par["Q"], h0=h0, P0=1.0)


def susun_petak_mingguan(s1: pd.DataFrame, cuaca: pd.DataFrame, par: dict,
                         dc: np.ndarray | None = None, mulai=None, selesai=None,
                         ambang: float = AMBANG_INDEKS) -> pd.DataFrame:
    """Snapshot setiap Senin dalam [mulai, selesai] (bawaan: seluruh deret cuaca).
    ambang: −1,645 untuk indeks kebasahan relatif (Sentinel-1), −0,40 m untuk muka air (SMAP)."""
    tanggal = pd.DatetimeIndex(pd.to_datetime(cuaca["tanggal"]))
    hujan, et = cuaca["hujan_mm"].to_numpy(float), cuaca["et_mm"].to_numpy(float)
    z, _ = deret_observasi(s1, tanggal)
    h, var = filter_semua(hujan, et, z, par)
    sig = np.sqrt(var)
    dalam = np.ones(len(tanggal), bool)
    if mulai is not None:
        dalam &= tanggal >= pd.Timestamp(mulai)
    if selesai is not None:
        dalam &= tanggal <= pd.Timestamp(selesai)
    senin = np.where((tanggal.weekday == 0) & dalam)[0]
    pid = [f"P{la:.5f}_{lo:.5f}" for la, lo in zip(s1["lat"], s1["lon"])]
    hujan30 = pd.Series(hujan).rolling(30, min_periods=1).sum().to_numpy()
    tabel = []
    for t in senin:
        d = pd.DataFrame({
            "petak_id": pid, "lat": s1["lat"].round(6), "lon": s1["lon"].round(6),
            "minggu": tanggal[t].date(),
            "indeks_basah": h[t].round(4), "sigma_indeks": sig[t].round(4),
            "p": peluang_kering(h[t], sig[t], ambang),
            "p_bawah": peluang_kering(h[t] + 1.645 * sig[t], sig[t], ambang),
            "p_atas": peluang_kering(h[t] - 1.645 * sig[t], sig[t], ambang),
            "hujan_30hari_mm": round(float(hujan30[t]), 1),
        })
        if dc is not None:
            d["dc"] = round(float(dc[t]), 1)
        tabel.append(d)
    return pd.concat(tabel, ignore_index=True)


# =============================================================================
# DIAGNOSTIK DAN UJI STATISTIK
# =============================================================================

def uji_prakiraan_citra(hujan, et, z, par: dict) -> dict:
    """Manfaat asimilasi tanpa data lapangan: prakiraan z pada citra berikutnya.

    Bandingkan galat prakiraan filter (h prediksi sebelum citra masuk) dengan
    persistensi (nilai citra sebelumnya di petak yang sama).
    """
    a, b, c, Q, R = par["a"], par["b"], par["c"], par["Q"], par["R"]
    T, N = z.shape
    ada = np.isfinite(z)
    pertama = np.where(ada.any(0), np.argmax(ada, axis=0), 0)
    h = np.where(ada.any(0), z[pertama, np.arange(N)], 0.0)
    P = np.full(N, 1.0)
    terakhir = np.full(N, np.nan)
    e_kf, e_ps = [], []
    for t in range(T):
        h = (1 - c) * h + a * hujan[t] - b * et[t]
        P = (1 - c) ** 2 * P + Q
        m = ada[t]
        if m.any():
            sudah = m & np.isfinite(terakhir)
            e_kf.append(z[t, sudah] - h[sudah])
            e_ps.append(z[t, sudah] - terakhir[sudah])
            K = P[m] / (P[m] + R)
            h[m] = h[m] + K * (z[t, m] - h[m])
            P[m] = (1 - K) * P[m]
            terakhir[m] = z[t, m]
    e_kf = np.concatenate(e_kf) if e_kf else np.array([])
    e_ps = np.concatenate(e_ps) if e_ps else np.array([])
    if len(e_kf) == 0:
        return {"rmse_filter": np.nan, "rmse_persistensi": np.nan, "n": 0}
    return {"rmse_filter": float(np.sqrt(np.mean(e_kf ** 2))),
            "rmse_persistensi": float(np.sqrt(np.mean(e_ps ** 2))), "n": int(len(e_kf))}


def _komponen_delong(y, s):
    from scipy.stats import rankdata
    pos, neg = s[y == 1], s[y == 0]
    m, n = len(pos), len(neg)
    r_all = rankdata(np.r_[pos, neg])
    r_pos, r_neg = rankdata(pos), rankdata(neg)
    auc = (r_all[:m].sum() - m * (m + 1) / 2) / (m * n)
    v10 = (r_all[:m] - r_pos) / n          # komponen struktural positif
    v01 = 1 - (r_all[m:] - r_neg) / m      # komponen struktural negatif
    return auc, v10, v01


def uji_delong(y, skor_a, skor_b) -> dict:
    """Uji DeLong (1988) untuk selisih AUC dua skor pada data yang sama (implementasi Sun & Xu 2014)."""
    y = np.asarray(y, dtype=int)
    a1, v10a, v01a = _komponen_delong(y, np.asarray(skor_a, dtype=float))
    a2, v10b, v01b = _komponen_delong(y, np.asarray(skor_b, dtype=float))
    s10 = np.cov(np.vstack([v10a, v10b]))
    s01 = np.cov(np.vstack([v01a, v01b]))
    S = s10 / len(v10a) + s01 / len(v01a)
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    zstat = (a1 - a2) / np.sqrt(var) if var > 0 else np.nan
    return {"auc_a": float(a1), "auc_b": float(a2), "selisih": float(a1 - a2),
            "z": float(zstat), "p_value": float(2 * norm.sf(abs(zstat))) if np.isfinite(zstat) else np.nan}

# =============================================================================
# 3b. Ambil data asli dari Google Earth Engine (dengan akun GEE pengguna)
# Sentinel-1 VV dirata-rata 100 m dalam skala linear (sekaligus meredam speckle), dari
# satu orbit relatif yang paling sering supaya geometri pengamatan sama. Referensi musim
# basah: Januari–Maret tahun yang sama. Air permanen dan permukiman (WorldCover) dibuang.
# Data diunduh langsung sebagai larik per potongan (computePixels, tanpa Google Drive), jadi bisa
# dijalankan di laptop.
# =============================================================================
PENGATURAN_BAWAAN = {
    # Kotak pilot ±13 × 13 km di KHG Sungai Kahayan–Sungai Sebangau [A, cek batas KHG]
    "lon_min": 113.98, "lon_max": 114.10, "lat_min": -2.42, "lat_max": -2.30,
    # Sesuai blueprint esai: latih 2018, uji 2019, musim kemarau Juni–Oktober
    # (2023 tidak dipakai karena Sentinel-1B mati Desember 2021).
    "tahun_latih": 2018, "tahun_uji": 2019, "bulan_mulai": 6, "bulan_selesai": 10,
    "k_minggu": 2, "ukuran_petak_m": 100, "khg_geojson": None,
}


def ee_mulai(project_id: str, autentikasi: bool = True):
    """Masuk ke Earth Engine. Bila belum pernah login, membuka jendela login Google."""
    import ee
    if not project_id:
        raise ValueError("Project ID Earth Engine kosong")
    try:
        ee.Initialize(project=project_id)
    except Exception:
        if not autentikasi:
            raise
        ee.Authenticate()
        ee.Initialize(project=project_id)
    return ee


def _geometri_geojson(g: dict) -> dict:
    if g.get("type") == "FeatureCollection":
        g = g["features"][0]
    return g.get("geometry", g) if g.get("type") == "Feature" else g


def batas_wilayah(pg: dict) -> tuple[float, float, float, float]:
    """(lon_min, lat_min, lon_max, lat_max) dari batas KHG bila ada, kalau tidak dari kotak."""
    if pg.get("khg_geojson"):
        xy = np.array(list(_titik(_geometri_geojson(pg["khg_geojson"])["coordinates"])))
        return (round(float(xy[:, 0].min()), 4), round(float(xy[:, 1].min()), 4),
                round(float(xy[:, 0].max()), 4), round(float(xy[:, 1].max()), 4))
    return pg["lon_min"], pg["lat_min"], pg["lon_max"], pg["lat_max"]


def _titik(c):
    if isinstance(c[0], (int, float)):
        yield c[:2]
    else:
        for x in c:
            yield from _titik(x)


def ee_wilayah(ee, pg: dict):
    if pg.get("khg_geojson"):
        return ee.Geometry(_geometri_geojson(pg["khg_geojson"]))
    return ee.Geometry.Rectangle([pg["lon_min"], pg["lat_min"], pg["lon_max"], pg["lat_max"]])


def _ee_s1_dasar(ee, aoi, mulai: str, selesai: str):
    """Citra Sentinel-1 IW VV yang menyentuh wilayah (termasuk potongan scene yang hanya sebagian)."""
    return (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(aoi).filterDate(mulai, selesai)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")).select("VV"))


def _jendela_basah(tahun: int) -> list:
    """Jendela referensi musim basah, dicoba berurutan [A]: Jan–Mar; Nov tahun lalu–Apr;
    lalu ditambah musim hujan sesudahnya (Nov–Apr tahun berikut). Sejak Sentinel-1B mati (Des 2021)
    satu orbit kadang hanya lewat 1–2 kali di Jan–Mar, jadi jendela sempit sering tidak cukup."""
    return [[(f"{tahun}0101", f"{tahun}0401")],
            [(f"{tahun - 1}1101", f"{tahun}0501")],
            [(f"{tahun - 1}1101", f"{tahun}0501"), (f"{tahun}1101", f"{tahun + 1}0501")]]


def pilih_orbit_s1(akuisisi: list, tahun: int, bulan_mulai: int, bulan_selesai: int,
                   min_tutup: float = 0.9) -> dict:
    """akuisisi: daftar (orbit, 'YYYYMMDD', bagian wilayah yang tertutup 0–1), satu per orbit per hari
    (potongan scene pada hari yang sama sudah digabung). Memilih satu orbit relatif (sudut pandang sama)
    yang punya ≥4 tanggal musim dan ≥3 tanggal musim basah, masing-masing menutup ≥ min_tutup wilayah."""
    awal = f"{tahun}{bulan_mulai - 1:02d}01"
    akhir = f"{tahun}{bulan_selesai + 2:02d}01" if bulan_selesai <= 10 else f"{tahun}1231"
    for batas_tutup in (min_tutup, 0.6):   # wilayah besar (batas KHG) bisa tidak pernah tertutup ≥90%
        ok = [(int(o), str(t)) for o, t, c in akuisisi if c is not None and float(c) >= batas_tutup]
        musim = {}
        for o, t in ok:
            if awal <= t < akhir:
                musim.setdefault(o, set()).add(t)
        for jendela in _jendela_basah(tahun):
            basah = {}
            for o, t in ok:
                if any(a <= t < b for a, b in jendela):
                    basah.setdefault(o, set()).add(t)
            calon = [o for o in musim if len(musim[o]) >= 4 and len(basah.get(o, ())) >= 3]
            if calon:
                o = max(calon, key=lambda o: (len(musim[o]), len(basah[o])))
                return {"orbit": o, "musim": sorted(musim[o]), "basah": sorted(basah[o]),
                        "jendela": jendela, "tutup": batas_tutup}
    ringkas = {}
    for o, t, c in akuisisi:
        r = ringkas.setdefault(int(o), [0, 0])
        r[0] += awal <= str(t) < akhir
        r[1] += c is not None and float(c) >= 0.6 and awal <= str(t) < akhir
    teks = "; ".join(f"orbit {o}: {a} citra musim, {b} menutup ≥60% wilayah" for o, (a, b) in sorted(ringkas.items()))
    raise S1KurangError(
        f"Sentinel-1 {tahun}: tidak ada orbit dengan ≥4 citra musim dan ≥3 citra musim basah yang menutup "
        f"wilayah ({teks or 'tidak ada citra'}).")


class S1KurangError(RuntimeError):
    pass


def ee_saran_tahun(ee, aoi, bulan_mulai: int, bulan_selesai: int, kecuali=()) -> list:
    """Tahun 2016 s.d. tahun lalu yang citra Sentinel-1-nya cukup di wilayah ini (dipanggil hanya bila gagal)."""
    ok = []
    for th in range(2016, date.today().year):
        if th in kecuali:
            continue
        try:
            pilih_orbit_s1(ee_daftar_akuisisi(ee, aoi, th), th, bulan_mulai, bulan_selesai)
            ok.append(th)
        except Exception:
            pass
    return ok


def ee_daftar_akuisisi(ee, aoi, tahun: int) -> list:
    """Satu baris per (orbit relatif, tanggal): potongan scene pada hari yang sama digabung (mosaik),
    lalu dihitung bagian wilayah yang benar-benar tertutup piksel VV. Satu panggilan getInfo."""
    kol = _ee_s1_dasar(ee, aoi, f"{tahun - 1}-11-01", f"{tahun + 1}-05-01")
    kol = kol.map(lambda im: im.set("kunci", ee.Number(im.get("relativeOrbitNumber_start")).format("%d")
                                    .cat("_").cat(im.date().format("YYYYMMdd"))))

    def tutup(k):
        m = kol.filter(ee.Filter.eq("kunci", k)).mosaic().select("VV").mask().rename("c").unmask(0)
        c = m.reduceRegion(ee.Reducer.mean(), aoi, 200, "EPSG:32750", bestEffort=True, maxPixels=1e9).get("c")
        return ee.Feature(None, {"kunci": k, "tutup": c})

    fc = ee.FeatureCollection(kol.aggregate_array("kunci").distinct().map(tutup)).getInfo()
    hasil = []
    for f in fc.get("features", []):
        p = f.get("properties", {})
        o, t = str(p.get("kunci", "_")).split("_", 1)
        if o and t:
            hasil.append((int(o), t, p.get("tutup")))
    return hasil


def ee_tumpukan_s1(ee, aoi, proj, tahun: int, bulan_mulai: int, bulan_selesai: int):
    """Satu citra berisi lon, lat, mu_vv, sd_vv, n_basah, dan satu pita vv_YYYYMMDD per akuisisi."""
    proj10 = ee.Projection("EPSG:32750").atScale(10)

    def ke_100m(img):
        # Mosaik satu hari tidak punya proyeksi bawaan yang sah untuk reduceResolution,
        # jadi diberi proyeksi 10 m dulu. dB → linear lewat exp(dB·ln10/10) supaya
        # proyeksinya tetap ikut citra (bukan citra konstanta).
        vv = img.select("VV").setDefaultProjection(proj10)
        linear = vv.multiply(np.log(10) / 10).exp()
        rata = linear.reduceResolution(ee.Reducer.mean(), maxPixels=1024).reproject(proj)
        return rata.log10().multiply(10).rename("VV")

    wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    wc100 = (wc.setDefaultProjection(wc.projection()).reduceResolution(ee.Reducer.mode(), maxPixels=1024)
             .reproject(proj))
    masker = wc100.neq(80).And(wc100.neq(50))   # bukan air permanen, bukan permukiman

    akuisisi = ee_daftar_akuisisi(ee, aoi, tahun)
    if not akuisisi:
        raise RuntimeError(f"Tidak ada citra Sentinel-1 di wilayah ini sekitar tahun {tahun}")
    try:
        pil = pilih_orbit_s1(akuisisi, tahun, bulan_mulai, bulan_selesai)
    except S1KurangError as e:
        saran = ee_saran_tahun(ee, aoi, bulan_mulai, bulan_selesai, kecuali=(tahun,))
        raise S1KurangError(f"{e} Tahun yang citranya cukup di wilayah ini: "
                            f"{', '.join(map(str, saran)) if saran else 'tidak ada (coba wilayah lain)'}. "
                            "Ganti tahun latih/uji ke salah satunya.") from None
    orbit = pil["orbit"]
    kol = _ee_s1_dasar(ee, aoi, f"{tahun - 1}-11-01", f"{tahun + 1}-05-01").filter(
        ee.Filter.eq("relativeOrbitNumber_start", orbit))

    def harian(d):   # mosaik semua potongan scene orbit ini pada tanggal d
        t0 = ee.Date.parse("YYYYMMdd", d)
        return ke_100m(kol.filterDate(t0, t0.advance(1, "day")).mosaic())

    basah = ee.ImageCollection([harian(d) for d in pil["basah"]])
    ref = ee.Image.cat([basah.mean().rename("mu_vv"),
                        basah.reduce(ee.Reducer.stdDev()).rename("sd_vv"),
                        basah.count().rename("n_basah")])
    pita = [harian(d).rename(f"vv_{d}") for d in pil["musim"]]
    lonlat = ee.Image.pixelLonLat().reproject(proj).rename(["lon", "lat"])
    img = ee.Image.cat([lonlat, ref] + pita).updateMask(masker)
    jendela = " + ".join(f"{a[:4]}-{a[4:6]} s.d. {b[:4]}-{b[4:6]}" for a, b in pil["jendela"])
    info = (f"{tahun}: orbit relatif {orbit}, {len(pil['musim'])} tanggal akuisisi musim, "
            f"{len(pil['basah'])} citra musim basah ({jendela}, akhir eksklusif)")
    return img, info


def ee_unduh_petak(ee, img, aoi, proj, batas: tuple, ukuran_ubin_km: float | None = None,
                   progres=None) -> pd.DataFrame:
    """Unduh semua petak 100 m sekaligus sebagai larik (computePixels), bukan titik demi titik.
    Grid sama dengan `proj` (UTM 50S, kelipatan 100 m). Bila terlalu besar atau server menolak,
    dibagi menjadi beberapa potongan baris."""
    utm = ee.Projection("EPSG:32750")
    ring = ee.Geometry.Rectangle(list(batas)).bounds(1, utm).coordinates().getInfo()[0]
    xs, ys = [p[0] for p in ring], [p[1] for p in ring]
    x0, y1 = np.floor(min(xs) / 100) * 100, np.ceil(max(ys) / 100) * 100
    W = int(np.ceil((max(xs) - x0) / 100))
    H = int(np.ceil((y1 - min(ys)) / 100))
    KOSONG = -9999.0
    # sameFootprint=False: piksel di luar wilayah juga diberi KOSONG (bukan 0), lalu dibuang.
    citra = img.clip(aoi).toFloat().unmask(KOSONG, False)
    n_pita = len(citra.bandNames().getInfo())
    baris_maks = max(1, int(24e6 / (4 * n_pita * max(W, 1))))   # ±24 MB per permintaan (batas 48 MB)
    if ukuran_ubin_km:
        baris_maks = min(baris_maks, max(1, int(ukuran_ubin_km * 10)))

    def ambil(r0, h, kedalaman=0):
        q = {"expression": citra, "fileFormat": "NUMPY_NDARRAY",
             "grid": {"dimensions": {"width": W, "height": h}, "crsCode": "EPSG:32750",
                      "affineTransform": {"scaleX": 100, "shearX": 0, "translateX": float(x0),
                                          "shearY": 0, "scaleY": -100, "translateY": float(y1 - 100 * r0)}}}
        try:
            return [ee.data.computePixels(q)]
        except ee.EEException as e:
            berat = any(k in str(e).lower() for k in ("memory", "timed out", "too many", "limit", "too large"))
            if kedalaman >= 3 or h < 2 or not berat:
                raise
            return ambil(r0, h // 2, kedalaman + 1) + ambil(r0 + h // 2, h - h // 2, kedalaman + 1)

    potong = list(range(0, H, baris_maks))
    tabel = []
    for n, r0 in enumerate(potong):
        for a in ambil(r0, min(baris_maks, H - r0)):
            d = pd.DataFrame({k: np.asarray(a[k], dtype=float).ravel() for k in a.dtype.names})
            tabel.append(d.replace(KOSONG, np.nan))
        if progres:
            progres((n + 1) / len(potong), f"potongan {n + 1}/{len(potong)} · grid {W} × {H} petak")
    d = pd.concat(tabel, ignore_index=True) if tabel else pd.DataFrame()
    if d.empty or "lat" not in d:
        return pd.DataFrame(columns=["lat", "lon", "mu_vv", "sd_vv", "n_basah"])
    d = d.dropna(subset=["lat", "lon"])
    d = d[(d["lat"] != 0) | (d["lon"] != 0)]
    return d.drop_duplicates(subset=["lat", "lon"]).reset_index(drop=True)


def ee_cuaca_harian(ee, aoi, tahun: int) -> pd.DataFrame:
    """Hujan CHIRPS, ET dan suhu maksimum ERA5-Land harian, rata-rata wilayah studi + 10 km
    (penyangga supaya selalu ada piksel ERA5-Land 11 km di dalamnya)."""
    mulai, selesai = f"{tahun}-01-01", f"{tahun + 1}-01-01"
    wil = aoi.buffer(10000, 100)

    def ch(img):
        v = img.reduceRegion(ee.Reducer.mean(), wil, 5566, bestEffort=True).get("precipitation")
        return ee.Feature(None, {"tanggal": img.date().format("YYYY-MM-dd"), "hujan_mm": v})

    def er(img):
        v = img.reduceRegion(ee.Reducer.mean(), wil, 11132, bestEffort=True)
        return ee.Feature(None, {"tanggal": img.date().format("YYYY-MM-dd"),
                                 "evap_m": v.get("total_evaporation_sum"), "tmax_k": v.get("temperature_2m_max")})

    a = ee.FeatureCollection(ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(mulai, selesai).map(ch))
    b = ee.FeatureCollection(ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(mulai, selesai)
                             .select(["total_evaporation_sum", "temperature_2m_max"]).map(er))
    # Nilai kosong (null) dibuang server dari properti, jadi kolom dipastikan ada lalu diisi interpolasi.
    da = pd.DataFrame([f["properties"] for f in a.getInfo()["features"]]).reindex(columns=["tanggal", "hujan_mm"])
    db = pd.DataFrame([f["properties"] for f in b.getInfo()["features"]]).reindex(
        columns=["tanggal", "evap_m", "tmax_k"])
    d = da.merge(db, on="tanggal", how="inner").sort_values("tanggal").reset_index(drop=True)
    if len(d) < 300:
        raise RuntimeError(f"Cuaca {tahun}: hanya {len(d)} hari CHIRPS + ERA5-Land yang lengkap")
    for k in ("hujan_mm", "evap_m", "tmax_k"):
        if d[k].isna().mean() > 0.05:
            raise RuntimeError(f"Cuaca {tahun}: {k} kosong di {100 * d[k].isna().mean():.0f}% hari")
        d[k] = d[k].interpolate(limit_direction="both")
    d["et_mm"] = -1000 * d["evap_m"]          # konvensi ECMWF: penguapan bernilai negatif
    d["tmax_c"] = d["tmax_k"] - 273.15
    return d[["tanggal", "hujan_mm", "et_mm", "tmax_c"]].sort_values("tanggal").reset_index(drop=True)


# =============================================================================
# 3c. Olah data asli sampai hasil esai (Lapis 1–5 + validasi)
# Keluaran di folder hasil/: petak_mingguan.csv (dibaca dasbor), hotspot_viirs.csv,
# parameter_kalman.csv, uji_prakiraan_citra.csv, metrik_validasi.csv, uji_delong.csv,
# tangkapan_hotspot.csv, gambar_peta_p.png, gambar_roc.png, gambar_tren.png, ringkasan.txt
# =============================================================================

def _potong_cuaca(cuaca: pd.DataFrame, tahun: int, bulan_mulai: int) -> pd.DataFrame:
    """Cuaca mulai 1 bulan sebelum musim (awal deret Sentinel-1; untuk Januari mulai 1 Januari);
    Drought Code dihitung sejak 1 Januari."""
    c = cuaca.copy()
    c["dc"] = drought_code(c["hujan_mm"], c["tmax_c"])
    awal = pd.Timestamp(f"{tahun}-{max(bulan_mulai - 1, 1):02d}-01")
    return c[pd.to_datetime(c["tanggal"]) >= awal].reset_index(drop=True)


def bersihkan_s1(d: pd.DataFrame) -> pd.DataFrame:
    d = d.drop(columns=["system:index", ".geo"], errors="ignore")
    d = d.dropna(subset=["mu_vv", "sd_vv"])
    return d[d["n_basah"] >= 3].reset_index(drop=True)


def olah_data_asli(s1: dict, cuaca: dict, hotspot: pd.DataFrame, pg: dict, folder,
                   progres=None) -> dict:
    """s1/cuaca: {tahun: DataFrame}. Hotspot hanya untuk validasi, tidak masuk ke model."""
    progres = progres or (lambda f, m: None)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    tl, tu, bm, bs, k = pg["tahun_latih"], pg["tahun_uji"], pg["bulan_mulai"], pg["bulan_selesai"], pg["k_minggu"]
    s1 = {th: bersihkan_s1(d) for th, d in s1.items()}
    c = {th: _potong_cuaca(cuaca[th], th, bm) for th in (tl, tu)}

    def deret(th):
        z, _ = deret_observasi(s1[th], pd.DatetimeIndex(pd.to_datetime(c[th]["tanggal"])))
        return c[th]["hujan_mm"].to_numpy(float), c[th]["et_mm"].to_numpy(float), z

    # Diagnosa data sebelum model: berapa petak, tanggal citra, dan nilai yang benar-benar terisi.
    for th in (tl, tu):
        kol = [k for k in s1[th].columns if str(k).startswith("vv_")]
        _, hari = deret_observasi(s1[th], pd.DatetimeIndex(pd.to_datetime(c[th]["tanggal"])))
        z = deret(th)[2]
        isi = float(np.isfinite(z[hari]).mean()) if len(hari) and z.size else 0.0
        pesan = (f"Data {th}: {len(s1[th]):,} petak valid, {len(kol)} tanggal citra, {len(hari)} cocok dengan "
                 f"tanggal cuaca ({c[th]['tanggal'].min()} s.d. {c[th]['tanggal'].max()}), "
                 f"{100 * isi:.0f}% nilai terisi")
        progres(0.02, pesan)
        if len(s1[th]) == 0 or len(hari) < 3 or isi < 0.05:
            raise RuntimeError("Data Sentinel-1 tidak cukup untuk dianalisis. " + pesan)

    # Lapis 2: MLE hanya pada musim latih, lalu dipakai apa adanya pada musim uji
    progres(0.05, f"Mengestimasi parameter Kalman (MLE) pada musim latih {tl}")
    hj, et_, z_latih = deret(tl)
    par = estimasi_parameter(hj, et_, z_latih, n_sampel=400)
    pd.Series(par).to_csv(folder / "parameter_kalman.csv")

    progres(0.35, "Menyaring semua petak dan menyusun petak_mingguan.csv")
    tabel = []
    for th in (tl, tu):
        t = susun_petak_mingguan(s1[th], c[th], par, c[th]["dc"].to_numpy(),
                                 mulai=f"{th}-{bm:02d}-01", selesai=f"{th}-{bs:02d}-28")
        t.to_csv(folder / f"petak_mingguan_{th}.csv", index=False, float_format="%.8g")
        tabel.append(t)
    petak_mingguan = pd.concat(tabel, ignore_index=True)
    petak_mingguan.to_csv(folder / "petak_mingguan.csv", index=False, float_format="%.8g")

    progres(0.55, f"Menguji prakiraan citra berikutnya pada musim uji {tu}")
    hj_u, et_u, z_uji = deret(tu)
    prak = uji_prakiraan_citra(hj_u, et_u, z_uji, par)
    pd.Series(prak).to_csv(folder / "uji_prakiraan_citra.csv")

    # Lapis 3–5: validasi terhadap hotspot minggu t+1..t+k pada musim uji
    progres(0.65, "Memvalidasi terhadap hotspot VIIRS (AUC, DeLong, tangkapan)")
    uji = siapkan_petak(pd.read_csv(folder / f"petak_mingguan_{tu}.csv"))
    satu = uji[uji["minggu"] == uji["minggu"].min()][["petak_id", "lat", "lon"]]
    hp = petakan_hotspot(hotspot, satu)
    y = label_api(uji[["petak_id", "minggu"]], hp[["petak_id", "minggu"]], k).astype(int)
    skor = {"SPARKS p": uji["p"], "Drought Code": uji["dc"], "Hujan 30 hari (dibalik)": -uji["hujan_30hari_mm"]}
    baris = []
    for nama, s in skor.items():
        m = metrik_prediksi(y, s)
        baris.append({"skor": nama, "AUC-ROC": m["auc_roc"], "AUC-PR": m["auc_pr"]})
    metrik = pd.DataFrame(baris)
    metrik["Brier (p)"] = [brier(y, uji["p"]), np.nan, np.nan]
    if 0 < y.sum() < len(y):
        dl = [uji_delong(y, uji["p"], s) | {"pembanding": n} for n, s in list(skor.items())[1:]]
    else:   # tanpa hotspot di musim uji, AUC dan DeLong tidak terdefinisi
        dl = [{"auc_a": np.nan, "auc_b": np.nan, "selisih": np.nan, "z": np.nan, "p_value": np.nan,
               "pembanding": n} for n in list(skor)[1:]]
        progres(0.66, f"Peringatan: tidak ada hotspot VIIRS yang jatuh di petak pada musim uji {tu}")
    metrik.to_csv(folder / "metrik_validasi.csv", index=False)
    pd.DataFrame(dl).to_csv(folder / "uji_delong.csv", index=False)
    # Tangkapan hotspot dengan anggaran sama (biaya seragam per blok 1 km², anggaran 20% blok)
    cap = uji_tangkapan(uji, hp, pd.Series(dtype=float), 20, k)
    cap.to_csv(folder / "tangkapan_hotspot.csv", index=False)
    kolom_cap = [x for x in ["SPARKS", "Drought Code", "Acak"] if x in cap]

    progres(0.85, "Menggambar peta, kurva ROC, dan tren untuk esai")
    gambar = _gambar_esai(uji, hp, y, skor, k, tu, folder)

    lon0, lat0, lon1, lat1 = batas_wilayah(pg)
    ringkasan = (
        f"RINGKASAN HASIL PILOT SPARKS (data asli)\n"
        f"Wilayah   : kotak {lon0}–{lon1} BT, {lat0}–{lat1} LS; {s1[tu].shape[0]:,} petak 100 m\n"
        f"Latih/uji : musim kemarau {tl} (estimasi parameter) / {tu} (uji), bulan {bm}–{bs}\n"
        f"Parameter : a={par['a']:.4g}, b={par['b']:.4g}, c={par['c']:.4g}, Q={par['Q']:.4g}, R={par['R']:.4g}\n"
        f"Prakiraan citra berikutnya (uji): RMSE filter {prak['rmse_filter']:.3f} vs persistensi "
        f"{prak['rmse_persistensi']:.3f} (satuan sd musim basah)\n"
        "AUC-ROC uji: " + ", ".join(f"{r['skor']} {r['AUC-ROC']:.3f}" for _, r in metrik.iterrows()) + "\n"
        f"DeLong (SPARKS vs DC): selisih {dl[0]['selisih']:.3f}, p = {dl[0]['p_value']:.3g}\n"
        "Tangkapan hotspot (anggaran 20%): "
        + ", ".join(f"{kk} {100 * v:.1f}%" for kk, v in cap[kolom_cap].mean().items()) + "\n"
        + ("PERHATIAN: tidak ada hotspot VIIRS di wilayah/musim uji, jadi AUC dan tangkapan tidak bisa "
           "dihitung (nan). Perluas wilayah atau pilih tahun uji dengan kebakaran.\n" if y.sum() == 0 else "")
        + "Catatan: p = Pr(indeks kebasahan relatif < −1,645), BUKAN Pr(muka air < 0,4 m), "
        "karena belum ada label SIPALAGA.\n")
    (folder / "ringkasan.txt").write_text(ringkasan, encoding="utf-8")
    progres(1.0, "Selesai")
    return {"parameter": par, "prakiraan": prak, "metrik": metrik, "delong": pd.DataFrame(dl),
            "tangkapan": cap, "ringkasan": ringkasan, "gambar": gambar,
            "n_baris": len(petak_mingguan), "n_label": int(y.sum())}


def _gambar_esai(uji, hp, y, skor, k, tahun_uji, folder: Path) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9})
    puncak = uji.groupby("minggu")["p"].mean().idxmax()
    d = uji[uji["minggu"] == puncak]
    fig, ax = plt.subplots(figsize=(5, 4.6))
    sc = ax.scatter(d["lon"], d["lat"], c=d["p"], s=1.2, cmap="Oranges", vmin=0, vmax=1, marker="s")
    h = hp[(hp["minggu"] > puncak) & (hp["minggu"] <= puncak + pd.Timedelta(weeks=k))]
    ax.scatter(h["longitude"], h["latitude"], s=10, facecolors="none", edgecolors="k", linewidths=0.6,
               label=f"hotspot VIIRS {k} minggu berikutnya")
    ax.set_title(f"Peluang kering p, minggu {puncak:%d %b %Y}")
    ax.set_xlabel("bujur")
    ax.set_ylabel("lintang")
    fig.colorbar(sc, ax=ax, label="p")
    ax.legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(folder / "gambar_peta_p.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.4))
    for nama, s in skor.items():
        r = kurva_roc(y, s)
        ax.plot(r["fpr"], r["tpr"], label=nama, lw=1.6)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("laju positif palsu")
    ax.set_ylabel("laju positif benar")
    ax.set_title(f"Kurva ROC musim uji {tahun_uji}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(folder / "gambar_roc.png")
    plt.close(fig)

    mingguan = (uji.groupby("minggu").agg(p_rata=("p", "mean"))
                .join(hp.groupby("minggu").size().rename("hotspot")).fillna(0))
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax2 = ax.twinx()
    ax.plot(mingguan.index, mingguan["p_rata"], color="tab:orange", marker="o", lw=1.6, label="rata-rata p")
    ax2.bar(mingguan.index, mingguan["hotspot"], width=4, color="tab:gray", alpha=0.5, label="hotspot")
    ax.set_ylabel("rata-rata p")
    ax2.set_ylabel("jumlah hotspot")
    ax.set_title(f"Peluang kering dan hotspot mingguan {tahun_uji}")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(folder / "gambar_tren.png")
    plt.close(fig)
    return [folder / "gambar_peta_p.png", folder / "gambar_roc.png", folder / "gambar_tren.png"]


VERSI_UNDUHAN = "s1-v4"   # naikkan bila cara mengambil data berubah, supaya unduhan lama tidak dipakai


def _kunci_pengaturan(pg: dict) -> str:
    isi = {x: pg.get(x) for x in pg if x != "k_minggu"} | {"versi": VERSI_UNDUHAN}
    return hashlib.md5(json.dumps(isi, sort_keys=True, default=str).encode()).hexdigest()[:10]


def _kotak_pusat(batas: tuple, km: float = 2.0) -> tuple:
    lon0, lat0, lon1, lat1 = batas
    cx, cy = (lon0 + lon1) / 2, (lat0 + lat1) / 2
    dx, dy = km / 2 / (111.32 * np.cos(np.radians(cy))), km / 2 / 111.32
    return (max(lon0, cx - dx), max(lat0, cy - dy), min(lon1, cx + dx), min(lat1, cy + dy))


def uji_koneksi(pg: dict, project_id: str, map_key: str, folder, progres=None,
                autentikasi: bool = True, ee=None, lewati_s1: tuple = ()) -> list:
    """Uji cepat (±1–2 menit) setiap tahap pada wilayah kecil 2 × 2 km di tengah wilayah studi:
    login, cuaca, FIRMS, dan Sentinel-1 kedua tahun. Galat muncul di sini, bukan setelah lama menunggu.
    Cuaca yang diambil disimpan dan dipakai lagi oleh proses penuh."""
    progres = progres or (lambda f, m: print(f"[{100 * f:5.1f}%] {m}", flush=True))
    folder = Path(folder)
    mentah = folder / "mentah"
    mentah.mkdir(parents=True, exist_ok=True)
    tl, tu, bm, bs = pg["tahun_latih"], pg["tahun_uji"], pg["bulan_mulai"], pg["bulan_selesai"]
    kunci, batas = _kunci_pengaturan(pg), batas_wilayah(pg)
    catatan = []

    def lulus(f, pesan):
        catatan.append("✓ " + pesan)
        progres(f, "✓ " + pesan)

    if not map_key:
        raise FIRMSError("MAP_KEY NASA FIRMS kosong (isi di .env atau kolom MAP_KEY)")
    progres(0.0, "Uji 1/4: masuk ke Google Earth Engine")
    ee = ee or ee_mulai(project_id, autentikasi)
    lulus(0.1, f"Earth Engine tersambung (project {project_id})")
    aoi = ee_wilayah(ee, pg)
    proj = ee.Projection("EPSG:32750").atScale(pg.get("ukuran_petak_m", 100))

    progres(0.15, "Uji 2/4: cuaca harian CHIRPS + ERA5-Land")
    for th in (tl, tu):
        f = mentah / f"cuaca_{th}_{kunci}.csv"
        if not f.exists():
            ee_cuaca_harian(ee, aoi, th).to_csv(f, index=False)
        c = pd.read_csv(f)
        lulus(0.3, f"Cuaca {th}: {len(c)} hari, hujan total {c['hujan_mm'].sum():,.0f} mm")

    progres(0.35, "Uji 3/4: NASA FIRMS (5 hari pertama musim)")
    bbox = (batas[0] - 0.01, batas[1] - 0.01, batas[2] + 0.01, batas[3] + 0.01)
    h = ambil_firms(map_key, bbox, date(tl, bm, 1), date(tl, bm, 5), "VIIRS_SNPP_SP", mentah / "firms")
    lulus(0.45, f"FIRMS menjawab ({len(h)} hotspot pada 1–5 {date(tl, bm, 1):%b} {tl})")

    for i, th in enumerate((tl, tu)):
        if th in lewati_s1:
            lulus(0.5 + 0.25 * (i + 1), f"Sentinel-1 {th}: sudah terunduh lengkap sebelumnya")
            continue
        progres(0.5 + 0.25 * i, f"Uji 4/4: Sentinel-1 {th} pada kotak 2 × 2 km di tengah wilayah")
        img, info = ee_tumpukan_s1(ee, aoi, proj, th, bm, bs)
        d = ee_unduh_petak(ee, img, aoi, proj, _kotak_pusat(batas), ukuran_ubin_km=3)
        n_vv = sum(str(k).startswith("vv_") for k in d.columns)
        if len(d) and not n_vv:
            raise RuntimeError(f"Sentinel-1 {th}: petak terambil tapi tanpa kolom vv_ (cek pita citra)")
        if len(d) and d["mu_vv"].notna().sum() == 0:
            raise RuntimeError(f"Sentinel-1 {th}: referensi musim basah kosong di semua petak uji")
        lulus(0.5 + 0.25 * (i + 1), f"{info}; {len(d)} petak uji terambil"
              + ("" if len(d) else " (tengah wilayah mungkin air/permukiman, proses penuh tetap jalan)"))
    return catatan


def jalankan_data_asli(pg: dict, project_id: str, map_key: str, folder, progres=None,
                       autentikasi: bool = True, ee=None) -> dict:
    """Seluruh pipeline: uji koneksi → Sentinel-1 → FIRMS → Kalman → validasi. Data mentah disimpan di
    folder/mentah, jadi menjalankan ulang dengan pengaturan sama tidak mengunduh lagi."""
    progres = progres or (lambda f, m: print(f"[{100 * f:5.1f}%] {m}", flush=True))
    folder = Path(folder)
    mentah = folder / "mentah"
    tl, tu, bm, bs = pg["tahun_latih"], pg["tahun_uji"], pg["bulan_mulai"], pg["bulan_selesai"]
    batas, kunci = batas_wilayah(pg), _kunci_pengaturan(pg)

    def s1_tersimpan(th):
        f = mentah / f"s1_{th}_{kunci}.csv"
        return f.exists() and not bersihkan_s1(pd.read_csv(f)).empty

    ee = ee or ee_mulai(project_id, autentikasi)
    uji_koneksi(pg, project_id, map_key, folder, progres=lambda x, m: progres(0.08 * x, m), ee=ee,
                lewati_s1=tuple(th for th in (tl, tu) if s1_tersimpan(th)))
    aoi = ee_wilayah(ee, pg)
    proj = ee.Projection("EPSG:32750").atScale(pg.get("ukuran_petak_m", 100))   # UTM 50S, petak 100 m

    s1, cuaca = {}, {}
    for i, th in enumerate((tl, tu)):
        cuaca[th] = pd.read_csv(mentah / f"cuaca_{th}_{kunci}.csv")
        cuaca[th].to_csv(folder / f"cuaca_{th}.csv", index=False)
        f = mentah / f"s1_{th}_{kunci}.csv"
        lo, hi = 0.08 + 0.28 * i, 0.08 + 0.28 * (i + 1)
        if s1_tersimpan(th):
            progres(hi, f"Sentinel-1 {th}: memakai unduhan sebelumnya")
            s1[th] = pd.read_csv(f)
        else:
            progres(lo, f"Sentinel-1 {th}: mengunduh semua petak")
            img, info = ee_tumpukan_s1(ee, aoi, proj, th, bm, bs)
            d = ee_unduh_petak(ee, img, aoi, proj, batas,
                               progres=lambda x, m, lo=lo, hi=hi, th=th: progres(lo + (hi - lo) * x,
                                                                                 f"Sentinel-1 {th}: {m}"))
            if bersihkan_s1(d).empty:
                raise RuntimeError(f"Sentinel-1 {th}: tidak ada petak gambut yang valid di wilayah ini")
            d.to_csv(f, index=False)
            s1[th] = d

    progres(0.64, "Mengambil hotspot VIIRS dari NASA FIRMS (hanya untuk validasi)")
    bbox = (batas[0] - 0.01, batas[1] - 0.01, batas[2] + 0.01, batas[3] + 0.01)
    hs = [ambil_firms(map_key, bbox, date(th, bm, 1), date(th, min(bs + 1, 12), 30), "VIIRS_SNPP_SP",
                      mentah / "firms") for th in (tl, tu)]
    hotspot = pd.concat(hs, ignore_index=True)
    hotspot.to_csv(folder / "hotspot_viirs.csv", index=False)

    hasil = olah_data_asli(s1, cuaca, hotspot, pg, folder,
                           progres=lambda x, m: progres(0.70 + 0.30 * x, m))
    hasil["hotspot"] = hotspot
    return hasil


# =============================================================================
# BAGIAN 4. DATA DASBOR SELURUH INDONESIA PER KABUPATEN (DATA SATELIT HARIAN, TERLAMBAT ±3–5 HARI)
# Pengguna memilih rentang tanggal. Sumber data (semua lewat Google Earth Engine):
#   NASA SMAP L4 v008 (9 km, tiap 3 jam, terlambat ±3–5 hari):
#     depth_to_water_table_from_surface_in_peat = muka air tanah gambut h (m, negatif = di bawah
#     permukaan; model PEATCLSM yang mengasimilasi kecerahan SMAP), presipitasi, evapotranspirasi,
#     suhu udara -> masukan dinamika Kalman dan Drought Code.
#   NASA FIRMS (MODIS ±1 km, harian) -> titik api per kabupaten per minggu: menyetel ambang level dan
#     menjadi fitur riwayat untuk XGBoost.
#   FAO GAUL 2025 -> batas 514 kabupaten/kota dan 38 provinsi, dikelompokkan ke 7 pulau/kepulauan.
# Setiap piksel gambut 9 km diolah dengan filter Kalman yang sama (BAGIAN 2–3): parameter diestimasi
# (MLE) per provinsi pada rentang tanggal yang sama setahun sebelumnya, lalu p = Pr(h < −0,40 m)
# (PP 57/2016). Peluang kering kabupaten = rata-rata p piksel gambutnya. Kabupaten tanpa piksel gambut =
# "bukan lahan gambut". Pekerjaan Earth Engine dibagi per provinsi/pulau dan dikirim paralel.
# =============================================================================
FOLDER = Path(__file__).resolve().parent
FOLDER_DATA = FOLDER / "data"
FOLDER_HASIL = FOLDER / "hasil"
BERKAS_PENGATURAN = FOLDER_DATA / "pengaturan.json"
BERKAS_KABUPATEN = FOLDER_DATA / "kabupaten_indonesia.geojson"
BERKAS_LOGO = Path(__file__).parent / "assets" / "logo_sparks.png"
BERKAS_KAB_MINGGUAN = FOLDER_DATA / "kabupaten_mingguan.csv"
BERKAS_HOTSPOT_KAB = FOLDER_DATA / "hotspot_kabupaten.csv"
GAUL_KABUPATEN = "FAO/GAUL/2025/level2"
GAUL_PROVINSI = "FAO/GAUL/2025/level1"
SMAP_L4 = "NASA/SMAP/SPL4SMGP/008"
PITA_MUKA_AIR = "depth_to_water_table_from_surface_in_peat"
TAHUN_DATA_PERTAMA = 2016          # SMAP L4 mulai 31 Maret 2015; 2016 = tahun penuh pertama
TITIK_MINIMUM = 1                  # kabupaten tanpa piksel gambut SMAP = bukan lahan gambut
LUAS_PIKSEL_SMAP_HA = 8114         # satu piksel EASE-Grid 2.0 9 km ≈ 9,008 km × 9,008 km
K_MINGGU = 2                       # [A] horizon kejadian api untuk kalibrasi ambang
JANGKA_PILIHAN = (1, 2, 3, 4)      # pilihan jangka prediksi XGBoost (minggu ke depan)
JANGKA_MAKS = max(JANGKA_PILIHAN)  # titik api diambil sampai sejauh ini sesudah rentang
SMAP_PERTAMA = date(2015, 4, 1)    # hari lengkap pertama SMAP L4

PULAU_PROVINSI = {
    "Sumatera": ["Aceh", "Sumatera Utara", "Sumatera Barat", "Riau", "Kepulauan Riau", "Jambi", "Bengkulu",
                 "Sumatera Selatan", "Kepulauan Bangka Belitung", "Lampung"],
    "Jawa": ["Banten", "Dki Jakarta", "Jawa Barat", "Jawa Tengah", "Daerah Istimewa Yogyakarta", "Jawa Timur"],
    "Bali-Nusa Tenggara": ["Bali", "Nusa Tenggara Barat", "Nusa Tenggara Timur"],
    "Kalimantan": ["Kalimantan Barat", "Kalimantan Tengah", "Kalimantan Selatan", "Kalimantan Timur",
                   "Kalimantan Utara"],
    "Sulawesi": ["Sulawesi Utara", "Gorontalo", "Sulawesi Tengah", "Sulawesi Barat", "Sulawesi Selatan",
                 "Sulawesi Tenggara"],
    "Maluku": ["Maluku", "Maluku Utara"],
    "Papua": ["Papua", "Papua Barat", "Papua Barat Daya", "Papua Tengah", "Papua Pegunungan", "Papua Selatan"],
}
PULAU = {pv: pl for pl, daftar in PULAU_PROVINSI.items() for pv in daftar}


def pulau_dari(provinsi: str) -> str:
    return PULAU.get(provinsi, "Lainnya")


def kunci_kab(provinsi: str, kabupaten: str) -> str:
    return f"{provinsi}|{kabupaten}"


def _slug(teks: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", teks.lower()).strip("_")


def pilih_tahun_latih(tahun_uji: int) -> int:
    """Tahun pembanding untuk estimasi parameter (MLE): tahun sebelumnya (2016 memakai 2017)."""
    return tahun_uji - 1 if tahun_uji - 1 >= TAHUN_DATA_PERTAMA else tahun_uji + 1


def ee_kabupaten(ee):
    fc = ee.FeatureCollection(GAUL_KABUPATEN).filter(ee.Filter.eq("GAUL0_NAME", "Indonesia"))
    return fc.map(lambda k: k.set("kunci", ee.String(k.get("GAUL1_NAME")).cat("|").cat(k.get("GAUL2_NAME"))))


def ee_provinsi(ee):
    fc = ee.FeatureCollection(GAUL_PROVINSI).filter(ee.Filter.eq("GAUL0_NAME", "Indonesia"))
    return fc.map(lambda k: k.set("kunci", k.get("GAUL1_NAME")))


def _bulatkan_koordinat(c, n: int = 3):
    if isinstance(c[0], (int, float)):
        return [round(c[0], n), round(c[1], n)]
    return [_bulatkan_koordinat(x, n) for x in c]


def unduh_batas_kabupaten(ee, f: Path = BERKAS_KABUPATEN) -> dict:
    """GeoJSON batas 514 kabupaten/kota Indonesia (disederhanakan ±1,5 km) beserta luas, provinsi, dan pulau.
    Disimpan sekali."""
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    fitur = []
    for pl, provs in PULAU_PROVINSI.items():   # per pulau supaya satu permintaan tidak terlalu besar
        fc = ee_kabupaten(ee).filter(ee.Filter.inList("GAUL1_NAME", provs)).map(
            lambda k: ee.Feature(k.geometry().simplify(1500), {
                "kabupaten": k.get("GAUL2_NAME"), "provinsi": k.get("GAUL1_NAME"),
                "luas_ha": k.geometry().area(1000).divide(1e4)}))
        for x in fc.getInfo()["features"]:
            geo = x["geometry"]
            if geo["type"] == "GeometryCollection":   # buang potongan garis/titik sisa penyederhanaan
                poli = [y for y in geo["geometries"] if y["type"] in ("Polygon", "MultiPolygon")]
                koord = [c for y in poli for c in ([y["coordinates"]] if y["type"] == "Polygon" else y["coordinates"])]
                geo = {"type": "MultiPolygon", "coordinates": koord}
            geo["coordinates"] = _bulatkan_koordinat(geo["coordinates"])
            p = x["properties"]
            p["luas_ha"] = round(float(p["luas_ha"]))
            p["kunci"] = kunci_kab(p["provinsi"], p["kabupaten"])
            p["pulau"] = pl
            fitur.append({"type": "Feature", "properties": p, "geometry": geo})
    out = {"type": "FeatureCollection", "features": sorted(fitur, key=lambda x: x["properties"]["kunci"])}
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    return out


BERKAS_PROVINSI = FOLDER_DATA / "provinsi_indonesia.geojson"
BBOX_INDONESIA = [94.0, -11.5, 141.5, 6.5]


def unduh_batas_provinsi(ee, f: Path = BERKAS_PROVINSI) -> dict:
    """GeoJSON batas 38 provinsi (disederhanakan ±3 km) untuk garis batas tebal dan label nama. Disimpan sekali."""
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    g = ee_provinsi(ee).map(lambda k: ee.Feature(k.geometry().simplify(3000), {"provinsi": k.get("GAUL1_NAME")})).getInfo()
    fitur = []
    for x in g["features"]:
        geo = x["geometry"]
        if geo["type"] == "GeometryCollection":
            poli = [y for y in geo["geometries"] if y["type"] in ("Polygon", "MultiPolygon")]
            geo = {"type": "MultiPolygon",
                   "coordinates": [c for y in poli for c in ([y["coordinates"]] if y["type"] == "Polygon" else y["coordinates"])]}
        geo["coordinates"] = _bulatkan_koordinat(geo["coordinates"])
        p = x["properties"]
        fitur.append({"type": "Feature", "properties": {"provinsi": p["provinsi"], "pulau": pulau_dari(p["provinsi"])},
                      "geometry": geo})
    out = {"type": "FeatureCollection", "features": fitur}
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    return out


def ee_titik_api(ee, mulai: date, selesai: date) -> pd.DataFrame:
    """Titik api FIRMS (MODIS ±1 km) se-Indonesia pada [mulai, selesai): satu titik per piksel api (lon, lat)."""
    img = (ee.ImageCollection("FIRMS").filterDate(f"{mulai:%Y-%m-%d}", f"{selesai:%Y-%m-%d}")
           .select(["T21", "confidence"]).map(_api_firms).max().toInt().selfMask().rename("api"))   # reduceToVectors butuh bilangan bulat
    fc = img.reduceToVectors(geometry=ee.Geometry.Rectangle(BBOX_INDONESIA), scale=1000, geometryType="centroid",
                             maxPixels=1e10, bestEffort=True, tileScale=4)
    fc = fc.map(lambda f: f.set({"lon": f.geometry().coordinates().get(0), "lat": f.geometry().coordinates().get(1)}))
    d = ee.data.computeFeatures({"expression": fc.select(["lon", "lat"]), "fileFormat": "PANDAS_DATAFRAME"})
    return pd.DataFrame(d)[["lon", "lat"]].round(4) if len(d) else pd.DataFrame(columns=["lon", "lat"])


def tanggal_terbaru_smap(ee) -> date:
    """Hari lengkap terakhir yang tersedia di SMAP L4 (biasanya 3–5 hari lalu)."""
    t = ee.Date(ee.ImageCollection(SMAP_L4).filterDate(f"{date.today().year - 1}-01-01", "2100-01-01")
                .aggregate_max("system:time_start")).format("YYYY-MM-dd").getInfo()
    return date.fromisoformat(t) - timedelta(days=1)   # hari terakhir bisa belum lengkap 8 citra


def jendela_periode(tahun: int, bm: int, bs: int, terbaru: date) -> tuple[date, date, date]:
    """(awal deret Kalman = 1 bulan sebelum periode, awal periode, akhir periode dipotong ke data terbaru)."""
    awal = date(tahun, max(bm - 1, 1), 1)
    akhir = (date(tahun, bs + 1, 1) - timedelta(days=1)) if bs < 12 else date(tahun, 12, 31)
    akhir = min(akhir, terbaru)
    if akhir < date(tahun, bm, 1):
        raise RuntimeError(f"Data SMAP baru tersedia sampai {terbaru:%d-%m-%Y}; pilih bulan yang sudah lewat.")
    return awal, date(tahun, bm, 1), akhir


def _geser_tahun(d: date, n: int) -> date:
    return (pd.Timestamp(d) + pd.DateOffset(years=n)).date()


def _segmen_tahun(a: date, b: date) -> list:
    """[a, b] dipotong per tahun kalender: [(a, 31 Des), (1 Jan, 31 Des), …, (1 Jan, b)]."""
    return [(max(a, date(th, 1, 1)), min(b, date(th, 12, 31))) for th in range(a.year, b.year + 1)]


def jendela_rentang(mulai: date, selesai: date, terbaru: date) -> dict:
    """Jendela "uji" (rentang pilihan pengguna, dipotong ke data terbaru) dan "latih" (untuk MLE Kalman).
    Rentang < 1 tahun: latih = tanggal yang sama setahun sebelumnya (bila sebelum SMAP tersedia, setahun
    sesudahnya). Rentang ≥ 1 tahun: latih = satu tahun tepat sebelum mulai; bila itu sebelum SMAP tersedia,
    tahun pertama rentang itu sendiri. Tiap jendela = (awal deret Kalman = 1 bulan sebelum mulai, mulai, selesai)."""
    selesai = min(selesai, terbaru)
    if selesai < mulai:
        raise RuntimeError(f"Data SMAP baru tersedia sampai {tgl_id(terbaru)}; pilih tanggal mulai sebelum itu.")
    awal = lambda m: max((pd.Timestamp(m) - pd.DateOffset(months=1)).date(), SMAP_PERTAMA)
    uji = (awal(mulai), mulai, selesai)
    if selesai >= _geser_tahun(mulai, 1):   # rentang panjang
        lm = _geser_tahun(mulai, -1)
        if lm >= SMAP_PERTAMA + timedelta(days=30):
            return {"uji": uji, "latih": (awal(lm), lm, mulai - timedelta(days=1))}
        return {"uji": uji, "latih": (awal(mulai), mulai, _geser_tahun(mulai, 1) - timedelta(days=1))}
    geser = -1 if _geser_tahun(mulai, -1) >= SMAP_PERTAMA + timedelta(days=30) else 1
    lm, ls = _geser_tahun(mulai, geser), _geser_tahun(selesai, geser)
    if ls > terbaru:
        raise RuntimeError("Tidak ada data setahun sebelum atau sesudah rentang ini untuk melatih model.")
    return {"uji": uji, "latih": (awal(lm), lm, ls)}


def tgl_id(d) -> str:
    """Tanggal dengan nama bulan Indonesia singkat, mis. 12 Okt 2026."""
    d = pd.Timestamp(d)
    return f"{d.day} {['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 'Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des'][d.month - 1]} {d.year}"


def _tumpukan_harian(ee, pita: str, mulai: date, selesai: date, reducer: str = "mean", awalan: str = "d_"):
    """Satu citra dengan satu pita per hari (nama awalan+YYYYMMDD) dari 8 citra 3-jam SMAP L4."""
    kol = ee.ImageCollection(SMAP_L4).select(pita)
    proj = kol.first().projection()
    hari = [mulai + timedelta(days=i) for i in range((selesai - mulai).days + 1)]
    # Ada hari tanpa citra SMAP; komposit kosong tidak punya pita dan merusak urutan nama pita.
    # Citra bertopeng penuh ini menjamin tiap hari tetap satu pita (nilai kosong = topeng).
    kosong = ee.ImageCollection([ee.Image.constant(0).toFloat().rename(pita).updateMask(0)])

    def satu(t):
        d = ee.Date(t)
        c = kol.filterDate(d, d.advance(1, "day")).map(lambda i: i.toFloat()).merge(kosong)
        return (c.max() if reducer == "max" else c.mean()).setDefaultProjection(proj)

    img = ee.ImageCollection([satu(f"{t:%Y-%m-%d}") for t in hari]).toBands()
    return img.rename([f"{awalan}{t:%Y%m%d}" for t in hari]), proj, hari


def ee_cuaca_wilayah(ee, fc, mulai: date, selesai: date) -> pd.DataFrame:
    """Hujan dan evapotranspirasi (mm/hari) serta suhu maksimum (°C) harian per fitur fc (kolom kunci) dari
    SMAP L4. Fluks SMAP dalam kg m⁻² s⁻¹ = mm/s, jadi × 86 400 = mm/hari."""
    hujan, proj, hari = _tumpukan_harian(ee, "precipitation_total_surface_flux", mulai, selesai, awalan="h_")
    et, _, _ = _tumpukan_harian(ee, "land_evapotranspiration_flux", mulai, selesai, awalan="e_")
    suhu, _, _ = _tumpukan_harian(ee, "temp_lowatmmodlay", mulai, selesai, "max", awalan="t_")
    img = ee.Image.cat([hujan.multiply(86400), et.multiply(86400), suhu.subtract(273.15)])
    r = img.reduceRegions(fc.select(["kunci"]), ee.Reducer.mean(), crs=proj, scale=9000, tileScale=4).getInfo()
    baris = []
    for f in r["features"]:
        p = f["properties"]
        for t in hari:
            s = f"{t:%Y%m%d}"
            baris.append({"kunci": p["kunci"], "tanggal": f"{t:%Y-%m-%d}", "hujan_mm": p.get("h_" + s),
                          "et_mm": p.get("e_" + s), "tmax_c": p.get("t_" + s)})
    d = pd.DataFrame(baris).sort_values(["kunci", "tanggal"])
    # hari tanpa citra SMAP: isi dengan interpolasi per wilayah
    kol = ["hujan_mm", "et_mm", "tmax_c"]
    d[kol] = d.groupby("kunci")[kol].transform(lambda s: s.interpolate(limit_direction="both"))
    return d.reset_index(drop=True)


ee_cuaca_kabupaten = ee_cuaca_wilayah   # nama lama


def ee_muka_air_piksel(ee, fc, mulai: date, selesai: date) -> pd.DataFrame:
    """Muka air gambut harian (m) per piksel SMAP 9 km di dalam kabupaten-kabupaten fc.
    Format sama dengan s1 BAGIAN 3 (kolom vv_YYYYMMDD; mu_vv = 0, sd_vv = 1 supaya z = h)."""
    h, proj, hari = _tumpukan_harian(ee, PITA_MUKA_AIR, mulai, selesai, awalan="vv_")
    # Piksel gambut = punya nilai pada hari mana pun. Hari kosong diisi −9999 (lalu NaN di Python) supaya
    # piksel tidak dibuang sampleRegions hanya karena satu hari tanpa citra.
    gambut = h.reduce(ee.Reducer.count()).gt(0)
    img = ee.Image.cat([ee.Image.pixelLonLat().rename(["lon", "lat"]), h.unmask(-9999)]).updateMask(gambut)
    r = img.sampleRegions(fc.map(lambda k: ee.Feature(k.geometry().simplify(500), {"kunci": k.get("kunci")})),
                          ["kunci"], projection=proj, tileScale=4, geometries=False).getInfo()
    d = pd.DataFrame([f["properties"] for f in r["features"]]).replace(-9999, np.nan)
    kolom = ["kunci", "lat", "lon", "mu_vv", "sd_vv", "n_basah"]
    if d.empty:
        return pd.DataFrame(columns=kolom)
    return d.assign(mu_vv=0.0, sd_vv=1.0, n_basah=3)[kolom + sorted(k for k in d.columns if k.startswith("vv_"))]


FIRMS_KEYAKINAN_MIN = 30   # MODIS: < 30 = keyakinan rendah (Giglio dkk. 2016); hanya nominal dan tinggi dipakai


def _api_firms(i):
    """Satu citra FIRMS → 1 bila piksel api dengan keyakinan ≥ FIRMS_KEYAKINAN_MIN, selain itu 0."""
    return i.select("T21").gt(0).And(i.select("confidence").gte(FIRMS_KEYAKINAN_MIN)).rename("T21").toFloat()


def ee_topeng_gambut(ee):
    """Piksel gambut SMAP L4 (pita muka air gambut hanya terisi di piksel gambut PEATCLSM)."""
    return (ee.ImageCollection(SMAP_L4).filterDate("2024-06-01", "2024-06-08").select(PITA_MUKA_AIR)
            .count().gt(0))


def ee_api_mingguan(ee, fc, senin: list) -> pd.DataFrame:
    """Jumlah piksel api FIRMS (MODIS ±1 km, keyakinan ≥ 30) DI DALAM PIKSEL GAMBUT SMAP per kabupaten per
    minggu (Senin–Minggu). Kebakaran di lahan non-gambut tidak dihitung, supaya sasaran prediksi sejalan
    dengan prediktornya (kekeringan gambut)."""
    firms = ee.ImageCollection("FIRMS").select(["T21", "confidence"])
    nol = ee.ImageCollection([ee.Image.constant(0).rename("T21").toFloat()])   # minggu tanpa citra tetap 0
    gambut = ee_topeng_gambut(ee)
    pita = [firms.filterDate(f"{s:%Y-%m-%d}", f"{s + timedelta(days=7):%Y-%m-%d}")
            .map(_api_firms).merge(nol).max().rename(f"w_{s:%Y%m%d}") for s in senin]
    img = ee.Image.cat(pita).updateMask(gambut)
    r = img.reduceRegions(fc.select(["kunci"]), ee.Reducer.sum(), scale=1000, tileScale=4).getInfo()
    baris = [{"petak_id": f["properties"]["kunci"], "minggu": f"{s:%Y-%m-%d}",
              "n": int(round(f["properties"].get(f"w_{s:%Y%m%d}") or 0))}
             for f in r["features"] for s in senin]
    return pd.DataFrame(baris, columns=["petak_id", "minggu", "n"]).query("n > 0").reset_index(drop=True)


def olah_provinsi(s1: dict, cuaca: dict, kab: dict, tl, tu, mulai: dict, selesai: dict,
                  ambang: float = AMBANG_MUKA_AIR_M) -> pd.DataFrame:
    """s1/kab: {peran: DataFrame piksel, Series kunci kabupaten per piksel}; cuaca: {peran: harian};
    mulai/selesai: {peran: tanggal}. Parameter Kalman (MLE) diestimasi pada peran tl, lalu peluang kering
    mingguan dihitung dan dirata-rata per kabupaten (kolom petak_id = kunci kabupaten)."""
    tanggal = {th: pd.DatetimeIndex(pd.to_datetime(cuaca[th]["tanggal"])) for th in (tl, tu)}
    z, _ = deret_observasi(s1[tl], tanggal[tl])
    par = estimasi_parameter(cuaca[tl]["hujan_mm"].to_numpy(float), cuaca[tl]["et_mm"].to_numpy(float), z, 400)
    tabel = []
    for th in (tl, tu):
        if s1[th].empty:
            continue
        dc = cuaca[th]["dc"].to_numpy() if "dc" in cuaca[th] else drought_code(cuaca[th]["hujan_mm"], cuaca[th]["tmax_c"])
        t = susun_petak_mingguan(s1[th], cuaca[th], par, dc, mulai=mulai[th], selesai=selesai[th], ambang=ambang)
        t["petak_id"] = np.tile(kab[th].to_numpy(), len(t) // len(s1[th]))
        g = t.groupby(["petak_id", "minggu"])
        tabel.append(pd.DataFrame({
            "p": g["p"].mean(), "p_bawah": g["p_bawah"].mean(), "p_atas": g["p_atas"].mean(),
            "muka_air_m": g["indeks_basah"].mean(), "n_titik": g.size(), "dc": g["dc"].first(),
            "hujan_30hari_mm": g["hujan_30hari_mm"].first(),
        }).reset_index())
    # Bila jendela latih tumpang tindih dengan rentang tampil (rentang panjang di awal masa SMAP), pakai hasil "uji".
    out = pd.concat(tabel, ignore_index=True).drop_duplicates(["petak_id", "minggu"], keep="last")
    out = out.reset_index(drop=True)
    for k, v in par.items():
        out.attrs[k] = v
    return out


# Permintaan Earth Engine yang berjalan bersamaan. Project nonkomersial yang belum diverifikasi ("Restricted
# Mode") hanya boleh sedikit permintaan bersamaan; lebih dari itu ditolak dengan "Too Many Requests".
EE_PARALEL = 3


def _coba_ulang_ee(fn, *arg, percobaan: int = 7):
    """Jalankan satu permintaan Earth Engine; bila ditolak karena batas permintaan bersamaan/kuota
    (Too Many Requests, 429, concurrency), tunggu makin lama (±5, 10, 20, … detik) lalu coba lagi."""
    import random
    import time
    for i in range(percobaan):
        try:
            return fn(*arg)
        except Exception as e:
            pesan = str(e).lower()
            sibuk = any(s in pesan for s in ("too many requests", "concurrency", "429", "quota", "rate limit"))
            if not sibuk or i == percobaan - 1:
                raise
            time.sleep(min(5 * 2 ** i, 120) + random.uniform(0, 3))


SIGMA_SMAP_M = 0.10   # [A] galat sistematik muka air SMAP L4 (PEATCLSM) terhadap muka air lapangan, meter


def lengkapi_cuaca(cuaca: pd.DataFrame) -> pd.DataFrame:
    """Cuaca harian per wilayah (kunci) yang sudah digabung dari semua potongan unduhan: buang duplikat, lalu
    Drought Code dan hujan 30 hari dihitung BERSAMBUNG sepanjang deret (tidak diulang tiap 1 Januari; di
    Indonesia tidak ada musim dingin yang mengatur ulang kekeringan). Bila ada lompatan tanggal (rentang
    pendek: jendela latih setahun sebelumnya terpisah), perhitungan dimulai ulang setelah lompatan itu."""
    c = (cuaca.dropna(subset=["hujan_mm", "tmax_c"]).drop_duplicates(["kunci", "tanggal"])
         .sort_values(["kunci", "tanggal"]).reset_index(drop=True))
    t = pd.to_datetime(c["tanggal"])
    blok = ((c["kunci"] != c["kunci"].shift()) | (t.diff().dt.days != 1)).cumsum()
    c["dc"] = np.nan
    c["hujan30"] = np.nan
    for _, i in c.groupby(blok).groups.items():
        c.loc[i, "dc"] = drought_code(c.loc[i, "hujan_mm"], c.loc[i, "tmax_c"])
        c.loc[i, "hujan30"] = c.loc[i, "hujan_mm"].rolling(30, min_periods=1).sum().to_numpy()
    return c


def olah_provinsi_smap(piksel: dict, cuaca_kab: pd.DataFrame, jendela: dict, latih: str = "latih",
                       ambang: float = AMBANG_MUKA_AIR_M, sigma_sis: float = SIGMA_SMAP_M) -> pd.DataFrame:
    """Peluang kering mingguan per kabupaten dari piksel SMAP satu provinsi.

    piksel: {peran: DataFrame kunci, lat, lon, …, vv_YYYYMMDD (muka air m)}; cuaca_kab: keluaran lengkapi_cuaca
    dengan kunci = kabupaten; jendela: {peran: (awal deret, awal tampil, akhir)}.
    - Tiap piksel memakai hujan dan ET kabupatennya sendiri (kab terlalu kecil untuk piksel 9 km memakai
      rata-rata provinsi).
    - Parameter Kalman (MLE) diestimasi pada peran `latih`, lalu filter dijalankan untuk semua peran.
    - p = Φ((ambang − ĥ)/σ), σ² = ragam filter + σ_sis². σ_sis = galat sistematik SMAP terhadap lapangan [A]:
      galat ini menetap dari hari ke hari, jadi tidak bisa dihaluskan filter (MLE menaksir R ≈ 0 karena
      keluaran SMAP L4 sendiri mulus)."""
    cu = {v: cuaca_kab.assign(tanggal=pd.to_datetime(cuaca_kab["tanggal"]))
                    .pivot_table(index="tanggal", columns="kunci", values=v)
          for v in ("hujan_mm", "et_mm", "dc", "hujan30")}

    def susun(pr):
        a, m0, b = jendela[pr]
        tanggal = pd.date_range(a, b)
        d = piksel[pr]
        kunci = d["kunci"].to_numpy()
        z, _ = deret_observasi(d, tanggal)

        def M(v):
            t = cu[v].reindex(index=tanggal).interpolate(limit_direction="both")
            rata = t.mean(axis=1)
            t = t.reindex(columns=np.unique(kunci)).apply(lambda kol: kol.fillna(rata))
            return t[kunci].to_numpy(float)
        return tanggal, kunci, z, M("hujan_mm"), M("et_mm"), M("dc"), M("hujan30"), m0, b

    # latih dulu, tampil terakhir: bila tumpang tindih, hasil jendela tampil yang dipakai (keep="last")
    urutan = [latih] + [pr for pr in jendela if pr != latih]
    data = {pr: susun(pr) for pr in urutan if pr in piksel and not piksel[pr].empty}
    _, _, z_l, H_l, E_l, *_ = data[latih]
    par = estimasi_parameter(H_l, E_l, z_l, 400)
    tabel = []
    for pr, (tanggal, kunci, z, H, E, DC, H30, m0, b) in data.items():
        ada = np.isfinite(z)
        h0 = np.where(ada.any(0), z[np.where(ada.any(0), np.argmax(ada, axis=0), 0), np.arange(z.shape[1])],
                      par["h_ref"])
        h, var = kalman_muka_air(H, E, z, par["R"], par["a"], par["b"], par["c"], par["h_ref"], par["Q"],
                                 h0=h0, P0=1.0)
        sig = np.sqrt(var + sigma_sis ** 2)
        for t in np.where((tanggal.weekday == 0) & (tanggal >= pd.Timestamp(m0)) & (tanggal <= pd.Timestamp(b)))[0]:
            d = pd.DataFrame({"petak_id": kunci, "p": peluang_kering(h[t], sig[t], ambang),
                              "p_bawah": peluang_kering(h[t] + 1.645 * np.sqrt(var[t]), sig[t], ambang),
                              "p_atas": peluang_kering(h[t] - 1.645 * np.sqrt(var[t]), sig[t], ambang),
                              "muka_air_m": h[t], "dc": DC[t], "hujan_30hari_mm": H30[t]})
            g = d.groupby("petak_id")
            tabel.append(g.mean().assign(n_titik=g.size(), minggu=tanggal[t].date()).reset_index())
    out = (pd.concat(tabel, ignore_index=True).drop_duplicates(["petak_id", "minggu"], keep="last")
           .reset_index(drop=True))
    out[["dc", "hujan_30hari_mm"]] = out[["dc", "hujan_30hari_mm"]].round(1)
    out = out[["petak_id", "minggu", "p", "p_bawah", "p_atas", "muka_air_m", "n_titik", "dc", "hujan_30hari_mm"]].copy()
    for k, v in {**par, "sigma_sis": sigma_sis}.items():
        out.attrs[k] = v
    return out


HARI_PEMANASAN_DC = 90   # cuaca diambil 90 hari sebelum tanggal mulai supaya Drought Code sudah "panas"
CELAH_BLOK_HARI = 14     # deret muka air dipisah menjadi blok bila ada lompatan tanggal lebih dari ini


def _tgl8(t: str) -> date:
    return date(int(t[:4]), int(t[4:6]), int(t[6:8]))


def cakupan_berkas(mentah: Path, awalan: str, slug: str) -> list:
    """Unduhan yang sudah tersimpan untuk satu jenis data dan satu wilayah: [(awal, akhir, berkas)], dari nama
    berkas {awalan}_{YYYYMMDD}_{YYYYMMDD}_{slug}.csv."""
    import re
    pola = re.compile(rf"^{re.escape(awalan)}_(\d{{8}})_(\d{{8}})_{re.escape(slug)}\.csv$")
    return sorted((_tgl8(m.group(1)), _tgl8(m.group(2)), f) for f in Path(mentah).glob(f"{awalan}_*.csv")
                  if (m := pola.match(f.name)))


def cakupan_api(mentah: Path, slug: str) -> list:
    """Seperti cakupan_berkas untuk titik api mingguan, dengan akhir efektif: minggu yang belum lengkap saat
    diunduh (Senin + 7 hari + 2 hari jeda FIRMS) dianggap belum ada, jadi akan diambil ulang."""
    out = []
    for s0, e, f in cakupan_berkas(mentah, f"api_gambut_k{FIRMS_KEYAKINAN_MIN}", slug):
        diunduh = date.fromtimestamp(f.stat().st_mtime)
        while e >= s0 and diunduh < e + timedelta(days=9):
            e -= timedelta(days=7)
        if e >= s0:
            out.append((s0, e + timedelta(days=6), f))
    return out


def celah_tanggal(a: date, b: date, cakupan: list) -> list:
    """Bagian [a, b] yang BELUM tercakup unduhan tersimpan, dipotong per tahun kalender (permintaan kecil)."""
    celah, kini = [], a
    for s0, e, *_ in sorted(cakupan):
        if e < kini or s0 > b:
            continue
        if s0 > kini:
            celah.append((kini, min(b, s0 - timedelta(days=1))))
        kini = max(kini, e + timedelta(days=1))
        if kini > b:
            break
    if kini <= b:
        celah.append((kini, b))
    return [(max(x, date(th, 1, 1)), min(y, date(th, 12, 31))) for x, y in celah for th in range(x.year, y.year + 1)]


def gabung_piksel(berkas: list) -> pd.DataFrame:
    """Semua unduhan muka air satu provinsi digabung menyamping per piksel (kolom harian vv_YYYYMMDD)."""
    id_ = ["kunci", "lat", "lon"]
    bagian = [d.set_index(id_)[[c for c in d.columns if c.startswith("vv_")]]
              for d in (pd.read_csv(f) for f in berkas) if not d.empty]
    if not bagian:
        return pd.DataFrame(columns=id_ + ["mu_vv", "sd_vv", "n_basah"])
    out = pd.concat(bagian, axis=1)
    out = out.loc[:, ~out.columns.duplicated(keep="last")]
    out = out[sorted(out.columns)].reset_index()
    return out.assign(mu_vv=0.0, sd_vv=1.0, n_basah=3)[id_ + ["mu_vv", "sd_vv", "n_basah"] +
                                                       [c for c in out.columns if c.startswith("vv_")]]


def blok_tanggal(piksel: pd.DataFrame) -> list:
    """Rentang tanggal bersambung [(awal, akhir)] dari kolom harian; lompatan > CELAH_BLOK_HARI memisah blok."""
    t = sorted(pd.Timestamp(c[3:]).date() for c in piksel.columns if c.startswith("vv_"))
    if not t:
        return []
    blok, a0 = [], t[0]
    for x, y in zip(t, t[1:]):
        if (y - x).days > CELAH_BLOK_HARI:
            blok.append((a0, x))
            a0 = y
    return blok + [(a0, t[-1])]


def gabung_api(berkas: list) -> pd.DataFrame:
    """Titik api mingguan dari semua unduhan; untuk tiap Senin dipakai unduhan TERBARU yang mencakupnya
    (minggu yang dulu belum lengkap tergantikan oleh unduhan sesudahnya)."""
    pilih = {}
    for s0, e, f in sorted(berkas, key=lambda x: x[2].stat().st_mtime):
        for i in range((e - s0).days // 7 + 1):
            pilih[f"{s0 + timedelta(weeks=i):%Y-%m-%d}"] = f
    baris = [d[d["minggu"].map(pilih) == f] for f in {f for *_, f in berkas}
             if not (d := pd.read_csv(f)).empty]
    return (pd.concat(baris, ignore_index=True) if baris else pd.DataFrame(columns=["petak_id", "minggu", "n"]))


def proses_rentang(mulai: date, selesai: date, project_id: str, progres=None, ee=None, folder=FOLDER_HASIL,
                   hanya_provinsi: list | None = None) -> dict:
    """Ambil dari Earth Engine HANYA tanggal di [mulai, selesai] yang belum pernah diunduh (muka air dan cuaca
    per provinsi, titik api per pulau), lalu olah SEMUA data yang pernah diunduh menjadi satu kumpulan di data/.
    Dasbor menampilkan rentang tanggal mana pun yang ada di kumpulan itu tanpa unduh ulang."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    progres = progres or (lambda f, m: print(f"[{100 * f:5.1f}%] {m}", flush=True))
    ee = ee or ee_mulai(project_id)
    mentah = Path(folder) / "mentah_smap"
    mentah.mkdir(parents=True, exist_ok=True)
    progres(0.02, "Mengecek data terbaru NASA SMAP dan batas kabupaten/kota")
    terbaru = tanggal_terbaru_smap(ee)
    selesai = min(selesai, terbaru)
    if selesai < mulai:
        raise RuntimeError(f"Data SMAP baru tersedia sampai {tgl_id(terbaru)}; pilih tanggal mulai sebelum itu.")
    batas = unduh_batas_kabupaten(ee)
    try:
        unduh_batas_provinsi(ee)   # hanya untuk tampilan peta; gagal pun tidak menghentikan proses
    except Exception:
        pass
    provinsi = sorted({f["properties"]["provinsi"] for f in batas["features"]})
    if hanya_provinsi:
        provinsi = [p for p in provinsi if p in hanya_provinsi]
    pulau = {pl: [pv for pv in provs if pv in provinsi] for pl, provs in PULAU_PROVINSI.items()}
    pulau = {pl: provs for pl, provs in pulau.items() if provs}
    fc = ee_kabupaten(ee)

    rencana = {}
    for pv in provinsi:
        sl, f_pv = _slug(pv), fc.filter(ee.Filter.eq("GAUL1_NAME", pv))
        for s0, e in celah_tanggal(mulai, selesai, cakupan_berkas(mentah, "muka_air", sl)):
            rencana[mentah / f"muka_air_{s0:%Y%m%d}_{e:%Y%m%d}_{sl}.csv"] = (ee_muka_air_piksel, (ee, f_pv, s0, e))
        awal_cuaca = max(mulai - timedelta(days=HARI_PEMANASAN_DC), SMAP_PERTAMA)
        for s0, e in celah_tanggal(awal_cuaca, selesai, cakupan_berkas(mentah, "cuaca_kab", sl)):
            rencana[mentah / f"cuaca_kab_{s0:%Y%m%d}_{e:%Y%m%d}_{sl}.csv"] = (ee_cuaca_wilayah, (ee, f_pv, s0, e))
    senin0 = mulai - timedelta(days=mulai.weekday())
    akhir_api = min(selesai + timedelta(weeks=JANGKA_MAKS), date.today())
    for pl in pulau:
        sl = f"h{JANGKA_MAKS}_{_slug(pl)}"
        for s0, e in celah_tanggal(senin0, akhir_api, cakupan_api(mentah, sl)):
            pertama = s0 + timedelta(days=(7 - s0.weekday()) % 7)
            senin = [pertama + timedelta(weeks=i) for i in range((e - pertama).days // 7 + 1)] if pertama <= e else []
            if senin:
                f = mentah / f"api_gambut_k{FIRMS_KEYAKINAN_MIN}_{senin[0]:%Y%m%d}_{senin[-1]:%Y%m%d}_{sl}.csv"
                rencana[f] = (ee_api_mingguan, (ee, fc.filter(ee.Filter.inList("GAUL1_NAME", pulau[pl])), senin))
    if rencana:
        with ThreadPoolExecutor(EE_PARALEL) as ex:
            jalan = {ex.submit(_coba_ulang_ee, fn, *arg): f for f, (fn, arg) in rencana.items()}
            for i, fut in enumerate(as_completed(jalan)):
                f = jalan[fut]
                fut.result().to_csv(f, index=False)
                progres(0.05 + 0.75 * (i + 1) / len(jalan), f"Terunduh: {f.stem} ({i + 1}/{len(jalan)})")
    else:
        progres(0.80, "Semua tanggal di rentang ini sudah pernah diunduh")
    return olah_semua(mentah, provinsi, pulau, progres, folder, mulai, selesai)


def olah_semua(mentah: Path, provinsi: list, pulau: dict, progres, folder, mulai: date, selesai: date) -> dict:
    """Olah SEMUA unduhan tersimpan menjadi data/kabupaten_mingguan.csv (satu kumpulan bersambung). Tiap blok
    tanggal bersambung difilter terpisah; parameter Kalman (MLE) per provinsi dari blok terpanjang."""
    tabel, parameter, gagal, tanpa_gambut, sampai = [], {}, [], [], []
    for i, pv in enumerate(provinsi):
        progres(0.82 + 0.16 * i / max(len(provinsi), 1), f"Menghitung peluang kering: {pv}")
        sl = _slug(pv)
        piksel = gabung_piksel([f for *_, f in cakupan_berkas(mentah, "muka_air", sl)])
        if piksel.empty:   # provinsi tanpa piksel gambut SMAP
            tanpa_gambut.append(pv)
            continue
        try:
            berkas_cuaca = [f for *_, f in cakupan_berkas(mentah, "cuaca_kab", sl)]
            cuaca = lengkapi_cuaca(pd.concat([pd.read_csv(f) for f in berkas_cuaca], ignore_index=True))
            blok = blok_tanggal(piksel)
            jendela = {f"blok{j}": (a, a, b) for j, (a, b) in enumerate(blok)}
            # MLE pada 365 hari terakhir blok terpanjang: cukup untuk 5 parameter dan menjaga olah ulang ±10 menit
            # (MLE seluruh deret multi-tahun ±30 menit). Baris mingguan jendela "latih" tertimpa hasil bloknya.
            a_b, _, b_l = jendela[max(jendela, key=lambda kk: (jendela[kk][2] - jendela[kk][0]).days)]
            a_l = max(a_b, b_l - timedelta(days=364))
            jendela = {"latih": (a_l, a_l, b_l), **jendela}
            t = olah_provinsi_smap({kk: piksel for kk in jendela}, cuaca, jendela, latih="latih")
        except Exception as e:   # satu provinsi gagal tidak menggagalkan yang lain
            gagal.append(f"{pv}: {e}")
            continue
        parameter[pv] = dict(t.attrs)
        tabel.append(t)
        sampai.append(blok[-1][1])
    if not tabel:
        raise RuntimeError("Tidak ada provinsi yang berhasil diolah. " + "; ".join(gagal))
    kab_mingguan = pd.concat(tabel, ignore_index=True)
    api = pd.concat([gabung_api(cakupan_berkas(mentah, f"api_gambut_k{FIRMS_KEYAKINAN_MIN}",
                                               f"h{JANGKA_MAKS}_{_slug(pl)}")) for pl in pulau], ignore_index=True)
    data_sampai = max(sampai)
    FOLDER_DATA.mkdir(exist_ok=True)
    kab_mingguan.to_csv(BERKAS_KAB_MINGGUAN, index=False, float_format="%.6g")
    api.to_csv(BERKAS_HOTSPOT_KAB, index=False)
    selesai = min(selesai, data_sampai)
    BERKAS_PENGATURAN.write_text(json.dumps({
        "mode": "kabupaten", "wilayah": "indonesia", "sumber": "SMAP L4",
        "mulai": f"{mulai:%Y-%m-%d}", "selesai": f"{selesai:%Y-%m-%d}",
        "data_awal": f"{pd.to_datetime(kab_mingguan['minggu']).min():%Y-%m-%d}", "data_sampai": f"{data_sampai:%Y-%m-%d}",
        # kunci lama, dipakai bagian lain yang masih berpikir per tahun/bulan
        "tahun_uji": selesai.year, "tahun_latih": mulai.year - 1, "bulan_mulai": mulai.month,
        "bulan_selesai": selesai.month}), encoding="utf-8")
    pd.DataFrame(parameter).T.to_csv(Path(folder) / "parameter_kalman_provinsi.csv")
    progres(1.0, "Selesai")
    return {"kabupaten_mingguan": kab_mingguan, "hotspot": api, "parameter": parameter, "gagal": gagal,
            "tanpa_gambut": tanpa_gambut, "data_sampai": data_sampai, "tahun_latih": mulai.year - 1}


def proses_kalimantan(tahun: int, bulan_mulai: int, bulan_selesai: int, project_id: str, map_key: str = "",
                      progres=None, ee=None, folder=FOLDER_HASIL, hanya_provinsi: list | None = None) -> dict:
    """Bentuk lama per tahun + bulan: bulan_mulai tanggal 1 sampai akhir bulan_selesai."""
    akhir = date(tahun, bulan_selesai + 1, 1) - timedelta(days=1) if bulan_selesai < 12 else date(tahun, 12, 31)
    return proses_rentang(date(tahun, bulan_mulai, 1), akhir, project_id, progres, ee, folder, hanya_provinsi)


# =============================================================================
# BAGIAN 5. DASBOR STREAMLIT
# Tiga tab: Peta Peluang Kering · Level Peringatan · Simulasi Anggaran, per kabupaten/kota
# (atau per provinsi) se-Kalimantan. Semua isian (kunci akses, tahun, bulan, minggu) di
# sidebar kiri; di tab 3 biaya dan anggaran.
# =============================================================================
APP_NAME = "SPARKS"
APP_VERSION = "0.6.0"
# Peta dasar tanpa label: nama negara berbahasa Inggris dan nama tempat bawaan menumpuk dengan poligon;
# label provinsi digambar sendiri dari data.
MAP_STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-nolabels-gl-style/style.json"
AMBANG_SEMENTARA = (0.25, 0.50, 0.75)  # [A] dipakai bila belum ada riwayat hotspot untuk kalibrasi
RASIO_BIAYA = 2                        # [A] tingkat kabupaten: melewatkan api 2x lebih mahal dari peringatan palsu
API_PER_10RIBU_HA = 1.0                # [A] "kebakaran meluas": ≥ 1 titik api VIIRS per 10.000 ha dalam k minggu

WARNA_KOSONG = [150, 150, 158, 28]     # bukan lahan gambut: hampir transparan supaya wilayah gambut menonjol
GARIS_KAB = [200, 212, 225, 55]        # garis kabupaten tipis dan samar
# Dua palet berbeda supaya peta "kekeringan sekarang" dan "risiko kebakaran ke depan" tidak tertukar:
# kekeringan = biru → kuning → oranye → merah (sama dengan peta peluang kering tab 1);
# kebakaran  = abu-abu → kuning api → oranye api → magenta.
WARNA_LEVEL = {"Aman": "#2c7bb6", "Waspada": "#f1e05a", "Siaga": "#fd9a44", "Awas": "#d7191c"}
WARNA_API = {"Aman": "#64748b", "Waspada": "#ffd43b", "Siaga": "#ff7a00", "Awas": "#d6249f"}
# Arti singkat tiap level (muat satu baris di legenda peta)
ARTI_LEVEL = {"Aman": "kondisi normal", "Waspada": "pantau rutin",
              "Siaga": "siapkan air", "Awas": "basahi segera"}
ARTI_API = {"Aman": "risiko rendah", "Waspada": "pantau titik api",
            "Siaga": "siapkan regu", "Awas": "patroli harian"}


# =============================================================================
# UTIL
# =============================================================================

def fmt(x, dec: int = 0) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    s = f"{x:,.{dec}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def rupiah(x) -> str:
    return f"Rp {fmt(x)}"


def hex_rgb(h: str, a: int = 230) -> list:
    return [int(h[i:i + 2], 16) for i in (1, 3, 5)] + [a]


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def tombol_unduh(df: pd.DataFrame, nama_berkas: str, label: str):
    """Tautan unduhan CSV dengan data tertanam di halaman. st.download_button meminta alamat unduhan ke server,
    jadi bila koneksi browser–server sempat putus (mis. saat server sibuk melatih model) muncul
    "Error: not connected to a server!". Tautan ini tidak bergantung pada koneksi itu."""
    import base64
    isi = base64.b64encode(b"\xef\xbb\xbf" + csv_bytes(df)).decode()   # BOM supaya Excel membaca UTF-8
    st.markdown(f'<a class="tombol-unduh" download="{nama_berkas}" href="data:text/csv;base64,{isi}">{label}</a>',
                unsafe_allow_html=True)


def tema_altair(chart):
    """Tema grafik seragam dengan kartu dan peta: huruf Inter, ukuran sama, garis bantu samar."""
    huruf = "Inter, -apple-system, Segoe UI, sans-serif"
    return (chart.configure(background="transparent", font=huruf, padding={"left": 6, "right": 10, "top": 8, "bottom": 4})
            .configure_axis(labelColor=C["text_muted"], titleColor=C["text_muted"], labelFontSize=11,
                            titleFontSize=11, titleFontWeight="normal", gridColor=C["border_soft"],
                            domainColor=C["border_active"], tickColor=C["border_active"])
            .configure_legend(labelColor=C["text"], titleColor=C["text_muted"], orient="top", labelFontSize=11,
                              symbolSize=80)
            .configure_title(color=C["text"], fontSize=12, anchor="start")
            .configure_view(stroke=None))


def persen(p) -> list:
    """Peluang sebagai teks. Peluang dari sebaran normal tidak pernah tepat 0 atau 1, jadi nilai yang
    dibulatkan ke 0%/100% ditulis "<1%"/">99%"."""
    def satu(x):
        if pd.isna(x):
            return "—"
        if x >= 0.995:
            return ">99%"
        if x < 0.005:
            return "<1%"
        return f"{100 * x:.0f}%" if x >= 0.1 else fmt(100 * x, 1) + "%"
    return [satu(x) for x in p]


# =============================================================================
# DATA (hasil olahan di data/; kunci cache = waktu ubah berkas)
# =============================================================================

@st.cache_resource(show_spinner="Membaca hasil olahan data satelit…")
def muat_data(mtime: float):
    batas = json.loads(BERKAS_KABUPATEN.read_text(encoding="utf-8"))
    for f in batas["features"]:   # berkas batas lama (Kalimantan saja) belum punya kolom pulau
        f["properties"].setdefault("pulau", pulau_dari(f["properties"]["provinsi"]))
    info = pd.DataFrame([f["properties"] for f in batas["features"]]).set_index("kunci")
    kab = pd.read_csv(BERKAS_KAB_MINGGUAN, parse_dates=["minggu"])
    kab.loc[kab["n_titik"] < TITIK_MINIMUM, ["p", "p_bawah", "p_atas"]] = np.nan
    kab = kab.join(info[["kabupaten", "provinsi", "pulau", "luas_ha"]], on="petak_id")
    hs = pd.read_csv(BERKAS_HOTSPOT_KAB, parse_dates=["minggu"]) if BERKAS_HOTSPOT_KAB.exists() else None
    pg = json.loads(BERKAS_PENGATURAN.read_text(encoding="utf-8"))
    return batas, kab, hs, pg


def label_kebakaran_meluas(kab: pd.DataFrame, hs: pd.DataFrame, k: int = K_MINGGU) -> pd.Series:
    """y = 1 bila pada minggu t+1..t+k ada ≥ API_PER_10RIBU_HA titik api per 10.000 ha di kabupaten itu.
    "Ada satu titik api" saja hampir selalu benar untuk kabupaten seluas ratusan ribu hektare."""
    # n = jumlah piksel api FIRMS per kabupaten-minggu (berkas lama: satu baris per titik api)
    hit = hs.groupby(["petak_id", "minggu"])["n"].sum() if "n" in hs else hs.groupby(["petak_id", "minggu"]).size()
    total = np.zeros(len(kab))
    for j in range(1, k + 1):
        kunci = pd.MultiIndex.from_arrays([kab["petak_id"], kab["minggu"] + pd.Timedelta(weeks=j)])
        total += hit.reindex(kunci).fillna(0).to_numpy()
    # kepadatan per luas GAMBUT kabupaten (titik api juga hanya yang di piksel gambut)
    return pd.Series(total / luas_gambut(kab).to_numpy() * 1e4 >= API_PER_10RIBU_HA, index=kab.index, name="api")


# --------------------------------------------------------------------------- XGBoost (tab 2, peta kedua)
# Memprediksi peluang KEBAKARAN MELUAS dalam 2 minggu (14 hari) berikutnya per kabupaten dari kondisi minggu
# ini. Hanya memakai informasi yang sudah tersedia pada hari Senin minggu itu (titik api minggu-minggu
# SEBELUMNYA, bukan sesudahnya), dan hanya dilatih pada minggu yang labelnya sudah teramati.
FITUR_XGB = {
    "p": "peluang kering (Kalman)", "muka_air_m": "muka air gambut",
    "dmuka_1": "perubahan muka air 1 minggu", "dmuka_2": "perubahan muka air 2 minggu",
    "hujan_30": "hujan 30 hari", "dc": "Drought Code",
    "api_lalu_1": "titik api minggu lalu", "api_lalu_2": "titik api 2 minggu lalu",
    "minggu_ke": "waktu dalam tahun (musim)",
}
def fitur_xgboost(kab: pd.DataFrame, hs: pd.DataFrame | None) -> pd.DataFrame:
    """9 fitur numerik per kabupaten-minggu (indeks sama dengan kab). Nilai kosong dibiarkan NaN (XGBoost
    menanganinya)."""
    urut = kab.sort_values(["petak_id", "minggu"])
    muka = urut["muka_air_m"] if "muka_air_m" in urut else pd.Series(np.nan, index=urut.index)
    g = muka.groupby(urut["petak_id"])
    X = pd.DataFrame({
        "p": kab["p"], "muka_air_m": muka, "dmuka_1": g.diff(1), "dmuka_2": g.diff(2),
        "hujan_30": kab.get("hujan_30hari_mm"), "dc": kab.get("dc"),
        "minggu_ke": kab["minggu"].dt.isocalendar().week.astype(int),
    }, index=kab.index)
    if hs is None or hs.empty:
        X["api_lalu_1"] = X["api_lalu_2"] = 0.0
    else:
        hit = hs.groupby(["petak_id", "minggu"])["n"].sum() if "n" in hs else hs.groupby(["petak_id", "minggu"]).size()
        for j in (1, 2):
            k = pd.MultiIndex.from_arrays([kab["petak_id"], kab["minggu"] - pd.Timedelta(weeks=j)])
            X[f"api_lalu_{j}"] = hit.reindex(k).fillna(0).to_numpy() / luas_gambut(kab).to_numpy() * 1e4
    return X[list(FITUR_XGB)].astype(float)


def latih_prediksi_xgboost(kab, hs, y, minggu: pd.Timestamp, rasio: float = RASIO_BIAYA,
                           k: int = K_MINGGU) -> dict | None:
    """Hasil XGBoost saja (nama lama; lihat latih_prediksi_model)."""
    semua = latih_prediksi_model(kab, hs, y, minggu, rasio, k, hanya=("XGBoost",))
    return None if semua is None else semua["model"]["XGBoost"]


def batas_latih(minggu: pd.Timestamp, k: int) -> pd.Timestamp:
    """Minggu terakhir yang labelnya sudah LENGKAP teramati pada Senin `minggu`. Label minggu t memakai
    kebakaran minggu t+1..t+k; minggu yang dimulai Senin itu sendiri belum terjadi, jadi t + k ≤ minggu − 1
    minggu → t ≤ minggu − (k + 1) minggu."""
    return minggu - pd.Timedelta(weeks=k + 1)


N_BLOK_UJI = 6


def oof_maju_waktu(model, X: pd.DataFrame, y: np.ndarray, minggu_s: pd.Series, k: int,
                   n_blok: int = N_BLOK_UJI) -> np.ndarray:
    """Validasi maju-waktu (rolling origin): minggu latih dibagi n_blok blok berurutan; tiap blok (kecuali
    yang pertama) ditebak model yang HANYA dilatih pada minggu ≤ awal blok − (k+1) minggu. Tidak ada informasi
    dari masa depan, termasuk label yang belum lengkap. Blok pertama tidak dinilai (NaN)."""
    from sklearn.base import clone
    minggu_u = np.sort(minggu_s.unique())
    oof = np.full(len(y), np.nan)
    for blok in np.array_split(minggu_u, n_blok)[1:]:
        if len(blok) == 0:
            continue
        uji = minggu_s.isin(blok).to_numpy()
        latih = (minggu_s <= pd.Timestamp(blok[0]) - pd.Timedelta(weeks=k + 1)).to_numpy()
        if y[latih].sum() < 10 or (1 - y[latih]).sum() < 10:
            continue
        m = clone(model).fit(X[latih], y[latih])
        oof[uji] = m.predict_proba(X[uji])[:, 1]
    return oof


METODE = ["XGBoost"]


# Batasan monoton dari ilmu gambut: risiko tidak boleh turun bila lebih kering / lebih banyak titik api, dan tidak
# boleh naik bila muka air atau hujan bertambah. Waktu dalam tahun tidak dibatasi (pola musim naik-turun).
MONOTON = {"p": 1, "muka_air_m": -1, "dmuka_1": -1, "dmuka_2": -1, "hujan_30": -1, "dc": 1,
           "api_lalu_1": 1, "api_lalu_2": 1, "minggu_ke": 0}


def _model_boosting() -> tuple[dict, list]:
    """XGBoost (300 pohon, kedalaman 3, laju belajar 0,05) dengan batasan monoton MONOTON."""
    from xgboost import XGBClassifier
    return {"XGBoost": XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                                     colsample_bytree=0.8, eval_metric="logloss", n_jobs=2, random_state=0,
                                     monotone_constraints=tuple(MONOTON[c] for c in FITUR_XGB))}, []


def _shap(nama: str, m, X: pd.DataFrame) -> np.ndarray:
    """Nilai SHAP (TreeSHAP bawaan XGBoost, satuan log-odds) per baris × fitur."""
    import xgboost as xgb
    return m.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]


def latih_prediksi_model(kab, hs, y, minggu: pd.Timestamp, rasio: float = RASIO_BIAYA,
                         k: int = K_MINGGU, hanya: tuple | None = None) -> dict | None:
    """Latih XGBoost.
    - Model akhir dilatih pada kabupaten-minggu ≤ batas_latih(minggu, k) (label sudah lengkap teramati).
    - Evaluasi dan ambang level dari validasi maju-waktu (oof_maju_waktu), bukan validasi silang acak per
      minggu: kebakaran berautokorelasi kuat, jadi mencampur minggu sesudahnya ke data latih memberi skor yang
      terlalu optimistis.
    Hasil: {"model": {nama: hasil}, "belum": [pustaka yang belum terpasang]}."""
    from sklearn.metrics import roc_auc_score
    if y is None:
        return None
    X0 = fitur_xgboost(kab, hs)
    latih = (kab["minggu"] <= batas_latih(minggu, k)) & kab["p"].notna()
    yl, mg = y[latih].astype(int).to_numpy(), kab.loc[latih, "minggu"]
    if yl.sum() < 20 or (1 - yl).sum() < 20 or mg.nunique() < 5:
        return None
    kini = (kab["minggu"] == minggu) & kab["p"].notna()
    nama_fitur = [FITUR_XGB[kol] for kol in X0.columns]
    idx_contoh = X0[latih].sample(min(4000, int(latih.sum())), random_state=0).index
    peringkat = X0.loc[idx_contoh].rank(pct=True).to_numpy().ravel()   # warna titik SHAP = peringkat nilai fitur
    model, belum = _model_boosting()
    if hanya:
        model = {n: m for n, m in model.items() if n in hanya}
    hasil = {}
    for nama, m in model.items():
        X = X0
        Xl, contoh = X[latih], X.loc[idx_contoh]
        oof_semua = oof_maju_waktu(m, Xl, yl, mg, k)
        dinilai = np.isfinite(oof_semua)
        oof, y_oof = oof_semua[dinilai], yl[dinilai]
        if y_oof.sum() < 5 or (1 - y_oof).sum() < 5:   # riwayat terlalu pendek untuk uji maju-waktu
            return None
        ambang = ambang_level(y_oof, oof, rasio) or AMBANG_SEMENTARA
        m.fit(Xl, yl)
        kontrib, kontrib_kini = _shap(nama, m, contoh), _shap(nama, m, X[kini])
        pred = pd.DataFrame({"petak_id": kab.loc[kini, "petak_id"].to_numpy(),
                             "prob": m.predict_proba(X[kini])[:, 1]})
        pred["pendorong"] = [", ".join(nama_fitur[j] for j in np.argsort(-r)[:2] if r[j] > 0) or "—"
                             for r in kontrib_kini]
        penting = pd.Series(np.abs(kontrib).mean(0), index=nama_fitur).sort_values(ascending=False)
        siaga = oof >= ambang[1]
        tp = int((siaga & (y_oof == 1)).sum())
        hasil[nama] = {
            "nama": nama, "pred": pred, "ambang": ambang, "auc": float(roc_auc_score(y_oof, oof)),
            "auc_kalman": float(roc_auc_score(y_oof, Xl.loc[dinilai, "p"])), "n_latih": int(latih.sum()),
            "n_uji": int(dinilai.sum()), "minggu_uji": int(mg[dinilai].nunique()),
            "n_api": int(yl.sum()), "jangka": k, "penting": penting, "penting_shap": penting,
            "shap": pd.DataFrame({"fitur": np.repeat([nama_fitur], len(contoh), axis=0).ravel(),
                                  "shap": kontrib.ravel(), "nilai": peringkat,
                                  "petak_id": np.repeat(kab.loc[idx_contoh, "petak_id"].to_numpy(),
                                                        len(nama_fitur)),
                                  "minggu": np.repeat(kab.loc[idx_contoh, "minggu"].to_numpy(), len(nama_fitur))}),
            "tepat": tp / max(int(siaga.sum()), 1), "tertangkap": tp / max(int(y_oof.sum()), 1),
            "oof": oof, "y_oof": y_oof, "minggu_oof": mg[dinilai].to_numpy(),
            "petak_oof": kab.loc[latih, "petak_id"].to_numpy()[dinilai], "fitur": list(X0.columns),
            "p_oof": Xl["p"].to_numpy()[dinilai]}
    return {"model": hasil, "belum": belum}


def _auc_cepat(y: np.ndarray, s: np.ndarray) -> float:
    from scipy.stats import rankdata
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((rankdata(s)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def uji_beda_auc(hasil: dict, acuan: str = "XGBoost", n_boot: int = 300, seed: int = 0) -> dict:
    """Selang kepercayaan 95% AUC dan nilai-p selisih AUC tiap model terhadap `acuan`, dari bootstrap
    BERKLASTER PER MINGGU (minggu diambil ulang, bukan baris), karena kab/kota dalam minggu yang sama saling
    berkorelasi. Semua model dinilai pada baris dan minggu yang sama."""
    rng = np.random.default_rng(seed)
    nama = list(hasil)
    y, mg = hasil[nama[0]]["y_oof"], hasil[nama[0]]["minggu_oof"]
    minggu_u = np.unique(mg)
    idx_minggu = {w: np.where(mg == w)[0] for w in minggu_u}
    boot = {n: [] for n in nama}
    for _ in range(n_boot):
        idx = np.concatenate([idx_minggu[w] for w in rng.choice(minggu_u, len(minggu_u), replace=True)])
        for n in nama:
            boot[n].append(_auc_cepat(y[idx], hasil[n]["oof"][idx]))
    out = {}
    for n in nama:
        b = np.array(boot[n])
        d = b - np.array(boot[acuan]) if acuan in boot else np.zeros_like(b)
        p = 1.0 if n == acuan else float(min(1.0, 2 * min(np.nanmean(d <= 0), np.nanmean(d >= 0))))
        out[n] = {"bawah": float(np.nanpercentile(b, 2.5)), "atas": float(np.nanpercentile(b, 97.5)), "p": p}
    return out


def bandingkan_metode(semua: dict | None, rasio: float = RASIO_BIAYA) -> dict | None:
    """Evaluasi tiap model dari prediksi maju-waktunya sendiri (baris dan minggu yang sama untuk semua model).
    Tiap model memakai batas Siaga dari metode biaya yang sama pada skornya sendiri, jadi presisi/recall/F1
    adil dibandingkan. SK 95% dan nilai-p selisih AUC (vs XGBoost) dari bootstrap berklaster per minggu."""
    from sklearn.metrics import roc_curve
    if semua is None:
        return None
    uji = uji_beda_auc(semua["model"])
    baris, roc, yl = [], [], None
    for nama in [m for m in METODE if m in semua["model"]]:
        h = semua["model"][nama]
        yl, s = h["y_oof"], h["oof"]
        pos = s >= h["ambang"][1]
        tp, fp = int((pos & (yl == 1)).sum()), int((pos & (yl == 0)).sum())
        fn, tn = int((~pos & (yl == 1)).sum()), int((~pos & (yl == 0)).sum())
        pres, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        baris.append({"metode": nama, "AUC": h["auc"], "bawah": uji[nama]["bawah"], "atas": uji[nama]["atas"],
                      "p": uji[nama]["p"], "Presisi": pres, "Recall": rec,
                      "F1": 2 * pres * rec / max(pres + rec, 1e-9), "TP": tp, "FP": fp, "FN": fn, "TN": tn})
        fpr, tpr, _ = roc_curve(yl, s)
        idx = np.unique(np.linspace(0, len(fpr) - 1, min(len(fpr), 120)).astype(int))
        roc.append(pd.DataFrame({"metode": nama, "fpr": fpr[idx], "tpr": tpr[idx]}))
    h0 = next(iter(semua["model"].values()))
    return {"tabel": pd.DataFrame(baris), "roc": pd.concat(roc, ignore_index=True),
            "dasar": float(yl.mean()), "n": int(len(yl)), "minggu_uji": h0["minggu_uji"],
            "belum": semua.get("belum", [])}


# Naikkan bila isi hasil model berubah: simpanan (cache) lama di server yang sedang berjalan otomatis diabaikan.
VERSI_MODEL = 17


def _pustaka_terpasang() -> tuple:
    return ()


def model_terpilih() -> str:
    return "XGBoost"


def semua_model(mtime: float, minggu: pd.Timestamp, k: int = K_MINGGU):
    return _semua_model(mtime, minggu, k, VERSI_MODEL)


@st.cache_data(show_spinner="Menghitung peringatan kebakaran untuk jangka ini (±15 detik)…")
def _semua_model(mtime: float, minggu: pd.Timestamp, k: int, versi: int):
    _, kab, hs, _ = muat_data(mtime)
    return latih_prediksi_model(kab, hs, label_hotspot(mtime, k), minggu, k=k)


def prediksi_risiko(mtime: float, minggu: pd.Timestamp, k: int = K_MINGGU, model: str | None = None):
    """Hasil satu model (pilihan pengguna di tab Level Peringatan) untuk peta, prioritas, detail, evaluasi."""
    semua = semua_model(mtime, minggu, k)
    if semua is None:
        return None
    m = model or model_terpilih()
    return semua["model"].get(m) or semua["model"].get("XGBoost")


prediksi_xgboost = prediksi_risiko   # nama lama


def masker_pulau(petak, pulau: str) -> np.ndarray:
    """True untuk kab/kota (kunci 'Provinsi|Kabupaten') yang berada di pulau terpilih."""
    petak = np.asarray(petak)
    if pulau == "Semua pulau":
        return np.ones(len(petak), bool)
    return np.array([pulau_dari(str(x).split("|")[0]) == pulau for x in petak], bool)


def hasil_pulau(h: dict | None, pulau: str, rentang: tuple | None = None) -> dict | None:
    """Hasil satu model, evaluasinya (prediksi uji maju-waktu, SHAP) dibatasi ke kab/kota di pulau terpilih.
    Model tetap dilatih se-Indonesia (lebih banyak contoh kebakaran); yang disaring hanya penilaiannya.
    None bila di pulau itu tidak ada cukup kejadian untuk dinilai."""
    from sklearn.metrics import roc_auc_score
    if h is None or (pulau == "Semua pulau" and rentang is None):
        return h
    m = masker_pulau(h["petak_oof"], pulau)
    if rentang is not None:   # hanya minggu uji di dalam rentang data yang dipilih
        mg = pd.to_datetime(h["minggu_oof"])
        m &= np.asarray((mg >= pd.Timestamp(rentang[0])) & (mg <= pd.Timestamp(rentang[1])))
    y = h["y_oof"][m]
    if m.sum() == 0 or y.sum() < 5 or (1 - y).sum() < 5:
        return None
    out = {**h, "oof": h["oof"][m], "y_oof": y, "minggu_oof": h["minggu_oof"][m], "petak_oof": h["petak_oof"][m],
           "n_uji": int(m.sum()), "minggu_uji": int(len(np.unique(h["minggu_oof"][m]))),
           "auc": float(roc_auc_score(y, h["oof"][m]))}
    if "p_oof" in h:
        out["p_oof"] = h["p_oof"][m]
        out["auc_kalman"] = float(roc_auc_score(y, np.nan_to_num(h["p_oof"][m])))
    ms = masker_pulau(h["shap"]["petak_id"], pulau)
    if rentang is not None and "minggu" in h["shap"]:   # SHAP dari kab/kota-minggu di dalam rentang terpilih
        mgs = pd.to_datetime(h["shap"]["minggu"])
        ms &= np.asarray((mgs >= pd.Timestamp(rentang[0])) & (mgs <= pd.Timestamp(rentang[1])))
    out["shap"] = h["shap"][ms]
    return out


def banding_metode(mtime: float, minggu: pd.Timestamp, k: int = K_MINGGU, pulau: str = "Semua pulau",
                   rentang: tuple | None = None):
    return _banding_metode(mtime, minggu, k, VERSI_MODEL, _pustaka_terpasang(), pulau, rentang)


@st.cache_data(show_spinner="Menilai model untuk wilayah ini…")
def _banding_metode(mtime: float, minggu: pd.Timestamp, k: int, versi: int, pustaka: tuple, pulau: str,
                    rentang: tuple | None = None):
    semua = semua_model(mtime, minggu, k)
    if semua is None:
        return None
    model = {n: hasil_pulau(h, pulau, rentang) for n, h in semua["model"].items()}
    if any(h is None for h in model.values()):
        return None
    return bandingkan_metode({**semua, "model": model})


@st.cache_resource(show_spinner=False)
def label_hotspot(mtime: float, k: int = K_MINGGU):
    _, kab, hs, _ = muat_data(mtime)
    if hs is None or hs.empty:
        return None
    return label_kebakaran_meluas(kab, hs, k)


@st.cache_data(show_spinner=False)
def ambang_minggu(mtime: float, minggu: pd.Timestamp):
    """Ambang dari kabupaten-minggu SEBELUM minggu terpilih (label t+1..t+k sudah teramati)."""
    _, kab, _, _ = muat_data(mtime)
    y = label_hotspot(mtime)
    if y is None:
        return AMBANG_SEMENTARA, "sementara"
    latih = (kab["minggu"] <= batas_latih(minggu, K_MINGGU)) & kab["p"].notna()
    if latih.sum() == 0 or y[latih].nunique() < 2:
        return AMBANG_SEMENTARA, "sementara"
    t = ambang_level(y[latih], kab.loc[latih, "p"], RASIO_BIAYA)
    return (t, "kalibrasi") if t else (AMBANG_SEMENTARA, "sementara")


# =============================================================================
# PETA
# =============================================================================
PILIHAN_PULAU = ["Semua pulau"] + list(PULAU_PROVINSI)


def saring_batas(batas: dict, pulau: str) -> dict:
    """Hanya fitur pulau terpilih: peta lebih ringan dan langsung fokus."""
    if pulau == "Semua pulau":
        return batas
    return {"type": "FeatureCollection",
            "features": [f for f in batas["features"]
                         if f["properties"].get("pulau", pulau_dari(f["properties"]["provinsi"])) == pulau]}


def view_untuk(batas: dict, lebar_px: int = 760, tinggi_px: int = 460) -> pdk.ViewState:
    """Pusat dan zoom peta dari kotak batas semua fitur (zoom web mercator, ubin 512 px)."""
    xy = np.array([c for f in batas["features"] for poli in _poligon(f["geometry"]) for c in poli[0]])
    lon0, lat0 = xy.min(0)
    lon1, lat1 = xy.max(0)
    z = min(np.log2(lebar_px * 360 / (512 * max(lon1 - lon0, 0.5))),
            np.log2(tinggi_px * 360 / (512 * max(lat1 - lat0, 0.5)))) - 0.15
    return pdk.ViewState(latitude=float((lat0 + lat1) / 2), longitude=float((lon0 + lon1) / 2),
                         zoom=float(z), pitch=0, min_zoom=float(z) - 0.8, max_zoom=11)


def _poligon(geo: dict) -> list:
    return [geo["coordinates"]] if geo["type"] == "Polygon" else geo["coordinates"]


@st.cache_resource(show_spinner=False)
def _provinsi_geojson(mtime: float):
    return json.loads(BERKAS_PROVINSI.read_text(encoding="utf-8")) if BERKAS_PROVINSI.exists() else None


def layer_provinsi(pulau: str) -> list:
    """Garis batas provinsi yang lebih tebal + label nama provinsi (dari data, bukan dari peta dasar)."""
    g = _provinsi_geojson(BERKAS_PROVINSI.stat().st_mtime if BERKAS_PROVINSI.exists() else 0.0)
    if g is None:
        return []
    semua = pulau == "Semua pulau"
    fitur = [f for f in g["features"] if semua or f["properties"]["pulau"] == pulau]
    if semua:   # tingkat nasional: cukup 7 nama pulau supaya label tidak menumpuk
        label = [{"nama": nama, "posisi": pos} for nama, pos in LABEL_PULAU.items()]
    else:       # satu pulau: nama provinsi di pusat poligon terbesarnya
        label = []
        for f in fitur:
            terbesar = max(_poligon(f["geometry"]), key=lambda poli: len(poli[0]))
            xy = np.array(terbesar[0])
            label.append({"nama": f["properties"]["provinsi"].replace("Daerah Istimewa ", "DI ").replace("Dki", "DKI"),
                          "posisi": [float(xy[:, 0].mean()), float(xy[:, 1].mean())]})
    return [
        pdk.Layer("GeoJsonLayer", {"type": "FeatureCollection", "features": fitur}, id="batas-provinsi",
                  filled=False, stroked=True, get_line_color=[235, 242, 250, 95 if semua else 150],
                  line_width_min_pixels=0.8 if semua else 1.3, pickable=False, parameters={"depthTest": False}),
        pdk.Layer("TextLayer", label, id="label-provinsi", get_position="posisi", get_text="nama",
                  # teks diapit backtick: tanpa itu pydeck menganggap "Arial" sebagai rumus dan label hilang
                  get_size=13 if semua else 11, get_color=[240, 245, 250, 235], font_family="`Arial, sans-serif`",
                  font_weight=700, outline_width=4, outline_color=[10, 22, 40, 230], font_settings={"sdf": True},
                  pickable=False, parameters={"depthTest": False}),
    ]


# Posisi label pulau pada tampilan seluruh Indonesia (sedikit di luar daratan supaya tidak menutupi warna).
LABEL_PULAU = {"Sumatera": [97.6, -2.6], "Jawa": [109.8, -8.9], "Bali-Nusa Tenggara": [119.5, -10.4],
               "Kalimantan": [113.0, 3.6], "Sulawesi": [121.2, 2.1], "Maluku": [129.6, 0.4], "Papua": [138.5, -1.2]}


def _di_dalam(x: np.ndarray, y: np.ndarray, cincin: np.ndarray) -> np.ndarray:
    """Uji titik-dalam-poligon (ray casting), vektor numpy."""
    dalam = np.zeros(len(x), dtype=bool)
    x0, y0 = cincin[:, 0], cincin[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    for a, b, c, d in zip(x0, y0, x1, y1):
        silang = (b > y) != (d > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xs = a + (y - b) * (c - a) / (d - b)
        dalam ^= silang & (x < xs)
    return dalam


@st.cache_data(show_spinner=False)
def pulau_titik_api(titik: pd.DataFrame, mtime: float) -> np.ndarray:
    """Pulau untuk tiap titik api: di dalam poligon provinsi mana; titik di garis pantai yang terpotong
    penyederhanaan batas (≤ ±10 km dari batas) ikut pulau terdekat; sisanya (laut lepas, negara lain) kosong."""
    g = _provinsi_geojson(mtime)
    x, y = titik["lon"].to_numpy(float), titik["lat"].to_numpy(float)
    hasil = np.full(len(titik), "", dtype=object)
    titik_batas, label_batas = [], []
    for f in g["features"]:
        pulau = f["properties"]["pulau"]
        for poli in _poligon(f["geometry"]):
            cincin = np.asarray(poli[0], float)
            titik_batas.append(cincin)
            label_batas += [pulau] * len(cincin)
            kotak_b = ((x >= cincin[:, 0].min()) & (x <= cincin[:, 0].max()) &
                       (y >= cincin[:, 1].min()) & (y <= cincin[:, 1].max()) & (hasil == ""))
            if kotak_b.any():
                idx = np.flatnonzero(kotak_b)
                hasil[idx[_di_dalam(x[idx], y[idx], cincin)]] = pulau
    sisa = np.flatnonzero(hasil == "")
    if len(sisa):
        from scipy.spatial import cKDTree
        jarak, j = cKDTree(np.vstack(titik_batas)).query(np.c_[x[sisa], y[sisa]])
        dekat = jarak < 0.1
        hasil[sisa[dekat]] = np.asarray(label_batas, dtype=object)[j[dekat]]
    return hasil


def saring_titik_api(titik: pd.DataFrame | None, pulau: str) -> pd.DataFrame | None:
    """Titik api di wilayah Indonesia yang ditampilkan saja (pulau terpilih, atau semua pulau)."""
    if titik is None or titik.empty or not BERKAS_PROVINSI.exists():
        return titik
    p = pulau_titik_api(titik, BERKAS_PROVINSI.stat().st_mtime)
    return titik[(p != "") if pulau == "Semua pulau" else (p == pulau)]


def layer_titik_api(titik: pd.DataFrame | None) -> list:
    if titik is None or titik.empty:
        return []
    return [pdk.Layer("ScatterplotLayer", titik, id="titik-api", get_position="[lon, lat]",
                      get_fill_color=[255, 236, 60, 165], stroked=False, get_radius=450, radius_units="meters",
                      radius_min_pixels=1, radius_max_pixels=2.2, pickable=False, parameters={"depthTest": False})]


def peta(layers: list, view: pdk.ViewState, tooltip_html: str | None, tinggi: int = 460, pulau: str = "Semua pulau",
         titik: pd.DataFrame | None = None, kunci: str | None = None):
    """Peta dengan batas dan label provinsi. kunci: bila diisi, kabupaten bisa diklik (pilihan dikembalikan)."""
    deck = pdk.Deck(
        layers=layers + layer_provinsi(pulau) + layer_titik_api(titik), initial_view_state=view, map_style=MAP_STYLE,
        tooltip={"html": tooltip_html, "style": {"backgroundColor": C["bg_soft"], "color": C["text"],
                                                  "border": f"1px solid {C['border']}", "borderRadius": "8px",
                                                  "fontSize": "12px"}} if tooltip_html else None)
    if kunci is None:
        st.pydeck_chart(deck, use_container_width=True, height=tinggi)
        return None
    return st.pydeck_chart(deck, use_container_width=True, height=tinggi, on_select="rerun",
                           selection_mode="single-object", key=kunci)


def layer_wilayah(batas: dict, nilai: pd.DataFrame, warna: dict, id_: str, garis: dict | None = None,
                  sorot: set | None = None) -> list:
    """Kabupaten diwarnai menurut nilainya. nilai: index kunci, kolom teks tooltip.
    warna/garis: {kunci: [r,g,b,a]}; sorot: kunci yang diberi garis tepi tebal (lapisan terpisah:
    tebal garis yang berbeda per fitur membuat sambungan garis meledak menjadi duri raksasa)."""
    fitur = []
    for f in batas["features"]:
        k = f["properties"]["kunci"]
        prop = {"kunci": k, "kabupaten": f["properties"]["kabupaten"], "wilayah": f["properties"]["kabupaten"]}
        prop.update({c: str(v) for c, v in (nilai.loc[k].items() if k in nilai.index else [])})
        prop.setdefault("p_txt", "—")
        prop["warna"] = warna.get(k, WARNA_KOSONG)
        prop["garis"] = (garis or {}).get(k, GARIS_KAB)
        fitur.append({"type": "Feature", "properties": prop, "geometry": f["geometry"]})
    lapis = [pdk.Layer("GeoJsonLayer", {"type": "FeatureCollection", "features": fitur}, id=id_,
                       filled=True, stroked=True, get_fill_color="properties.warna",
                       get_line_color="properties.garis", line_width_min_pixels=0.6,
                       pickable=True, auto_highlight=True, parameters={"depthTest": False})]
    if sorot:
        tepi = [f for f in fitur if f["properties"]["kunci"] in sorot]
        lapis.append(pdk.Layer("GeoJsonLayer", {"type": "FeatureCollection", "features": tepi}, id=id_ + "-sorot",
                               filled=False, stroked=True, get_line_color=[0, 212, 255, 255],
                               line_width_min_pixels=3, line_joint_rounded=True, pickable=False,
                               parameters={"depthTest": False}))
    return lapis


def legenda_level(judul: str, palet: dict, arti: dict):
    """Legenda rapi di bawah peta: 5 kolom sama lebar (4 level + bukan gambut), nama di atas, arti di bawah."""
    sel = [(palet[l], l, arti[l]) for l in LEVEL] + [("rgba(150,150,158,.25)", "Bukan gambut", "tidak dipantau")]
    isi = "".join(f'<div class="lg2"><i style="background:{w}"></i><div><b>{n}</b><span>{a}</span></div></div>'
                  for w, n, a in sel)
    st.markdown(f'<div class="legenda-grid">{isi}</div>', unsafe_allow_html=True)


# =============================================================================
# BARIS FILTER (di atas, menggantikan sidebar)
# =============================================================================

def project_id_aktif() -> str:
    """Project ID Earth Engine terkini: isian di tombol pengaturan (sesi ini), kalau tidak dari berkas .env
    (dibaca ulang, jadi ganti project tidak perlu restart dasbor)."""
    isian = (st.session_state.get("ee_id") or "").strip()
    if isian:
        return isian
    f = FOLDER / ".env"
    for b in (f.read_text(encoding="utf-8").splitlines() if f.exists() else []):
        if b.startswith("EE_PROJECT_ID="):
            return b.split("=", 1)[1].strip()
    return os.getenv("EE_PROJECT_ID", "")


def simpan_kunci_env(project_id: str):
    """Simpan Project ID ke .env supaya tidak perlu diisi lagi. Baris lain (mis. NASA_FIRMS_MAP_KEY untuk
    analisis esai) dibiarkan apa adanya."""
    f = FOLDER / ".env"
    baris = [b for b in (f.read_text(encoding="utf-8").splitlines() if f.exists() else [])
             if not b.startswith("EE_PROJECT_ID=")]
    baris.append(f"EE_PROJECT_ID={project_id}")
    f.write_text("\n".join(baris) + "\n", encoding="utf-8")
    os.environ["EE_PROJECT_ID"] = project_id


def senin_belum_ada(mulai: date, selesai: date, semua_minggu: list) -> list:
    """Senin di [mulai, selesai] (sampai data SMAP terbaru, ±5 hari lalu) yang belum ada di kumpulan data."""
    ada = {pd.Timestamp(m).date() for m in semua_minggu}
    akhir = min(selesai, date.today() - timedelta(days=5))
    pertama = mulai + timedelta(days=(7 - mulai.weekday()) % 7)
    return [pertama + timedelta(weeks=i) for i in range(max((akhir - pertama).days // 7 + 1, 0))
            if pertama + timedelta(weeks=i) not in ada]


def ringkas_senin(senin: list) -> str:
    """[Senin…] → '1 Des – 22 Des 2025, 5 Jan 2026' (minggu berurutan digabung jadi satu rentang)."""
    bagian, i = [], 0
    while i < len(senin):
        j = i
        while j + 1 < len(senin) and (senin[j + 1] - senin[j]).days == 7:
            j += 1
        a, b = senin[i], senin[j] + timedelta(days=6)
        bagian.append(tgl_id(a) if i == j else f"{tgl_id(a)[:-5] if a.year == b.year else tgl_id(a)} – {tgl_id(b)}")
        i = j + 1
    return ", ".join(bagian)


def bar_filter(pg: dict | None, semua_minggu: list | None) -> dict:
    """Baris atas: Wilayah · Dari · Sampai · Prediksi untuk · Data. Kondisi = Senin terakhir sampai tanggal
    "Sampai" (tidak ada pengaturan minggu terpisah). Tombol Data: ambil data dari satelit dan Project ID."""
    hari_ini = date.today()
    semua_minggu = semua_minggu or []
    if pg and "mulai" in pg:
        awal0, akhir0 = date.fromisoformat(pg["mulai"]), date.fromisoformat(pg["selesai"])
    else:
        awal0, akhir0 = date(hari_ini.year, 1, 1), hari_ini
    with st.container(border=True):
        k = st.columns([1.1, 0.95, 0.95, 1.15, 0.7], vertical_alignment="bottom")
        pulau = k[0].selectbox("Wilayah", PILIHAN_PULAU, key="pulau")
        mulai = k[1].date_input("Dari tanggal", value=awal0, min_value=date(2015, 5, 1), max_value=hari_ini,
                                format="DD/MM/YYYY", key="tgl_mulai",
                                help="Awal periode yang dilihat (grafik perkembangan dan penilaian model).")
        selesai = k[2].date_input("Sampai tanggal", value=akhir0, min_value=date(2015, 5, 1), max_value=hari_ini,
                                  format="DD/MM/YYYY", key="tgl_selesai",
                                  help="Peta, status peringatan, dan rencana patroli dihitung dari kondisi Senin terakhir "
                                       "sampai tanggal ini. Isi tanggal lampau untuk melihat peringatan pada masa itu.")
        k[3].selectbox("Prediksi untuk", JANGKA_PILIHAN, index=JANGKA_PILIHAN.index(K_MINGGU), key="jangka",
                       format_func=lambda j: f"{j} minggu ke depan",
                       help="Jangka pendek lebih akurat; jangka panjang memberi waktu persiapan lebih lama.")
        siap = selesai >= mulai
        hilang = senin_belum_ada(mulai, selesai, semua_minggu) if siap else []
        with k[4].popover("Data", icon=":material/database:", use_container_width=True):
            if not siap:
                st.caption("Tanggal akhir harus sesudah tanggal awal.")
            elif hilang:
                st.caption(f"Belum pernah diambil: **{ringkas_senin(hilang)}**. Ambil dari satelit (±"
                           f"{max(2, round(30 * 7 * len(hilang) / 365))} menit + ±11 menit olah).")
            else:
                st.caption("Semua tanggal di rentang ini sudah tersimpan.")
            tombol = st.button("Ambil data dari satelit", type="primary" if hilang or not semua_minggu else "secondary",
                               use_container_width=True, key="ambil", disabled=not siap)
            if semua_minggu:
                st.caption(f"Data tersimpan: {tgl_id(semua_minggu[0])} – "
                           f"{tgl_id(pd.Timestamp(pg.get('data_sampai', semua_minggu[-1])) if pg else semua_minggu[-1])}")
            project_id = st.text_input("Project ID Google Earth Engine", project_id_aktif(), key="ee_id",
                                       help="Project milik akun Google yang sedang login (code.earthengine.google.com).")
            if project_id.strip() and project_id.strip() != os.getenv("EE_PROJECT_ID", ""):
                simpan_kunci_env(project_id.strip())
        mg = [m for m in semua_minggu if mulai <= pd.Timestamp(m).date() <= selesai]
        minggu = pd.Timestamp(mg[-1]) if mg else None
        jangka = st.session_state.get("jangka") or K_MINGGU
        if minggu is not None:
            awal_p, akhir_p = minggu + pd.Timedelta(days=7), minggu + pd.Timedelta(days=7 * (jangka + 1) - 1)
            ket = (f"Kondisi: <b>Senin {tgl_id(minggu)}</b> · peringatan untuk <b>{tgl_id(awal_p)[:-5]} – "
                   f"{tgl_id(akhir_p)}</b>")
        else:
            ket = "Belum ada data tersimpan untuk rentang ini."
        if hilang and semua_minggu:
            ket += " · sebagian tanggal belum diambil, buka <b>Data</b> untuk mengambilnya"
        st.markdown(f'<div class="bar-info">{ket}</div>', unsafe_allow_html=True)
    return {"pulau": pulau, "minggu": minggu, "mulai": mulai, "selesai": selesai, "hilang": hilang,
            "permintaan": {"mulai": mulai, "selesai": selesai, "project_id": project_id.strip()} if tombol else None}


def jalankan_permintaan(r: dict):
    if not r["project_id"]:
        st.error("Isi Project ID Google Earth Engine di tombol Data terlebih dahulu.")
        return
    bar = st.progress(0.0, "Memulai…")
    catatan = st.status("Mengambil dan mengolah data satelit seluruh Indonesia…", expanded=False)

    def progres(f, pesan):
        bar.progress(min(max(float(f), 0.0), 1.0), f"{100 * f:.0f}% · {pesan}")
        catatan.write(pesan)

    try:
        hasil = proses_rentang(r["mulai"], r["selesai"], r["project_id"], progres=progres)
        st.session_state["gagal_proses"] = hasil["gagal"]
    except Exception as e:
        catatan.update(label="Gagal", state="error")
        if "restricted mode" in str(e).lower() or "too many requests" in str(e).lower():
            st.error("Earth Engine sedang membatasi project ini (Restricted Mode: hanya sedikit permintaan boleh "
                     "berjalan bersamaan). Bagian yang sudah terunduh tersimpan; tunggu 1–2 menit lalu tekan "
                     "**Ambil data** lagi untuk melanjutkan. Supaya batasnya hilang, daftarkan project untuk "
                     "penggunaan nonkomersial di https://code.earthengine.google.com/register.")
        else:
            st.error(f"Data belum berhasil diambil: {e}. Tekan tombol lagi untuk melanjutkan; "
                     "bagian yang sudah terunduh tidak diulang.")
        return
    simpan_kunci_env(r["project_id"])
    bar.empty()
    catatan.update(label="Selesai", state="complete")
    st.rerun()


# =============================================================================
# TAB 1 — PETA PELUANG KERING
# =============================================================================

def titik_api_minggu(minggu: pd.Timestamp) -> pd.DataFrame | None:
    """Titik api 7 hari sebelum Senin terpilih (yang sudah diketahui pada hari itu). Disimpan di hasil/.
    Kegagalan (mis. koneksi) TIDAK disimpan di cache, jadi dicoba lagi pada muat ulang berikutnya."""
    try:
        return _titik_api_minggu(minggu)
    except Exception:
        return None


@st.cache_data(show_spinner="Mengambil titik api 7 hari terakhir dari NASA FIRMS…")
def _titik_api_minggu(minggu: pd.Timestamp) -> pd.DataFrame:
    akhir = minggu.date()
    mulai = akhir - timedelta(days=7)
    f = FOLDER_HASIL / "mentah_smap" / f"titik_api_k{FIRMS_KEYAKINAN_MIN}_{mulai:%Y%m%d}.csv"
    if f.exists():
        return pd.read_csv(f)
    ee = ee_mulai(project_id_aktif(), autentikasi=False)
    d = ee_titik_api(ee, mulai, akhir)
    f.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(f, index=False)
    return d


def api_per_minggu(hs: pd.DataFrame | None, kunci: set) -> pd.Series:
    if hs is None or hs.empty:
        return pd.Series(dtype=float)
    h = hs[hs["petak_id"].isin(kunci)]
    return (h.groupby("minggu")["n"].sum() if "n" in h else h.groupby("minggu").size()).astype(float)


def grafik_waktu(seri: pd.DataFrame, minggu, judul_y: str, ambang_y: float | None = None, tinggi: int = 170):
    """Deret mingguan: garis (sumbu kiri) + batang titik api (sumbu kanan) + penanda minggu terpilih."""
    dasar = alt.Chart(seri).encode(x=alt.X("minggu:T", title=None, axis=alt.Axis(format="%d %b", labelAngle=0,
                                                                                   tickCount=8)))
    # Tiap batang tepat selebar satu minggu (Senin s.d. Minggu, dengan celah tipis) supaya tidak saling menimpa.
    seri = seri.assign(akhir=pd.to_datetime(seri["minggu"]) + pd.Timedelta(days=6))
    batang = alt.Chart(seri).mark_bar(color="#ffb800", opacity=0.55).encode(
        x=alt.X("minggu:T", title=None, axis=alt.Axis(format="%d %b", labelAngle=0, tickCount=8)), x2="akhir:T",
        y=alt.Y("api:Q", title="titik api / minggu", axis=alt.Axis(titleColor="#ffb800", orient="right", format="~s")),
        y2=alt.datum(0),
        tooltip=[alt.Tooltip("minggu:T", title="minggu", format="%d %b %Y"), alt.Tooltip("api:Q", title="titik api")])
    garis = dasar.mark_line(color=C["accent"], strokeWidth=2.4, interpolate="monotone").encode(
        y=alt.Y("nilai:Q", title=judul_y, axis=alt.Axis(titleColor=C["accent"])),
        tooltip=[alt.Tooltip("minggu:T", title="minggu", format="%d %b %Y"),
                 alt.Tooltip("nilai:Q", title=judul_y, format=".2f")])
    kini = alt.Chart(pd.DataFrame({"minggu": [pd.Timestamp(minggu)]})).mark_rule(
        color="#ffffff", strokeDash=[3, 3], opacity=0.7).encode(x="minggu:T")
    kiri = garis + kini
    if ambang_y is not None:
        kiri = kiri + alt.Chart(pd.DataFrame({"y": [ambang_y]})).mark_rule(color=C["danger"], strokeDash=[5, 4]).encode(
            y="y:Q")
    ch = alt.layer(batang, kiri).resolve_scale(y="independent").properties(height=tinggi)
    st.altair_chart(tema_altair(ch), use_container_width=True)


def detail_kabupaten(kunci: str, d: pd.DataFrame, kab_tampil: pd.DataFrame, hs, minggu, k: int):
    """Panel kanan tab Situasi setelah kab/kota diklik/dipilih: status + grafik muka air dan titik api."""
    b = d[d["petak_id"] == kunci].iloc[0]
    st.markdown(f'<div class="detail-judul">{b["kabupaten"]}<span>{b["provinsi"]}</span></div>',
                unsafe_allow_html=True)
    if pd.isna(b["p"]):
        st.caption("Tidak ada lahan gambut yang terpantau di kab/kota ini.")
        return
    aksen = {"Aman": "success", "Waspada": "warning", "Siaga": "orange", "Awas": "danger"}
    api = api_per_minggu(hs, {kunci})
    lv_api = b.get("level_api", "—")
    kpi_grid([
        {"label": "Peringatan kebakaran", "value": lv_api, "accent": aksen.get(lv_api, ""),
         "hint": f"{k} minggu ke depan" + (f" · {ARTI_API[lv_api]}" if lv_api in ARTI_API else "")},
        {"label": "Kekeringan sekarang", "value": b["level"], "accent": aksen.get(b["level"], ""),
         "hint": f"muka air {fmt(abs(b['muka_air_m']), 2)} m di bawah permukaan" if pd.notna(b.get("muka_air_m")) else ""},
        {"label": "Luas gambut", "value": luas_txt(float(luas_gambut(d[d["petak_id"] == kunci]).iloc[0]))},
        {"label": "Titik api minggu lalu", "value": fmt(api.get(minggu - pd.Timedelta(days=7), 0))},
    ], kolom=2)
    deret = kab_tampil[kab_tampil["petak_id"] == kunci][["minggu", "muka_air_m"]].rename(columns={"muka_air_m": "nilai"})
    deret["api"] = deret["minggu"].map(api).fillna(0)
    st.markdown('<span class="sub">Garis = kedalaman muka air gambut (m); garis merah = batas aman 0,4 m di bawah '
                'permukaan. Batang = titik api per minggu.</span>', unsafe_allow_html=True)
    grafik_waktu(deret, minggu, "muka air (m)", ambang_y=AMBANG_MUKA_AIR_M, tinggi=135)


def kotak(kunci: str, isi: bool = False):
    """Kotak bertepi untuk satu bahasan. Bawaannya memanjang mengikuti kolom tertinggi di baris yang sama,
    supaya tidak ada ruang kosong; isi=True untuk kotak yang tingginya mengikuti isinya saja."""
    return st.container(border=True, height="content" if isi else "stretch", key=kunci)


def judul_kotak(judul: str, sub: str | None = None):
    st.markdown(f'<div class="kotak-judul">{judul}{f"<span>{sub}</span>" if sub else ""}</div>',
                unsafe_allow_html=True)


PILIH_KOSONG = "Pilih kab/kota untuk melihat detail"


def _tutup_detail():
    st.session_state["pilih_kab"] = PILIH_KOSONG


def tab_situasi(batas: dict, d1: pd.DataFrame, kab_tampil: pd.DataFrame, hs, minggu, ambang, mtime: float, view,
                pulau: str):
    """Tab Situasi Terbaru: kondisi kekeringan gambut minggu terpilih (pengamatan), peta kekeringan, detail
    kab/kota, dan perkembangan kekeringan serta titik api dari minggu ke minggu."""
    k = st.session_state.get("jangka") or K_MINGGU
    h = prediksi_risiko(mtime, minggu, k)
    d = d1.assign(level=beri_level_aman(d1["p"], ambang))
    if h is not None:   # dipakai panel detail kab/kota
        d = d.merge(h["pred"][["petak_id", "prob"]], on="petak_id", how="left")
        d["level_api"] = beri_level_aman(d["prob"], h["ambang"])
    else:
        d["prob"], d["level_api"] = np.nan, "—"
    g = d[d["p"].notna()]
    wil = "seluruh Indonesia" if pulau == "Semua pulau" else pulau

    # Kalimat ringkasan + 4 angka (semuanya pengamatan, bukan prediksi)
    n_awas, n_siaga = int((g["level"] == "Awas").sum()), int((g["level"] == "Siaga").sum())
    if n_awas + n_siaga == 0:
        kalimat = f"Senin <b>{tgl_id(minggu)}</b>: tidak ada kab/kota bergambut di {wil} yang kekeringannya Siaga atau Awas."
    else:
        terbanyak = g[g["level"].isin(["Awas", "Siaga"])]["provinsi"].value_counts().index[0]
        kalimat = (f"Senin <b>{tgl_id(minggu)}</b>: {n_awas} kab/kota bergambut di {wil} kekeringannya <b>Awas</b> dan "
                   f"{n_siaga} <b>Siaga</b>, terbanyak di {terbanyak}.")
    # Pembanding minggu sebelumnya (pengamatan): titik api di gambut dan luas gambut kering
    kk = set(d1["petak_id"])
    m1, m2 = minggu - pd.Timedelta(days=7), minggu - pd.Timedelta(days=14)
    api_s = api_per_minggu(hs, kk)
    api_lalu, api_sblm = float(api_s.get(m1, 0)), float(api_s.get(m2, 0))
    lalu = kab_tampil[(kab_tampil["minggu"] == m1) & kab_tampil["petak_id"].isin(kk) & kab_tampil["p"].notna()]
    kering, kering_lalu = float((g["p"] * luas_gambut(g)).sum()), float((lalu["p"] * luas_gambut(lalu)).sum())

    def ubah(a: float, b: float, satuan: str = "") -> str:
        if not b:
            return ""
        r = (a - b) / b
        arah = "▲" if r > 0.005 else ("▼" if r < -0.005 else "■")
        return f"{arah} {fmt(abs(100 * r))}% vs minggu lalu" + satuan

    # Kalimat tren dari data pengamatan: arah titik api dan provinsi dengan kenaikan terbesar
    tren = ""
    if hs is not None and api_sblm:
        h2 = hs[hs["petak_id"].isin(kk) & hs["minggu"].isin([m1, m2])].assign(
            prov=lambda x: x["petak_id"].str.split("|").str[0])
        per = h2.pivot_table(index="prov", columns="minggu", values="n", aggfunc="sum").reindex(columns=[m2, m1]).fillna(0)
        naik = (per[m1] - per[m2]).sort_values(ascending=False)
        tren = (f"{naik.index[0]} (+{fmt(naik.iloc[0])} titik)" if len(naik) and naik.iloc[0] > 0 else "")
    info_strip(kalimat + (f" Kenaikan titik api terbesar minggu lalu: {tren}." if tren else ""),
               "warn" if n_awas else "")
    kpi_grid([
        {"label": "Kekeringan · Awas", "value": f"{n_awas} kab/kota", "accent": "danger",
         "hint": ARTI_LEVEL["Awas"]},
        {"label": "Kekeringan · Siaga", "value": f"{n_siaga} kab/kota", "accent": "orange",
         "hint": ARTI_LEVEL["Siaga"]},
        {"label": "Pengamatan · Gambut kering", "value": luas_txt(kering) if len(g) else "—", "accent": "warning",
         "hint": f"Senin {tgl_id(minggu)[:-5]} · {ubah(kering, kering_lalu)}"},
        {"label": "Pengamatan · Titik api di gambut", "value": f"{fmt(api_lalu)} titik",
         "hint": f"{tgl_id(m1)[:-5]}–{tgl_id(minggu - pd.Timedelta(days=1))[:-5]} · {ubah(api_lalu, api_sblm)}"
},
    ], kolom=4)

    # Pilihan kab/kota: dari klik peta atau kotak pilihan
    urut = g.sort_values(["p", "prob"], ascending=False)
    label = {f"{r.kabupaten} ({r.provinsi})": r.petak_id for r in urut.itertuples()}
    pilihan = [PILIH_KOSONG] + list(label)
    klik = ((st.session_state.get("peta_utama") or {}).get("selection") or {}).get("objects", {}).get("wilayah-utama") or []
    if klik:
        obj = klik[0].get("properties", klik[0])
        nama = next((n for n, kk in label.items() if kk == obj.get("kunci")), None)
        if nama and obj.get("kunci") != st.session_state.get("_klik_terakhir"):
            st.session_state["pilih_kab"] = nama
        st.session_state["_klik_terakhir"] = obj.get("kunci")
    if st.session_state.get("pilih_kab") not in pilihan:
        st.session_state["pilih_kab"] = PILIH_KOSONG

    c1, c2 = st.columns([2.2, 1])
    with c2:
        kanan = kotak("kotak_kanan")
        with kanan:
            dipilih = st.selectbox("Lihat detail kab/kota", pilihan, key="pilih_kab", label_visibility="collapsed",
                                   help="Atau klik kab/kota di peta.")
    kunci_pilih = label.get(dipilih)
    with c1, kotak("kotak_peta", isi=True):
        kj, kt = st.columns([2.3, 1], vertical_alignment="center")
        tampil_api = kt.toggle("Titik api 7 hari terakhir", key="tampil_api")
        with kj:
            judul_kotak(f"Kekeringan gambut (Senin, {tgl_id(minggu)})", "seberapa kering lahan gambut saat ini")
        nilai = d.set_index("petak_id").assign(wilayah=d["kabupaten"].to_numpy(),
                                               p_txt=[persen([x])[0] for x in d["p"]])[["wilayah", "level", "p_txt"]]
        warna = {kk: hex_rgb(WARNA_LEVEL[l], 225) for kk, l in zip(d["petak_id"], d["level"]) if l in WARNA_LEVEL}
        titik = saring_titik_api(titik_api_minggu(minggu), pulau) if tampil_api else None
        peta(layer_wilayah(batas, nilai, warna, "wilayah-utama", sorot={kunci_pilih} if kunci_pilih else None), view,
             "<b>{wilayah}</b><br/>Kekeringan: {level}<br/>Peluang gambut kering: {p_txt}"
             "<br/><i>klik untuk detail</i>", tinggi=320, pulau=pulau, titik=titik, kunci="peta_utama")
        legenda_level("Kekeringan", WARNA_LEVEL, ARTI_LEVEL)
        if tampil_api and titik is None:
            st.caption("Titik api belum bisa diambil (periksa koneksi internet atau Project ID di tombol Data).")
        elif tampil_api:
            st.caption(f"Titik kuning = {fmt(len(titik))} titik api NASA FIRMS (pengamatan) "
                       f"{tgl_id(minggu - pd.Timedelta(days=7))[:-5]}–{tgl_id(minggu - pd.Timedelta(days=1))}.")
    with c2, kanan:
        if kunci_pilih:
            detail_kabupaten(kunci_pilih, d, kab_tampil, hs, minggu, k)
            st.button("Kembali ke daftar", on_click=_tutup_detail, use_container_width=True)
        else:
            daftar = urut[urut["level"].isin(["Awas", "Siaga"])]
            judul_kotak("Paling kering", "kab/kota dengan kekeringan Awas atau Siaga · urut dari peluang kering tertinggi")
            if daftar.empty:
                st.caption("Tidak ada kab/kota dengan kekeringan Awas atau Siaga.")
            else:
                tabel = pd.DataFrame({"Kab/kota": daftar["kabupaten"], "Provinsi": daftar["provinsi"],
                                      "Kekeringan": daftar["level"], "Peluang kering": persen(daftar["p"])})
                st.dataframe(tabel, use_container_width=True, hide_index=True, height="stretch",
                             column_config={"Kab/kota": st.column_config.TextColumn(width=110),
                                            "Provinsi": st.column_config.TextColumn(width=110),
                                            "Kekeringan": st.column_config.TextColumn(width=75),
                                            "Peluang kering": st.column_config.TextColumn(width=70)})
                tombol_unduh(tabel, f"sparks_paling_kering_{minggu:%Y%m%d}.csv", "Unduh daftar (CSV)")
    grafik_perkembangan(d1, kab_tampil, hs, minggu, pulau)


def grafik_perkembangan(d1: pd.DataFrame, kab_tampil: pd.DataFrame, hs, minggu, pulau: str):
    """Kekeringan dan titik api dari minggu ke minggu, untuk wilayah terpilih (bagian bawah tab Situasi Terbaru)."""
    kk = set(d1["petak_id"])
    dk = kab_tampil[kab_tampil["petak_id"].isin(kk) & kab_tampil["p"].notna()]
    wilayah = "seluruh Indonesia" if pulau == "Semua pulau" else pulau
    with kotak("kotak_tren", isi=True):
        judul_kotak(f"Kekeringan gambut dan titik api ({wilayah})",
                    "garis = rata-rata peluang lahan gambut kering (%) · batang = titik api di lahan gambut · "
                    "garis putus-putus = minggu terpilih")
        if dk.empty:
            st.caption("Belum ada data.")
        else:
            seri = dk.groupby("minggu").apply(lambda x: np.average(x["p"], weights=luas_gambut(x)) * 100,
                                              include_groups=False).rename("nilai").reset_index()
            seri["api"] = seri["minggu"].map(api_per_minggu(hs, kk)).fillna(0)
            grafik_waktu(seri, minggu, "peluang kering (%)", tinggi=150)


# =============================================================================
# TAB 2 — LEVEL PERINGATAN: KONDISI SEKARANG (KALMAN) + PREDIKSI 14 HARI (XGBOOST)
# =============================================================================

def beri_level_aman(p, ambang) -> np.ndarray:
    lv = beri_level(np.nan_to_num(np.asarray(p, float), nan=0.0), ambang).astype(object)
    lv[pd.isna(p)] = "Bukan gambut"
    return lv


WARNA_METODE = {"XGBoost": "#22c55e",
                "Peluang kering saja": "#f1e05a"}


def rentang_terpilih() -> tuple | None:
    a, b = st.session_state.get("tgl_mulai"), st.session_state.get("tgl_selesai")
    return (a, b) if a and b else None


def _model_tersedia(mtime: float, minggu, pulau: str = "Semua pulau"):
    k = st.session_state.get("jangka") or K_MINGGU
    h = prediksi_xgboost(mtime, minggu, k)
    if h is None:
        info_strip("Riwayat kebakaran sebelum minggu ini belum cukup untuk melatih model. Pilih rentang tanggal yang "
                   "lebih panjang di tombol Data.", "warn")
        return k, None
    hp = hasil_pulau(h, pulau, rentang_terpilih())
    if hp is None:
        info_strip("Di wilayah dan rentang data ini terlalu sedikit kejadian kebakaran besar untuk menilai model. "
                   "Pilih wilayah lain atau perpanjang rentang data di tombol Data.", "warn")
    return k, hp


def _nama_wilayah(pulau: str) -> str:
    return "seluruh Indonesia" if pulau == "Semua pulau" else pulau


def tab_model(mtime: float, minggu, pulau: str = "Semua pulau"):
    """Untuk analis/juri: cara kerja, akurasi, prediksi vs kenyataan, faktor pendorong. Model yang dipilih di sini
    dipakai di seluruh dasbor."""
    kt, _ = st.columns([1.1, 2.9], vertical_alignment="bottom")
    tampilan = kt.selectbox("Tampilan", ["Cara kerja", "Hasil uji model"], key="tampilan_model")
    if tampilan == "Cara kerja":
        cara_kerja()
        return
    k, h = _model_tersedia(mtime, minggu, pulau)
    if h is None:
        return
    wil = _nama_wilayah(pulau)
    r = rentang_terpilih()
    if r:
        wil += f" ({tgl_id(r[0])} – {tgl_id(r[1])})"
    # Hanya tiga grafik inti: seberapa tepat peringatan, prediksi vs kenyataan, dan faktor pendorongnya
    banding = banding_metode(mtime, minggu, k, pulau, rentang_terpilih())
    d = pd.DataFrame({"minggu": pd.to_datetime(h["minggu_oof"]), "prob": h["oof"], "aktual": h["y_oof"],
                      "petak_id": h["petak_oof"]})
    st.markdown(f'<div class="ket-uji">Hasil uji model peringatan {k} minggu ke depan untuk <b>{wil}</b>, diuji pada '
                'minggu-minggu yang belum pernah dilihat model.</div>', unsafe_allow_html=True)
    c4, c5 = st.columns([1, 1.5])
    with c4, kotak("kotak_konfusi"):
        r = banding["tabel"].iloc[0] if banding is not None else None
        judul_kotak("Ketepatan peringatan",
                    "Siaga/Awas dibanding kejadian sebenarnya" + (f" · AUC {fmt(r['AUC'], 2)}" if r is not None else ""))
        grafik_konfusi(banding)
    with c5, kotak("kotak_pva_waktu"):
        judul_kotak("Prediksi vs kenyataan per minggu", "batang = kab/kota yang benar-benar terbakar · garis = prediksi model")
        grafik_pva_waktu(d, k, h["nama"])
    with kotak("kotak_shap", isi=True):
        judul_kotak("Faktor pendorong risiko kebakaran", "ke kanan = menaikkan risiko · penjelasan model, bukan bukti sebab-akibat")
        grafik_shap(h)


def cara_kerja():
    """Penjelasan runtut cara SPARKS bekerja, dalam satu tempat (bukan tersebar di tiap grafik)."""
    langkah = [
        ("1. Data satelit", "NASA SMAP: muka air tanah gambut, hujan, penguapan, dan suhu setiap hari (9 km). NASA "
         "FIRMS: titik api. Semua diambil lewat Google Earth Engine."),
        ("2. Kekeringan gambut", "Muka air SMAP dihaluskan dengan filter Kalman. <b>Peluang kering</b> = peluang muka "
         "air turun lebih dari 40 cm di bawah permukaan, batas kerusakan gambut menurut PP 57/2016."),
        ("3. Peringatan kebakaran", "Model machine learning XGBoost memakai 9 variabel "
         "untuk menebak apakah akan terjadi <b>kebakaran besar</b> (≥ 1 titik api per 10.000 ha gambut) dalam 1–4 "
         "minggu ke depan. Status Aman/Waspada/Siaga/Awas ditentukan dari riwayat kebakaran."),
        ("4. Rencana patroli", "Dengan biaya dan anggaran yang diisi, sistem memilih kab/kota yang menjangkau luas "
         "gambut berisiko terbesar (optimasi knapsack)."),
    ]
    kol = st.columns(4)
    for c, (judul, isi) in zip(kol, langkah):
        with c, kotak(f"kotak_cara_{judul[0]}"):
            judul_kotak(judul)
            st.markdown(f'<div class="sub" style="font-size:.85rem;line-height:1.5;padding-bottom:10px">{isi}</div>',
                        unsafe_allow_html=True)
    variabel = [
        ("Peluang kering", "peluang muka air gambut lebih dari 40 cm di bawah permukaan (0–100%)", "SMAP + filter Kalman"),
        ("Muka air gambut", "kedalaman muka air tanah gambut minggu ini (meter)", "SMAP + filter Kalman"),
        ("Perubahan muka air 1 minggu", "selisih muka air dibanding minggu lalu (makin turun = makin kering)", "SMAP"),
        ("Perubahan muka air 2 minggu", "selisih muka air dibanding 2 minggu lalu", "SMAP"),
        ("Hujan 30 hari", "total curah hujan 30 hari terakhir di kab/kota (mm)", "SMAP"),
        ("Drought Code", "indeks kekeringan lapisan tanah dalam, dari hujan dan suhu (makin besar makin kering)",
         "dihitung dari SMAP"),
        ("Titik api minggu lalu", "jumlah titik api di lahan gambut per 10.000 ha gambut, minggu lalu", "NASA FIRMS"),
        ("Titik api 2 minggu lalu", "jumlah titik api di lahan gambut per 10.000 ha gambut, 2 minggu lalu", "NASA FIRMS"),
        ("Waktu dalam tahun", "nomor minggu dalam setahun (1–53), untuk menangkap pola musim kemarau", "tanggal"),
    ]
    with kotak("kotak_variabel", isi=True):
        judul_kotak("Variabel yang dipakai model peringatan (9 variabel, semuanya angka)",
                    "satu baris data = satu kab/kota pada satu minggu · yang ditebak: apakah terjadi kebakaran besar "
                    "(≥ 1 titik api per 10.000 ha gambut) dalam 1–4 minggu ke depan (ya/tidak)")
        sel = "".join(f'<div class="var"><b>{n}</b><span>{a}</span><i>{sb}</i></div>' for n, a, sb in variabel)
        st.markdown(f'<div class="var-grid">{sel}</div>', unsafe_allow_html=True)
    with kotak("kotak_batasan", isi=True):
        judul_kotak("Batasan yang perlu diketahui")
        st.markdown('<div class="sub" style="font-size:.85rem;line-height:1.6;padding-bottom:10px">'
                    "• Satu piksel SMAP mewakili ±8.100 ha, jadi perbedaan di dalam satu kab/kota tidak terlihat.<br>"
                    "• Muka air SMAP adalah hasil model, belum dikalibrasi dengan pengukuran lapangan (SIPALAGA).<br>"
                    "• Peringatan berlaku untuk satu periode (mis. 14 hari), bukan per hari.<br>"
                    "• Model diuji pada minggu-minggu yang belum pernah dilihatnya: sekitar 4 dari 10 kebakaran besar "
                    "sudah diberi Siaga/Awas lebih dulu (lihat Hasil uji model).</div>", unsafe_allow_html=True)


def grafik_pva_waktu(d: pd.DataFrame, k: int, nama: str):
    seri = d.groupby("minggu").agg(Diprediksi=("prob", "sum"), Aktual=("aktual", "sum")).reset_index()
    seri["periode"] = [f"{tgl_id(m + pd.Timedelta(days=7))[:-5]} – {tgl_id(m + pd.Timedelta(days=7 * (k + 1) - 1))}"
                       for m in seri["minggu"]]
    x = alt.X("minggu:T", title="minggu kondisi (Senin)", axis=alt.Axis(format="%b %y", labelAngle=0, tickCount=8))
    tip = [alt.Tooltip("minggu:T", title="kondisi Senin", format="%d %b %Y"),
           alt.Tooltip("periode:N", title="periode diprediksi"), alt.Tooltip("Aktual:Q", title="terjadi (kab/kota)"),
           alt.Tooltip("Diprediksi:Q", title="diprediksi", format=".1f")]
    batang = alt.Chart(seri).mark_bar(color="#ff7a00", opacity=0.55).encode(
        x=x, y=alt.Y("Aktual:Q", title="kab/kota dengan kebakaran meluas"), tooltip=tip)
    garis = alt.Chart(seri).mark_line(color=WARNA_METODE.get(nama, C["accent"]), strokeWidth=2.4).encode(
        x=x, y="Diprediksi:Q", tooltip=tip)
    st.altair_chart(tema_altair((batang + garis).properties(height=300)), use_container_width=True)
    r = float(np.corrcoef(seri["Diprediksi"], seri["Aktual"])[0, 1]) if len(seri) > 2 and seri["Aktual"].std() > 0 else np.nan
    mae = float((seri["Diprediksi"] - seri["Aktual"]).abs().mean())
    st.markdown(f'<span class="sub">{fmt(len(seri))} minggu diuji · korelasi prediksi–aktual r = {fmt(r, 2)} · '
                f'rata-rata selisih {fmt(mae, 1)} kab/kota per minggu · total diprediksi '
                f'{fmt(seri["Diprediksi"].sum())} vs terjadi {fmt(seri["Aktual"].sum())} kab/kota-minggu.</span>',
                unsafe_allow_html=True)


def grafik_konfusi(b: dict | None):
    if b is None:
        st.caption("Belum bisa dihitung.")
        return
    t_ = b["tabel"].set_index("metode")
    r = t_.loc[model_terpilih() if model_terpilih() in t_.index else t_.index[0]]
    sel = pd.DataFrame([
        {"aktual": "Terjadi kebakaran", "prediksi": "Siaga/Awas", "n": r.TP, "kode": "TP · tertangkap"},
        {"aktual": "Terjadi kebakaran", "prediksi": "Aman/Waspada", "n": r.FN, "kode": "FN · terlewat"},
        {"aktual": "Tidak terjadi", "prediksi": "Siaga/Awas", "n": r.FP, "kode": "FP · peringatan palsu"},
        {"aktual": "Tidak terjadi", "prediksi": "Aman/Waspada", "n": r.TN, "kode": "TN · benar aman"}])
    sel["porsi"] = sel["n"] / sel.groupby("aktual")["n"].transform("sum")
    sel["teks"] = [f"{fmt(n)} ({fmt(100 * q)}%)" for n, q in zip(sel["n"], sel["porsi"])]
    sel["benar"] = sel["kode"].str.startswith(("TP", "TN"))
    x = alt.X("prediksi:N", sort=["Siaga/Awas", "Aman/Waspada"], title=None,
              axis=alt.Axis(labelAngle=0, orient="top", labelExpr="'Diprediksi ' + datum.value"))
    y = alt.Y("aktual:N", sort=["Terjadi kebakaran", "Tidak terjadi"], title=None, axis=alt.Axis(labelLimit=120))
    kotak_ = alt.Chart(sel).mark_rect(cornerRadius=6, stroke=C["bg"], strokeWidth=3).encode(
        x=x, y=y, color=alt.Color("porsi:Q", legend=None,
                                  scale=alt.Scale(domain=[0, 1], range=[C["bg_soft"], "#1f7a4d"])),
        tooltip=["aktual:N", "prediksi:N", alt.Tooltip("n:Q", title="kab/kota-minggu"),
                 alt.Tooltip("porsi:Q", title="bagian dari baris", format=".0%")])
    teks = alt.Chart(sel).mark_text(fontSize=15, fontWeight="bold", dy=-8, color=C["text"]).encode(
        x=x, y=y, text="teks:N")
    kode = alt.Chart(sel).mark_text(fontSize=10, dy=12, color=C["text_muted"]).encode(x=x, y=y, text="kode:N")
    st.altair_chart(tema_altair((kotak_ + teks + kode).properties(height=165)), use_container_width=True)
    kpi_grid([
        {"label": "Presisi", "value": f"{fmt(100 * r.Presisi)}%", "accent": "success",
         "hint": "Siaga/Awas yang benar terbakar"},
        {"label": "Recall", "value": f"{fmt(100 * r.Recall)}%", "accent": "success",
         "hint": "kebakaran yang tertangkap"},
        {"label": "F1", "value": fmt(r.F1, 2), "hint": f"tebak acak ≈ {fmt(2 * b['dasar'] / (1 + b['dasar']), 2)}"},
    ], kolom=3)
    st.markdown(f'<span class="sub">Uji maju-waktu: {fmt(b["n"])} kab/kota-minggu dari {b.get("minggu_uji", "—")} '
                f'minggu; kebakaran meluas di lahan gambut pada {fmt(100 * b["dasar"])}%. Persen = bagian dari '
                'baris.</span>', unsafe_allow_html=True)


def grafik_shap(h: dict):
    s = h["shap"]
    urut = list(h["penting_shap"].index)
    contoh = s.groupby("fitur").sample(n=min(400, int(s["fitur"].value_counts().min())), random_state=0)
    contoh = contoh.assign(geser=np.random.default_rng(0).uniform(-1, 1, len(contoh)))
    titik = alt.Chart(contoh).mark_circle(size=11, opacity=0.75).encode(
        x=alt.X("shap:Q", title="kontribusi pada skor risiko (log-odds) · kanan = menaikkan risiko"),
        y=alt.Y("fitur:N", sort=urut, title=None, axis=alt.Axis(labelLimit=180)),
        yOffset=alt.YOffset("geser:Q", scale=alt.Scale(domain=[-3, 3])),
        color=alt.Color("nilai:Q", title=["nilai", "faktor"],
                        scale=alt.Scale(domain=[0, 1], range=["#2c7bb6", "#e2e8f0", "#d7191c"]),
                        legend=alt.Legend(orient="right", format="%", values=[0, 0.5, 1], gradientLength=110,
                                          labelExpr="datum.value == 0 ? 'rendah' : datum.value == 1 ? 'tinggi' : ''")),
        tooltip=["fitur:N", alt.Tooltip("shap:Q", title="kontribusi", format=".2f"),
                 alt.Tooltip("nilai:Q", title="peringkat nilai", format=".0%")])
    nol = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=C["text_dim"]).encode(x="x:Q")
    st.altair_chart(tema_altair((nol + titik).properties(height=395)), use_container_width=True)
    st.markdown('<span class="sub">Urut dari faktor paling berpengaruh (rata-rata |SHAP|). Contoh: titik merah di '
                'kanan pada "titik api minggu lalu" = banyak titik api minggu lalu menaikkan risiko.</span>',
                unsafe_allow_html=True)


# =============================================================================
# TAB 3 — SIMULASI ANGGARAN PATROLI
# =============================================================================

def luas_gambut(d: pd.DataFrame) -> pd.Series:
    """Luas gambut kabupaten (ha) = jumlah piksel gambut SMAP × luas piksel, paling besar luas kabupaten.
    Data lama (titik sampel Sentinel-1) tidak punya ukuran ini, jadi memakai luas kabupaten."""
    if "muka_air_m" not in d or "n_titik" not in d:
        return d["luas_ha"]
    return np.minimum(d["n_titik"] * LUAS_PIKSEL_SMAP_HA, d["luas_ha"])


def luas_txt(ha: float) -> str:
    if ha >= 1e6:
        return f"{fmt(ha / 1e6, 1)} juta ha"
    return f"{fmt(ha / 1e3, 1)} ribu ha" if ha >= 1e3 else f"{fmt(ha)} ha"


def _rapikan_rupiah(kunci: str):
    """Isian rupiah ditulis ulang dengan pemisah ribuan titik (10000000 → 10.000.000)."""
    import re
    angka = re.sub(r"\D", "", st.session_state.get(kunci) or "")
    st.session_state[kunci] = f"{int(angka):,}".replace(",", ".") if angka else ""


def isian_rupiah(kol, label: str, kunci: str, contoh: str, bantuan: str | None = None) -> int:
    import re
    kol.text_input(label, key=kunci, placeholder=contoh, on_change=_rapikan_rupiah, args=(kunci,), help=bantuan)
    return int(re.sub(r"\D", "", st.session_state.get(kunci) or "") or 0)


_JS_RUPIAH = """
<script>
// Titik ribuan langsung saat mengetik pada isian rupiah (penanda: placeholder diawali "mis. ").
const doc = window.parent.document;
const setter = Object.getOwnPropertyDescriptor(window.parent.HTMLInputElement.prototype, "value").set;
function rapikan(e) {
  const el = e.target;
  const pos = el.selectionStart ?? el.value.length;
  const digitSebelum = el.value.slice(0, pos).replace(/\\D/g, "").length;
  const angka = el.value.replace(/\\D/g, "").replace(/^0+(?=\\d)/, "");
  const baru = angka.replace(/\\B(?=(\\d{3})+(?!\\d))/g, ".");
  if (baru === el.value) return;
  setter.call(el, baru);
  el.dispatchEvent(new Event("input", { bubbles: true }));   // samakan nilai yang dipegang React/Streamlit
  let i = 0, d = 0;
  while (i < baru.length && d < digitSebelum) { if (/\\d/.test(baru[i])) d++; i++; }
  el.setSelectionRange(i, i);
}
function pasang() {
  doc.querySelectorAll('input[placeholder^="mis. "]').forEach(el => {
    if (el.dataset.rupiah) return;
    el.dataset.rupiah = "1";
    el.addEventListener("input", rapikan);
  });
}
pasang();
new MutationObserver(pasang).observe(doc.body, { childList: true, subtree: true });
</script>
"""


def pasang_format_rupiah():
    import streamlit.components.v1 as components
    components.html(_JS_RUPIAH, height=0)


LEVEL_PATROLI = ["Awas", "Siaga", "Waspada"]   # status Aman tidak masuk daftar patroli


def tab_anggaran(batas: dict, d1: pd.DataFrame, minggu, view, pulau: str = "Semua pulau", mtime: float = 0.0):
    """Peringatan kebakaran dan rencana patroli. Pengguna mengisi total anggaran per status (Awas, Siaga, Waspada);
    anggaran tiap status dibagi ke kab/kota berstatus sama sebanding luas gambutnya, jadi jumlah semua bagian =
    anggaran minggu ini. Urutan prioritas = luas gambut berisiko terbakar (skor peringatan × luas gambut); kab/kota
    Aman tidak dipatroli khusus."""
    k = st.session_state.get("jangka") or K_MINGGU
    h = prediksi_risiko(mtime, minggu, k) if mtime else None
    semua = d1[d1["p"].notna()].copy()
    if h is not None:
        semua = semua.merge(h["pred"][["petak_id", "prob"]], on="petak_id", how="left")
        semua["skor"] = semua["prob"].fillna(0.0)
        semua["level_api"] = beri_level_aman(semua["skor"], h["ambang"])
    else:   # model belum bisa dilatih: pakai peluang kering dan level kekeringan
        semua["skor"] = semua["p"]
        semua["level_api"] = beri_level_aman(semua["p"], AMBANG_SEMENTARA)
    semua["luas_g"] = luas_gambut(semua)
    semua["v"] = semua["skor"] * semua["luas_g"]
    kand = semua[semua["level_api"].isin(LEVEL_PATROLI)].sort_values("v", ascending=False).reset_index(drop=True)
    n_aman = int((semua["level_api"] == "Aman").sum())
    total, n = float(kand["v"].sum()), len(kand)

    # 0) Ringkasan peringatan kebakaran (prediksi XGBoost) untuk periode ke depan
    awal, akhir = minggu + pd.Timedelta(days=7), minggu + pd.Timedelta(days=7 * (k + 1) - 1)
    periode = f"{tgl_id(awal)[:-5]} – {tgl_id(akhir)}"
    hit = semua["level_api"].value_counts()
    wil = "seluruh Indonesia" if pulau == "Semua pulau" else pulau
    if h is None:
        info_strip("Model peringatan belum bisa dihitung untuk minggu ini; status memakai peluang kering.", "warn")
    elif hit.get("Awas", 0) + hit.get("Siaga", 0) == 0:
        info_strip(f"Untuk <b>{periode}</b>, tidak ada kab/kota bergambut di {wil} yang berstatus Siaga atau Awas.", "")
    else:
        terbanyak = semua[semua["level_api"].isin(["Awas", "Siaga"])]["provinsi"].value_counts().index[0]
        info_strip(f"Peringatan kebakaran <b>{periode}</b>: {hit.get('Awas', 0)} kab/kota berstatus <b>Awas</b> dan "
                   f"{hit.get('Siaga', 0)} <b>Siaga</b> di {wil}, terbanyak di {terbanyak}.",
                   "warn" if hit.get("Awas", 0) else "")
    kpi_grid([{"label": f"Peringatan · {l}", "value": f"{hit.get(l, 0)} kab/kota",
               "accent": {"Awas": "danger", "Siaga": "orange", "Waspada": "warning"}[l],
               "hint": f"{periode} · {ARTI_API[l]}"} for l in LEVEL_PATROLI]
             + [{"label": "Gambut berisiko terbakar", "value": luas_txt(total), "hint": "skor risiko × luas gambut"}],
             kolom=4)

    # 1) Anggaran per status (diisi) dibagi ke tiap kab/kota berstatus sama, sebanding luas gambutnya
    kolom_kiri, kolom_kanan = st.columns([1.45, 1])
    with kolom_kiri, kotak("kotak_input"):
        kiri = st.container()
        with kiri:
            judul_kotak("Anggaran patroli per status",
                        "total anggaran per minggu (Rp) untuk semua kab/kota berstatus itu · dibagi ke tiap kab/kota "
                        "sesuai luas gambutnya · Aman tidak dipatroli khusus")
        ka, ks, kw, kb = kiri.columns(4)
        bantu = " Dibagi ke tiap kab/kota berstatus sama sesuai luas gambutnya."
        biaya = {"Awas": isian_rupiah(ka, "Awas", "biaya_awas", "mis. 1.000.000.000.000",
                                      "Total anggaran semua kab/kota Awas per minggu (patroli & pemadaman dini)." + bantu),
                 "Siaga": isian_rupiah(ks, "Siaga", "biaya_siaga", "mis. 100.000.000.000",
                                       "Total anggaran semua kab/kota Siaga per minggu (siapkan regu pemadam)." + bantu),
                 "Waspada": isian_rupiah(kw, "Waspada", "biaya_waspada", "mis. 50.000.000.000",
                                         "Total anggaran semua kab/kota Waspada per minggu (pantau titik api)." + bantu)}
        B = float(sum(biaya.values()))
        kb.markdown(f'<div class="total-anggaran"><span>Anggaran minggu ini</span><b>{rupiah(B) if B else "—"}</b>'
                    '</div>', unsafe_allow_html=True)
        pasang_format_rupiah()
        # anggaran kab/kota = anggaran statusnya × luas gambutnya ÷ total luas gambut kab/kota berstatus sama
        luas_status = kand.groupby("level_api")["luas_g"].transform("sum") if n else pd.Series(dtype=float)
        c = (kand["level_api"].map(biaya).to_numpy(float) * (kand["luas_g"] / luas_status).to_numpy(float)) if n \
            else np.zeros(0)
        kurang = [l for l in LEVEL_PATROLI if (kand["level_api"] == l).any() and biaya[l] <= 0]
        siap = n > 0 and not kurang
        baris_r = ""
        for l in LEVEL_PATROLI:
            sel = kand[kand["level_api"] == l]
            rata = biaya[l] / len(sel) if len(sel) and biaya[l] else None
            baris_r += (f'<tr><td><i style="background:{WARNA_API[l]}"></i>{l}</td><td>{len(sel)}</td>'
                        f'<td>{fmt(sel["luas_g"].sum())} ha</td>'
                        f'<td>{rupiah(biaya[l]) if biaya[l] else "belum diisi"}</td>'
                        f'<td>{rupiah(rata) if rata else "—"}</td></tr>')
        kiri.markdown(
            '<table class="rincian"><thead><tr><th>Status</th><th>Kab/kota</th><th>Luas gambut</th>'
            '<th>Anggaran status</th><th>Rata-rata per kab/kota</th></tr></thead><tbody>' + baris_r +
            f'<tr class="total pertama"><td>Total</td><td>{n}</td><td>{fmt(kand["luas_g"].sum())} ha</td>'
            f'<td>{rupiah(B) if B else "—"}</td><td></td></tr></tbody></table>', unsafe_allow_html=True)
        with kiri:
            sia = [l for l in LEVEL_PATROLI if biaya[l] > 0 and not (kand["level_api"] == l).any()]
            if n == 0:
                info_strip("Tidak ada kab/kota berstatus Waspada, Siaga, atau Awas, jadi tidak ada yang perlu dipatroli.", "")
            elif not siap:
                info_strip("Isi anggaran untuk " + ", ".join(kurang) + ". Anggaran tiap status dibagi ke kab/kota "
                           "berstatus sama sesuai luas gambutnya: gambut lebih luas, bagian lebih besar"
                           + (f" ({n_aman} kab/kota berstatus Aman tidak masuk daftar)." if n_aman else "."), "")
            else:
                i = int(np.argmax(c))
                info_strip(f"<b>{rupiah(B)}</b> dibagi ke <b>{n} kab/kota</b> sesuai luas gambutnya. Bagian terbesar: "
                           f"<b>{kand.loc[i, 'kabupaten']}</b> ({kand.loc[i, 'level_api']}, {rupiah(c[i])})."
                           + (f" Anggaran {', '.join(sia)} tidak terpakai karena tidak ada kab/kota berstatus itu."
                              if sia else ""), "warn" if sia else "ok")
    with kolom_kanan, kotak("kotak_kurva"):
        if n == 0:
            judul_kotak("Pembagian anggaran per kab/kota")
            st.caption("Tidak ada kab/kota yang perlu dipatroli.")
        else:
            top = kand.assign(c=c).sort_values("c" if siap else "luas_g", ascending=False).head(12)
            if siap:
                judul_kotak("Pembagian anggaran per kab/kota", "12 kab/kota dengan bagian terbesar · sebanding luas gambut")
                top["nilai"], judul_x = top["c"] / 1e9, "anggaran (miliar Rp)"
            else:
                judul_kotak("Luas gambut kab/kota yang dipatroli",
                            "12 terluas · anggaran nanti dibagi sebanding luas ini")
                top["nilai"], judul_x = top["luas_g"] / 1e3, "luas gambut (ribu ha)"
            top["teks"] = [rupiah(x) if siap else luas_txt(a) for x, a in zip(top["c"], top["luas_g"])]
            y = alt.Y("kabupaten:N", sort=list(top["kabupaten"]), title=None,
                      axis=alt.Axis(labelLimit=150, labelOverlap=False))
            batang = alt.Chart(top).mark_bar(cornerRadiusEnd=3).encode(
                x=alt.X("nilai:Q", title=judul_x, axis=alt.Axis(labelExpr="replace(format(datum.value, ',.0f'), /,/g, '.')")),
                y=y, color=alt.Color("level_api:N", title=None, legend=alt.Legend(orient="top"),
                                     scale=alt.Scale(domain=LEVEL_PATROLI, range=[WARNA_API[l] for l in LEVEL_PATROLI])),
                tooltip=[alt.Tooltip("kabupaten:N", title="Kab/kota"), alt.Tooltip("provinsi:N", title="Provinsi"),
                         alt.Tooltip("level_api:N", title="Status"), alt.Tooltip("teks:N", title="Nilai")])
            st.altair_chart(tema_altair(batang.properties(height=330)), use_container_width=True)

    # 2) Peta | daftar prioritas dan anggaran tiap kab/kota
    c1, c2 = st.columns([1.45, 1])
    anggaran = dict(zip(kand["petak_id"], c)) if siap else {}
    with c1, kotak("kotak_peta_anggaran", isi=True):
        judul_kotak(f"Peringatan kebakaran ({k} minggu ke depan)",
                    "arahkan kursor ke kab/kota untuk melihat bagian anggarannya")
        ang = [rupiah(anggaran[kk]) if kk in anggaran else
               ("belum diisi" if l in LEVEL_PATROLI else "tidak dipatroli khusus")
               for kk, l in zip(semua["petak_id"], semua["level_api"])]
        nilai = semua.set_index("petak_id").assign(
            wilayah=semua["kabupaten"].to_numpy(), lv=semua["level_api"].to_numpy(),
            luas_txt=[luas_txt(a) for a in semua["luas_g"]], ang=ang)[["wilayah", "lv", "luas_txt", "ang"]]
        warna = {kk: hex_rgb(WARNA_API[l], 225) for kk, l in zip(semua["petak_id"], semua["level_api"]) if l in WARNA_API}
        peta(layer_wilayah(batas, nilai, warna, "wilayah-anggaran"), view,
             "<b>{wilayah}</b><br/>Peringatan kebakaran: {lv}<br/>"
             "Luas gambut: {luas_txt}<br/>Anggaran patroli: {ang}", tinggi=345, pulau=pulau)
        legenda_level("Peringatan", WARNA_API, ARTI_API)
    with c2, kotak("kotak_prioritas"):
        judul_kotak("Urutan prioritas patroli",
                    "kab/kota Waspada, Siaga, Awas · urut dari luas gambut berisiko terbakar · anggaran sebanding luas gambut")
        tabel = pd.DataFrame({
            "#": np.arange(1, n + 1), "Kab/kota": kand["kabupaten"], "Provinsi": kand["provinsi"],
            "Status": kand["level_api"], "Luas gambut (ha)": [fmt(a) for a in kand["luas_g"]],
            "Anggaran (Rp)": [fmt(b) if b > 0 else "" for b in c],
        })
        st.dataframe(tabel[["#", "Kab/kota", "Status", "Luas gambut (ha)", "Anggaran (Rp)"]],
                     use_container_width=True, hide_index=True, height="stretch",
                     column_config={"#": st.column_config.TextColumn(width=30),
                                    "Kab/kota": st.column_config.TextColumn(width=130),
                                    "Status": st.column_config.TextColumn(width=65),
                                    "Luas gambut (ha)": st.column_config.TextColumn(width=100),
                                    "Anggaran (Rp)": st.column_config.TextColumn(width=130)})
        tombol_unduh(tabel, f"sparks_patroli_{minggu:%Y%m%d}.csv", "Unduh daftar patroli (CSV)")


# =============================================================================
# MAIN
# =============================================================================

def css_ringkas():
    """Tata letak rapat: tanpa sidebar, kartu dan judul lebih kecil, supaya jarang perlu menggulir."""
    st.markdown(f"""
    <style>
    section[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
    [data-testid="stExpandSidebarButton"] {{ display: none !important; }}
    .block-container {{ padding-top: 0.5rem !important; padding-bottom: .5rem !important; }}
    .topbar {{ padding: 4px 16px; margin-bottom: 6px; }}
    .topbar .brand-logo {{ height: 42px; width: auto; display: block; margin: -2px 0; }}
    .topbar .brand-title {{ font-size: 1.35rem; letter-spacing: 3px; }}
    header[data-testid="stHeader"] {{ height: 0; min-height: 0; background: transparent; }}
    [data-testid="stToolbar"] {{ top: 4px; }}
    div[class*="st-key-kotak"] {{ padding: 9px 12px !important; }}
    [data-testid="stVerticalBlock"] {{ gap: .7rem; }}
    .topbar .brand-title .hijau {{ color: #4cbf86; }}
    .topbar .brand-title .api {{ background: linear-gradient(180deg, #ffa22b, #e5461d); -webkit-background-clip: text;
                                 background-clip: text; color: transparent; }}
    .topbar .brand-sub {{ font-size: .62rem; margin-top: 2px; }}
    .kpi-grid {{ gap: 8px; margin-bottom: 8px; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); }}
    .kpi-box {{ padding: 8px 12px; }}
    .kpi-box .kpi-icon {{ font-size: .85rem; margin-bottom: 2px; }}
    .kpi-box .kpi-label {{ font-size: .6rem; letter-spacing: .8px; }}
    .kpi-box .kpi-value {{ font-size: 1.3rem; margin-top: 2px; }}
    .kpi-box .kpi-hint {{ font-size: .65rem; margin-top: 2px; }}
    .feature-bar {{ padding: 5px 10px; margin-bottom: 4px; font-size: .72rem; }}
    .info-strip {{ margin: 0 0 10px; padding: 9px 14px; font-size: .85rem; }}
    .legend-row {{ padding: 6px 12px; margin-top: 4px; font-size: .72rem; }}
    .bar-info {{ font-size: .75rem; color: {C['text_muted']}; margin-top: -4px; }}
    .bar-info b {{ color: {C['text']}; }}
    .sub {{ font-size: .75rem; color: {C['text_muted']}; }}
    .chip {{ display: inline-flex; align-items: center; gap: 5px; padding: 2px 9px; margin: 0 6px 6px 0;
             border: 1px solid; border-radius: 12px; font-size: .74rem; background: {C['bg_soft']}; }}
    .chip i {{ display: inline-block; width: 9px; height: 9px; border-radius: 50%; }}
    a.tombol-unduh {{ display: block; text-align: center; padding: 7px 12px; margin-top: 6px; border-radius: 10px;
                      background: {C['panel']}; border: 1px solid {C['border']}; color: {C['text']} !important;
                      font-weight: 600; font-size: .85rem; text-decoration: none; }}
    a.tombol-unduh:hover {{ border-color: {C['accent']}; color: {C['accent']} !important; }}
    .legenda {{ display: flex; flex-wrap: wrap; align-items: center; gap: 4px 12px; font-size: .74rem;
                color: {C['text_muted']}; margin-top: 4px; }}
    .legenda b {{ color: {C['text']}; font-weight: 600; margin-right: 2px; }}
    .legenda .lg {{ display: inline-flex; align-items: center; gap: 5px; }}
    .legenda .lg i {{ display: inline-block; width: 14px; height: 11px; border-radius: 2px; }}
    .var-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin: 8px 0 6px; }}
    .var-grid .var {{ background: rgba(10, 22, 40, .45); border: 1px solid {C['border']}; border-radius: 10px;
                      padding: 8px 12px; font-size: .8rem; line-height: 1.4; }}
    .var-grid .var b {{ display: block; color: {C['text']}; }}
    .var-grid .var span {{ display: block; color: {C['text_muted']}; }}
    .var-grid .var i {{ display: block; color: {C['accent']}; font-style: normal; font-size: .72rem; margin-top: 2px; }}
    .total-anggaran span {{ display: block; font-size: .875rem; color: {C['text']}; margin-bottom: 8px; }}
    .total-anggaran b {{ display: block; padding: 9px 12px; border-radius: 8px; border: 1px dashed {C['border_active']};
                         color: {C['accent']}; font-size: .95rem; }}
    table.rincian {{ width: 100%; border-collapse: collapse; font-size: .8rem; margin: 6px 0 10px; }}
    table.rincian th {{ text-align: left; font-weight: 600; color: {C['text_muted']}; padding: 5px 8px;
                        border-bottom: 1px solid {C['border']}; }}
    table.rincian td {{ padding: 5px 8px; color: {C['text']}; border-bottom: 1px solid rgba(44,85,130,.35); }}
    table.rincian th:not(:first-child), table.rincian td:not(:first-child) {{ text-align: right; }}
    table.rincian td i {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 7px; }}
    table.rincian tr.total td {{ font-weight: 700; }}
    table.rincian tr.pertama td {{ border-top: 1px solid {C['border']}; }}
    table.rincian tr.selisih td {{ font-weight: 700; color: {C['accent']}; border-bottom: none; }}
    /* Label tombol geser (mis. Titik api) tidak dipotong "…"; tombol rata kanan di samping judul peta */
    .st-key-tampil_api {{ display: flex; justify-content: flex-end; }}
    .st-key-tampil_api label, .st-key-tampil_api label p {{ white-space: nowrap !important; overflow: visible !important;
                                                           text-overflow: clip !important; max-width: none !important; }}
    .legenda-grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 6px 12px; margin-top: 10px;
                     padding-bottom: 14px; }}
    .legenda-grid .lg2 {{ display: flex; gap: 8px; align-items: flex-start; font-size: .74rem; line-height: 1.3;
                          white-space: nowrap; }}
    .legenda-grid .lg2 i {{ flex: none; width: 14px; height: 14px; border-radius: 3px; margin-top: 2px;
                            border: 1px solid rgba(200,212,225,.35); }}
    .legenda-grid .lg2 b {{ display: block; color: {C['text']}; font-weight: 600; }}
    .legenda-grid .lg2 span {{ color: {C['text_muted']}; }}
    .detail-judul {{ font-size: 1.15rem; font-weight: 700; color: {C['text']}; margin: 2px 0 8px;
                     border-left: 3px solid {C['accent']}; padding-left: 10px; line-height: 1.25; }}
    .detail-judul span {{ display: block; font-size: .74rem; font-weight: 500; color: {C['text_muted']}; }}
    /* Kotak bahasan dan baris filter: lebih terang dari latar halaman, bertepi jelas dan berbayang */
    div[class*="st-key-kotak"], [data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) {{
        background: linear-gradient(180deg, #18375c 0%, #142e4d 100%) !important;
        border: 1px solid #2c5582 !important; border-radius: 14px !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, .45), inset 0 1px 0 rgba(255, 255, 255, .05); }}
    div[class*="st-key-kotak"] {{ padding: 10px 14px !important; }}
    .stApp, [data-testid="stAppViewContainer"] {{ background: #070f1c !important; }}
    .kotak-judul {{ font-size: .98rem; font-weight: 700; color: {C['text']}; line-height: 1.3; margin: 4px 0 12px;
                    padding: 1px 0 2px 10px; border-left: 3px solid {C['accent']}; }}
    .kotak-judul span {{ display: block; font-size: .74rem; font-weight: 400; color: {C['text_muted']};
                         line-height: 1.4; margin-top: 4px; }}
    /* Layar HP (≤ 640 px): kolom Streamlit sudah bertumpuk sendiri; yang di bawah ini merapikan bagian buatan sendiri */
    @media (max-width: 640px) {{
        .block-container {{ padding-left: .7rem !important; padding-right: .7rem !important; }}
        .topbar {{ padding: 8px 12px; flex-wrap: wrap; gap: 6px; }}
        .topbar .brand-sub {{ letter-spacing: .5px; font-size: .58rem; }}
        .topbar .status-pill.live {{ display: none; }}
        .kpi-grid {{ grid-template-columns: repeat(2, 1fr) !important; }}
        .kpi-box .kpi-value {{ font-size: 1.1rem; }}
        .legenda-grid {{ grid-template-columns: repeat(3, 1fr); }}
        .legenda-grid .lg2 {{ white-space: normal; }}
        table.rincian {{ display: block; overflow-x: auto; font-size: .72rem; }}
        table.rincian th, table.rincian td {{ white-space: nowrap; padding: 4px 6px; }}
        .st-key-kotak_kanan [data-testid="stDataFrame"],
        .st-key-kotak_prioritas [data-testid="stDataFrame"] {{ height: 330px !important; min-height: 330px; }}
    }}
    .ket-uji {{ font-size: .8rem; color: {C['text_muted']}; margin: 2px 0 12px; }}
    .ket-uji b {{ color: {C['text']}; font-weight: 600; }}
    [data-baseweb="tab-list"] {{ gap: 6px; }}
    [data-testid="stVerticalBlockBorderWrapper"] > div {{ padding: 2px 4px; }}
    div[data-testid="stTabs"] {{ margin-top: -6px; }}
    .footer {{ margin-top: 10px; padding: 8px 0 4px; }}
    </style>
    """, unsafe_allow_html=True)


def topbar(periode: str | None):
    pill = (f'<span class="status-pill live"><span class="pulse"></span>DATA SATELIT ASLI</span>'
            f'<span class="status-pill info" title="Data SMAP tersedia ±3–5 hari setelah pengamatan; titik api '
            f'±1–2 hari.">{periode}</span>' if periode
            else '<span class="status-pill offline">BELUM ADA DATA</span>')
    logo = (f'<img class="brand-logo" src="data:image/png;base64,{base64.b64encode(BERKAS_LOGO.read_bytes()).decode()}">'
            if BERKAS_LOGO.exists() else "")
    st.markdown(f"""
    <div class="topbar">
        <div class="brand">
            {logo}
            <div>
                <div class="brand-title"><span class="hijau">SPAR</span><span class="api">K</span><span class="hijau">S</span></div>
                <div class="brand-sub">Peringatan dini kekeringan lahan gambut · Indonesia</div>
            </div>
        </div>
        <div class="status">{pill}</div>
    </div>
    """, unsafe_allow_html=True)


def footer():
    st.markdown(f'<div class="footer"><b>{APP_NAME} {APP_VERSION}</b> · UEC LOGIKA UI 2026 · '
                'data: NASA SMAP L4 (muka air gambut, terlambat ±3–5 hari), NASA FIRMS (titik api, ±1–2 hari), '
                'FAO GAUL 2025 (batas wilayah), lewat Google Earth Engine · diperbarui saat data diambil, bukan '
                'real-time</div>', unsafe_allow_html=True)


def ada_data() -> bool:
    if not (BERKAS_KAB_MINGGUAN.exists() and BERKAS_KABUPATEN.exists() and BERKAS_PENGATURAN.exists()):
        return False
    return json.loads(BERKAS_PENGATURAN.read_text(encoding="utf-8")).get("mode") == "kabupaten"


def main():
    st.set_page_config(page_title=f"{APP_NAME} — Peringatan Dini Kekeringan Gambut",
                       page_icon=str(BERKAS_LOGO) if BERKAS_LOGO.exists() else None, layout="wide", initial_sidebar_state="collapsed")
    inject_css()
    css_ringkas()
    if not ada_data():
        topbar(None)
        f = bar_filter(None, None)
        if f["permintaan"]:
            jalankan_permintaan(f["permintaan"])
        empty_state("", "Belum ada data. Buka tombol <b>Data</b>, pilih rentang tanggal, isi Project ID Google Earth "
                          "Engine, lalu tekan <b>Ambil data dari satelit</b>.")
        footer()
        return
    mtime = BERKAS_KAB_MINGGUAN.stat().st_mtime
    batas, kab, hs, pg = muat_data(mtime)
    semua_minggu = sorted(kab["minggu"].unique())
    m0 = st.session_state.get("tgl_mulai") or date.fromisoformat(pg.get("mulai", f"{semua_minggu[0]:%Y-%m-%d}"))
    s0 = st.session_state.get("tgl_selesai") or date.fromisoformat(pg.get("selesai", f"{semua_minggu[-1]:%Y-%m-%d}"))
    topbar(f"Data satelit s.d. {tgl_id(pg.get('data_sampai', f'{semua_minggu[-1]:%Y-%m-%d}'))}")
    f = bar_filter(pg, semua_minggu)
    if f["permintaan"]:
        jalankan_permintaan(f["permintaan"])
    # Rentang tampil = tanggal pilihan pengguna, langsung dari kumpulan data tersimpan (tanpa unduh).
    tampil = kab[(kab["minggu"] >= pd.Timestamp(f["mulai"])) & (kab["minggu"] <= pd.Timestamp(f["selesai"]))]
    gagal = st.session_state.get("gagal_proses") or []
    if gagal:
        info_strip(f"{len(gagal)} provinsi gagal diolah dan tampil abu-abu: " + "; ".join(gagal[:3])
                   + ". Buka <b>Data</b> dan ambil lagi untuk mencoba ulang.", "warn")
    if tampil.empty or f["minggu"] is None:
        info_strip("Belum ada data tersimpan untuk rentang ini. Buka <b>Data</b> lalu tekan <b>Ambil data dari "
                   "satelit</b> (hanya tanggal yang belum pernah diambil).", "warn")
        return
    minggu, pulau = f["minggu"], f["pulau"]

    # Satu baris per kabupaten di pulau terpilih (yang tidak punya piksel gambut tampil abu-abu).
    batas_p = saring_batas(batas, pulau)
    if not batas_p["features"]:
        info_strip(f"Belum ada batas wilayah untuk {pulau}.", "warn")
        return
    view = view_untuk(batas_p)
    semua = pd.DataFrame([x["properties"] for x in batas_p["features"]]).rename(columns={"kunci": "petak_id"})
    d1 = semua[["petak_id", "kabupaten", "provinsi", "luas_ha"]].merge(
        tampil[tampil["minggu"] == minggu][[k for k in ("petak_id", "p", "p_bawah", "p_atas", "n_titik", "muka_air_m")
                                            if k in tampil]], on="petak_id", how="left")
    tampil_p = tampil[tampil["petak_id"].isin(set(semua["petak_id"]))]
    ambang, sumber_ambang = ambang_minggu(mtime, minggu)   # batas level & XGBoost dilatih se-Indonesia

    t1, t3, t4 = st.tabs(["Situasi Terbaru", "Peringatan & Rencana Patroli", "Tentang Model"])
    with t1:
        tab_situasi(batas_p, d1, tampil_p, hs, minggu, ambang, mtime, view, pulau)
    with t3:
        tab_anggaran(batas_p, d1, minggu, view, pulau, mtime)
    with t4:
        tab_model(mtime, minggu, pulau)
    footer()


def cli(argv: list[str]) -> None:
    """python app.py proses --mulai 2026-01-01 --selesai 2026-10-19 : data dasbor tanpa dasbor.
    python app.py esai : analisis Bab III blueprint (kotak pilot Sentinel-1 100 m, latih 2018 / uji 2019,
    Juni–Oktober) ke folder hasil/."""
    import argparse
    if argv and argv[0] == "esai":
        pg = dict(PENGATURAN_BAWAAN)
        hasil = jalankan_data_asli(pg, os.getenv("EE_PROJECT_ID", ""), os.getenv("NASA_FIRMS_MAP_KEY", ""), FOLDER_HASIL)
        print(hasil["ringkasan"])
        return
    pa = argparse.ArgumentParser(prog="python app.py proses", description="Ambil dan olah data SPARKS")
    pa.add_argument("--mulai", help="tanggal mulai YYYY-MM-DD (bersama --selesai menggantikan --tahun/bulan)")
    pa.add_argument("--selesai", help="tanggal selesai YYYY-MM-DD")
    pa.add_argument("--tahun", type=int)
    pa.add_argument("--bulan-mulai", type=int, default=PENGATURAN_BAWAAN["bulan_mulai"])
    pa.add_argument("--bulan-selesai", type=int, default=PENGATURAN_BAWAAN["bulan_selesai"])
    pa.add_argument("--provinsi", nargs="*", help="batasi ke provinsi tertentu (untuk uji)")
    pa.add_argument("--project", default=os.getenv("EE_PROJECT_ID", ""), help="bawaan: dari .env")
    a = pa.parse_args(argv)
    if a.mulai and a.selesai:
        hasil = proses_rentang(date.fromisoformat(a.mulai), date.fromisoformat(a.selesai), a.project,
                               hanya_provinsi=a.provinsi)
    elif a.tahun:
        hasil = proses_kalimantan(a.tahun, a.bulan_mulai, a.bulan_selesai, a.project, hanya_provinsi=a.provinsi)
    else:
        pa.error("isi --mulai dan --selesai, atau --tahun")
    print(f"Selesai: data s.d. {hasil['data_sampai']}; gagal: {hasil['gagal'] or 'tidak ada'}; "
          f"tanpa gambut: {', '.join(hasil.get('tanpa_gambut', [])) or '-'}")


# `streamlit run app.py` menjalankan berkas ini sebagai __main__. Saat diimpor
# (misalnya `import app` di notebook atau uji), dasbor tidak ikut dijalankan.
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("proses", "esai"):
        cli(sys.argv[2:] if sys.argv[1] == "proses" else ["esai"])
    else:
        main()
