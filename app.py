import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Nifty Smallcap 250 Scanner",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background: #05080f; }
  .block-container { padding-top: 1rem; padding-bottom: 1rem; }
  h1 { color: #e2e8f0 !important; font-family: monospace; font-size: 1.4rem !important; }
  .stDataFrame { background: #070b14; }
  div[data-testid="metric-container"] {
    background: #070b14; border: 1px solid #0d1f35;
    border-radius: 8px; padding: 12px;
  }
  div[data-testid="metric-container"] label { color: #334155 !important; font-size: 11px !important; }
  div[data-testid="metric-container"] div { color: #22c55e !important; font-size: 1.8rem !important; }
  .stButton > button {
    background: #0f2040; color: #38bdf8; border: 1px solid #1e3a5f;
    font-family: monospace; border-radius: 6px;
  }
  .stButton > button:hover { background: #1e3a5f; border-color: #38bdf8; }
  .stTabs [data-baseweb="tab"] { color: #475569; font-family: monospace; font-size: 12px; }
  .stTabs [aria-selected="true"] { color: #22c55e !important; }
  .stProgress > div > div { background: linear-gradient(90deg, #0ea5e9, #22c55e); }
  footer { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Universe ──────────────────────────────────────────────────────
SYMBOLS = [
    "AAVAS","AARTIDRUGS","AARTIIND","ABCAPITAL","ACE","AEGISLOG","AJANTPHARM",
    "AKZOINDIA","ALEMBICPH","ALLCARGO","AMBER","ANGELONE","APARINDS","APLLTD",
    "APOLLOPIPE","ASAHIINDIA","ASHOKA","ASTERDM","ASTRAZEN","AVANTIFEED",
    "BAJAJCON","BALMLAWRIE","BASF","BAYERCROP","BEML","BIKAJI","BIRLACORP",
    "BLUESTARCO","BSOFT","CANFINHOME","CAPLIPOINT","CAMS","CASTROLIND",
    "CENTURYPLY","CERA","CHAMBLFERT","COCHINSHIP","CONCORDBIO","CROMPTON",
    "CSBBANK","DATAMATICS","DELHIVERY","DHANUKA","ECLERX","ELGIEQUIP",
    "EMAMILTD","ENGINERSIN","EPIGRAL","EQUITASBNK","FAIRCHEM","FDC",
    "FINEORG","FLUOROCHEM","GABRIEL","GAEL","GILLETTE","GLAND","GLAXO",
    "GMDCLTD","GNFC","GODFRYPHLP","GOLDIAM","GPIL","GRANULES","GREENPANEL",
    "GRINDWELL","GSFC","GUFICBIO","HEG","HERITGFOOD","HIKAL","HMVL",
    "HOMEFIRST","HUDCO","IEX","IFBIND","IIFLSEC","INDGN","INDIACEM",
    "INDORAMA","INDOSTAR","INFIBEAM","IRCON","JAMNAAUTO","JBCHEPHARM",
    "JKLAKSHMI","JKPAPER","JMFINANCIL","JUBLINGREA","JUSTDIAL","KALPATPOWR",
    "KANSAINER","KARURVYSYA","KCP","KFINTECH","KNRCON","KOLTEPATIL","KPIL",
    "KSB","KSOLVES","LATENTVIEW","LEMONTREE","LINDEINDIA","LLOYDSENGG",
    "LUXIND","MAHINDCIE","MASTEK","METROPOLIS","MFSL","MOIL","MOLDTKPAC",
    "MONTECARLO","MRPL","MSTCLTD","NACLIND","NATCOPHARM","NAVA","NESCO",
    "NILKAMAL","NLCINDIA","NOCIL","NRBBEARING","NUVAMA","NCC","OLECTRA",
    "ORIENTELEC","ORIENTPAPER","PANAMAPET","PARADEEP","PCBL","PENIND",
    "PFIZER","PGHL","PILANIINVS","PNBHOUSING","PNCINFRA","POLYMED","PRAJIND",
    "PRICOLLTD","PRINCEPIPE","PRSMJOHNSN","PRUDENT","PSPPROJECT","QUESS",
    "RAILTEL","RAJESHEXPO","RALLIS","RAMCOIND","RATNAMANI","RAYMOND","RBLBANK",
    "REDINGTON","RHIM","RITES","ROSSARI","ROUTE","RPGLIFE","SAFARI","SAREGAMA",
    "SBFC","SCHAEFFLER","SEQUENT","SHARDACROP","SHYAMMETL","SKFINDIA",
    "SKIPPER","SMLISUZU","SNOWMAN","SOLARA","SOMANYCER","SOUTHBANK","SPANDANA",
    "SPENCERS","SPIC","STARCEMENT","STCINDIA","SUBROS","SUDARSCHEM","SUMICHEM",
    "SUPRIYA","SURYAROSNI","SUVENPHAR","SYMPHONY","TANLA","TATAINVEST",
    "TATAMETALI","TBOTEK","TEAMLEASE","THYROCARE","TIMETECHNO","TINPLATE",
    "TITAGARH","TTKPRESTIGE","TVSHLTD","TVSSCS","UJJIVANSFB","UNOMINDA",
    "VAKRANGEE","VENKEYS","VESUVIUS","VINATIORGA","VIPIND","VOLTAMP","VTL",
    "WABAG","WELCORP","WELSPUNLIV","YATHARTH","ZENSARTECH","ZODIACLOTH",
    "KRBL","NYKAA","KAYNES","RVNL","IRFC","NBCC","RPOWER","SUZLON","HFCL",
    "DBREALTY","RAILVIKAS","CIGNITITEC","CLEAN","SPANDANA",
]
SYMBOLS_NS = [s + ".NS" for s in SYMBOLS]

# ── Indicators ────────────────────────────────────────────────────
def calc_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_l = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_macd(series):
    ema12  = series.ewm(span=12, adjust=False).mean()
    ema26  = series.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist   = macd - signal
    return macd, signal, hist

def calc_stoch(df, k=14, d=3):
    low_k  = df["Low"].rolling(k).min()
    high_k = df["High"].rolling(k).max()
    stoch_k = ((df["Close"] - low_k) / (high_k - low_k).replace(0, np.nan)) * 100
    stoch_d = stoch_k.rolling(d).mean()
    return stoch_k, stoch_d

def calc_mfi(df, period=14):
    tp  = (df["High"] + df["Low"] + df["Close"]) / 3
    mfv = tp * df["Volume"]
    pos = mfv.where(tp > tp.shift(1), 0)
    neg = mfv.where(tp < tp.shift(1), 0)
    pos_sum = pos.rolling(period).sum()
    neg_sum = neg.rolling(period).sum()
    return 100 - (100 / (1 + pos_sum / neg_sum.replace(0, np.nan)))

def calc_obv(df):
    direction = np.sign(df["Close"].diff())
    return (direction * df["Volume"]).cumsum()

def calc_cmf(df, period=20):
    hl   = df["High"] - df["Low"]
    mfv  = ((df["Close"] - df["Low"] - (df["High"] - df["Close"])) / hl.replace(0, np.nan)) * df["Volume"]
    return mfv.rolling(period).sum() / df["Volume"].rolling(period).sum()

def calc_bb(series, period=20, std=2):
    mid  = series.rolling(period).mean()
    band = series.rolling(period).std()
    upper = mid + std * band
    lower = mid - std * band
    width = (upper - lower) / mid
    return upper, mid, lower, width

def classify(sym, df):
    if df is None or len(df) < 35:
        return None
    df = df.copy()
    C, H, L, V = df["Close"], df["High"], df["Low"], df["Volume"]
    n = len(df)

    # Indicators
    df["RSI"]       = calc_rsi(C)
    df["OBV"]       = calc_obv(df)
    df["CMF"]       = calc_cmf(df)
    df["MFI"]       = calc_mfi(df)
    df["EMA9"]      = C.ewm(span=9,  adjust=False).mean()
    df["EMA21"]     = C.ewm(span=21, adjust=False).mean()
    df["EMA55"]     = C.ewm(span=55, adjust=False).mean()
    df["MACD"], df["MACD_SIG"], df["MACD_HIST"] = calc_macd(C)
    df["STOCH_K"], df["STOCH_D"] = calc_stoch(df)
    df["BB_U"], df["BB_M"], df["BB_L"], df["BB_W"] = calc_bb(C)
    df["VOL_RATIO"] = V / V.rolling(20).mean()
    df["OBV_SLOPE"] = df["OBV"].diff(3)
    df.dropna(inplace=True)
    if len(df) < 5:
        return None

    last  = df.iloc[-1]
    prev  = df.iloc[-2]

    vr = last["VOL_RATIO"]

    # Signals
    s = {
        # LEADING
        "VOL_BUILD"     : 1.3 <= vr < 2.0,
        "OBV_RISING"    : last["OBV_SLOPE"] > 0,
        "CMF_TURNING"   : last["CMF"] > 0 and prev["CMF"] <= 0,
        "RSI_OFF_SOLD"  : 30 < last["RSI"] < 55 and last["RSI"] > prev["RSI"],
        "BB_SQUEEZE"    : last["BB_W"] < 0.06,
        "STOCH_CROSS"   : last["STOCH_K"] > last["STOCH_D"] and prev["STOCH_K"] <= prev["STOCH_D"] and prev["STOCH_K"] < 35,
        "MFI_RISING"    : 20 < last["MFI"] < 55 and last["MFI"] > prev["MFI"],
        # COINCIDENT
        "MACD_CROSS"    : last["MACD"] > last["MACD_SIG"] and prev["MACD"] <= prev["MACD_SIG"],
        "P_GT_EMA21"    : last["Close"] > last["EMA21"],
        "EMA9_GT_21"    : last["EMA9"] > last["EMA21"],
        "VOL_2X"        : vr >= 2.0,
        "MACD_HIST_POS" : last["MACD_HIST"] > 0,
    }

    LEADING_K    = ["VOL_BUILD","OBV_RISING","CMF_TURNING","RSI_OFF_SOLD","BB_SQUEEZE","STOCH_CROSS","MFI_RISING"]
    COINCIDENT_K = ["MACD_CROSS","P_GT_EMA21","EMA9_GT_21","VOL_2X","MACD_HIST_POS"]
    W = dict(VOL_BUILD=3,OBV_RISING=3,CMF_TURNING=2,RSI_OFF_SOLD=3,BB_SQUEEZE=2,STOCH_CROSS=3,MFI_RISING=2,
             MACD_CROSS=1,P_GT_EMA21=1,EMA9_GT_21=1,VOL_2X=1,MACD_HIST_POS=1)
    MAX_L  = sum(W[k] for k in LEADING_K)
    l_score = sum(W[k] for k in LEADING_K if s[k])
    c_score = sum(W[k] for k in COINCIDENT_K if s[k])
    l_pct   = l_score / MAX_L

    if   l_pct >= 0.45 and l_score >= 6: bucket = "⚡ Leading"
    elif l_pct >= 0.25 or c_score >= 2:  bucket = "↔ Coincident"
    else:                                  bucket = "○ Neutral"

    c = df["Close"]
    m2 = (c.iloc[-1]-c.iloc[-3])/c.iloc[-3]*100 if len(c)>3 else None
    m3 = (c.iloc[-1]-c.iloc[-4])/c.iloc[-4]*100 if len(c)>4 else None
    m5 = (c.iloc[-1]-c.iloc[-6])/c.iloc[-6]*100 if len(c)>6 else None

    return {
        "Symbol"         : sym.replace(".NS",""),
        "Bucket"         : bucket,
        "Score"          : f"{l_score}/{MAX_L}",
        "Leading Signals": " · ".join(k.replace("_"," ") for k in LEADING_K if s[k]) or "—",
        "Coincident"     : " · ".join(k.replace("_"," ") for k in COINCIDENT_K if s[k]) or "—",
        "RSI"            : round(last["RSI"], 1),
        "Vol×"           : round(vr, 2),
        "Close ₹"        : round(last["Close"], 2),
        "2D %"           : round(m2, 2) if m2 else None,
        "3D %"           : round(m3, 2) if m3 else None,
        "5D %"           : round(m5, 2) if m5 else None,
        "BB Squeeze"     : "🔥" if last["BB_W"] < 0.06 else "",
        "_bucket_raw"    : bucket,
        "_l_pct"         : l_pct,
        "_big_move"      : max((abs(m2 or 0), abs(m3 or 0))) >= 8,
    }

def fetch_one(symbol):
    try:
        df = yf.download(symbol, period="60d", interval="1d",
                         auto_adjust=True, progress=False)
        if df is None or len(df) < 25:
            return symbol, None
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return symbol, df[["Open","High","Low","Close","Volume"]].dropna()
    except:
        return symbol, None

# ── Main UI ───────────────────────────────────────────────────────
st.markdown("## 📊 Nifty Smallcap 250 — Daily Signal Scanner")
st.caption(f"Universe: {len(SYMBOLS)} stocks · Data: Yahoo Finance · Indicators: RSI, MACD, Stoch, MFI, OBV, CMF, BB, EMA")

col1, col2, col3 = st.columns([1,1,4])
with col1:
    run = st.button("▶  Run Scan", use_container_width=True)
with col2:
    st.caption(f"Last run: {datetime.now().strftime('%d %b %Y')}")

# ── Auto-run or manual ────────────────────────────────────────────
should_run = run or ("results" not in st.session_state)

if should_run:
    st.session_state.results = []
    progress_bar  = st.progress(0)
    status_text   = st.empty()
    results = []
    failed  = 0

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(fetch_one, sym): sym for sym in SYMBOLS_NS}
        done = 0
        for future in as_completed(futures):
            sym, df = future.result()
            done += 1
            progress_bar.progress(done / len(SYMBOLS_NS))
            status_text.caption(f"Fetching… {done}/{len(SYMBOLS_NS)} · {sym.replace('.NS','')} · Failed: {failed}")
            if df is not None:
                res = classify(sym, df)
                if res:
                    results.append(res)
            else:
                failed += 1

    progress_bar.empty()
    status_text.empty()
    st.session_state.results = results
    st.session_state.scan_time = datetime.now()
    st.session_state.failed = failed
    st.success(f"✓ Scan complete — {len(results)} stocks analysed · {failed} failed · {datetime.now().strftime('%H:%M:%S')}")

results = st.session_state.get("results", [])

if not results:
    st.info("Click **▶ Run Scan** to start.")
    st.stop()

# ── Summary metrics ───────────────────────────────────────────────
df_all   = pd.DataFrame(results)
leading  = df_all[df_all["_bucket_raw"]=="⚡ Leading"]
coincident = df_all[df_all["_bucket_raw"]=="↔ Coincident"]
neutral  = df_all[df_all["_bucket_raw"]=="○ Neutral"]
movers   = df_all[df_all["_big_move"]==True]

m1,m2,m3,m4,m5 = st.columns(5)
m1.metric("⚡ Leading",    len(leading),    "Pre-move signals")
m2.metric("↔ Coincident", len(coincident), "Starting now")
m3.metric("○ Neutral",    len(neutral),    "No signal")
m4.metric("🚀 Big Movers", len(movers),    "8%+ in 2–3 days")
m5.metric("📡 Scanned",   len(df_all),     f"{st.session_state.get('failed',0)} failed")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────
DISPLAY_COLS = ["Symbol","Bucket","Score","Leading Signals","Coincident","RSI","Vol×","Close ₹","2D %","3D %","5D %","BB Squeeze"]

tab1, tab2, tab3, tab4 = st.tabs(["⚡ Leading", "↔ Coincident", "🚀 Big Movers", "All Stocks"])

def colour_pct(val):
    if val is None or pd.isna(val): return ""
    if val >= 8:  return "color: #e879f9; font-weight: bold"
    if val >= 3:  return "color: #22c55e"
    if val <= -3: return "color: #ef4444"
    return "color: #64748b"

def colour_bucket(val):
    if "Leading"    in str(val): return "color: #22c55e; font-weight: bold"
    if "Coincident" in str(val): return "color: #f59e0b; font-weight: bold"
    return "color: #64748b"

def show_table(df):
    if df.empty:
        st.info("No stocks in this category.")
        return
    disp = df[DISPLAY_COLS].sort_values("Score", ascending=False).reset_index(drop=True)
    styled = disp.style\
        .applymap(colour_bucket, subset=["Bucket"])\
        .applymap(colour_pct,    subset=["2D %","3D %","5D %"])\
        .format({"RSI":"{:.1f}","Vol×":"{:.2f}x","Close ₹":"₹{:.2f}",
                 "2D %":lambda v: f"+{v:.2f}%" if v and v>0 else (f"{v:.2f}%" if v else "—"),
                 "3D %":lambda v: f"+{v:.2f}%" if v and v>0 else (f"{v:.2f}%" if v else "—"),
                 "5D %":lambda v: f"+{v:.2f}%" if v and v>0 else (f"{v:.2f}%" if v else "—"),
                 })\
        .set_properties(**{"font-family":"monospace","font-size":"12px"})
    st.dataframe(styled, use_container_width=True, height=min(600, 45+len(disp)*36))

with tab1: show_table(leading)
with tab2: show_table(coincident)
with tab3: show_table(movers.sort_values("2D %", ascending=False, key=lambda x: x.abs()))
with tab4: show_table(df_all)

st.markdown("---")
st.caption("⚠️ For research purposes only · Not financial advice · Nifty Smallcap 250 list updated semi-annually")
