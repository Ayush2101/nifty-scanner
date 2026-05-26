"""
backtest.py — Exit Signal Backtester
Validates exit signals against 1 year of historical data
across known high-momentum stocks (our 17-stock study universe).

Methodology:
  1. For each stock, fetch 1Y of daily OHLCV
  2. Roll through every trading day computing all 13 exit signals
  3. For each day a signal fires, check: did the stock fall 15%+
     within the next 15 trading days?  → True Positive
  4. If it fired but stock didn't fall 15% → False Positive
  5. Find all 15%+ drops that had NO signal in prior 5 days → Missed (False Negative)
  6. Output: Precision / Recall / F1 / Avg Lead Time for each signal
"""

import yfinance as yf
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")

# ── Backtest Universe ─────────────────────────────────────────
BACKTEST_STOCKS = [
    ("LLOYDSENGG.NS", "LLOYDSENGG",   "Engineering"),
    ("HFCL.NS",       "HFCL",         "Telecom"),
    ("STLTECH.NS",    "Sterlite Tech","Telecom"),
    ("MCX.NS",        "MCX",          "Exchange"),
    ("MTARTECH.NS",   "MTAR Tech",    "Aerospace"),
    ("NCC.NS",        "NCC",          "Infra"),
    ("RKFORGE.NS",    "Ramkrishna",   "Auto Anc"),
    ("DREDGECORP.NS", "Dredging Corp","Marine"),
    ("NATIONALUM.NS", "NALCO",        "Metals"),
    ("BAJAJCON.NS",   "Bajaj Consumer","FMCG"),
    ("BLUESTARCO.NS", "Blue Star",    "Electricals"),
    ("GRINDWELL.NS",  "Grindwell",    "Abrasives"),
    ("GPIL.NS",       "GPIL",         "Steel"),
    ("TITAGARH.NS",   "Titagarh",     "Railways"),
    ("OLECTRA.NS",    "Olectra",      "EV"),
    ("KFINTECH.NS",   "KFin Tech",    "Fintech"),
    ("ROUTE.NS",      "Route Mobile", "Tech"),
]

# ── Indicator Functions ───────────────────────────────────────
def calc_rsi(series, p=14):
    d = series.diff()
    g = d.clip(lower=0); l = -d.clip(upper=0)
    ag = g.ewm(alpha=1/p, min_periods=p).mean()
    al = l.ewm(alpha=1/p, min_periods=p).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - 100/(1+rs)

def calc_macd(series):
    e12 = series.ewm(span=12, adjust=False).mean()
    e26 = series.ewm(span=26, adjust=False).mean()
    m   = e12 - e26
    s   = m.ewm(span=9, adjust=False).mean()
    return m, s, m - s

def calc_stoch(df, k=14, d=3):
    lk = df["Low"].rolling(k).min()
    hk = df["High"].rolling(k).max()
    sk = ((df["Close"]-lk)/(hk-lk).replace(0,np.nan))*100
    return sk, sk.rolling(d).mean()

def calc_mfi(df, p=14):
    tp  = (df["High"]+df["Low"]+df["Close"])/3
    mfv = tp * df["Volume"]
    pos = mfv.where(tp>tp.shift(1), 0)
    neg = mfv.where(tp<tp.shift(1), 0)
    return 100-100/(1+pos.rolling(p).sum()/neg.rolling(p).sum().replace(0,np.nan))

def calc_obv(df):
    return (np.sign(df["Close"].diff())*df["Volume"]).cumsum()

def calc_cmf(df, p=20):
    hl  = df["High"]-df["Low"]
    mfv = ((df["Close"]-df["Low"]-(df["High"]-df["Close"]))/hl.replace(0,np.nan))*df["Volume"]
    return mfv.rolling(p).sum()/df["Volume"].rolling(p).sum()

def calc_bb(series, p=20, s=2):
    mid = series.rolling(p).mean()
    std = series.rolling(p).std()
    return mid+s*std, mid, mid-s*std, (2*s*std)/mid.replace(0,np.nan)

def calc_atr(df, p=14):
    hl = df["High"]-df["Low"]
    hc = (df["High"]-df["Close"].shift(1)).abs()
    lc = (df["Low"] -df["Close"].shift(1)).abs()
    return pd.concat([hl,hc,lc],axis=1).max(axis=1).rolling(p).mean()

# ── Compute all exit signals on full DataFrame ─────────────────
def compute_signals(df):
    """Add all exit signal columns to df in one pass."""
    C, H, L, V = df["Close"], df["High"], df["Low"], df["Volume"]

    df["RSI"]                          = calc_rsi(C)
    df["MACD"], df["MSIG"], df["MHIST"]= calc_macd(C)
    df["SK"], df["SD"]                 = calc_stoch(df)
    df["MFI"]                          = calc_mfi(df)
    df["OBV"]                          = calc_obv(df)
    df["CMF"]                          = calc_cmf(df)
    df["BBU"],df["BBM"],df["BBL"],df["BBW"] = calc_bb(C)
    df["ATR"]                          = calc_atr(df)
    df["EMA21"]                        = C.ewm(span=21, adjust=False).mean()
    df["EMA55"]                        = C.ewm(span=55, adjust=False).mean()
    df["VOL_RATIO"]                    = V / V.rolling(20).mean()
    df["OBV_SLOPE5"]                   = df["OBV"].diff(5)

    # Red/Green volume ratio
    df["IS_RED"]    = (C < C.shift(1)).astype(float)
    df["RED_VOL"]   = df["Volume"] * df["IS_RED"]
    df["GREEN_VOL"] = df["Volume"] * (1 - df["IS_RED"])
    df["RED_VOL_5"] = df["RED_VOL"].rolling(5).mean()
    df["GRN_VOL_5"] = df["GREEN_VOL"].rolling(5).mean()
    df["VOL_DOWN"]  = df["RED_VOL_5"] / df["GRN_VOL_5"].replace(0,1)

    # ── Signal columns ────────────────────────────────────────
    r, rp     = df["RSI"], df["RSI"].shift(1)
    sk, skp   = df["SK"], df["SK"].shift(1)
    sd, sdp   = df["SD"], df["SD"].shift(1)
    mfi, mfip = df["MFI"], df["MFI"].shift(1)
    cmf, cmfp = df["CMF"], df["CMF"].shift(1)
    m, ms     = df["MACD"], df["MSIG"]
    mp, msp   = m.shift(1), ms.shift(1)
    mh,mhp,mhp2 = df["MHIST"], df["MHIST"].shift(1), df["MHIST"].shift(2)
    obv_sl    = df["OBV_SLOPE5"]
    bbu, bbm  = df["BBU"], df["BBM"]
    e21, e55  = df["EMA21"], df["EMA55"]
    vr        = df["VOL_RATIO"]

    # Phase 1 — Warning
    df["SIG_RSI_OB_TURN"]    = ((r > 68) & (r < rp)).astype(int)
    df["SIG_OBV_DIST"]       = (obv_sl < 0).astype(int)
    df["SIG_MFI_OB_TURN"]    = ((mfi > 65) & (mfi < mfip)).astype(int)
    df["SIG_MACD_HIST_FALL"] = ((mh < mhp) & (mhp < mhp2)).astype(int)
    df["SIG_STOCH_OB_CROSS"] = ((sk < sd) & (skp >= sdp) & (skp > 75)).astype(int)
    df["SIG_VOL_DRY"]        = ((vr < 0.7) & (C > e21)).astype(int)
    df["SIG_EMA_WEAK"]       = ((e21 > e55) & ((e21-e55)/e55 < 0.015)).astype(int)

    # Phase 2 — Confirmed
    df["SIG_P_BELOW_EMA21"]  = (C < e21).astype(int)
    df["SIG_MACD_BEAR"]      = ((m < ms) & (mp >= msp)).astype(int)
    df["SIG_CMF_NEG"]        = ((cmf < 0) & (cmfp >= 0)).astype(int)
    df["SIG_RSI_BELOW_50"]   = ((r < 50) & (rp >= 50)).astype(int)
    df["SIG_BB_REJECT"]      = ((H.shift(1) >= bbu.shift(1)*0.99) & (C < bbm)).astype(int)
    df["SIG_VOL_SPIKE_DN"]   = (df["VOL_DOWN"] > 1.5).astype(int)

    df.dropna(inplace=True)
    return df

ALL_SIGS = [
    "SIG_RSI_OB_TURN","SIG_OBV_DIST","SIG_MFI_OB_TURN","SIG_MACD_HIST_FALL",
    "SIG_STOCH_OB_CROSS","SIG_VOL_DRY","SIG_EMA_WEAK",
    "SIG_P_BELOW_EMA21","SIG_MACD_BEAR","SIG_CMF_NEG","SIG_RSI_BELOW_50",
    "SIG_BB_REJECT","SIG_VOL_SPIKE_DN",
]
SIG_NAMES = {
    "SIG_RSI_OB_TURN"   : "RSI Overbought Turn",
    "SIG_OBV_DIST"      : "OBV Distribution",
    "SIG_MFI_OB_TURN"   : "MFI Overbought Turn",
    "SIG_MACD_HIST_FALL": "MACD Hist Declining",
    "SIG_STOCH_OB_CROSS": "Stoch OB Cross Down",
    "SIG_VOL_DRY"       : "Volume Dry Up",
    "SIG_EMA_WEAK"      : "EMA Weakening",
    "SIG_P_BELOW_EMA21" : "Price Below EMA21",
    "SIG_MACD_BEAR"     : "MACD Bearish Cross",
    "SIG_CMF_NEG"       : "CMF Turned Negative",
    "SIG_RSI_BELOW_50"  : "RSI Broke Below 50",
    "SIG_BB_REJECT"     : "BB Upper Rejection",
    "SIG_VOL_SPIKE_DN"  : "Volume Spike on Down Days",
}
PHASE = {
    "SIG_RSI_OB_TURN":"⚠️ Warning","SIG_OBV_DIST":"⚠️ Warning",
    "SIG_MFI_OB_TURN":"⚠️ Warning","SIG_MACD_HIST_FALL":"⚠️ Warning",
    "SIG_STOCH_OB_CROSS":"⚠️ Warning","SIG_VOL_DRY":"⚠️ Warning",
    "SIG_EMA_WEAK":"⚠️ Warning","SIG_P_BELOW_EMA21":"🔴 Confirmed",
    "SIG_MACD_BEAR":"🔴 Confirmed","SIG_CMF_NEG":"🔴 Confirmed",
    "SIG_RSI_BELOW_50":"🔴 Confirmed","SIG_BB_REJECT":"🔴 Confirmed",
    "SIG_VOL_SPIKE_DN":"🔴 Confirmed",
}

# ── Find drawdown events ──────────────────────────────────────
def find_drop_events(df, min_drop_pct=15, min_prior_rally=40,
                     rally_window=90, drop_window=20):
    """
    Find every day that was a local peak preceding a 15%+ drop,
    where the prior rally was at least 40%.
    Returns list of peak indices.
    """
    C = df["Close"].values
    n = len(C)
    peaks = []

    for i in range(rally_window, n - drop_window):
        prior_low   = C[max(0,i-rally_window):i].min()
        future_min  = C[i:i+drop_window].min()

        # Must have rallied 40%+ to reach this point
        if C[i] / prior_low < 1 + min_prior_rally/100:
            continue
        # Must be a local high (within ±5 bars)
        local_max = C[max(0,i-5):min(n,i+6)].max()
        if C[i] < local_max * 0.97:
            continue
        # Must fall 15%+ in next 20 days
        if (future_min - C[i]) / C[i] > -min_drop_pct/100:
            continue

        peaks.append(i)

    # Deduplicate: remove peaks within 15 bars of each other
    filtered = []
    for p in peaks:
        if not filtered or p - filtered[-1] > 15:
            filtered.append(p)

    return filtered

# ── Backtest one stock ────────────────────────────────────────
def backtest_stock(symbol, name, sector,
                   signal_window=10, drop_check=15, min_drop=15,
                   min_rally=40):
    """
    For each confirmed drop event (peak), check which signals
    fired in the prior `signal_window` days.
    Returns list of event dicts.
    """
    try:
        raw = yf.download(symbol, period="1y", interval="1d",
                          auto_adjust=True, progress=False)
        if raw is None or len(raw) < 60:
            return []
        raw.columns = [c[0] if isinstance(c,tuple) else c for c in raw.columns]
        df = raw[["Open","High","Low","Close","Volume"]].dropna().copy()
        df = compute_signals(df)

        peaks = find_drop_events(df, min_drop_pct=min_drop,
                                 min_prior_rally=min_rally)
        events = []

        for peak_idx in peaks:
            if peak_idx < signal_window:
                continue

            peak_price = float(df["Close"].iloc[peak_idx])
            peak_date  = df.index[peak_idx]
            future_min = float(df["Close"].iloc[peak_idx:peak_idx+drop_check].min())
            actual_drop= (future_min - peak_price) / peak_price * 100

            # Check each signal: did it fire in last signal_window days?
            event = {
                "symbol"     : name,
                "sector"     : sector,
                "peak_date"  : str(peak_date.date()),
                "peak_price" : round(peak_price,2),
                "drop_pct"   : round(actual_drop,1),
            }

            for sig in ALL_SIGS:
                window = df[sig].iloc[peak_idx-signal_window:peak_idx]
                fired  = int(window.sum() > 0)
                # If fired: how many days before peak?
                if fired:
                    first_fire = window[window>0].index[0]
                    days_before = (df.index[peak_idx] - first_fire).days
                else:
                    days_before = None
                event[sig]              = fired
                event[sig+"_DAYS"]      = days_before

            events.append(event)

        return events

    except Exception as e:
        return []

# ── False positive test ───────────────────────────────────────
def false_positive_test(symbol, drop_check=15, drop_threshold=10):
    """
    Find days where a signal fired but stock did NOT fall 10%+
    in next 15 days. Count false positives per signal.
    """
    try:
        raw = yf.download(symbol, period="1y", interval="1d",
                          auto_adjust=True, progress=False)
        if raw is None or len(raw) < 60:
            return {}
        raw.columns = [c[0] if isinstance(c,tuple) else c for c in raw.columns]
        df = raw[["Open","High","Low","Close","Volume"]].dropna().copy()
        df = compute_signals(df)
        C  = df["Close"].values
        n  = len(C)

        fp = {s: 0 for s in ALL_SIGS}
        tp = {s: 0 for s in ALL_SIGS}

        for i in range(len(df)-drop_check):
            future_min   = C[i:i+drop_check].min()
            future_drop  = (future_min - C[i]) / C[i] * 100
            actually_fell = future_drop <= -drop_threshold

            for sig in ALL_SIGS:
                if df[sig].iloc[i] == 1:
                    if actually_fell:
                        tp[sig] += 1
                    else:
                        fp[sig] += 1

        return {"tp": tp, "fp": fp}
    except:
        return {}

# ── Main backtest runner ──────────────────────────────────────
def run_backtest(progress_callback=None):
    all_events = []
    fp_totals  = {s: 0 for s in ALL_SIGS}
    tp_totals  = {s: 0 for s in ALL_SIGS}
    total      = len(BACKTEST_STOCKS)

    for i, (sym, name, sector) in enumerate(BACKTEST_STOCKS):
        if progress_callback:
            progress_callback(i, total, name)

        events = backtest_stock(sym, name, sector)
        all_events.extend(events)

        fpr = false_positive_test(sym)
        if fpr:
            for s in ALL_SIGS:
                fp_totals[s] += fpr["fp"].get(s, 0)
                tp_totals[s] += fpr["tp"].get(s, 0)

    return all_events, fp_totals, tp_totals

# ── Aggregate results ─────────────────────────────────────────
def aggregate(events, fp_totals, tp_totals):
    if not events:
        return pd.DataFrame()

    rows = []
    for sig in ALL_SIGS:
        fires   = sum(e[sig] for e in events)
        total   = len(events)
        hit_rate= fires / total * 100 if total else 0

        # Average days before peak (when fired)
        days_list = [e[sig+"_DAYS"] for e in events
                     if e[sig]==1 and e[sig+"_DAYS"] is not None]
        avg_days  = round(np.mean(days_list),1) if days_list else None

        # Precision = TP / (TP + FP)
        tp = tp_totals.get(sig, 0)
        fp = fp_totals.get(sig, 0)
        precision = tp/(tp+fp)*100 if (tp+fp)>0 else 0

        # F1 Score
        recall = hit_rate  # using hit_rate as recall proxy
        f1 = 2*(precision*recall)/(precision+recall) if (precision+recall)>0 else 0

        rows.append({
            "Signal"         : SIG_NAMES[sig],
            "Phase"          : PHASE[sig],
            "Hit Rate %"     : round(hit_rate,1),
            "Precision %"    : round(precision,1),
            "F1 Score"       : round(f1,1),
            "Avg Lead Days"  : avg_days if avg_days else "—",
            "Fires / Total"  : f"{fires}/{total}",
            "Verdict"        : (
                "✅ Strong"   if hit_rate>=75 and precision>=60 else
                "🟡 Moderate" if hit_rate>=50 or precision>=50  else
                "❌ Weak"
            ),
            "_hit"           : hit_rate,
            "_prec"          : precision,
            "_f1"            : f1,
            "_sig"           : sig,
        })

    df = pd.DataFrame(rows).sort_values("_f1", ascending=False).reset_index(drop=True)
    return df
