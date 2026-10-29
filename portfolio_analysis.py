"""Portfolio Analysis tab: diversification, risk contribution, correlation and exposure analytics.

Risk figures come from historical weekly total returns of each fund/ETF. Exposures combine
live yfinance fund data (sectors, asset mix, top holdings) with FUND_PROFILES below for the
breakdowns yfinance does not provide (geography, market-cap mix).
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
import yfinance as yf
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

# S&P 500 benchmark styling (same as app.py), so blue always means "benchmark"
BENCHMARK_COLOR = "#2c7fb8"
# Scenario colors: adjacent categorical slots, validated as a CVD-safe pair
BASE_COLOR = "#1baf7a"
ACTIVE_COLOR = "#eb6834"
NEUTRAL_COLOR = "#8a8984"
# Ordinal blue ramp for Large -> Mid -> Small cap
CAP_COLORS = {"Large Cap": "#184f95", "Mid Cap": "#3987e5", "Small Cap": "#86b6ef",
              "Non-equity": NEUTRAL_COLOR, "Unclassified": "#c3c2b7"}

WEEKS_PER_YEAR = 52
MIN_WEEKS = 26  # funds with less history than this are left out of the risk model

SECTOR_LABELS = {
    "realestate": "Real Estate", "consumer_cyclical": "Consumer Cyclical",
    "basic_materials": "Basic Materials", "consumer_defensive": "Consumer Defensive",
    "technology": "Technology", "communication_services": "Communication Services",
    "financial_services": "Financial Services", "utilities": "Utilities",
    "industrials": "Industrials", "energy": "Energy", "healthcare": "Healthcare",
}

US, CAN, EUR, JPN, APAC, EM, GLOBAL = (
    "United States", "Canada", "Europe", "Japan", "Asia Pacific (dev. ex-Japan)",
    "Emerging Markets", "Global (non-regional)",
)
# Approximate regional split of global equities (MSCI ACWI), the geography benchmark
ACWI_GEO = {US: 0.64, EUR: 0.13, JPN: 0.05, APAC: 0.03, CAN: 0.03, EM: 0.12}
# Fallback for funds whose name/category only says "international" / "foreign"
INTL_DEVELOPED_GEO = {EUR: 0.58, JPN: 0.22, APAC: 0.10, CAN: 0.10}

# Geography (share of the whole fund) and market-cap mix (share of its equity), taken
# approximately from fund factsheets because yfinance does not expose them. "assets"
# overrides yfinance's asset mix where it is missing or too coarse (gold, crypto).
# Add a ticker here when the Data notes list it as inferred or unclassified.
FUND_PROFILES = {
    "FZROX": {"geo": {US: 1.0}, "cap": {"Large Cap": 0.72, "Mid Cap": 0.19, "Small Cap": 0.09}},
    "FNILX": {"geo": {US: 1.0}, "cap": {"Large Cap": 0.86, "Mid Cap": 0.14}},
    "BKLC": {"geo": {US: 1.0}, "cap": {"Large Cap": 0.86, "Mid Cap": 0.14}},
    "FZIPX": {"geo": {US: 1.0}, "cap": {"Large Cap": 0.05, "Mid Cap": 0.60, "Small Cap": 0.35}},
    "VTWO": {"geo": {US: 1.0}, "cap": {"Mid Cap": 0.10, "Small Cap": 0.90}},
    "FTEC": {"geo": {US: 1.0}, "cap": {"Large Cap": 0.88, "Mid Cap": 0.08, "Small Cap": 0.04}},
    "FIDU": {"geo": {US: 1.0}, "cap": {"Large Cap": 0.75, "Mid Cap": 0.20, "Small Cap": 0.05}},
    "FCLD": {"geo": {US: 0.92, EUR: 0.03, CAN: 0.03, EM: 0.02},
             "cap": {"Large Cap": 0.45, "Mid Cap": 0.35, "Small Cap": 0.20}},
    "FDRV": {"geo": {US: 0.55, EM: 0.20, EUR: 0.12, JPN: 0.10, APAC: 0.03},
             "cap": {"Large Cap": 0.55, "Mid Cap": 0.25, "Small Cap": 0.20}},
    "FDIG": {"geo": {US: 0.70, CAN: 0.12, EM: 0.08, EUR: 0.05, APAC: 0.05},
             "cap": {"Large Cap": 0.35, "Mid Cap": 0.30, "Small Cap": 0.35}},
    "ROKT": {"geo": {US: 0.95, EUR: 0.05}, "cap": {"Large Cap": 0.45, "Mid Cap": 0.30, "Small Cap": 0.25}},
    "IBBQ": {"geo": {US: 0.88, EUR: 0.10, EM: 0.02}, "cap": {"Large Cap": 0.35, "Mid Cap": 0.25, "Small Cap": 0.40}},
    "BIZD": {"geo": {US: 1.0}, "cap": {"Large Cap": 0.05, "Mid Cap": 0.45, "Small Cap": 0.50}},
    "REXC": {"geo": {APAC: 0.40, US: 0.25, EM: 0.20, CAN: 0.08, EUR: 0.07},
             "cap": {"Large Cap": 0.10, "Mid Cap": 0.30, "Small Cap": 0.60}},
    "FZILX": {"geo": {EUR: 0.42, JPN: 0.15, APAC: 0.08, CAN: 0.08, EM: 0.27},
              "cap": {"Large Cap": 0.80, "Mid Cap": 0.17, "Small Cap": 0.03}},
    "AVDV": {"geo": {EUR: 0.40, JPN: 0.26, APAC: 0.18, CAN: 0.16}, "cap": {"Mid Cap": 0.15, "Small Cap": 0.85}},
    "DXJ": {"geo": {JPN: 1.0}, "cap": {"Large Cap": 0.80, "Mid Cap": 0.18, "Small Cap": 0.02}},
    "FPADX": {"geo": {EM: 1.0}, "cap": {"Large Cap": 0.85, "Mid Cap": 0.14, "Small Cap": 0.01}},
    "AGG": {"geo": {US: 1.0}},
    "PFIAX": {"geo": {US: 0.70, EUR: 0.15, EM: 0.15}, "assets": {"Bonds": 1.0}},
    "IAUM": {"geo": {GLOBAL: 1.0}, "assets": {"Gold / Commodities": 1.0}},
    "BTC": {"geo": {GLOBAL: 1.0}, "assets": {"Crypto": 1.0}},
    "ETH": {"geo": {GLOBAL: 1.0}, "assets": {"Crypto": 1.0}},
}

LOOKBACKS = {"1 year": 1, "3 years": 3, "5 years": 5}


# ---------------------------------------------------------------------------
# Data fetching (cached: yfinance is slow and these change at most daily)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_weekly_returns(tickers, start, end):
    """Weekly total returns (dividends reinvested) for each ticker, one column per ticker."""
    try:
        px = yf.download(list(tickers), start=start, end=end, progress=False, auto_adjust=True)
    except Exception:
        return pd.DataFrame()
    if px.empty or "Close" not in px.columns.get_level_values(0):
        return pd.DataFrame()
    close = px["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)
    # Weekly sampling avoids the stale-close bias daily data has across time zones
    weekly = close.resample("W-FRI").last()
    return weekly.pct_change(fill_method=None).iloc[1:]


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def fetch_fund_profile(ticker):
    """Name, category, sector weights, asset mix and top holdings from yfinance."""
    out = {"name": ticker, "category": None, "quote_type": None, "sectors": {}, "assets": {},
           "holdings": [], "country": None, "market_cap": None}
    t = yf.Ticker(ticker)
    try:
        info = t.info or {}
        out["name"] = info.get("longName") or info.get("shortName") or ticker
        out["category"] = info.get("category")
        out["quote_type"] = info.get("quoteType")
        out["country"] = info.get("country")
        out["market_cap"] = info.get("marketCap")
        if out["quote_type"] == "EQUITY" and info.get("sector"):
            out["sectors"] = {info["sector"]: 1.0}
    except Exception:
        pass
    if out["quote_type"] == "EQUITY":
        return out
    try:
        fd = t.funds_data
        out["sectors"] = {SECTOR_LABELS.get(k, k): float(v) for k, v in (fd.sector_weightings or {}).items() if v}
        out["assets"] = {k: float(v) for k, v in (fd.asset_classes or {}).items() if v}
        th = fd.top_holdings
        if th is not None and not th.empty:
            out["holdings"] = [(sym, row["Name"], float(row["Holding Percent"])) for sym, row in th.iterrows()]
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# Risk model
# ---------------------------------------------------------------------------

def annualized_covariance(rets):
    """Pairwise annualized covariance, nudged to the nearest positive semi-definite matrix.

    Pairwise estimation keeps the full lookback for long-lived funds even when a newer
    fund has a shorter history; that can leave tiny negative eigenvalues, which are clipped.
    """
    cov = rets.cov(min_periods=MIN_WEEKS).fillna(0.0) * WEEKS_PER_YEAR
    vals, vecs = np.linalg.eigh(cov.values)
    if vals.min() < 0:
        cov = pd.DataFrame(vecs @ np.diag(np.clip(vals, 0, None)) @ vecs.T, index=cov.index, columns=cov.columns)
    return cov


def annualized_returns(rets):
    """Geometric annualized return of each column over its own available history."""
    out = {}
    for col in rets.columns:
        r = rets[col].dropna()
        out[col] = (1 + r).prod() ** (WEEKS_PER_YEAR / len(r)) - 1 if len(r) else np.nan
    return pd.Series(out)


def scenario_risk(weights, cov, betas, mu, spy_vol, rf_rate):
    """Portfolio-level metrics plus per-fund marginal and total risk contributions."""
    w = weights.reindex(cov.index).fillna(0.0)
    S = cov.values
    wv = w.values
    vols = np.sqrt(np.diag(S))
    port_vol = float(np.sqrt(wv @ S @ wv))
    mcr = S @ wv / port_vol if port_vol > 0 else np.zeros_like(wv)
    rc = wv * mcr
    port_beta = float(np.nansum(wv * betas.reindex(cov.index).values))
    port_ret = float(np.nansum(wv * mu.reindex(cov.index).values))
    weighted_vol = float(wv @ vols)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr_to_port = np.where(vols > 0, mcr / vols, np.nan)
    funds = pd.DataFrame({
        "Weight": wv,
        "Volatility": vols,
        "Beta": betas.reindex(cov.index).values,
        "CorrToPortfolio": corr_to_port,
        "MCR": mcr,
        "RiskContribution": rc / port_vol if port_vol > 0 else 0.0,
    }, index=cov.index)
    summary = {
        "Volatility": port_vol,
        "WeightedVol": weighted_vol,
        "DiversificationRatio": weighted_vol / port_vol if port_vol > 0 else np.nan,
        "Beta": port_beta,
        "MarketRisk": float(abs(port_beta) * spy_vol),
        "Return": port_ret,
        "Sharpe": (port_ret - rf_rate) / port_vol if port_vol > 0 else np.nan,
        "EffectiveHoldings": 1 / float((wv ** 2).sum()) if wv.any() else np.nan,
    }
    return summary, funds


def risk_signal(row):
    if row["Weight"] <= 0:
        return ""
    ratio = row["RiskContribution"] / row["Weight"]
    if ratio >= 1.25:
        return "⚠️ Trim candidate: adds more risk than its weight"
    if ratio <= 0.75 or row["CorrToPortfolio"] < 0.3:
        return "✅ Diversifier: adds less risk than its weight"
    return "In line with its weight"


# ---------------------------------------------------------------------------
# Exposures
# ---------------------------------------------------------------------------

def _text(prof):
    return f"{prof.get('category') or ''} {prof.get('name') or ''}".lower()


def asset_mix(ticker, prof):
    """Asset-class split of a fund; returns (mix, source) where source is 'profile', 'yfinance' or 'inferred'."""
    if "assets" in FUND_PROFILES.get(ticker, {}):
        return FUND_PROFILES[ticker]["assets"], "profile"
    if prof["quote_type"] == "EQUITY":
        return {"Equity": 1.0}, "yfinance"
    text = _text(prof)
    a = prof["assets"]
    other_label = "Crypto" if any(k in text for k in ("digital asset", "bitcoin", "ether")) else (
        "Gold / Commodities" if any(k in text for k in ("commodit", "gold", "precious")) else "Other")
    mix = {
        "Equity": a.get("stockPosition", 0.0),
        "Bonds": a.get("bondPosition", 0.0) + a.get("preferredPosition", 0.0) + a.get("convertiblePosition", 0.0),
        "Cash": a.get("cashPosition", 0.0),
        other_label: a.get("otherPosition", 0.0),
    }
    # Leveraged/hedged funds report small negative cash; drop it rather than net it off
    mix = {k: max(v, 0.0) for k, v in mix.items()}
    total = sum(mix.values())
    if total > 0:
        return {k: v / total for k, v in mix.items() if v > 0}, "yfinance"
    if other_label != "Other":
        return {other_label: 1.0}, "inferred"
    if "bond" in text or "treasury" in text or "fixed income" in text:
        return {"Bonds": 1.0}, "inferred"
    return {"Unclassified": 1.0}, "unclassified"


def geography_mix(ticker, prof):
    if "geo" in FUND_PROFILES.get(ticker, {}):
        return FUND_PROFILES[ticker]["geo"], "profile"
    if prof["quote_type"] == "EQUITY" and prof.get("country"):
        country = prof["country"]
        region = {"United States": US, "Canada": CAN, "Japan": JPN}.get(country)
        if region is None:
            region = EM if country in ("China", "India", "Brazil", "Taiwan", "South Korea", "Mexico") else EUR
        return {region: 1.0}, "yfinance"
    text = _text(prof)
    if "japan" in text:
        return {JPN: 1.0}, "inferred"
    if any(k in text for k in ("emerging", "china", "india")):
        return {EM: 1.0}, "inferred"
    if "europe" in text:
        return {EUR: 1.0}, "inferred"
    if any(k in text for k in ("foreign", "international", "ex-us", "ex us")):
        return dict(INTL_DEVELOPED_GEO), "inferred"
    return {"Unclassified": 1.0}, "unclassified"


def cap_mix(ticker, prof):
    """Market-cap split of the fund's equity sleeve."""
    if "cap" in FUND_PROFILES.get(ticker, {}):
        return FUND_PROFILES[ticker]["cap"], "profile"
    if prof["quote_type"] == "EQUITY" and prof.get("market_cap"):
        mc = prof["market_cap"]
        return {"Large Cap" if mc >= 10e9 else "Mid Cap" if mc >= 2e9 else "Small Cap": 1.0}, "yfinance"
    cat = (prof.get("category") or "").lower()
    if "small/mid" in cat or "smid" in cat:
        return {"Mid Cap": 0.5, "Small Cap": 0.5}, "inferred"
    for word, label in (("large", "Large Cap"), ("mid", "Mid Cap"), ("small", "Small Cap")):
        if word in cat:
            return {label: 1.0}, "inferred"
    return {"Unclassified": 1.0}, "unclassified"


def _accumulate(weights, mix_fn):
    total = {}
    for t, w in weights.items():
        for k, v in mix_fn(t).items():
            total[k] = total.get(k, 0.0) + w * v
    return pd.Series(total, dtype=float)


def compute_exposures(weights, profiles):
    """All exposure breakdowns for one set of weights (index: ticker, values sum to 1)."""
    equity = {t: asset_mix(t, profiles[t])[0].get("Equity", 0.0) for t in weights.index}

    assets = _accumulate(weights, lambda t: asset_mix(t, profiles[t])[0])

    def sector_mix(t):
        s = profiles[t]["sectors"]
        tot = sum(s.values())
        return {k: equity[t] * v / tot for k, v in s.items()} if tot > 0 else (
            {"Unclassified": equity[t]} if equity[t] > 0 else {})
    sectors = _accumulate(weights, sector_mix)
    sectors = sectors / sectors.sum() if sectors.sum() > 0 else sectors

    geo = _accumulate(weights, lambda t: geography_mix(t, profiles[t])[0])
    regional = geo.drop(GLOBAL, errors="ignore")
    regional = regional / regional.sum() if regional.sum() > 0 else regional

    cap = _accumulate(weights, lambda t: {k: equity[t] * v for k, v in cap_mix(t, profiles[t])[0].items()})
    non_equity = 1.0 - sum(w * equity[t] for t, w in weights.items())
    if non_equity > 1e-4:
        cap["Non-equity"] = non_equity

    holdings = {}
    for t, w in weights.items():
        for sym, name, pct in profiles[t]["holdings"]:
            h = holdings.setdefault(sym, {"Name": name, "Weight": 0.0, "Via": []})
            h["Weight"] += w * pct
            h["Via"].append(f"{t} {w * pct:.2%}")
    holdings = pd.DataFrame.from_dict(holdings, orient="index")

    return {
        "assets": assets, "sectors": sectors, "geo": regional,
        "global_share": float(geo.get(GLOBAL, 0.0)), "cap": cap, "holdings": holdings,
    }


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _is_dark():
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


def exposure_chart(series, title, benchmark=None, benchmark_name=None, top_n=None):
    """Horizontal bars per category for each scenario, with an optional benchmark tick per row.

    series: list of (name, color, pd.Series of fractions). Rows follow the first series, largest first.
    """
    order = series[0][2].sort_values(ascending=False)
    order = order[order > 0.0005]  # hide rounding-dust rows like 0.0% cash
    if top_n:
        order = order.head(top_n)
    labels = list(order.index)
    positions = list(range(len(labels)))
    fig = go.Figure()
    for name, color, s in series:
        vals = s.reindex(labels).fillna(0.0) * 100
        fig.add_trace(go.Bar(
            x=vals, y=positions, orientation="h", name=name, marker=dict(color=color, line=dict(width=0)),
            text=[f"{v:.1f}%" for v in vals], textposition="outside", cliponaxis=False,
            customdata=labels, hovertemplate="%{customdata}<br>" + name + ": %{x:.1f}%<extra></extra>",
        ))
    if benchmark is not None:
        for i, lab in enumerate(labels):
            b = benchmark.get(lab)
            if b is None or pd.isna(b):
                continue
            fig.add_shape(type="line", x0=b * 100, x1=b * 100, y0=i - 0.45, y1=i + 0.45,
                          line=dict(color=BENCHMARK_COLOR, width=2))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=benchmark_name,
                                 line=dict(color=BENCHMARK_COLOR, width=2)))
    peaks = [s.max() for _, _, s in series if not s.empty] + ([max(benchmark.values())] if benchmark else [])
    max_x = (np.nanmax(peaks) if peaks else 0.0) * 100
    fig.update_yaxes(type="category", tickvals=positions, ticktext=labels, autorange="reversed", showgrid=False)
    fig.update_layout(
        title=title, barmode="group", bargap=0.25, bargroupgap=0.05,
        height=max(260, (28 + 14 * len(series)) * len(labels) + 140),
        xaxis=dict(ticksuffix="%", range=[0, max_x * 1.15 + 1], showgrid=True),
        margin=dict(l=10, r=20, t=80, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        showlegend=len(series) > 1 or benchmark is not None,
    )
    return fig


def market_cap_chart(scenarios):
    order = ["Large Cap", "Mid Cap", "Small Cap", "Non-equity", "Unclassified"]
    rows = [name for name, _ in scenarios]
    fig = go.Figure()
    for seg in order:
        vals = [float(cap.get(seg, 0.0)) * 100 for _, cap in scenarios]
        if not any(v > 0.05 for v in vals):
            continue
        dark_fill = seg in ("Large Cap", "Mid Cap", "Non-equity")
        fig.add_trace(go.Bar(
            x=vals, y=rows, orientation="h", name=seg,
            marker=dict(color=CAP_COLORS[seg], line=dict(width=2, color="rgba(0,0,0,0)")),
            text=[f"{v:.1f}%<br>{seg}" if v >= 6 else "" for v in vals], textposition="inside",
            insidetextanchor="middle", insidetextfont=dict(color="#ffffff" if dark_fill else "#0b0b0b"),
            hovertemplate="%{y} · " + seg + ": %{x:.1f}%<extra></extra>",
        ))
    fig.update_layout(
        barmode="stack", height=110 + 70 * len(rows), title="Market Cap Exposure (% of portfolio)",
        xaxis=dict(range=[0, 100], ticksuffix="%", showgrid=False),
        yaxis=dict(autorange="reversed"), margin=dict(l=10, r=20, t=80, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    return fig


def correlation_heatmap(corr):
    # Cluster so funds that move together sit next to each other
    if len(corr) > 2:
        dist = squareform(np.clip(1 - corr.values, 0, 2), checks=False)
        order = corr.index[leaves_list(linkage(dist, method="average"))]
        corr = corr.loc[order, order]
    mid = "#383835" if _is_dark() else "#f0efec"
    n = len(corr)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=list(corr.columns), y=list(corr.index), zmin=-1, zmax=1,
        colorscale=[[0, "#2a78d6"], [0.5, mid], [1, "#e34948"]],
        texttemplate="%{z:.2f}", textfont=dict(size=11 if n <= 16 else 9),
        hovertemplate="%{y} vs %{x}<br>Correlation: %{z:.2f}<extra></extra>",
        colorbar=dict(title="ρ", thickness=12), xgap=2, ygap=2,
    ))
    fig.update_layout(
        title="Return Correlation (weekly, clustered)", height=max(420, 34 * n + 180),
        xaxis=dict(tickangle=-45, side="bottom"), yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


# ---------------------------------------------------------------------------
# Tab
# ---------------------------------------------------------------------------

def _pct(x, digits=1):
    return f"{x * 100:.{digits}f}%" if pd.notna(x) else "n/a"


def render_portfolio_analysis(holdings, name_map, rf_rate, cash_value=0.0):
    """holdings: {symbol: current market value} of funds still held (cash excluded)."""
    st.header("Portfolio Analysis")
    st.caption(
        "Where to add and where to trim. **Base** = your current holdings. **Active** = the weights you "
        "set below (trim, add, or try new tickers). Risk figures use historical weekly total returns, so "
        "they describe how these funds have behaved, not a forecast."
    )
    if not holdings:
        st.info("No current holdings to analyse.")
        return

    total_value = sum(holdings.values())
    base = pd.Series(holdings, dtype=float) / total_value

    lookback = st.selectbox("Risk lookback window", list(LOOKBACKS), index=1)

    # --- Scenario editor ---
    st.subheader("Scenario Weights")
    editor_df = pd.DataFrame({
        "Ticker": base.index,
        "Name": [name_map.get(t, t) for t in base.index],
        "Value ($)": [holdings[t] for t in base.index],
        "Base %": (base * 100).round(2).values,
        "Active %": (base * 100).round(2).values,
    }).sort_values("Value ($)", ascending=False, ignore_index=True)
    edited = st.data_editor(
        editor_df, key="active_weights", num_rows="dynamic", hide_index=True, width="stretch",
        disabled=["Name", "Value ($)", "Base %"],
        column_config={
            "Value ($)": st.column_config.NumberColumn(format="$%.0f"),
            "Base %": st.column_config.NumberColumn(format="%.2f"),
            "Active %": st.column_config.NumberColumn(format="%.2f", min_value=0.0, max_value=100.0,
                                                      help="Set to 0 to model selling the fund"),
        },
    )
    st.caption(f"Cash / money market (${cash_value:,.0f}) is excluded. Add a row with any ticker to test "
               "a new fund; Active weights are scaled to sum to 100%.")
    edited = edited.dropna(subset=["Ticker"])
    edited["Ticker"] = edited["Ticker"].astype(str).str.strip().str.upper()
    edited = edited[(edited["Ticker"] != "") & (edited["Active %"].fillna(0) > 0)]
    active = edited.groupby("Ticker")["Active %"].sum()
    if active.sum() <= 0:
        st.warning("Active weights are all zero; using Base weights for Active.")
        active = base.copy()
    else:
        if abs(active.sum() - 100) > 0.05:
            st.caption(f"Active weights summed to {active.sum():.1f}% and were scaled to 100%.")
        active = active / active.sum()

    tickers = sorted(set(base.index) | set(active.index))

    # --- Historical returns ---
    end = datetime.today()
    start = end - timedelta(days=365 * LOOKBACKS[lookback] + 10)
    with st.spinner("Downloading price history..."):
        rets = fetch_weekly_returns(tuple(tickers + ["SPY"]), start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    if rets.empty:
        st.error("Could not download price history for the risk model.")
        return
    counts = rets.notna().sum()
    usable = [t for t in tickers if counts.get(t, 0) >= MIN_WEEKS]
    skipped = [t for t in tickers if t not in usable]
    if skipped:
        st.warning(f"Not enough price history (< {MIN_WEEKS} weeks) for: {', '.join(skipped)}. "
                   "They are left out of the risk figures; weights are rescaled over the rest.")
    short = [f"{t} ({counts[t]} wks)" for t in usable if counts[t] < counts.max() - 4]
    if short:
        st.caption("Shorter history than the window (statistics use what is available): " + ", ".join(short))
    if not usable:
        st.error("No fund has enough price history for the risk model.")
        return

    cols = usable + (["SPY"] if counts.get("SPY", 0) >= MIN_WEEKS else [])
    cov_all = annualized_covariance(rets[cols])
    cov = cov_all.loc[usable, usable]
    if "SPY" in cov_all:
        spy_var = cov_all.loc["SPY", "SPY"]
        betas = cov_all.loc[usable, "SPY"] / spy_var
        spy_vol = float(np.sqrt(spy_var))
    else:
        betas, spy_vol = pd.Series(np.nan, index=usable), np.nan
    mu = annualized_returns(rets[usable])

    def norm(w):
        w = w.reindex(usable).fillna(0.0)
        return w / w.sum() if w.sum() > 0 else w

    base_w, active_w = norm(base), norm(active)
    has_active = not np.allclose(base_w.values, active_w.values, atol=1e-4)
    base_sum, base_funds = scenario_risk(base_w, cov, betas, mu, spy_vol, rf_rate)
    act_sum, act_funds = scenario_risk(active_w, cov, betas, mu, spy_vol, rf_rate)
    scenarios = [("Base", BASE_COLOR, base_w)] + ([("Active", ACTIVE_COLOR, active_w)] if has_active else [])

    # --- Diversification ratio & summary ---
    st.subheader("Diversification Ratio")
    summary_rows = [
        ("Diversification Ratio", lambda s: f"{s['DiversificationRatio']:.2f}"),
        ("Portfolio Volatility", lambda s: _pct(s["Volatility"])),
        ("Beta vs S&P 500", lambda s: f"{s['Beta']:.2f}"),
        ("Historical Return (annualized)", lambda s: _pct(s["Return"])),
        ("Sharpe Ratio", lambda s: f"{s['Sharpe']:.2f}"),
        ("Effective No. of Holdings", lambda s: f"{s['EffectiveHoldings']:.1f}"),
    ]
    summary = pd.DataFrame(
        {"Base": [f(base_sum) for _, f in summary_rows], "Active": [f(act_sum) for _, f in summary_rows]},
        index=[n for n, _ in summary_rows],
    )
    st.dataframe(summary, width="stretch")
    st.markdown("📈 **Guide:** 1 = no benefit, 1.1–1.3 low, 1.3–1.6 moderate, 1.6–2.0 good, >2.0 excellent. "
                "Effective no. of holdings = 1 / Σ weight², i.e. how many equal-sized funds the "
                "concentration is equivalent to.")

    # --- Diversification benefit ---
    st.subheader("Diversification Benefit")
    measures = ["Weighted Avg. Fund Volatility", "Portfolio Volatility", "Undiversifiable (Market) Risk"]
    keys = ["WeightedVol", "Volatility", "MarketRisk"]
    fig = go.Figure()
    for name, color, _ in scenarios:
        s = base_sum if name == "Base" else act_sum
        vals = [s[k] * 100 for k in keys]
        fig.add_trace(go.Bar(x=measures, y=vals, name=name, marker=dict(color=color, line=dict(width=0)),
                             text=[f"{v:.2f}%" for v in vals], textposition="outside", cliponaxis=False,
                             hovertemplate="%{x}<br>" + name + ": %{y:.2f}%<extra></extra>"))
    fig.update_layout(barmode="group", height=420, yaxis=dict(title="Annualized Std Dev (%)", ticksuffix="%"),
                      margin=dict(t=40), showlegend=has_active,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0))
    st.plotly_chart(fig, width="stretch")
    st.caption("The gap between the first two bars is the risk removed by diversification. "
               "Undiversifiable risk = |portfolio beta| × S&P 500 volatility: the market risk no amount "
               "of fund-mixing within equities removes.")

    # --- Risk contribution ---
    st.subheader("Marginal Contribution to Risk")
    view = st.radio("Scenario", ["Base", "Active"] if has_active else ["Base"], horizontal=True, key="risk_view")
    funds = (base_funds if view == "Base" else act_funds).copy()
    funds = funds[funds["Weight"] > 0].sort_values("RiskContribution", ascending=False)
    color = BASE_COLOR if view == "Base" else ACTIVE_COLOR

    fig = go.Figure()
    fig.add_trace(go.Bar(x=funds.index, y=funds["Weight"] * 100, name="Portfolio weight",
                         marker=dict(color=NEUTRAL_COLOR, line=dict(width=0)),
                         hovertemplate="%{x}<br>Weight: %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Bar(x=funds.index, y=funds["RiskContribution"] * 100, name="Share of portfolio risk",
                         marker=dict(color=color, line=dict(width=0)),
                         hovertemplate="%{x}<br>Risk contribution: %{y:.1f}%<extra></extra>"))
    fig.update_layout(title="Weight vs Share of Portfolio Risk", barmode="group", height=420,
                      yaxis=dict(title="% of portfolio", ticksuffix="%"), xaxis=dict(title="Ticker"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0))
    st.plotly_chart(fig, width="stretch")

    mcr_sorted = funds.sort_values("MCR", ascending=False)
    fig = go.Figure(go.Bar(x=mcr_sorted.index, y=mcr_sorted["MCR"] * 100, marker=dict(color=color, line=dict(width=0)),
                           hovertemplate="%{x}<br>MCR: %{y:.2f}% vol per 100% weight<extra></extra>"))
    fig.update_layout(title="Marginal Contribution to Risk (∂σ / ∂w)", height=380,
                      yaxis=dict(title="MCR (% vol)", ticksuffix="%"), xaxis=dict(title="Ticker"), margin=dict(t=60))
    st.plotly_chart(fig, width="stretch")
    st.caption("MCR = how much portfolio volatility rises per unit of extra weight in that fund. "
               "Funding a purchase by trimming a high-MCR fund and adding to a low-MCR fund lowers total risk.")

    st.markdown("**Rebalancing signals**")
    signals = pd.DataFrame({
        "Name": [name_map.get(t, t) for t in funds.index],
        "Weight": funds["Weight"].map(_pct),
        "Share of Risk": funds["RiskContribution"].map(_pct),
        "Risk / Weight": (funds["RiskContribution"] / funds["Weight"]).map(lambda x: f"{x:.2f}"),
        "Volatility": funds["Volatility"].map(_pct),
        "Beta": funds["Beta"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "n/a"),
        "Corr. to Portfolio": funds["CorrToPortfolio"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "n/a"),
        "Hist. Return": mu.reindex(funds.index).map(_pct),
        "Signal": funds.apply(risk_signal, axis=1),
    }, index=funds.index)
    st.dataframe(signals, width="stretch")
    st.caption("Risk / Weight above 1 means the fund adds more risk than its size. Pair this with Hist. Return: "
               "a diversifier with a weak return may still not be worth adding.")

    # --- Correlation ---
    st.subheader("Correlation Matrix")
    corr_tickers = [t for t in usable if base_w.get(t, 0) > 0 or active_w.get(t, 0) > 0]
    vols = np.sqrt(np.diag(cov.loc[corr_tickers, corr_tickers]))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov.loc[corr_tickers, corr_tickers] / np.outer(vols, vols)
    st.plotly_chart(correlation_heatmap(corr.fillna(0.0)), width="stretch")
    st.caption("Red = funds move together (little diversification between them); "
               "blue = they offset each other.")

    # --- Exposures ---
    exposure_tickers = sorted(set(base.index) | set(active.index))
    with st.spinner("Loading fund profiles..."):
        profiles = {t: fetch_fund_profile(t) for t in exposure_tickers}
        spy_profile = fetch_fund_profile("SPY")
    base_exp = compute_exposures(base, profiles)
    exp_scen = [("Base", BASE_COLOR, base_exp)]
    if has_active:
        exp_scen.append(("Active", ACTIVE_COLOR, compute_exposures(active, profiles)))

    st.subheader("Asset Class Exposure")
    st.plotly_chart(exposure_chart([(n, c, e["assets"]) for n, c, e in exp_scen], "Asset Class (% of portfolio)"),
                    width="stretch")

    st.subheader("Sector Exposure")
    spy_sectors = spy_profile["sectors"]
    spy_total = sum(spy_sectors.values())
    spy_sectors = {k: v / spy_total for k, v in spy_sectors.items()} if spy_total else None
    eq_share = base_exp["assets"].get("Equity", 0.0)
    st.plotly_chart(exposure_chart([(n, c, e["sectors"]) for n, c, e in exp_scen], "Sector (% of equity holdings)",
                                   benchmark=spy_sectors, benchmark_name="S&P 500"), width="stretch")
    st.caption(f"Equity is {eq_share:.1%} of the Base portfolio; sectors are shown as a share of that equity "
               "so they compare directly with the S&P 500.")

    st.subheader("Geographic Exposure")
    st.plotly_chart(exposure_chart([(n, c, e["geo"]) for n, c, e in exp_scen], "Region (% of regional holdings)",
                                   benchmark=ACWI_GEO, benchmark_name="Global equity market (MSCI ACWI, approx.)"),
                    width="stretch")
    st.caption(f"{base_exp['global_share']:.1%} of the Base portfolio (crypto, gold) has no region and is excluded "
               "from this chart.")

    st.subheader("Market Cap Exposure")
    st.plotly_chart(market_cap_chart([(n, e["cap"]) for n, _, e in exp_scen]), width="stretch")

    st.subheader("Top Underlying Holdings (look-through)")
    h = base_exp["holdings"]
    if h.empty:
        st.info("No holdings data available for these funds.")
    else:
        h = h.sort_values("Weight", ascending=False).head(15)
        labels = [f"{n} ({s})" for s, n in zip(h.index, h["Name"])]
        fig = go.Figure(go.Bar(
            x=h["Weight"] * 100, y=labels, orientation="h", marker=dict(color=BASE_COLOR, line=dict(width=0)),
            text=[f"{v:.2f}%" for v in h["Weight"] * 100], textposition="outside", cliponaxis=False,
            customdata=["<br>".join(v) for v in h["Via"]],
            hovertemplate="%{y}: %{x:.2f}% of portfolio<br>via:<br>%{customdata}<extra></extra>",
        ))
        fig.update_layout(title="Largest Stocks Held Through Your Funds (Base, % of portfolio)",
                          height=30 * len(h) + 140, yaxis=dict(autorange="reversed"),
                          xaxis=dict(ticksuffix="%", range=[0, h["Weight"].max() * 115 + 0.2]),
                          margin=dict(l=10, r=20, t=60, b=40))
        st.plotly_chart(fig, width="stretch")
        st.caption("Built from each fund's top-10 holdings only, so it understates the totals. "
                   "Hover a bar to see which funds hold the stock. The same name across several funds is overlap.")

    with st.expander("Data notes: where the exposure data comes from"):
        rows = []
        for t in exposure_tickers:
            p = profiles[t]
            rows.append({
                "Ticker": t, "Name": p["name"], "Category": p["category"] or "",
                "Sectors": "yfinance" if p["sectors"] else "none",
                "Asset mix": asset_mix(t, p)[1], "Geography": geography_mix(t, p)[1],
                "Market cap": cap_mix(t, p)[1],
            })
        st.dataframe(pd.DataFrame(rows).set_index("Ticker"), width="stretch")
        st.caption("'profile' = approximate factsheet figures kept in FUND_PROFILES (portfolio_analysis.py); "
                   "'inferred' = guessed from the fund's name or category; 'unclassified' = no data. "
                   "Add an entry to FUND_PROFILES to fix inferred or unclassified funds.")
