import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import newton
import plotly.graph_objs as go

def xnpv(rate, cashflows):
    t0 = cashflows[0][0]
    return sum([cf / (1 + rate) ** ((t - t0).days / 365) for t, cf in cashflows])

def xirr(cashflows):
    try:
        return newton(lambda r: xnpv(r, cashflows), 0.1)
    except Exception:
        return np.nan

def read_uploaded_files(files):
    dfs = []
    for file in files:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def preprocess_trades(df):
    df.columns = [col.strip() for col in df.columns]
    df["Run Date"] = pd.to_datetime(df["Run Date"])
    df = df[df["Action"].notna()]
    df = df[df["Action"].str.contains("YOU BOU|YOU SOLD|DIVIDEND", case=False, na=False)]
    return df

def get_symbol_list(df):
    return df["Symbol"].unique()

def get_price_history(symbol, start, end):
    px = yf.download(symbol, start=start, end=end)
    return px["Close"].dropna()

def get_cashflows_for_xirr(trades, prices, end_date):
    cf = []
    qty = 0
    for _, row in trades.iterrows():
        amt = float(row["Amount ($)"])
        if "YOU BOU" in row["Action"]:
            cf.append((row["Run Date"], amt))
            qty += row["Quantity"]
        elif "YOU SOLD" in row["Action"]:
            cf.append((row["Run Date"], amt))
            qty -= row["Quantity"]
        elif "DIVIDEND" in row["Action"]:
            cf.append((row["Run Date"], amt))
    if qty > 0 and end_date in prices.index:
        cf.append((end_date, float(qty * float(prices.loc[end_date]))))
    return cf

def get_portfolio_value_curve(df, prices):
    all_days = pd.date_range(prices.index.min(), prices.index.max(), freq='D')
    trades = df.sort_values('Run Date').set_index('Run Date')
    qty = 0
    cost = 0
    realized = 0
    value_curve = []
    for date in all_days:
        if date in trades.index:
            for _, row in trades.loc[[date]].iterrows():
                amt = float(row["Amount ($)"])
                if "YOU BOU" in row["Action"]:
                    qty += row["Quantity"]
                    cost += abs(amt)
                elif "YOU SOLD" in row["Action"]:
                    sell_qty = row["Quantity"]
                    if qty == 0:
                        continue
                    avg_cost = cost / qty if qty > 0 else 0
                    realized += (row["Price ($)"] - avg_cost) * sell_qty
                    cost -= avg_cost * sell_qty
                    qty -= sell_qty
                elif "DIVIDEND" in row["Action"]:
                    realized += amt
        mkt_val = prices.loc[date] * qty if date in prices.index else np.nan
        value_curve.append({
            "Date": date,
            "PortfolioValue": mkt_val + realized if not np.isnan(mkt_val) else np.nan,
            "Realized": realized,
            "Unrealized": (mkt_val - cost) if qty > 0 and not np.isnan(mkt_val) else 0,
            "TotalProfit": (mkt_val - cost + realized) if qty > 0 and not np.isnan(mkt_val) else realized
        })
    return pd.DataFrame(value_curve).set_index("Date")

def get_sp500_shadow_value_curve(trades, sp500_prices):
    all_days = pd.date_range(sp500_prices.index.min(), sp500_prices.index.max(), freq='D')
    trades = trades.sort_values('Run Date').set_index('Run Date')
    units = 0.0
    realized = 0.0
    value_curve = []
    for date in all_days:
        if date in trades.index and date in sp500_prices.index:
            for _, row in trades.loc[[date]].iterrows():
                amt = float(row["Amount ($)"])
                px = float(sp500_prices.loc[date])
                if "YOU BOU" in row["Action"]:
                    units += abs(amt) / px
                elif "YOU SOLD" in row["Action"]:
                    units -= row["Quantity"]
                elif "DIVIDEND" in row["Action"]:
                    realized += amt
        mkt_val = float(sp500_prices.loc[date]) * float(units) if date in sp500_prices.index else np.nan
        value_curve.append({
            "Date": date,
            "S&P500Value": mkt_val + realized if not np.isnan(mkt_val) else np.nan
        })
    return pd.DataFrame(value_curve).set_index("Date")

def compute_period_returns(portfolio_value, freq='M'):
    period_val = portfolio_value.resample(freq).last().dropna()
    returns = period_val.pct_change().dropna()
    return returns

def main():
    st.title("Portfolio vs S&P500 Analysis")

    uploaded_files = st.file_uploader(
        "Upload one or more files (Excel/CSV) from Fidelity.",
        type=["xls", "xlsx", "csv"],
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("Please upload your trade files.")
        return

    df = read_uploaded_files(uploaded_files)
    df = preprocess_trades(df)
    symbols = get_symbol_list(df)
    t0 = df["Run Date"].min() - pd.Timedelta(days=5)
    t1 = datetime.today()

    # --- 1. Portfolio Level Metrics ---
    st.header("1. Portfolio-level Metrics")

    all_prices = {}
    for symbol in symbols:
        all_prices[symbol] = get_price_history(symbol, t0, t1)
    sp500_prices = get_price_history("^GSPC", t0, t1)

    cf_all = []
    for symbol in symbols:
        trades = df[df["Symbol"] == symbol]
        px = all_prices[symbol]
        cf_all.extend(get_cashflows_for_xirr(trades, px, sp500_prices.index[-1]))
    cf_all = sorted(cf_all, key=lambda x: x[0])
    cf_all_scalar = [(d, float(a)) for d, a in cf_all]

    # S&P500 shadow XIRR: same cashflows, but applied to S&P500, final sale at last price
    cf_sp500 = []
    units = 0.0
    for date, amt in cf_all_scalar:
        px_idx = sp500_prices.index[sp500_prices.index <= date]
        if len(px_idx) == 0:
            continue
        px = float(sp500_prices.loc[px_idx.max()])
        amt_float = float(amt)
        if amt_float < 0:
            units += abs(amt_float) / px
            cf_sp500.append((date, amt_float))
        else:
            cf_sp500.append((date, amt_float))
    units_held = float(units) if not isinstance(units, (pd.Series, np.ndarray)) else float(units.iloc[-1])
    if units_held > 0:
        cf_sp500.append((sp500_prices.index[-1], float(units_held * float(sp500_prices.iloc[-1]))))
    xirr_sp500 = xirr(cf_sp500)
    xirr_portfolio = xirr(cf_all_scalar)

    port_val_df = None
    for symbol in symbols:
        trades = df[df["Symbol"] == symbol]
        px = all_prices[symbol]
        val_curve = get_portfolio_value_curve(trades, px)[["PortfolioValue"]]
        if port_val_df is None:
            port_val_df = val_curve
        else:
            port_val_df = port_val_df.add(val_curve, fill_value=0)
    total_invested = -sum([float(amt) for date, amt in cf_all_scalar if amt < 0])
    total_end_value = port_val_df["PortfolioValue"].iloc[-1]
    total_profit = total_end_value - total_invested
    total_return_pct = (total_end_value / total_invested - 1) if total_invested != 0 else 0

    sp500_val_curve = get_sp500_shadow_value_curve(df, sp500_prices)
    sp500_end_value = sp500_val_curve["S&P500Value"].iloc[-1]

    st.write(f"**Portfolio XIRR:** {xirr_portfolio:.2%}")
    st.write(f"**S&P500 XIRR (shadow):** {xirr_sp500:.2%}")
    st.write(f"**Total Return (%):** {total_return_pct:.2%}")
    st.write(f"**Total Profit ($):** {total_profit:,.2f}")

    # --- 2. Cumulative Profits ($) vs S&P500 ---
    st.header("2. Portfolio Cumulative Profits vs S&P500")
    base_invested = -sum([float(amt) for date, amt in cf_all_scalar if amt < 0])
    plot_df = pd.DataFrame({
        "Portfolio": port_val_df["PortfolioValue"] - base_invested,
        "S&P500": sp500_val_curve["S&P500Value"] - base_invested
    }).dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Portfolio"], mode='lines', name="Portfolio"))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["S&P500"], mode='lines', name="S&P500"))
    fig.update_layout(title="Cumulative Profits: Portfolio vs S&P500", yaxis_title="Profit ($)", xaxis_title="Date")
    st.plotly_chart(fig, use_container_width=True)

    # --- 3. Annual Returns Bar Chart ---
    st.header("3. Annual Returns (%)")
    annual_returns = port_val_df["PortfolioValue"].resample('Y').last().pct_change().dropna() * 100
    sp500_annual = sp500_val_curve["S&P500Value"].resample('Y').last().pct_change().dropna() * 100
    years = annual_returns.index.strftime("%Y")
    sp500_annual_vals = sp500_annual.reindex(annual_returns.index, fill_value=np.nan).values
    fig = go.Figure(data=[
        go.Bar(name='Portfolio', x=years, y=annual_returns.values),
        go.Bar(name='S&P500', x=years, y=sp500_annual_vals)
    ])
    fig.update_layout(barmode='group', title="Annual Returns (%)", yaxis_title="Return (%)", xaxis_title="Year")
    st.plotly_chart(fig, use_container_width=True)

    # --- 4. Monthly Returns Bar Chart ---
    st.header("4. Monthly Returns (%)")
    monthly_returns = port_val_df["PortfolioValue"].resample('M').last().pct_change().dropna() * 100
    sp500_monthly = sp500_val_curve["S&P500Value"].resample('M').last().pct_change().dropna() * 100
    months = monthly_returns.index.strftime("%Y-%m")
    sp500_monthly_vals = sp500_monthly.reindex(monthly_returns.index, fill_value=np.nan).values
    fig = go.Figure(data=[
        go.Bar(name='Portfolio', x=months, y=monthly_returns.values),
        go.Bar(name='S&P500', x=months, y=sp500_monthly_vals)
    ])
    fig.update_layout(barmode='group', title="Monthly Returns (%)", yaxis_title="Return (%)", xaxis_title="Month")
    st.plotly_chart(fig, use_container_width=True)

    # --- 5. Symbol-level comparison & XIRR table ---
    st.header("5. Symbol Movements vs S&P500 (XIRR Table)")
    symbol_xirr = []
    symbol_total_return = []
    for symbol in symbols:
        trades = df[df["Symbol"] == symbol]
        px = all_prices[symbol]
        cf = get_cashflows_for_xirr(trades, px, px.index[-1])
        cf_scalar = [(d, float(a)) for d, a in cf]
        sym_xirr = xirr(cf_scalar)
        cf_sp = []
        units = 0.0
        for date, amt in cf_scalar:
            px_idx = sp500_prices.index[sp500_prices.index <= date]
            if len(px_idx) == 0:
                continue
            px_sp = float(sp500_prices.loc[px_idx.max()])
            if amt < 0:
                units += abs(amt) / px_sp
            cf_sp.append((date, amt))
        units_held = float(units) if not isinstance(units, (pd.Series, np.ndarray)) else float(units.iloc[-1])
        if units_held > 0:
            cf_sp.append((sp500_prices.index[-1], float(units_held * float(sp500_prices.iloc[-1]))))
        sym_xirr_sp = xirr(cf_sp)
        invested = -sum([float(amt) for d, amt in cf_scalar if amt < 0])
        qty_held = 0
        for _, row in trades.iterrows():
            if "YOU BOU" in row["Action"]:
                qty_held += row["Quantity"]
            elif "YOU SOLD" in row["Action"]:
                qty_held -= row["Quantity"]
        end_value = px.iloc[-1] * qty_held
        total_ret = (end_value / invested - 1) if invested != 0 else 0
        symbol_xirr.append((symbol, sym_xirr, sym_xirr_sp))
        symbol_total_return.append((symbol, total_ret * 100))

    sym_df = pd.DataFrame(symbol_total_return, columns=["Symbol", "TotalReturn"])
    sp500_return = (sp500_end_value / total_invested - 1) * 100 if total_invested != 0 else 0
    fig = go.Figure(data=[
        go.Bar(name='Symbol', x=sym_df["Symbol"], y=sym_df["TotalReturn"]),
        go.Bar(name='S&P500', x=sym_df["Symbol"], y=[sp500_return]*len(sym_df))
    ])
    fig.update_layout(barmode='group', title="Symbol Total Return (%) vs S&P500", yaxis_title="Return (%)")
    st.plotly_chart(fig, use_container_width=True)

    table_df = pd.DataFrame(symbol_xirr, columns=["Symbol", "XIRR", "S&P500 XIRR"])
    st.dataframe(table_df.style.format({"XIRR": "{:.2%}", "S&P500 XIRR": "{:.2%}"}))

if __name__ == "__main__":
    main()
