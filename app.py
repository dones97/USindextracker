import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objs as go

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

def get_profit_curve(trades, prices):
    # Returns a DataFrame with:
    #   'Unrealized Profit', 'Realized Profit', 'Total Profit', 'Portfolio Value', 'Qty Held', 'Cost Basis'
    all_days = pd.date_range(prices.index.min(), prices.index.max(), freq='D')
    trades = trades.sort_values('Run Date')
    trades = trades.set_index('Run Date')

    qty = 0
    cost = 0
    realized = 0
    profit_curve = []
    for date in all_days:
        # Process trades on this date (could be multiple)
        if date in trades.index:
            for _, row in trades.loc[[date]].iterrows():
                if "YOU BOU" in row["Action"]:
                    qty += row["Quantity"]
                    cost += abs(row["Amount ($)"])
                elif "YOU SOLD" in row["Action"]:
                    sell_qty = row["Quantity"]
                    if qty == 0: continue
                    avg_cost = cost / qty if qty > 0 else 0
                    realized += (row["Price ($)"] - avg_cost) * sell_qty
                    cost -= avg_cost * sell_qty
                    qty -= sell_qty
                elif "DIVIDEND" in row["Action"]:
                    realized += row["Amount ($)"]
        # Mark-to-market
        if date in prices.index:
            mkt_val = qty * prices.loc[date]
        else:
            mkt_val = np.nan
        unrealized = mkt_val - cost if qty > 0 else 0
        total_profit = unrealized + realized
        port_val = mkt_val + realized if not np.isnan(mkt_val) else np.nan
        profit_curve.append({
            "Date": date,
            "Unrealized Profit": unrealized,
            "Realized Profit": realized,
            "Total Profit": total_profit,
            "Qty Held": qty,
            "Cost Basis": cost,
            "Portfolio Value": port_val
        })
    curve_df = pd.DataFrame(profit_curve).set_index("Date")
    return curve_df

def get_sp500_shadow_curve(trades, sp500_prices):
    # Simulate S&P500 investment: same cash flows, dates, applied to S&P500
    all_days = pd.date_range(sp500_prices.index.min(), sp500_prices.index.max(), freq='D')
    trades = trades.sort_values('Run Date')
    trades = trades.set_index('Run Date')
    units = 0.0
    realized = 0.0
    shadow_curve = []
    for date in all_days:
        # Process trades on this date
        if date in trades.index and date in sp500_prices.index:
            for _, row in trades.loc[[date]].iterrows():
                px = sp500_prices.loc[date]
                if "YOU BOU" in row["Action"]:
                    units += abs(row["Amount ($)"]) / px
                elif "YOU SOLD" in row["Action"]:
                    # Sell same $ as original sale, notional, not actual units
                    sell_px = px
                    sell_amt = row["Quantity"] * sell_px
                    units -= row["Quantity"]  # Remove equivalent units
                    # No realized gain since cost basis is not tracked here
                elif "DIVIDEND" in row["Action"]:
                    realized += row["Amount ($)"]  # add to shadow realized
        if date in sp500_prices.index:
            mkt_val = units * sp500_prices.loc[date]
        else:
            mkt_val = np.nan
        total_val = mkt_val + realized if not np.isnan(mkt_val) else np.nan
        shadow_curve.append({
            "Date": date,
            "S&P500 Value": mkt_val,
            "S&P500 Realized": realized,
            "S&P500 Total Value": total_val
        })
    curve_df = pd.DataFrame(shadow_curve).set_index("Date")
    return curve_df

def compute_period_returns(portfolio_value, freq='M'):
    # Portfolio value should be a Series indexed by date
    period_val = portfolio_value.resample(freq).last().dropna()
    returns = period_val.pct_change().dropna()
    return returns

def main():
    st.title("Index Fund Tracker: Symbol & S&P500 Comparison")

    uploaded_files = st.file_uploader(
        "Upload one or more files (Excel/CSV) from Fidelity.",
        type=["xls", "xlsx", "csv"],
        accept_multiple_files=True
    )

    if uploaded_files:
        df = read_uploaded_files(uploaded_files)
        df = preprocess_trades(df)
        st.write("Combined Trade Data", df.head())

        symbols = get_symbol_list(df)
        sel_symbols = st.multiselect("Select fund symbols to analyze:", symbols, default=list(symbols))
        t0 = df["Run Date"].min() - pd.Timedelta(days=5)
        t1 = datetime.today()

        # Get S&P500 price history once
        sp500_prices = get_price_history("^GSPC", t0, t1)

        for symbol in sel_symbols:
            st.header(f"Analysis for {symbol}")
            trades = df[df["Symbol"] == symbol]
            prices = get_price_history(symbol, t0, t1)
            if prices.empty or trades.empty:
                st.warning(f"Not enough data for {symbol}.")
                continue

            curve = get_profit_curve(trades, prices)
            shadow_curve = get_sp500_shadow_curve(trades, sp500_prices)

            # Cumulative Profits Line Chart (Symbol vs S&P500)
            st.subheader("Cumulative Profits (Unrealized + Realized, $)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=curve.index, y=curve["Total Profit"], mode='lines', name=f"{symbol}"))
            fig.add_trace(go.Scatter(x=shadow_curve.index, y=shadow_curve["S&P500 Total Value"] - shadow_curve["S&P500 Total Value"].iloc[0], mode='lines', name="S&P500 Shadow"))
            fig.update_layout(title=f"Cumulative Profits: {symbol} vs S&P500", yaxis_title="Profit ($)", xaxis_title="Date")
            st.plotly_chart(fig, use_container_width=True)

            # Monthly Returns (%)
            st.subheader("Monthly Returns (%)")
            sym_returns = compute_period_returns(curve["Portfolio Value"], freq='M') * 100
            sp500_returns = compute_period_returns(shadow_curve["S&P500 Total Value"], freq='M') * 100
            returns_df = pd.DataFrame({f"{symbol}": sym_returns, "S&P500 Shadow": sp500_returns}).dropna()
            fig = go.Figure(data=[
                go.Bar(name=symbol, x=returns_df.index.astype(str), y=returns_df[symbol]),
                go.Bar(name='S&P500 Shadow', x=returns_df.index.astype(str), y=returns_df["S&P500 Shadow"])
            ])
            fig.update_layout(barmode='group', title=f"Monthly % Returns: {symbol} vs S&P500", yaxis_title="Return (%)", xaxis_title="Month")
            st.plotly_chart(fig, use_container_width=True)

            # Annual Returns (%)
            st.subheader("Annual Returns (%)")
            sym_returns_y = compute_period_returns(curve["Portfolio Value"], freq='Y') * 100
            sp500_returns_y = compute_period_returns(shadow_curve["S&P500 Total Value"], freq='Y') * 100
            returns_y_df = pd.DataFrame({f"{symbol}": sym_returns_y, "S&P500 Shadow": sp500_returns_y}).dropna()
            fig = go.Figure(data=[
                go.Bar(name=symbol, x=returns_y_df.index.astype(str), y=returns_y_df[symbol]),
                go.Bar(name='S&P500 Shadow', x=returns_y_df.index.astype(str), y=returns_y_df["S&P500 Shadow"])
            ])
            fig.update_layout(barmode='group', title=f"Annual % Returns: {symbol} vs S&P500", yaxis_title="Return (%)", xaxis_title="Year")
            st.plotly_chart(fig, use_container_width=True)

            # Metrics Table
            st.subheader("Summary Metrics")
            st.table({
                "Total Realized Profit": [curve["Realized Profit"].iloc[-1]],
                "Total Unrealized Profit": [curve["Unrealized Profit"].iloc[-1]],
                "Total Profit": [curve["Total Profit"].iloc[-1]],
                "Portfolio Value": [curve["Portfolio Value"].iloc[-1]],
                "Latest Quantity Held": [curve["Qty Held"].iloc[-1]]
            })

if __name__ == "__main__":
    main()
