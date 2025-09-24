import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta
from io import BytesIO

# Helper for XIRR calculation
def xnpv(rate, cashflows):
    t0 = cashflows[0][0]
    return sum([cf / (1 + rate) ** ((t - t0).days / 365) for t, cf in cashflows])

def xirr(cashflows, guess=0.1):
    # cashflows: list of (datetime, amount)
    from scipy.optimize import newton
    if all(cf[1] == 0 for cf in cashflows):
        return np.nan
    try:
        return newton(lambda r: xnpv(r, cashflows), guess)
    except (RuntimeError, OverflowError):
        return np.nan

def read_excel_files(files):
    dfs = []
    for file in files:
        dfs.append(pd.read_excel(file))
    df = pd.concat(dfs, ignore_index=True)
    return df

def preprocess_trades(df):
    # Standardize column names
    df.columns = [col.strip() for col in df.columns]
    df["Run Date"] = pd.to_datetime(df["Run Date"])
    df = df[df["Action"].notna()]
    # Filter for actual trades (exclude electronic/cash balance updates)
    df = df[df["Action"].str.contains("YOU BOU|YOU SOLD|DIVIDEND", case=False, na=False)]
    return df

def get_cashflows(df, symbol):
    rows = df[df["Symbol"] == symbol]
    cashflows = []
    for _, row in rows.iterrows():
        dt = row["Run Date"]
        amt = row["Amount ($)"]
        if "YOU BOU" in row["Action"]:
            cashflows.append((dt, amt))  # amt is negative for buy
        elif "YOU SOLD" in row["Action"] or "DIVIDEND" in row["Action"]:
            cashflows.append((dt, amt))  # amt is positive for sell/dividend
    # Assume a final sell at latest price for remaining units
    latest_date = df["Run Date"].max()
    qty = rows[rows["Action"].str.contains("YOU BOU")]["Quantity"].sum() - rows[rows["Action"].str.contains("YOU SOLD")]["Quantity"].sum()
    if qty > 0:
        price = yf.download(symbol, start=latest_date, end=latest_date + pd.Timedelta(days=1))["Close"]
        if not price.empty:
            cashflows.append((latest_date, qty * price.values[0]))
    return sorted(cashflows, key=lambda x: x[0])

def get_price_history(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    return df["Close"]

def calculate_returns(df, symbols, start, end):
    result = []
    for symbol in symbols:
        cashflows = get_cashflows(df, symbol)
        irr = xirr(cashflows)
        result.append({"Symbol": symbol, "XIRR": irr})
    return pd.DataFrame(result)

def calculate_monthly_returns(price_df):
    return price_df.resample('M').ffill().pct_change().dropna()

def calculate_annual_returns(price_df):
    return price_df.resample('Y').ffill().pct_change().dropna()

def plot_cumulative_returns(prices, label):
    cum_ret = (prices / prices.iloc[0]) - 1
    st.line_chart(cum_ret.rename(label))

def main():
    st.title("Fidelity Index Fund Returns Tracker & S&P500 Comparison")

    uploaded_files = st.file_uploader(
        "Upload one or more Excel files exported from Fidelity (buy/sell/dividend trades).",
        type=["xls", "xlsx"],
        accept_multiple_files=True
    )

    if uploaded_files:
        df = read_excel_files(uploaded_files)
        df = preprocess_trades(df)
        st.write("Combined Trade Data", df.head())

        symbols = df["Symbol"].unique()
        st.sidebar.write("Symbols in your trades:")
        sel_symbols = st.sidebar.multiselect("Select funds to analyze:", symbols, default=list(symbols))

        start = df["Run Date"].min() - pd.Timedelta(days=10)
        end = datetime.today()

        # Get S&P500 price history
        sp500 = get_price_history("^GSPC", start, end)
        if sp500.empty:
            st.warning("Could not fetch S&P500 data.")
            return

        st.header("Portfolio vs S&P500: Cumulative Returns")
        plot_cumulative_returns(sp500, "S&P500")

        for symbol in sel_symbols:
            prices = get_price_history(symbol, start, end)
            if prices.empty:
                st.warning(f"Could not fetch price data for {symbol}")
                continue
            plot_cumulative_returns(prices, symbol)

            # Monthly/Annual Returns
            st.subheader(f"{symbol}: Monthly Returns")
            st.bar_chart(calculate_monthly_returns(prices).rename("Monthly Return"))

            st.subheader(f"{symbol}: Annual Returns")
            st.bar_chart(calculate_annual_returns(prices).rename("Annual Return"))

        # Portfolio XIRR
        st.header("Portfolio XIRR")
        result_df = calculate_returns(df, sel_symbols, start, end)
        st.write(result_df)

        # S&P500 XIRR (simulate single buy at earliest date, sell at latest)
        sp500_cf = [
            (start, -sp500.iloc[0]),
            (end, sp500.iloc[-1])
        ]
        sp500_xirr = xirr(sp500_cf)
        st.write(f"S&P500 XIRR for same period: {sp500_xirr:.2%}")

        # ToDo: Add more advanced cumulative/relative profit calculation, more visualizations

if __name__ == "__main__":
    main()
