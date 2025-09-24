import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import newton
import plotly.graph_objects as go

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
        if file.name.endswith(".csv"):
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

def get_cashflows(df):
    # Returns list of (datetime, amount) tuples for the whole portfolio
    cashflows = []
    for _, row in df.iterrows():
        dt = row["Run Date"]
        amt = row["Amount ($)"]
        cashflows.append((dt, amt))
    return sorted(cashflows, key=lambda x: x[0])

def get_snp500_shadow(cashflows, snp_prices):
    # Simulate S&P500 shadow portfolio by applying cashflows to S&P500 at each date
    snp_prices = snp_prices.copy()
    snp_prices = snp_prices.asfreq("D").ffill()
    units = 0.0
    shadow_value = []
    for date, amount in cashflows:
        # Find the closest trading date on or before the cashflow date
        if date in snp_prices.index:
            px_date = date
        else:
            possible_dates = snp_prices.index[snp_prices.index <= date]
            if len(possible_dates) == 0:
                continue  # skip if no price available before this date
            px_date = possible_dates.max()
        px = snp_prices.loc[px_date]
        units += amount / px
        shadow_value.append((px_date, units * px))
    # Expand this to all dates for charting
    all_dates = snp_prices.index
    values = []
    for d in all_dates:
        px = snp_prices.loc[d]
        values.append(units * px)
    shadow_df = pd.DataFrame({
        "date": all_dates,
        "snp_shadow_value": values
    }).set_index("date")
    return shadow_df

def get_portfolio_value(cashflows, price_history, latest_qty=None):
    # Calculate cumulative portfolio value over time, applying each cashflow, using portfolio's own prices
    price_history = price_history.asfreq("D").ffill()
    value = 0.0
    values = []
    qty = 0.0
    for d in price_history.index:
        # Apply any cashflow at this day
        for cf_date, cf_amt in [cf for cf in cashflows if cf[0] == d]:
            # For buy/sell, adjust qty
            if cf_amt < 0:
                qty += abs(cf_amt) / price_history.loc[d]
            else:
                qty -= abs(cf_amt) / price_history.loc[d]
            value += cf_amt
        # Mark-to-market: add the current portfolio value
        mtm = qty * price_history.loc[d]
        values.append(value + mtm)
    portfolio_df = pd.DataFrame({
        "date": price_history.index,
        "portfolio_value": values
    }).set_index("date")
    return portfolio_df

def monthly_profits(df, price_history):
    # Compute monthly profit in dollars (realized+unrealized)
    df = df.copy()
    df["Month"] = df["Run Date"].dt.to_period("M")
    # Cashflow per month
    monthly_cf = df.groupby("Month")["Amount ($)"].sum()
    # Mark-to-market at month end
    price_monthly = price_history.resample("M").last()
    # You'd need to track quantity per month for accurate unrealized profit
    return monthly_cf, price_monthly

def main():
    st.title("US Index Tracker: Portfolio vs S&P500 ($ Profits)")

    uploaded_files = st.file_uploader(
        "Upload one or more Excel or CSV files exported from Fidelity (buy/sell/dividend trades).",
        type=["xls", "xlsx", "csv"],
        accept_multiple_files=True
    )

    if uploaded_files:
        df = read_uploaded_files(uploaded_files)
        df = preprocess_trades(df)
        st.write("Combined Trade Data", df.head())

        cashflows = get_cashflows(df)
        start = min([cf[0] for cf in cashflows]) - timedelta(days=5)
        end = datetime.today()

        # Get S&P500 price history
        snp500 = yf.download("^GSPC", start=start, end=end)["Close"]
        snp500 = snp500.dropna()
        st.write(f"S&P500 data from {snp500.index[0].date()} to {snp500.index[-1].date()}")

        # Portfolio value over time (mark-to-market)
        # For simplicity, assume all trades are in one index fund or use average price
        # For more accuracy, expand to handle multiple symbols (sum over all)
        # Here, let's use the last traded symbol as the main
        main_symbol = df["Symbol"].value_counts().index[0]
        price_hist = yf.download(main_symbol, start=start, end=end)["Close"].dropna()
        # Mark-to-market: calculate cumulative profit
        # For demo, we just use running sum of cashflows (not actual holding value)
        cf_df = pd.DataFrame(cashflows, columns=["date", "amount"])
        cf_df = cf_df.groupby("date").sum()  # Aggregate cashflows per date
        cf_df = cf_df.reindex(price_hist.index, fill_value=0)  # No more duplicates
        cum_profit = cf_df["amount"].cumsum()

        # S&P500 shadow
        shadow_df = get_snp500_shadow(cashflows, snp500)
        # Align indexes
        plot_df = pd.DataFrame({
            "Portfolio": cum_profit,
            "S&P500 Shadow": shadow_df["snp_shadow_value"]
        }).fillna(method="ffill").dropna()

        # Plot cumulative profits ($)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Portfolio"], mode="lines", name="Portfolio"))
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["S&P500 Shadow"], mode="lines", name="S&P500 Shadow"))
        fig.update_layout(title="Cumulative Profits: Portfolio vs S&P500 ($)",
                          yaxis_title="Profits ($)", xaxis_title="Date")
        st.plotly_chart(fig, use_container_width=True)

        # Monthly Profits
        df["Month"] = df["Run Date"].dt.to_period("M")
        monthly_cf = df.groupby("Month")["Amount ($)"].sum()
        snp_prices_monthly = snp500.resample("M").last()
        # Simulate S&P500 monthly profits by applying cashflows to S&P500 at each month
        shadow_monthly = []
        units = 0.0
        for month, amt in monthly_cf.items():
            dt = snp_prices_monthly.index[snp_prices_monthly.index.to_period("M") == month][0]
            px = snp_prices_monthly.loc[dt]
            units += amt / px
            shadow_monthly.append(units * px)
        shadow_monthly_profit = pd.Series(shadow_monthly, index=monthly_cf.index)
        clustered_df = pd.DataFrame({
            "Portfolio": monthly_cf,
            "S&P500 Shadow": shadow_monthly_profit.diff().fillna(shadow_monthly_profit)
        })

        fig = go.Figure(data=[
            go.Bar(name='Portfolio', x=clustered_df.index.astype(str), y=clustered_df["Portfolio"]),
            go.Bar(name='S&P500 Shadow', x=clustered_df.index.astype(str), y=clustered_df["S&P500 Shadow"])
        ])
        fig.update_layout(barmode='group', title="Monthly Profits: Portfolio vs S&P500 ($)",
                          yaxis_title="Profits ($)", xaxis_title="Month")
        st.plotly_chart(fig, use_container_width=True)

        # Annual Profits
        df["Year"] = df["Run Date"].dt.to_period("Y")
        annual_cf = df.groupby("Year")["Amount ($)"].sum()
        snp_prices_annual = snp500.resample("Y").last()
        shadow_annual = []
        units = 0.0
        for year, amt in annual_cf.items():
            dt = snp_prices_annual.index[snp_prices_annual.index.to_period("Y") == year][0]
            px = snp_prices_annual.loc[dt]
            units += amt / px
            shadow_annual.append(units * px)
        shadow_annual_profit = pd.Series(shadow_annual, index=annual_cf.index)
        clustered_annual = pd.DataFrame({
            "Portfolio": annual_cf,
            "S&P500 Shadow": shadow_annual_profit.diff().fillna(shadow_annual_profit)
        })

        fig = go.Figure(data=[
            go.Bar(name='Portfolio', x=clustered_annual.index.astype(str), y=clustered_annual["Portfolio"]),
            go.Bar(name='S&P500 Shadow', x=clustered_annual.index.astype(str), y=clustered_annual["S&P500 Shadow"])
        ])
        fig.update_layout(barmode='group', title="Annual Profits: Portfolio vs S&P500 ($)",
                          yaxis_title="Profits ($)", xaxis_title="Year")
        st.plotly_chart(fig, use_container_width=True)

        # XIRR
        st.header("Portfolio XIRR vs S&P500 XIRR")
        xirr_portfolio = xirr(cashflows)
        # S&P500 XIRR: treat as investing each cashflow in S&P500, get final value, withdraw all at end
        shadow_cashflows = []
        units = 0.0
        for date, amt in cashflows:
            px_date = snp500.index[snp500.index.get_loc(date, method='ffill')]
            px = snp500.loc[px_date]
            units += amt / px
            shadow_cashflows.append((date, -amt))  # for S&P500, reverse sign to simulate buy
        # Final withdrawal
        px_final = snp500.iloc[-1]
        shadow_cashflows.append((snp500.index[-1], units * px_final))
        xirr_snp500 = xirr(shadow_cashflows)

        st.write(f"Portfolio XIRR: {xirr_portfolio:.2%}")
        st.write(f"S&P500 XIRR (same cashflow timing): {xirr_snp500:.2%}")

if __name__ == "__main__":
    main()
