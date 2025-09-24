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
    close_px = px["Close"].dropna()
    close_px = close_px[~close_px.index.duplicated(keep='last')]
    close_px = close_px.sort_index()
    return close_px

def build_portfolio_profit_curve(trades, prices):
    prices = prices[~prices.index.duplicated(keep='last')]
    prices = prices.sort_index()
    all_days = pd.date_range(prices.index.min(), prices.index.max(), freq='D')

    # Prepare all trades sorted by date
    trades = trades.sort_values('Run Date')
    trades = trades.set_index('Run Date')

    # Track lots: each buy is added as a lot (qty, cost per unit)
    lots = []
    realized_profits = []  # (date, realized profit from sells/dividends)
    profit_curve = []
    for date in all_days:
        # Process all trades for this date
        if date in trades.index:
            for _, row in trades.loc[[date]].iterrows():
                qty = row["Quantity"]
                amt = float(row["Amount ($)"])
                if "YOU BOU" in row["Action"]:
                    price = float(row["Price ($)"])
                    # Add a new lot for each buy
                    lots.append({"qty": qty, "cost": price})
                elif "YOU SOLD" in row["Action"]:
                    # Remove from lots FIFO
                    sell_qty = qty
                    sell_price = float(row["Price ($)"])
                    profit = 0
                    while sell_qty > 0 and lots:
                        lot = lots[0]
                        lot_qty = lot["qty"]
                        lot_cost = lot["cost"]
                        if lot_qty <= sell_qty:
                            this_qty = lot_qty
                            lots.pop(0)
                        else:
                            this_qty = sell_qty
                            lot["qty"] -= sell_qty
                        profit += (sell_price - lot_cost) * this_qty
                        sell_qty -= this_qty
                    # If more sold than held, treat as all sold from existing lots (should not happen)
                    realized_profits.append((date, profit))
                elif "DIVIDEND" in row["Action"]:
                    realized_profits.append((date, amt))

        # At each date, compute:
        # - Total quantity held and weighted average cost
        # - Unrealized profit = (current price - lot cost) * qty for all open lots
        # - Realized profit so far = sum of realized_profits up to now
        if date in prices.index:
            px = float(prices.loc[date])
        else:
            px = np.nan

        unrealized = 0.0
        for lot in lots:
            unrealized += (px - lot["cost"]) * lot["qty"] if not np.isnan(px) else 0.0
        realized = sum([p for d, p in realized_profits if d <= date])
        cumulative = realized + unrealized
        profit_curve.append({
            "Date": date,
            "Unrealized": unrealized,
            "Realized": realized,
            "TotalProfit": cumulative,
            "CurrentValue": sum([lot["qty"] for lot in lots]) * px if not np.isnan(px) else np.nan,
            "CurrentQty": sum([lot["qty"] for lot in lots])
        })
    return pd.DataFrame(profit_curve).set_index("Date")

def portfolio_cashflows_for_xirr(df, prices, end_date):
    # Cash outflows (buys, negative), inflows (sells, positive), dividends (positive), plus final value of holdings
    cashflows = []
    for _, row in df.iterrows():
        amt = float(row["Amount ($)"])
        cashflows.append((row["Run Date"], amt))
    # Add a final inflow of value of all holdings at end_date (simulate complete liquidation)
    # Calculate total qty and cost per symbol
    lots = []
    for _, row in df.iterrows():
        qty = row["Quantity"]
        if "YOU BOU" in row["Action"]:
            lots.append({"qty": qty, "cost": float(row["Price ($)"])})
        elif "YOU SOLD" in row["Action"]:
            sell_qty = qty
            # Remove from lots FIFO
            while sell_qty > 0 and lots:
                lot = lots[0]
                lot_qty = lot["qty"]
                if lot_qty <= sell_qty:
                    this_qty = lot_qty
                    lots.pop(0)
                else:
                    this_qty = sell_qty
                    lot["qty"] -= sell_qty
                sell_qty -= this_qty
    # Now lots contains all open lots for all symbols
    if len(lots) > 0 and end_date in prices.index:
        px = float(prices.loc[end_date])
        total_qty = sum([lot["qty"] for lot in lots])
        if total_qty > 0:
            cashflows.append((end_date, total_qty * px))
    return cashflows

def compute_period_returns(profit_curve, freq='M'):
    # Use portfolio value curve (including realized and unrealized) to get monthly/annual returns
    values = profit_curve['CurrentValue'] + profit_curve['Realized']
    # Fill NAs forward for days with missing prices
    values = values.ffill()
    period_val = values.resample(freq).last().dropna()
    returns = period_val.pct_change().dropna()
    return returns

def main():
    st.title("Portfolio Analysis (No S&P500, Correct Realized/Unrealized Profits)")

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

    # Get price history for all symbols
    all_prices = {}
    for symbol in symbols:
        all_prices[symbol] = get_price_history(symbol, t0, t1)

    # Build and sum profit curves for all symbols
    profit_curves = {}
    for symbol in symbols:
        trades = df[df["Symbol"] == symbol]
        prices = all_prices[symbol]
        profit_curves[symbol] = build_portfolio_profit_curve(trades, prices)

    # Portfolio-level curve: sum across all symbols
    all_dates = sorted(set().union(*[curve.index for curve in profit_curves.values()]))
    portfolio_curve = pd.DataFrame(index=pd.DatetimeIndex(all_dates))
    for k, curve in profit_curves.items():
        for col in ['Unrealized', 'Realized', 'TotalProfit', 'CurrentValue', 'CurrentQty']:
            if col not in portfolio_curve.columns:
                portfolio_curve[col] = 0.0
            # Align index, fill missing as 0 for addition, then add
            vals = curve[col].reindex(portfolio_curve.index, fill_value=0.0)
            portfolio_curve[col] += vals
    # Fill NAs in price-driven columns forward for value calculations
    for col in ['CurrentValue', 'CurrentQty']:
        portfolio_curve[col] = portfolio_curve[col].replace(0, np.nan).ffill().fillna(0.0)
    # Realized profit line should not be cumulative sum; it's already cumulative

    # --- 1. Portfolio Level Metrics ---
    st.header("1. Portfolio-level Metrics")
    # For XIRR and total return, use all trades and total value at end
    # Reconstruct cashflows for XIRR from all trades
    # For prices, use a "portfolio price" as weighted sum, or just use the sum from the curve
    end_date = portfolio_curve.index[-1]
    # Use sum of symbol prices for final liquidation value
    values_for_xirr = portfolio_curve['CurrentValue'].iloc[-1]
    cashflows = []
    for _, row in df.iterrows():
        amt = float(row["Amount ($)"])
        cashflows.append((row["Run Date"], amt))
    if values_for_xirr > 0:
        cashflows.append((end_date, values_for_xirr))
    xirr_portfolio = xirr(cashflows)
    total_invested = -sum([amt for date, amt in cashflows if amt < 0])
    total_end_value = values_for_xirr + portfolio_curve["Realized"].iloc[-1]
    total_profit = total_end_value - total_invested
    total_return_pct = (total_end_value / total_invested - 1) if total_invested != 0 else 0

    st.write(f"**Portfolio XIRR:** {xirr_portfolio:.2%}")
    st.write(f"**Total Return (%):** {total_return_pct:.2%}")
    st.write(f"**Total Profit ($):** {total_profit:,.2f}")

    # --- 2. Cumulative Profit Chart ---
    st.header("2. Portfolio Cumulative Profits")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio_curve.index, y=portfolio_curve["TotalProfit"], mode='lines', name="Cumulative Profit"))
    fig.update_layout(title="Cumulative Profits", yaxis_title="Profit ($)", xaxis_title="Date")
    st.plotly_chart(fig, use_container_width=True)

    # --- 3. Annual Returns Bar Chart ---
    st.header("3. Annual Returns (%)")
    annual_returns = compute_period_returns(portfolio_curve, freq='Y') * 100
    years = annual_returns.index.strftime("%Y")
    fig = go.Figure(data=[
        go.Bar(name='Portfolio', x=years, y=annual_returns.values)
    ])
    fig.update_layout(barmode='group', title="Annual Returns (%)", yaxis_title="Return (%)", xaxis_title="Year")
    st.plotly_chart(fig, use_container_width=True)

    # --- 4. Monthly Returns Bar Chart ---
    st.header("4. Monthly Returns (%)")
    monthly_returns = compute_period_returns(portfolio_curve, freq='M') * 100
    months = monthly_returns.index.strftime("%Y-%m")
    fig = go.Figure(data=[
        go.Bar(name='Portfolio', x=months, y=monthly_returns.values)
    ])
    fig.update_layout(barmode='group', title="Monthly Returns (%)", yaxis_title="Return (%)", xaxis_title="Month")
    st.plotly_chart(fig, use_container_width=True)

    # --- 5. Symbol-level comparison & XIRR table ---
    st.header("5. Symbol Movements (XIRR Table)")
    symbol_xirr = []
    symbol_total_return = []
    for symbol in symbols:
        trades = df[df["Symbol"] == symbol]
        px = all_prices[symbol]
        curve = profit_curves[symbol]
        # Cashflows and final liquidation for symbol XIRR
        end_date = curve.index[-1]
        # For symbol, use symbol's current value at end
        symbol_value = curve['CurrentValue'].iloc[-1]
        symbol_cashflows = []
        for _, row in trades.iterrows():
            amt = float(row["Amount ($)"])
            symbol_cashflows.append((row["Run Date"], amt))
        if symbol_value > 0:
            symbol_cashflows.append((end_date, symbol_value))
        sym_xirr = xirr(symbol_cashflows)
        invested = -sum([amt for date, amt in symbol_cashflows if amt < 0])
        end_value = curve['CurrentValue'].iloc[-1] + curve['Realized'].iloc[-1]
        total_ret = (end_value / invested - 1) if invested != 0 else 0
        symbol_xirr.append((symbol, sym_xirr))
        symbol_total_return.append((symbol, total_ret * 100))

    # Bar chart: total return % per symbol
    sym_df = pd.DataFrame(symbol_total_return, columns=["Symbol", "TotalReturn"])
    fig = go.Figure(data=[
        go.Bar(name='Symbol', x=sym_df["Symbol"], y=sym_df["TotalReturn"])
    ])
    fig.update_layout(barmode='group', title="Symbol Total Return (%)", yaxis_title="Return (%)")
    st.plotly_chart(fig, use_container_width=True)

    # Data Table: XIRR per symbol
    table_df = pd.DataFrame(symbol_xirr, columns=["Symbol", "XIRR"])
    st.dataframe(table_df.style.format({"XIRR": "{:.2%}"}))

if __name__ == "__main__":
    main()
