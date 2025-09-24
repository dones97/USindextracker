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

def get_symbol_name_map(df):
    # Try to get a descriptive fund name for each symbol
    # If there's a 'Description' or 'Fund Name' column, use that; else fallback to symbol
    name_col = None
    for col in ["Description", "Fund Name", "Security Description", "Name"]:
        if col in df.columns:
            name_col = col
            break
    if name_col:
        name_map = df.groupby("Symbol")[name_col].agg(lambda x: x.value_counts().idxmax()).to_dict()
    else:
        name_map = {s: s for s in df["Symbol"].unique()}
    return name_map

def get_price_history(symbol, start, end):
    px = yf.download(symbol, start=start, end=end, progress=False)
    close_px = px["Close"].dropna()
    close_px = close_px[~close_px.index.duplicated(keep='last')]
    close_px = close_px.sort_index()
    return close_px

def build_portfolio_profit_curve(trades, prices):
    prices = prices[~prices.index.duplicated(keep='last')]
    prices = prices.sort_index()
    all_days = pd.date_range(prices.index.min(), prices.index.max(), freq='D')
    ff_prices = prices.reindex(all_days).ffill()
    trades = trades.sort_values('Run Date')
    trades = trades.set_index('Run Date')

    lots = []
    realized_profits = []  # (date, realized profit from sells/dividends)
    profit_curve = []
    for date in all_days:
        if date in trades.index:
            for _, row in trades.loc[[date]].iterrows():
                qty = row["Quantity"]
                amt = float(row["Amount ($)"])
                if "YOU BOU" in row["Action"]:
                    price = float(row["Price ($)"])
                    lots.append({"qty": qty, "cost": price})
                elif "YOU SOLD" in row["Action"]:
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
                    realized_profits.append((date, profit))
                elif "DIVIDEND" in row["Action"]:
                    realized_profits.append((date, amt))
        price_val = ff_prices.loc[date]
        if isinstance(price_val, pd.Series):
            price_val = price_val.iloc[-1]
        try:
            px = float(price_val)
        except Exception:
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

def compute_monthly_returns(curve):
    value = curve["CurrentValue"] + curve["Realized"]
    value = value.ffill()
    month_ends = value.resample("M").last()
    month_starts = value.resample("M").first()
    returns = []
    months = month_ends.index
    for i in range(len(months)):
        end = months[i]
        start_value = month_starts.iloc[i]
        end_value = month_ends.iloc[i]
        if pd.isna(start_value) or start_value == 0:
            returns.append(np.nan)
            continue
        ret = (end_value - start_value) / start_value * 100
        returns.append(ret)
    return pd.Series(returns, index=months)

def compute_annual_returns(curve):
    value = curve["CurrentValue"] + curve["Realized"]
    value = value.ffill()
    year_ends = value.resample("Y").last()
    year_starts = value.resample("Y").first()
    years = year_ends.index
    returns = []
    for i in range(len(years)):
        start_value = year_starts.iloc[i]
        end_value = year_ends.iloc[i]
        if pd.isna(start_value) or start_value == 0:
            returns.append(np.nan)
            continue
        ret = (end_value - start_value) / start_value * 100
        returns.append(ret)
    return pd.Series(returns, index=years)

def main():
    st.title("Portfolio Analysis (Correct Realized/Unrealized Profits)")

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
    # --- Fund name lookup ---
    symbol_name_map = get_symbol_name_map(df)
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
            vals = curve[col].reindex(portfolio_curve.index, fill_value=0.0)
            portfolio_curve[col] += vals
    for col in ['CurrentValue', 'CurrentQty']:
        portfolio_curve[col] = portfolio_curve[col].replace(0, np.nan).ffill().fillna(0.0)

    # --- 1. Portfolio Level Metrics ---
    st.header("1. Portfolio-level Metrics")
    end_date = portfolio_curve.index[-1]
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

    # --- 2. Symbol-level comparison & XIRR table ---
    st.header("2. Fund Movements (XIRR Table)")
    symbol_xirr = []
    symbol_total_return = []
    for symbol in symbols:
        trades = df[df["Symbol"] == symbol]
        px = all_prices[symbol]
        curve = profit_curves[symbol]
        end_date = curve.index[-1]
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
        # Use fund name
        fund_name = symbol_name_map.get(symbol, symbol)
        symbol_xirr.append((symbol, fund_name, sym_xirr))
        symbol_total_return.append((fund_name, total_ret * 100))

    # Bar chart: total return % per fund
    sym_df = pd.DataFrame(symbol_total_return, columns=["Fund Name", "TotalReturn"])
    fig = go.Figure(data=[
        go.Bar(name='Fund', x=sym_df["Fund Name"], y=sym_df["TotalReturn"])
    ])
    fig.update_layout(barmode='group', title="Fund Total Return (%)", yaxis_title="Return (%)")
    st.plotly_chart(fig, use_container_width=True)

    # Data Table: XIRR per fund
    table_df = pd.DataFrame(symbol_xirr, columns=["Symbol", "Fund Name", "XIRR"])
    # Show only Fund Name and XIRR for clarity
    st.dataframe(table_df[["Fund Name", "XIRR"]].style.format({"XIRR": "{:.2%}"}))

    # --- 3. Portfolio Cumulative Profits ---
    st.header("3. Portfolio Cumulative Profits")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio_curve.index, y=portfolio_curve["TotalProfit"], mode='lines', name="Cumulative Profit"))
    fig.update_layout(title="Cumulative Profits", yaxis_title="Profit ($)", xaxis_title="Date")
    st.plotly_chart(fig, use_container_width=True)

    # --- 4. Annual Returns Bar Chart ---
    st.header("4. Annual Returns (%)")
    annual_returns = compute_annual_returns(portfolio_curve)
    years = [d.strftime("%Y") for d in annual_returns.index]
    fig = go.Figure(data=[
        go.Bar(name='Portfolio', x=years, y=annual_returns.values)
    ])
    fig.update_layout(barmode='group', title="Annual Returns (%)", yaxis_title="Return (%)", xaxis_title="Year")
    st.plotly_chart(fig, use_container_width=True)

    # --- 5. Monthly Returns Bar Chart ---
    st.header("5. Monthly Returns (%)")
    monthly_returns = compute_monthly_returns(portfolio_curve)
    months = [d.strftime("%Y-%m") for d in monthly_returns.index]
    fig = go.Figure(data=[
        go.Bar(name='Portfolio', x=months, y=monthly_returns.values)
    ])
    fig.update_layout(barmode='group', title="Monthly Returns (%)", yaxis_title="Return (%)", xaxis_title="Month")
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
