import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import newton
import plotly.graph_objs as go
import os
import glob

def xnpv(rate, cashflows):
    t0 = cashflows[0][0]
    return sum([cf / (1 + rate) ** ((t - t0).days / 365) for t, cf in cashflows])

def xirr(cashflows):
    try:
        return newton(lambda r: xnpv(r, cashflows), 0.1)
    except Exception:
        return np.nan

def load_trade_data(uploaded_files, data_dir="data"):
    dfs = []
    
    # 1. Load files from the local data directory
    if os.path.exists(data_dir):
        for file_path in glob.glob(os.path.join(data_dir, "*.csv")):
            dfs.append(pd.read_csv(file_path))
        for file_path in glob.glob(os.path.join(data_dir, "*.xlsx")):
            dfs.append(pd.read_excel(file_path))
        for file_path in glob.glob(os.path.join(data_dir, "*.xls")):
            dfs.append(pd.read_excel(file_path))

    # 2. Load manually uploaded files
    if uploaded_files:
        for file in uploaded_files:
            if file.name.endswith('.csv'):
                dfs.append(pd.read_csv(file))
            else:
                dfs.append(pd.read_excel(file))
                
    if not dfs:
        return pd.DataFrame()
        
    # Combine and deduplicate
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Deduplicate exact rows to avoid double-counting trades overlapping across quarters/uploads
    combined_df.drop_duplicates(inplace=True)
    
    return combined_df

def classify_action(row):
    action = str(row["Action"]).upper()
    if "YOU BOUGHT" in action:
        return "BUY"
    if "YOU SOLD" in action:
        return "SELL"
    if "REINVESTMENT" in action:
        return "REINVEST"
    if "DIVIDEND" in action:
        # Reinvestment lines often say "DIVIDEND RECEIVED" then "REINVESTMENT"? 
        # Based on file check, "DIVIDEND RECEIVED" is separate from "REINVESTMENT"
        # We will handle "REINVESTMENT" checks in profit curve building, 
        # but here we just want to ID it as a dividend type if not reinvestment
        return "DIVIDEND"
    if "TRANSFER" in action:
        if "RECEIVED" in action:
            return "DEPOSIT"
        if "PAID" in action:
            return "WITHDRAWAL"
    # Fallback for explicit cash movements if any (e.g. check deposits not labeled transfer)
    if "DEPOSIT" in action: return "DEPOSIT"
    if "WITHDRAWAL" in action: return "WITHDRAWAL"
    return "OTHER"

def preprocess_trades(df):
    df.columns = [col.strip() for col in df.columns]
    df["Run Date"] = pd.to_datetime(df["Run Date"]).dt.normalize()
    
    # helper for amount parsing
    def parse_amount(x):
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).replace('$', '').replace(',', '').strip()
        if '(' in s and ')' in s:
            s = '-' + s.replace('(', '').replace(')', '')
        # Handle cases where - is at end? Or other formats?
        return float(s)
        
    if "Amount ($)" in df.columns:
        df["Amount ($)"] = df["Amount ($)"].apply(parse_amount)
    
    df["ActionType"] = df.apply(classify_action, axis=1)
    
    # Filter to relevant types
    relevant_types = ["BUY", "SELL", "DIVIDEND", "REINVEST", "DEPOSIT", "WITHDRAWAL"]
    df = df[df["ActionType"].isin(relevant_types)]
    
    return df

def get_symbol_list(df):
    syms = df["Symbol"].dropna().unique()
    # clean strings
    syms = [str(s).strip() for s in syms]
    # remove empty
    return [s for s in syms if s and s.upper() != "NAN" and s != "."]

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
    # Special handling for Cash/Money Market
    if symbol == "SPAXX" or symbol == "Cash":
        # Create a dummy series of 1.0
        dates = pd.date_range(start, end, freq='D')
        return pd.Series(1.0, index=dates)
        
    try:
        px = yf.download(symbol, start=start, end=end, progress=False)
        if px.empty: 
            return pd.Series()
        
        # yfinance might return multi-index columns if only 1 ticker but formatted that way
        if "Close" in px.columns:
            close_px = px["Close"]
        else:
            return pd.Series()
            
        # If it's a DataFrame (multi-symbol or check), squeeze it
        if isinstance(close_px, pd.DataFrame):
            close_px = close_px.iloc[:, 0]
            
        close_px = close_px.dropna()
        close_px = close_px[~close_px.index.duplicated(keep='last')]
        # Ensure index is timezone-naive and normalized
        if close_px.index.tz is not None:
             close_px.index = close_px.index.tz_localize(None)
        close_px.index = close_px.index.normalize()
        
        close_px = close_px.sort_index()
        return close_px
    except Exception:
        return pd.Series()

def build_portfolio_profit_curve(trades, prices):
    # Check if prices is empty or has no valid dates
    if prices.empty or prices.index.isna().all():
        return pd.DataFrame(columns=['Unrealized', 'Realized', 'TotalProfit', 'CurrentValue', 'CurrentQty'])
    
    prices = prices[~prices.index.duplicated(keep='last')]
    prices = prices.sort_index()
    
    if prices.empty:
         return pd.DataFrame(columns=['Unrealized', 'Realized', 'TotalProfit', 'CurrentValue', 'CurrentQty'])

    all_days = pd.date_range(prices.index.min(), prices.index.max(), freq='D')
    ff_prices = prices.reindex(all_days).ffill()
    trades = trades.sort_values('Run Date')
    trades = trades.set_index('Run Date')

    lots = []
    realized_profits = []  # (date, realized profit from sells/dividends)
    profit_curve = []
    
    # Track accumulated cashflows for this symbol (Internal Rate of Return perspective)
    # Cash In: Cost of Buys. Cash Out: Proceeds of Sells + Dividends.
    cum_cash_in = 0.0
    cum_cash_out = 0.0
    
    for date in all_days:
        if date in trades.index:
            # Handle multiple trades on same day
            day_trades = trades.loc[[date]]
            for _, row in day_trades.iterrows():
                qty = row["Quantity"]
                amt = float(row["Amount ($)"])
                atype = row["ActionType"]
                
                # PRICE handling:
                # Some rows might imply price = Amount/Qty if Price is missing/zero
                # But usually Price column is there.
                trade_price = float(row["Price ($)"]) if pd.notna(row["Price ($)"]) and row["Price ($)"] != 0 else 0
                if trade_price == 0 and qty != 0:
                     trade_price = abs(amt / qty)

                if atype == "BUY" or atype == "REINVEST":
                    # For REINVEST, it's effectively a BUY.
                    # The Income aspect of Reinvest is handled by the DIVIDEND row usually present.
                    # If strictly one row "Dividend Reinvestment", then we count it as both Income and Buy?
                    # Safer assumption: Treat REINVEST as BUY. The DIVIDEND row handles the profit realization.
                    lots.append({"qty": qty, "cost": trade_price})
                    cost_basis = qty * trade_price
                    cum_cash_in += cost_basis
                    
                elif atype == "SELL":
                    sell_qty = abs(qty) # Ensure positive
                    sell_price = trade_price
                    
                    # Logic: FIFO
                    profit_from_sale = 0
                    cost_of_sold = 0
                    
                    remaining_to_sell = sell_qty
                    while remaining_to_sell > 0 and lots:
                        lot = lots[0]
                        if lot["qty"] <= remaining_to_sell:
                            # Consume entire lot
                            profit_from_sale += (sell_price - lot["cost"]) * lot["qty"]
                            cost_of_sold += lot["cost"] * lot["qty"]
                            remaining_to_sell -= lot["qty"]
                            lots.pop(0)
                        else:
                            # Partial lot
                            profit_from_sale += (sell_price - lot["cost"]) * remaining_to_sell
                            cost_of_sold += lot["cost"] * remaining_to_sell
                            lot["qty"] -= remaining_to_sell
                            remaining_to_sell = 0
                            
                    realized_profits.append((date, profit_from_sale))
                    cum_cash_out += (cost_of_sold + profit_from_sale) # This equals Proceeds
                    
                elif atype == "DIVIDEND":
                    realized_profits.append((date, amt))
                    cum_cash_out += amt

        # Daily Valuation
        price_val = ff_prices.loc[date]
        if isinstance(price_val, pd.Series):
            price_val = price_val.iloc[-1]
        
        try:
            px = float(price_val)
        except Exception:
            px = np.nan

        current_qty = sum([lot["qty"] for lot in lots])
        
        # Helper for NaN price
        val_px = 0.0 if np.isnan(px) else px
        
        # Calculate Metrics
        current_value = current_qty * val_px
        cost_basis_held = sum([lot["qty"] * lot["cost"] for lot in lots])
        
        unrealized = current_value - cost_basis_held if not np.isnan(px) else 0.0
        
        # Realized is cumulative sum of realized events
        cum_realized = sum([p for d, p in realized_profits if d <= date])
        
        # Total Profit = Realized + Unrealized
        # Alternative: MarketValue + CashOut - CashIn
        # Let's use Realized + Unrealized as it is cleaner for display
        total_profit = cum_realized + unrealized

        profit_curve.append({
            "Date": date,
            "Unrealized": unrealized,
            "Realized": cum_realized,
            "TotalProfit": total_profit,
            "CurrentValue": current_value,
            "CurrentQty": current_qty
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
        end_value = month_ends.iloc[i]
        start_value = month_starts.iloc[i]
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
        end_value = year_ends.iloc[i]
        start_value = year_starts.iloc[i]
        if pd.isna(start_value) or start_value == 0:
            returns.append(np.nan)
            continue
        ret = (end_value - start_value) / start_value * 100
        returns.append(ret)
    return pd.Series(returns, index=years)

def wrap_labels(labels, width=15):
    wrapped = []
    for label in labels:
        # Try to break at spaces, otherwise just insert <br> every width chars
        parts = []
        while len(label) > width:
            idx = label[:width].rfind(' ')
            if idx == -1:
                idx = width
            parts.append(label[:idx])
            label = label[idx:].lstrip()
        parts.append(label)
        wrapped.append('<br>'.join(parts))
    return wrapped

def main():
    st.title("Portfolio Analysis (Correct Realized/Unrealized Profits)")

    uploaded_files = st.file_uploader(
        "Upload one or more files (Excel/CSV) from Fidelity (Optional).",
        type=["xls", "xlsx", "csv"],
        accept_multiple_files=True
    )

    df = load_trade_data(uploaded_files)
    
    if df.empty:
        st.info("Please upload your trade files, or ensure they are present in the 'data' folder.")
        return

    df = preprocess_trades(df)
    symbols = get_symbol_list(df)
    symbol_name_map = get_symbol_name_map(df)
    t0 = df["Run Date"].min() - pd.Timedelta(days=5)
    t1 = datetime.today()

    # Get price history for all symbols
    all_prices = {}
    failed_symbols = []
    for symbol in symbols:
        prices = get_price_history(symbol, t0, t1)
        if prices.empty:
            failed_symbols.append(symbol)
            st.warning(f"⚠️ Could not retrieve price data for symbol: {symbol}. This symbol will be skipped.")
        all_prices[symbol] = prices

    # Build and sum profit curves for all symbols
    profit_curves = {}
    for symbol in symbols:
        if symbol in failed_symbols:
            continue  # Skip symbols with no price data
        trades = df[df["Symbol"] == symbol]
        prices = all_prices[symbol]
        curve = build_portfolio_profit_curve(trades, prices)
        if not curve.empty:
            profit_curves[symbol] = curve

    # Check if we have any valid profit curves
    if not profit_curves:
        st.error("❌ No valid price data could be retrieved for any symbols. Please check that the symbols in your file are correct and try again.")
        return
    
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
    current_portfolio_value = portfolio_curve['CurrentValue'].iloc[-1]

    # Calculate Net Invested and Portfolio XIRR using External Flows (Deposits/Withdrawals)
    # Filter for external flows
    ext_flows = df[df["ActionType"].isin(["DEPOSIT", "WITHDRAWAL"])].copy()
    
    # Sign convention for XIRR:
    # Deposits (Investments) -> Negative
    # Withdrawals (Returns) -> Positive
    # Final Value -> Positive
    
    xirr_flows = []
    net_invested = 0.0
    
    for _, row in ext_flows.iterrows():
        dt = row["Run Date"]
        amt = row["Amount ($)"]
        atype = row["ActionType"]
        
        # In file, Deposit is Positive, Withdrawal is Negative (usually).
        # Check signs. Assuming Dep +, With -
        if atype == "DEPOSIT":
             # For XIRR: Negative flow (money going in)
             xirr_flows.append((dt, -1 * abs(amt)))
             net_invested += abs(amt)
        elif atype == "WITHDRAWAL":
             # For XIRR: Positive flow (money coming out)
             xirr_flows.append((dt, abs(amt))) # File has negative, so abs() is positive
             net_invested -= abs(amt) # Reduces net invested
             
    # Append Final Value
    xirr_flows.append((end_date, current_portfolio_value))
    
    xirr_portfolio = xirr(xirr_flows)
    
    total_profit = current_portfolio_value - net_invested
    total_return_pct = (total_profit / net_invested) if net_invested != 0 else 0

    st.write(f"**Portfolio XIRR:** {xirr_portfolio:.2%}")
    st.write(f"**Total Return (%):** {total_return_pct:.2%}")
    st.write(f"**Total Profit ($):** {total_profit:,.2f}")
    
    st.info(f"debug: Net Invested: ${net_invested:,.2f}, Current Value: ${current_portfolio_value:,.2f}")

    # --- 2. Symbol-level comparison & XIRR table ---
    st.header("2. Fund Movements (XIRR Table)")
    symbol_xirr = []
    symbol_total_return = []
    
    # Sort symbols by current value (descending) for better display, or just alphabetical
    # Using original order for now but filtered
    for symbol in symbols:
        if symbol not in profit_curves:
            continue
            
        trades = df[df["Symbol"] == symbol]
        # curve is guaranteed to exist now
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
        
        # Simple Return for Symbol: (Value + CashOut) / CashIn - 1?
        # Or just (Profit / CashIn)
        # CashIn = Cost of Buys
        cash_in = sum([abs(row["Amount ($)"]) for _, row in trades.iterrows() if row["ActionType"] in ["BUY", "REINVEST"] and row["Amount ($)"] < 0]) 
        # Note: BUY amount usually negative in file?
        # Let's be careful. If preprocess_trades didn't enforce sign, we check.
        # User file inspection: BUY -Amount? No, typically "Amount" is cost.
        # Let's rely on 'amount' being negative for money spent if we didn't force it?
        # In preprocess: we strip $ and ().
        # Let's Assume: BUY cost is implied by Quantity * Price?
        # Better: use logic from profit curve building (Cost Basis)
        
        # Let's grab cumulative profit from curve
        sym_profit = curve['TotalProfit'].iloc[-1]
        
        # Invested estimate: Current Cost Basis? No, that misses sold items.
        # Sum of all BUYS cost.
        buy_rows = trades[trades["ActionType"].isin(["BUY", "REINVEST"])]
        # Cost is roughly sum of amounts (since file likely has negative amounts for buys or we treat them as cost)
        # Actually in file "Amount ($)" for BUY is usually negative.
        # Let's sum abs(Amount) for BUYs.
        invested = buy_rows["Amount ($)"].abs().sum()
        
        total_ret = (sym_profit / invested) if invested != 0 else 0
        fund_name = symbol_name_map.get(symbol, symbol)
        symbol_xirr.append((symbol, fund_name, sym_xirr))
        symbol_total_return.append((fund_name, total_ret * 100))

    # Bar chart: total return % per fund
    sym_df = pd.DataFrame(symbol_total_return, columns=["Fund Name", "TotalReturn"])
    wrapped_fund_names = wrap_labels(sym_df["Fund Name"].tolist(), width=15)
    fig = go.Figure(data=[
        go.Bar(x=wrapped_fund_names, y=sym_df["TotalReturn"])
    ])
    fig.update_layout(
        barmode='group',
        title="Fund Total Return (%)",
        yaxis_title="Return (%)",
        xaxis_title="Fund",
        xaxis_tickangle=0,
        margin=dict(b=120),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Data Table: XIRR per fund
    table_df = pd.DataFrame(symbol_xirr, columns=["Symbol", "Fund Name", "XIRR"])
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
