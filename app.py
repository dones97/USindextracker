import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import newton, brentq
import plotly.graph_objs as go
import plotly.colors as pcolors
import os
import glob

from portfolio_analysis import render_portfolio_analysis

# S&P 500 benchmark styling, shared by every chart
BENCHMARK_COLOR = "#2c7fb8"

def xnpv(rate, cashflows):
    t0 = cashflows[0][0]
    return sum([cf / (1 + rate) ** ((t - t0).days / 365) for t, cf in cashflows])

def xirr(cashflows):
    try:
        return newton(lambda r: xnpv(r, cashflows), 0.1)
    except Exception:
        pass
    # Newton can fail on short or deeply negative histories; fall back to a bracketed solve
    try:
        return brentq(lambda r: xnpv(r, cashflows), -0.99, 100.0)
    except Exception:
        return np.nan

def classify_action(row):
    action = str(row["Action"]).upper()
    if "YOU BOUGHT" in action:
        return "BUY"
    if "YOU SOLD" in action:
        return "SELL"
    if "REINVESTMENT" in action:
        return "REINVEST"
    if "DISTRIBUTION" in action:
        # Fidelity books stock splits as a "DISTRIBUTION" of new shares (no cash moves;
        # Amount is just the market value). A distribution with no shares is cash income.
        try:
            qty = float(row.get("Quantity", 0) or 0)
        except (TypeError, ValueError):
            qty = 0.0
        return "SHARE_DIST" if qty > 0 else "DIVIDEND"
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
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]
    # Coerce errors to NaT to handle Fidelity's textual footer rows, then drop those rows
    df["Run Date"] = pd.to_datetime(df["Run Date"], format='mixed', dayfirst=False, errors='coerce')
    df = df.dropna(subset=["Run Date"])
    df["Run Date"] = df["Run Date"].dt.normalize()
    
    # helper for amount parsing
    def parse_amount(x):
        if pd.isna(x):
            return 0.0
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).replace('$', '').replace(',', '').strip()
        if '(' in s and ')' in s:
            s = '-' + s.replace('(', '').replace(')', '')
        if not s:
            return 0.0
        try:
            return float(s)
        except Exception:
            return 0.0
        
    if "Amount ($)" in df.columns:
        df["Amount ($)"] = df["Amount ($)"].apply(parse_amount)
    if "Quantity" in df.columns:
        df["Quantity"] = df["Quantity"].apply(lambda x: parse_amount(x) if pd.notna(x) else 0.0)
    if "Price ($)" in df.columns:
        df["Price ($)"] = df["Price ($)"].apply(lambda x: parse_amount(x) if pd.notna(x) else 0.0)
    
    df["ActionType"] = df.apply(classify_action, axis=1)
    
    # Filter to relevant types
    relevant_types = ["BUY", "SELL", "DIVIDEND", "REINVEST", "DEPOSIT", "WITHDRAWAL", "SHARE_DIST"]
    df = df[df["ActionType"].isin(relevant_types)]
    
    return df

def load_trade_data(uploaded_files=None, data_dir="data", return_info=False):
    raw_dfs = []
    repo_file_names = []
    uploaded_file_names = []
    
    # 1. Load files from repository directory
    if data_dir and os.path.exists(data_dir):
        exts = ["*.csv", "*.xlsx", "*.xls"]
        file_paths = []
        for ext in exts:
            file_paths.extend(glob.glob(os.path.join(data_dir, ext)))
        file_paths = sorted(file_paths)
        
        for fp in file_paths:
            try:
                base_name = os.path.basename(fp)
                if fp.endswith(".csv"):
                    df = pd.read_csv(fp)
                else:
                    df = pd.read_excel(fp)
                raw_dfs.append((base_name, "repository", df))
                repo_file_names.append(base_name)
            except Exception:
                pass

    # 2. Load manually uploaded files
    if uploaded_files:
        for f in uploaded_files:
            try:
                fname = getattr(f, "name", "uploaded_file")
                if hasattr(f, "seek"):
                    f.seek(0)
                if fname.endswith(".csv"):
                    df = pd.read_csv(f)
                else:
                    df = pd.read_excel(f)
                if hasattr(f, "seek"):
                    f.seek(0)
                raw_dfs.append((fname, "upload", df))
                uploaded_file_names.append(fname)
            except Exception:
                pass

    if not raw_dfs:
        empty_info = {
            "total_raw_rows": 0,
            "final_rows": 0,
            "num_duplicates": 0,
            "repo_files_count": 0,
            "uploaded_files_count": 0,
            "repo_files": [],
            "uploaded_files": []
        }
        return (pd.DataFrame(), empty_info) if return_info else pd.DataFrame()

    # Preprocess each file individually to produce normalized fields
    processed_dfs = []
    for fname, source_type, raw_df in raw_dfs:
        try:
            pdf = preprocess_trades(raw_df)
            if pdf.empty:
                continue
            pdf["_source_file"] = fname
            pdf["_source_type"] = source_type
            
            # Construct normalized trade signature components
            amt_s = pdf["Amount ($)"].replace(-0.0, 0.0).round(2).astype(str)
            qty_s = pdf["Quantity"].replace(-0.0, 0.0).round(6).astype(str)
            price_s = pdf["Price ($)"].replace(-0.0, 0.0).round(4).astype(str)
            sym_s = pdf["Symbol"].fillna("__CASH__").astype(str).str.strip().str.upper()
            action_s = pdf["ActionType"].astype(str).str.strip().str.upper()
            date_s = pdf["Run Date"].dt.strftime("%Y-%m-%d")
            
            # Signature key representing the exact financial transaction
            pdf["_sig_key"] = date_s + "|" + action_s + "|" + sym_s + "|" + amt_s + "|" + qty_s + "|" + price_s
            # Intra-file occurrence index preserves multiple legitimate identical trades placed in a single statement
            pdf["_occ"] = pdf.groupby("_sig_key").cumcount()
            pdf["_dedup_id"] = pdf["_sig_key"] + "|" + pdf["_occ"].astype(str)
            
            processed_dfs.append(pdf)
        except Exception:
            pass

    if not processed_dfs:
        empty_info = {
            "total_raw_rows": 0,
            "final_rows": 0,
            "num_duplicates": 0,
            "repo_files_count": len(repo_file_names),
            "uploaded_files_count": len(uploaded_file_names),
            "repo_files": repo_file_names,
            "uploaded_files": uploaded_file_names
        }
        return (pd.DataFrame(), empty_info) if return_info else pd.DataFrame()

    combined_df = pd.concat(processed_dfs, ignore_index=True)
    total_raw_rows = len(combined_df)

    # Identify duplicate rows across files
    is_duplicate = combined_df.duplicated(subset=["_dedup_id"], keep="first")
    num_duplicates = int(is_duplicate.sum())
    
    # Keep only the unique set of entries
    clean_df = combined_df[~is_duplicate].copy()
    
    # Sort chronologically by Run Date
    clean_df.sort_values(by="Run Date", inplace=True)
    clean_df.reset_index(drop=True, inplace=True)

    info = {
        "total_raw_rows": total_raw_rows,
        "final_rows": len(clean_df),
        "num_duplicates": num_duplicates,
        "repo_files_count": len(repo_file_names),
        "uploaded_files_count": len(uploaded_file_names),
        "repo_files": repo_file_names,
        "uploaded_files": uploaded_file_names,
    }

    # Clean up internal helper columns
    clean_df.drop(columns=["_sig_key", "_occ", "_dedup_id", "_source_file", "_source_type"], inplace=True, errors="ignore")

    return (clean_df, info) if return_info else clean_df

def extract_daily_cash(uploaded_files=None, data_dir="data"):
    files_data = []
    if os.path.exists(data_dir):
        for f in glob.glob(os.path.join(data_dir, "*.csv")):
            try: files_data.append(pd.read_csv(f))
            except Exception: pass
        for f in glob.glob(os.path.join(data_dir, "*.xlsx")):
            try: files_data.append(pd.read_excel(f))
            except Exception: pass
        for f in glob.glob(os.path.join(data_dir, "*.xls")):
            try: files_data.append(pd.read_excel(f))
            except Exception: pass
    if uploaded_files:
        for file in uploaded_files:
            try:
                if hasattr(file, 'seek'):
                    file.seek(0)
                fname = getattr(file, 'name', 'file.csv')
                if fname.endswith('.csv'):
                    files_data.append(pd.read_csv(file))
                else:
                    files_data.append(pd.read_excel(file))
                if hasattr(file, 'seek'):
                    file.seek(0)
            except Exception: pass
            
    if not files_data:
        return pd.Series(dtype=float)
        
    # Sort files by newest date first so latest records win if dates overlap
    file_tuples = []
    for f_df in files_data:
        if 'Run Date' in f_df.columns:
            dts = pd.to_datetime(f_df['Run Date'], format='mixed', errors='coerce').dropna()
            max_dt = dts.max() if not dts.empty else pd.Timestamp.min
            file_tuples.append((max_dt, f_df))
    file_tuples.sort(key=lambda x: x[0], reverse=True)
    
    records = []
    for _, f_df in file_tuples:
        if 'Cash Balance ($)' not in f_df.columns or 'Run Date' not in f_df.columns:
            continue
        temp = f_df.copy()
        temp['_Date'] = pd.to_datetime(temp['Run Date'], format='mixed', errors='coerce').dt.normalize()
        temp = temp.dropna(subset=['_Date', 'Cash Balance ($)'])
        for dt, group in temp.groupby('_Date', sort=False):
            val = group['Cash Balance ($)'].iloc[0]
            if pd.notna(val):
                s = str(val).replace('$', '').replace(',', '').strip()
                try: records.append({'Date': dt, 'Cash': float(s)})
                except Exception: pass
    if not records:
        return pd.Series(dtype=float)
    cash_df = pd.DataFrame(records).drop_duplicates(subset=['Date'], keep='first').set_index('Date').sort_index()
    return cash_df['Cash']

def get_symbol_list(df):
    syms = df["Symbol"].dropna().unique()
    # clean strings
    syms = [str(s).strip() for s in syms]
    return [s for s in syms if s and s.upper() != "NAN" and s != "." and s.upper() not in ["SPAXX", "CASH", "FDRXX"]]

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

def get_price_history(symbol, start, end, adjusted=False):
    # Special handling for Cash/Money Market
    if symbol == "SPAXX" or symbol == "Cash":
        # Create a dummy series of 1.0
        dates = pd.date_range(start, end, freq='D')
        return pd.Series(1.0, index=dates)

    try:
        px = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
        if px.empty:
            return pd.Series()

        # Adjusted close includes reinvested dividends (total return)
        price_col = "Adj Close" if adjusted else "Close"
        # yfinance might return multi-index columns if only 1 ticker but formatted that way
        if price_col in px.columns:
            close_px = px[price_col]
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

def get_risk_free_rate(start, end):
    try:
        irx = yf.download("^IRX", start=start, end=end, progress=False, auto_adjust=False)
        if irx.empty: return 0.04  # Fallback to 4%
        if "Close" in irx.columns:
            close_px = irx["Close"]
        else:
            return 0.04
        if isinstance(close_px, pd.DataFrame):
            close_px = close_px.iloc[:, 0]
        # Yield is given as percentage (e.g., 4.5 for 4.5%), so divide by 100 for decimal
        return float(close_px.mean()) / 100.0
    except Exception:
        return 0.04

def get_split_adjustments(symbol):
    """Retrieve stock split history for a symbol from yfinance.
    
    Returns a list of (date, ratio) tuples sorted by date.
    For a 3-for-1 split, ratio = 3.0, meaning:
      - Quantity should be multiplied by 3
      - Price should be divided by 3
    """
    try:
        ticker = yf.Ticker(symbol)
        splits = ticker.splits
        if splits.empty:
            return []
        result = []
        for dt, ratio in splits.items():
            # Normalize the date (remove timezone, normalize to midnight)
            if hasattr(dt, 'tz') and dt.tz is not None:
                dt = dt.tz_localize(None)
            dt = pd.Timestamp(dt).normalize()
            if ratio != 0 and ratio != 1.0:
                result.append((dt, float(ratio)))
        return sorted(result, key=lambda x: x[0])
    except Exception:
        return []

def get_cumulative_split_factor(trade_date, split_history):
    """Calculate the cumulative split factor for a trade made on trade_date.
    
    For splits that occurred AFTER the trade_date, multiply the factors together.
    This tells us how many current shares one original share became.
    """
    factor = 1.0
    for split_date, ratio in split_history:
        if split_date > trade_date:
            factor *= ratio
    return factor

def merge_share_distributions(trades, split_history, tolerance_days=7):
    """Add splits implied by SHARE_DIST rows that yfinance does not already report.

    A distribution of D new shares on top of H held shares is a (H + D) / H split.
    If yfinance already has a split within tolerance_days of the distribution, it is
    left alone so the split is not applied twice.
    """
    splits = list(split_history or [])
    dists = trades[trades["ActionType"] == "SHARE_DIST"].sort_values("Run Date")
    for _, dist in dists.iterrows():
        dist_date = dist["Run Date"]
        if any(abs((s_date - dist_date).days) <= tolerance_days for s_date, _ in splits):
            continue
        prior = trades[trades["Run Date"] < dist_date]
        held = (
            prior.loc[prior["ActionType"].isin(["BUY", "REINVEST", "SHARE_DIST"]), "Quantity"].sum()
            - prior.loc[prior["ActionType"] == "SELL", "Quantity"].abs().sum()
        )
        if held > 0:
            splits.append((dist_date, (held + dist["Quantity"]) / held))
    return sorted(splits, key=lambda x: x[0])

def simulate_spy_portfolio(trades, spy_prices):
    if spy_prices.empty:
        return pd.DataFrame(), np.nan
    
    spy_prices = spy_prices[~spy_prices.index.duplicated(keep='last')].sort_index()
    all_days = pd.date_range(spy_prices.index.min(), spy_prices.index.max(), freq='D')
    ff_spy = spy_prices.reindex(all_days).ffill()
    
    # Filter trades to only those that moved money in/out of stocks
    valid_actions = trades[trades["ActionType"].isin(["BUY", "SELL", "REINVEST"])]
    daily_flows = valid_actions.groupby("Run Date")["Amount ($)"].sum()
    
    spy_curve = []
    spy_qty = 0.0
    spy_invested = 0.0
    spy_xirr_flows = []
    
    for date in all_days:
        px = ff_spy.loc[date]
        if pd.isna(px): continue
            
        if date in daily_flows.index:
            flow = daily_flows.loc[date]
            # flow is negative for money invested (BUY), positive for money returned (SELL)
            shares = -flow / px
            spy_qty += shares
            spy_invested += -flow
            spy_xirr_flows.append((date, flow))
            
        current_val = spy_qty * px
        unrealized = current_val - spy_invested
        
        spy_curve.append({
            "Date": date,
            "CurrentValue": current_val,
            "TotalProfit": unrealized,
            "Unrealized": unrealized,
            "Realized": 0.0
        })
        
    if not spy_curve:
        return pd.DataFrame(), np.nan
        
    spy_df = pd.DataFrame(spy_curve).set_index("Date")
    last_val = spy_df["CurrentValue"].iloc[-1]
    if last_val > 0:
        spy_xirr_flows.append((spy_df.index[-1], last_val))
    spy_xirr_val = xirr(spy_xirr_flows)
    
    return spy_df, spy_xirr_val

def build_portfolio_profit_curve(trades, prices, split_history=None):
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
    
    # Adjust trade quantities and prices for stock splits
    # yfinance returns split-adjusted prices, so we need to adjust the trade data to match
    if split_history:
        trades = trades.copy()
        adjusted_qtys = []
        adjusted_prices = []
        for _, row in trades.iterrows():
            factor = get_cumulative_split_factor(row["Run Date"], split_history)
            adjusted_qtys.append(row["Quantity"] * factor)
            price_val = row["Price ($)"]
            if pd.notna(price_val) and price_val != 0:
                adjusted_prices.append(price_val / factor)
            else:
                adjusted_prices.append(price_val)
        trades["Quantity"] = adjusted_qtys
        trades["Price ($)"] = adjusted_prices
    
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
    month_ends = curve.resample("ME").last()
    month_starts = curve.resample("ME").first()
    returns = []
    months = month_ends.index
    for i in range(len(months)):
        start_val = month_starts["CurrentValue"].iloc[i]
        end_val = month_ends["CurrentValue"].iloc[i]
        start_profit = month_starts["TotalProfit"].iloc[i]
        end_profit = month_ends["TotalProfit"].iloc[i]
        
        profit_change = end_profit - start_profit
        unrealized_change = month_ends["Unrealized"].iloc[i] - month_starts["Unrealized"].iloc[i]
        net_flows = (end_val - start_val) - unrealized_change
        
        denom = start_val + (net_flows / 2.0)
        if denom <= 0:
            denom = end_val / 2.0 if end_val > 0 else 0
            
        if denom <= 0:
            returns.append(0.0)
        else:
            returns.append((profit_change / denom) * 100)
    return pd.Series(returns, index=months)

def compute_annual_returns(curve):
    year_ends = curve.resample("YE").last()
    year_starts = curve.resample("YE").first()
    years = year_ends.index
    returns = []
    for i in range(len(years)):
        start_val = year_starts["CurrentValue"].iloc[i]
        end_val = year_ends["CurrentValue"].iloc[i]
        start_profit = year_starts["TotalProfit"].iloc[i]
        end_profit = year_ends["TotalProfit"].iloc[i]
        
        profit_change = end_profit - start_profit
        unrealized_change = year_ends["Unrealized"].iloc[i] - year_starts["Unrealized"].iloc[i]
        net_flows = (end_val - start_val) - unrealized_change
        
        denom = start_val + (net_flows / 2.0)
        if denom <= 0:
            denom = end_val / 2.0 if end_val > 0 else 0
            
        if denom <= 0:
            returns.append(0.0)
        else:
            returns.append((profit_change / denom) * 100)
    return pd.Series(returns, index=years)

def benchmark_return_for_trades(trades, spy_adj, invested, end_date):
    """S&P 500 total return (%) had the fund's own buys/sells gone into SPY instead.

    Uses the same dollar flows on the same dates, valued at end_date (today for a
    held fund, the last trade date for a closed one) and divided by the same
    invested amount as the fund's total return, so the two are directly comparable.
    """
    if spy_adj.empty or invested == 0:
        return np.nan
    flows = trades[trades["ActionType"].isin(["BUY", "SELL"])]
    spy_qty = 0.0
    net_cash = 0.0
    for _, row in flows.iterrows():
        px = spy_adj.asof(row["Run Date"])
        if pd.isna(px):
            px = spy_adj.iloc[0]
        amt = float(row["Amount ($)"])
        spy_qty += -amt / px
        net_cash += amt
    end_px = spy_adj.asof(end_date)
    if pd.isna(end_px):
        return np.nan
    return (spy_qty * end_px + net_cash) / invested * 100

def benchmark_xirr_for_trades(trades, spy_adj, end_date):
    """S&P 500 XIRR (%) had the fund's own buys/sells gone into SPY instead.

    Same dollar flows on the same dates as benchmark_return_for_trades, with the
    SPY position valued at end_date as the final inflow.
    """
    if spy_adj.empty:
        return np.nan
    flows = trades[trades["ActionType"].isin(["BUY", "SELL"])]
    spy_qty = 0.0
    cashflows = []
    for _, row in flows.iterrows():
        px = spy_adj.asof(row["Run Date"])
        if pd.isna(px):
            px = spy_adj.iloc[0]
        amt = float(row["Amount ($)"])
        spy_qty += -amt / px
        cashflows.append((row["Run Date"], amt))
    end_px = spy_adj.asof(end_date)
    if pd.isna(end_px) or not cashflows:
        return np.nan
    if spy_qty * end_px > 0:
        cashflows.append((end_date, spy_qty * end_px))
    return xirr(cashflows) * 100

def benchmark_period_returns(spy_adj, period_ends, freq, port_start, port_end):
    """S&P 500 total return (%) for each period, clipped to the portfolio's lifetime.

    A period that began before the portfolio existed (or is still in progress)
    only counts the days the portfolio was actually invested.
    """
    results = []
    for p_end in period_ends:
        p_start = p_end.replace(day=1) if freq == "M" else p_end.replace(month=1, day=1)
        start_ref = max(p_start - pd.Timedelta(days=1), port_start)
        end_ref = min(p_end, port_end)
        if spy_adj.empty or start_ref >= end_ref:
            results.append(np.nan)
            continue
        px0, px1 = spy_adj.asof(start_ref), spy_adj.asof(end_ref)
        results.append((px1 / px0 - 1) * 100 if pd.notna(px0) and pd.notna(px1) and px0 else np.nan)
    return pd.Series(results, index=period_ends)

def databar_chart(labels, values, benchmarks, title, benchmark_name="S&P 500"):
    """Horizontal data-bar chart: label | value % | bar, with a solid benchmark line per row.

    All rows share one x scale, and bars are shaded red (negative) to green (positive)
    on a symmetric scale so equal magnitudes get equal intensity.
    """
    values = list(values)
    benchmarks = list(benchmarks)
    n = len(values)
    vals_arr = np.array(values, dtype=float)
    bench_arr = np.array(benchmarks, dtype=float)

    max_abs = np.nanmax(np.abs(vals_arr)) if n else 1.0
    max_abs = max_abs if max_abs > 0 else 1.0
    colors = pcolors.sample_colorscale("RdYlGn", [(v / max_abs + 1) / 2 for v in np.nan_to_num(vals_arr)])

    all_x = np.concatenate([vals_arr, bench_arr, [0.0]])
    x_min, x_max = np.nanmin(all_x), np.nanmax(all_x)
    pad = (x_max - x_min) * 0.05 or 1.0
    x_range = [x_min - pad if x_min < 0 else 0, x_max + pad if x_max > 0 else 0]

    # Row positions 0..n-1 on a category axis keep bars, labels and benchmark lines aligned
    # (labels like "2025" or "2025-07" would otherwise be auto-typed as numbers/dates).
    positions = list(range(n))
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values, y=positions, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        customdata=np.column_stack([labels, bench_arr]).tolist() if n else [],
        hovertemplate="%{customdata[0]}<br>Return: %{x:.1f}%<br>" + benchmark_name
                      + ": %{customdata[1]:.1f}%<extra></extra>",
        showlegend=False,
    ))
    # Invisible trace on the top axis so its tick labels are drawn
    fig.add_trace(go.Scatter(
        x=x_range, y=[0, 0], xaxis="x2", mode="markers", marker=dict(opacity=0),
        hoverinfo="skip", showlegend=False,
    ))

    for i, b in enumerate(benchmarks):
        if pd.isna(b):
            continue
        fig.add_shape(type="line", x0=b, x1=b, y0=i - 0.45, y1=i + 0.45, xref="x", yref="y",
                      line=dict(color=BENCHMARK_COLOR, width=2))
    # Legend entry for the benchmark lines (shapes don't appear in legends)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines", name=benchmark_name,
        line=dict(color=BENCHMARK_COLOR, width=2),
    ))

    # Return % column sits in the gap between the row labels and the bars
    pct_col_px = 90
    # Reserve the left part of the figure for labels so separators can span label, % and bar
    max_label = max((len(str(l)) for l in labels), default=0)
    label_frac = min(0.45, (max_label * 7 + pct_col_px + 30) / 950)
    for i, v in enumerate(values):
        fig.add_annotation(
            x=label_frac, xref="paper", xanchor="right", xshift=-10, y=i, yref="y",
            text=f"<b>{v:.1f}%</b>" if pd.notna(v) else "n/a", showarrow=False, font=dict(size=13),
        )
    # Subtle separators between rows so each label reads across to its bar
    for i in range(n + 1):
        fig.add_shape(type="line", x0=0, x1=1, xref="paper", y0=i - 0.5, y1=i - 0.5, yref="y",
                      line=dict(color="rgba(128,128,128,0.25)", width=1), layer="below")

    height = max(260, 30 * n + 160)
    margin = dict(l=10, r=20, t=110, b=50)
    plot_h = height - margin["t"] - margin["b"]
    fig.update_yaxes(
        type="category", tickvals=positions, ticktext=labels, autorange="reversed",
        ticklabelstandoff=pct_col_px, automargin=False, showgrid=False,
    )
    fig.update_layout(
        title=title,
        height=height,
        bargap=0.3,
        margin=margin,
        xaxis=dict(range=x_range, domain=[label_frac, 1], ticksuffix="%", zeroline=True, zerolinewidth=1, showgrid=True),
        xaxis2=dict(overlaying="x", side="top", matches="x", domain=[label_frac, 1], ticksuffix="%", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1 + 30 / plot_h, xanchor="right", x=1.0),
    )
    return fig

def main():
    st.title("Portfolio Analysis (Correct Realized/Unrealized Profits)")

    uploaded_files = st.file_uploader(
        "Upload one or more files (Excel/CSV) from Fidelity (Optional).",
        type=["xls", "xlsx", "csv"],
        accept_multiple_files=True
    )

    df, load_info = load_trade_data(uploaded_files, return_info=True)
    
    if df.empty:
        st.info("Please upload your trade files, or ensure they are present in the 'data' folder.")
        return

    # Inform the user when duplicate entries across overlapping reports were identified and removed
    if load_info.get("num_duplicates", 0) > 0:
        st.info(
            f"ℹ️ **Overlapping Trade Reports Merged:** Detected and resolved **{load_info['num_duplicates']} duplicate trade entries** "
            f"across reports ({load_info['repo_files_count']} repository file(s) and {load_info['uploaded_files_count']} uploaded file(s)). "
            f"A single unified set of transactions is used."
        )
    elif load_info.get("uploaded_files_count", 0) > 0:
        st.success(
            f"✅ Successfully loaded {load_info['uploaded_files_count']} uploaded report(s) alongside {load_info['repo_files_count']} repository file(s) with no duplicates."
        )

    df = preprocess_trades(df)
    symbols = get_symbol_list(df)
    symbol_name_map = get_symbol_name_map(df)
    t0 = df["Run Date"].min() - pd.Timedelta(days=5)
    t1 = datetime.today()

    # Get price history and split adjustments for all symbols
    all_prices = {}
    all_splits = {}
    failed_symbols = []
    for symbol in symbols:
        prices = get_price_history(symbol, t0, t1)
        if prices.empty:
            failed_symbols.append(symbol)
            st.warning(f"⚠️ Could not retrieve price data for symbol: {symbol}. This symbol will be skipped.")
        all_prices[symbol] = prices
        all_splits[symbol] = get_split_adjustments(symbol)

    # Build and sum profit curves for all symbols
    profit_curves = {}
    for symbol in symbols:
        if symbol in failed_symbols:
            continue  # Skip symbols with no price data
        trades = df[df["Symbol"] == symbol]
        prices = all_prices[symbol]
        split_history = merge_share_distributions(trades, all_splits.get(symbol, []))
        curve = build_portfolio_profit_curve(trades, prices, split_history)
        if not curve.empty:
            profit_curves[symbol] = curve

    # Check if we have any valid profit curves
    if not profit_curves:
        st.error("❌ No valid price data could be retrieved for any symbols. Please check that the symbols in your file are correct and try again.")
        return
    
    tab_returns, tab_analysis = st.tabs(["Returns Tracking", "Portfolio Analysis"])

    with tab_returns:
        # Portfolio-level curve: sum across all symbols
        all_dates = sorted(set().union(*[curve.index for curve in profit_curves.values()]))
        portfolio_curve = pd.DataFrame(index=pd.DatetimeIndex(all_dates))
        for k, curve in profit_curves.items():
            for col in ['Unrealized', 'Realized', 'TotalProfit', 'CurrentValue', 'CurrentQty']:
                if col not in portfolio_curve.columns:
                    portfolio_curve[col] = 0.0
                vals = curve[col].reindex(portfolio_curve.index).ffill().fillna(0.0)
                portfolio_curve[col] += vals
        for col in ['CurrentValue', 'CurrentQty']:
            portfolio_curve[col] = portfolio_curve[col].replace(0, np.nan).ffill().fillna(0.0)

        # Extract daily cash balance and incorporate into portfolio curve
        cash_series = extract_daily_cash(uploaded_files).reindex(portfolio_curve.index).ffill().fillna(0.0)
    
        # Track SPAXX/Cash dividends (income earned on cash)
        cash_divs = df[df["Symbol"].isin(["SPAXX", "CASH", "FDRXX"]) & (df["ActionType"] == "DIVIDEND")]
        if not cash_divs.empty:
            daily_cash_div = cash_divs.groupby("Run Date")["Amount ($)"].sum()
            cum_cash_div = daily_cash_div.reindex(portfolio_curve.index).fillna(0.0).cumsum()
        else:
            cum_cash_div = pd.Series(0.0, index=portfolio_curve.index)

        portfolio_curve["HoldingsValue"] = portfolio_curve["CurrentValue"]
        portfolio_curve["Cash"] = cash_series
        portfolio_curve["CurrentValue"] += cash_series
        portfolio_curve["Realized"] += cum_cash_div
        portfolio_curve["TotalProfit"] = portfolio_curve["Realized"] + portfolio_curve["Unrealized"]

        # End of Portfolio-level curve aggregation
        end_date = portfolio_curve.index[-1]
        current_portfolio_value = portfolio_curve['CurrentValue'].iloc[-1]
        current_holdings_value = portfolio_curve['HoldingsValue'].iloc[-1]
        current_cash_value = portfolio_curve['Cash'].iloc[-1]

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

        # --- Calculate Risk-Free Rate and SPY Benchmark ---
        rf_rate = get_risk_free_rate(t0, t1)
        spy_prices = get_price_history("SPY", t0, t1)
        # Total-return SPY series (dividends reinvested) for the per-row benchmark lines
        spy_adj = get_price_history("SPY", t0, t1, adjusted=True)
        port_start = df["Run Date"].min()

        valid_trades_for_spy = df[df["Symbol"].isin(symbols)]
        spy_curve, spy_xirr_val = simulate_spy_portfolio(valid_trades_for_spy, spy_prices)
    
        spy_profit = spy_curve["TotalProfit"].iloc[-1] if not spy_curve.empty else 0.0
    
        # Calculate Volatilities
        portfolio_monthly = compute_monthly_returns(portfolio_curve)
        port_vol = portfolio_monthly.std() * np.sqrt(12) / 100.0  # Annualized volatility as decimal
    
        if not spy_curve.empty:
            spy_monthly = compute_monthly_returns(spy_curve)
            spy_vol = spy_monthly.std() * np.sqrt(12) / 100.0
        else:
            spy_vol = np.nan
        
        # Calculate Sharpe Ratios
        port_sharpe = (xirr_portfolio - rf_rate) / port_vol if port_vol > 0 else np.nan
        spy_sharpe = (spy_xirr_val - rf_rate) / spy_vol if spy_vol > 0 else np.nan

        # --- 1. Portfolio Level Metrics ---
        st.header("1. Portfolio-level Metrics")
        st.caption(f"Risk-Free Rate (T-Bill historical average used for Sharpe): {rf_rate:.2%}")
    
        # Get realized and unrealized gains from the portfolio curve
        portfolio_realized = portfolio_curve['Realized'].iloc[-1]
        portfolio_unrealized = portfolio_curve['Unrealized'].iloc[-1]
    
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Your Portfolio")
            st.write(f"**Total Profit ($):** {total_profit:,.2f}")
            st.write(f"**Realized Gains/Losses ($):** {portfolio_realized:,.2f}")
            st.write(f"**Unrealized Gains/Losses ($):** {portfolio_unrealized:,.2f}")
            st.write(f"**Portfolio XIRR:** {xirr_portfolio:.2%}")
            st.write(f"**Annualized Volatility:** {port_vol:.2%}")
            st.write(f"**Sharpe Ratio:** {port_sharpe:.2f}")

        with col2:
            st.subheader("S&P 500 Benchmark (If fully invested)")
            st.write(f"**Total Profit ($):** {spy_profit:,.2f}")
            st.write(f"**S&P 500 XIRR:** {spy_xirr_val:.2%}")
            st.write(f"**Annualized Volatility:** {spy_vol:.2%}")
            st.write(f"**Sharpe Ratio:** {spy_sharpe:.2f}")
    
        st.info(f"Net Invested: ${net_invested:,.2f} | Current Value: ${current_portfolio_value:,.2f} (Holdings: ${current_holdings_value:,.2f}, Cash / SPAXX: ${current_cash_value:,.2f})")

        # --- 2. Symbol-level comparison & XIRR table ---
        st.header("2. Fund Performance vs S&P 500")
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
                if row["ActionType"] == "SHARE_DIST":
                    continue  # No cash changes hands on a share distribution
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
            is_held = curve['CurrentQty'].iloc[-1] > 1e-4
            # Benchmark over the holding period: to today if still held, else to the last trade
            hold_end = curve.index[-1] if is_held else trades["Run Date"].max()
            spy_ret = benchmark_return_for_trades(trades, spy_adj, invested, hold_end)
            spy_xirr_pct = benchmark_xirr_for_trades(trades, spy_adj, hold_end)
            symbol_xirr.append((symbol, fund_name, sym_xirr * 100, spy_xirr_pct, is_held))
            symbol_total_return.append((fund_name, total_ret * 100, spy_ret, is_held))

        show_past = st.checkbox("Show past investments (fully sold funds/ETFs)", value=False)
        st.subheader("Total Return")

        # Data-bar chart: total return % per fund vs S&P 500 over the same holding period
        sym_df = pd.DataFrame(symbol_total_return, columns=["Fund Name", "TotalReturn", "SPYReturn", "Held"])
        if not show_past:
            sym_df = sym_df[sym_df["Held"]]
        sym_df = sym_df.sort_values(by="TotalReturn", ascending=False)

        if sym_df.empty:
            st.info("No funds/ETFs to show. Tick the box above to include past investments.")
        else:
            fig = databar_chart(
                sym_df["Fund Name"].tolist(), sym_df["TotalReturn"], sym_df["SPYReturn"],
                title="Fund Total Return (%)",
                benchmark_name="S&P 500 (same cash flows, same holding period)",
            )
            st.plotly_chart(fig, width="stretch")

        # Data-bar chart: XIRR per fund vs S&P 500 XIRR on the same cash flows
        st.subheader("Annualized Return (XIRR)")
        xirr_df = pd.DataFrame(symbol_xirr, columns=["Symbol", "Fund Name", "XIRR", "SPYXIRR", "Held"])
        if not show_past:
            xirr_df = xirr_df[xirr_df["Held"]]
        xirr_df = xirr_df.sort_values(by="XIRR", ascending=False, na_position="last")
        if not xirr_df.empty:
            fig = databar_chart(
                xirr_df["Fund Name"].tolist(), xirr_df["XIRR"], xirr_df["SPYXIRR"],
                title="Fund XIRR (%)",
                benchmark_name="S&P 500 XIRR (same cash flows, same holding period)",
            )
            st.plotly_chart(fig, width="stretch")

        st.header("3. Cumulative Profits Comparison")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=portfolio_curve.index, y=portfolio_curve["TotalProfit"], mode='lines', name="Actual Portfolio", line=dict(color='red')))
        if not spy_curve.empty:
            fig.add_trace(go.Scatter(x=spy_curve.index, y=spy_curve["TotalProfit"], mode='lines', name="S&P 500 (Hypothetical)", line=dict(color=BENCHMARK_COLOR, width=2)))
        fig.update_layout(title="Cumulative Profits vs S&P 500", yaxis_title="Profit ($)", xaxis_title="Date", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        st.plotly_chart(fig, width="stretch")

        # --- 4. Annual Returns Bar Chart ---
        st.header("4. Annual Returns (%)")
        port_end = portfolio_curve.index[-1]
        annual_returns = compute_annual_returns(portfolio_curve)
        # Drop periods that ended before the first trade (curve starts a few days early)
        annual_returns = annual_returns[annual_returns.index >= port_start]
        annual_spy = benchmark_period_returns(spy_adj, annual_returns.index, "Y", port_start, port_end)
        years = [d.strftime("%Y") for d in annual_returns.index]
        fig = databar_chart(
            years, annual_returns.values, annual_spy.values,
            title="Annual Returns (%)",
            benchmark_name="S&P 500 (portfolio's active dates only)",
        )
        st.plotly_chart(fig, width="stretch")

        # --- 5. Monthly Returns Bar Chart ---
        st.header("5. Monthly Returns (%)")
        monthly_returns = compute_monthly_returns(portfolio_curve)
        monthly_returns = monthly_returns[monthly_returns.index >= port_start]
        monthly_spy = benchmark_period_returns(spy_adj, monthly_returns.index, "M", port_start, port_end)
        months = [d.strftime("%Y-%m") for d in monthly_returns.index]
        fig = databar_chart(
            months, monthly_returns.values, monthly_spy.values,
            title="Monthly Returns (%)",
            benchmark_name="S&P 500 (portfolio's active dates only)",
        )
        st.plotly_chart(fig, width="stretch")

    with tab_analysis:
        # Current market value of each fund still held; cash is handled separately
        holdings = {
            sym: curve["CurrentValue"].iloc[-1] for sym, curve in profit_curves.items()
            if curve["CurrentQty"].iloc[-1] > 1e-4 and curve["CurrentValue"].iloc[-1] > 0
        }
        render_portfolio_analysis(holdings, symbol_name_map, rf_rate, current_cash_value)

if __name__ == "__main__":
    main()
