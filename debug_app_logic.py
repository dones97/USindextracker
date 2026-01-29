import pandas as pd
import yfinance as yf
import numpy as np

# Copy paste functions from app.py to ensure identical logic
def classify_action(row):
    action = str(row["Action"]).upper()
    if "YOU BOUGHT" in action:
        return "BUY"
    if "YOU SOLD" in action:
        return "SELL"
    if "REINVESTMENT" in action:
        return "REINVEST"
    if "DIVIDEND" in action:
        return "DIVIDEND"
    if "TRANSFER" in action:
        if "RECEIVED" in action:
            return "DEPOSIT"
        if "PAID" in action:
            return "WITHDRAWAL"
    if "DEPOSIT" in action: return "DEPOSIT"
    if "WITHDRAWAL" in action: return "WITHDRAWAL"
    return "OTHER"

def preprocess_trades(df):
    df.columns = [col.strip() for col in df.columns]
    df["Run Date"] = pd.to_datetime(df["Run Date"])
    
    def parse_amount(x):
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).replace('$', '').replace(',', '').strip()
        if '(' in s and ')' in s:
            s = '-' + s.replace('(', '').replace(')', '')
        return float(s)
        
    if "Amount ($)" in df.columns:
        df["Amount ($)"] = df["Amount ($)"].apply(parse_amount)
    
    df["ActionType"] = df.apply(classify_action, axis=1)
    
    relevant_types = ["BUY", "SELL", "DIVIDEND", "REINVEST", "DEPOSIT", "WITHDRAWAL"]
    df = df[df["ActionType"].isin(relevant_types)]
    
    return df

# Main logic test
try:
    df = pd.read_excel('History_for_Account_Z33464548.xlsx')
    df = preprocess_trades(df)
    
    print("Preprocessed DataFrame:")
    print(df['ActionType'].value_counts())
    
    print("\nSample BUYs:")
    print(df[df['ActionType'] == 'BUY'][['Run Date', 'Symbol', 'Quantity', 'Amount ($)']].head())

    print("\nSymbols found:")
    symbols = df["Symbol"].unique()
    print(symbols)
    
    # Check specifically for the 'missing symbol' which might be NaN or empty
    print("\nChecking for weird symbols:")
    for s in symbols:
        print(f"'{s}' type: {type(s)}")
        
except Exception as e:
    print(e)
