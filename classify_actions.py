import pandas as pd
import shutil
import os
import time

original_file = r'c:\Users\dones\OneDrive\Documents\Investments\USindextracker\History_for_Account_Z33464548.xlsx'
temp_file = r'c:\Users\dones\OneDrive\Documents\Investments\USindextracker\temp_history_analysis_3.xlsx'

def classify_action(action_str):
    if not isinstance(action_str, str):
        return "UNKNOWN (NaN)"
    u = action_str.upper()
    if "YOU BOUGHT" in u:
        return "BUY"
    if "YOU SOLD" in u:
        return "SELL"
    if "DIVIDEND" in u:
        if "REINVESTMENT" in u:
            return "DIVIDEND_REINVEST"
        return "DIVIDEND"
    if "REINVESTMENT" in u:
        return "REINVESTMENT"
    if "TRANSFER" in u:
        if "RECEIVED" in u:
            return "DEPOSIT"
        if "PAID" in u:
            return "WITHDRAWAL"
    if "CASH" in u:
        return "CASH_MOVEMENT"
    return "OTHER"

try:
    shutil.copy2(original_file, temp_file)
    time.sleep(1)
    df = pd.read_excel(temp_file)
    
    print("--- RAW ACTION ANALYSIS ---")
    if 'Action' in df.columns:
        df['ActionType'] = df['Action'].apply(classify_action)
        print(df['ActionType'].value_counts())
        
        print("\n--- SAMPLE RAW VALUES FOR EACH TYPE ---")
        for atype in df['ActionType'].unique():
            print(f"\nType: {atype}")
            samples = df[df['ActionType'] == atype]['Action'].head(2).tolist()
            for s in samples:
                print(f"  - {s[:100]}...") # Truncate to avoid log limit
    else:
        print("No Action column found")

except Exception as e:
    print(f"Error: {e}")
finally:
    if os.path.exists(temp_file):
        os.remove(temp_file)
