import pandas as pd
import shutil
import os
import time

original_file = r'c:\Users\dones\OneDrive\Documents\Investments\USindextracker\History_for_Account_Z33464548.xlsx'
temp_file = r'c:\Users\dones\OneDrive\Documents\Investments\USindextracker\temp_history_analysis_4.xlsx'

try:
    shutil.copy2(original_file, temp_file)
    time.sleep(1)
    df = pd.read_excel(temp_file)
    
    # Filter for Divs and Reinvests
    divs = df[df['Action'].str.contains('DIVIDEND', na=False, case=False)]
    reinvests = df[df['Action'].str.contains('REINVESTMENT', na=False, case=False)]
    
    print(f"DIVIDEND rows: {len(divs)}")
    print(f"REINVESTMENT rows: {len(reinvests)}")
    
    print("\nSample DIVIDEND rows:")
    print(divs[['Run Date', 'Amount ($)']].head(5))
    
    print("\nSample REINVESTMENT rows:")
    print(reinvests[['Run Date', 'Amount ($)']].head(5))
    
    # Check for same date and amount
    print("\nChecking for matches:")
    matches = 0
    for idx, row in reinvests.iterrows():
        # Look for a dividend on the same date with similar amount (maybe opposite sign?)
        date = row['Run Date']
        amt = abs(row['Amount ($)'])
        
        # Check if there is a dividend on this date with this amount
        match = divs[(divs['Run Date'] == date) & (divs['Amount ($)'].abs() - amt < 0.01)]
        if not match.empty:
            matches += 1
            
    print(f"\nFound {matches} REINVESTMENT rows that have a matching DIVIDEND row on the same date.")

except Exception as e:
    print(f"Error: {e}")
finally:
    if os.path.exists(temp_file):
        os.remove(temp_file)
