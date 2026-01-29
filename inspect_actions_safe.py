import pandas as pd
import os
import shutil
import time

original_file = r'c:\Users\dones\OneDrive\Documents\Investments\USindextracker\History_for_Account_Z33464548.xlsx'
temp_file = r'c:\Users\dones\OneDrive\Documents\Investments\USindextracker\temp_history_analysis.xlsx'

print(f"Attempting to copy {original_file} to {temp_file}...")

try:
    shutil.copy2(original_file, temp_file)
    print("Copy successful.")
    
    # Give it a moment to ensure file handle is released/ready
    time.sleep(1)
    
    if not os.path.exists(temp_file):
        print(f"Temp file not found at: {temp_file}")
    else:
        df = pd.read_excel(temp_file)
        print("UNIQUE ACTIONS:")
        if 'Action' in df.columns:
            actions = df['Action'].dropna().unique()
            for action in actions:
                print(f"'{action}'")
                
            print("\nSAMPLE ROWS FOR EACH ACTION:")
            for action in actions:
                print(f"\n--- Action: {action} ---")
                sample = df[df['Action'] == action].head(2)
                # Print relevant columns
                cols = ['Run Date', 'Action', 'Symbol', 'Description', 'Quantity', 'Price ($)', 'Amount ($)', 'Commission ($)']
                # Only print columns that exist
                existing_cols = [c for c in cols if c in df.columns]
                print(sample[existing_cols].to_string(index=False))
        else:
            print("Column 'Action' not found in dataframe.")
            print("Columns are:", df.columns.tolist())

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    if os.path.exists(temp_file):
        try:
            os.remove(temp_file)
            print("Temp file removed.")
        except Exception as e:
            print(f"Could not remove temp file: {e}")
