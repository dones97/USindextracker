import pandas as pd
import os

file_path = r'c:\Users\dones\OneDrive\Documents\Investments\USindextracker\History_for_Account_Z33464548.xlsx'

try:
    if not os.path.exists(file_path):
        print(f"File not found at: {file_path}")
    else:
        df = pd.read_excel(file_path)
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
