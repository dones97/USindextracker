import pandas as pd
import sys

# Read the Excel file
df = pd.read_excel('History_for_Account_Z33464548.xlsx')

print("=" * 80)
print("ACTION TYPES AND COUNTS:")
print("=" * 80)
print(df['Action'].value_counts())

print("\n" + "=" * 80)
print("SAMPLE TRANSACTIONS (First 50 rows):")
print("=" * 80)
cols_to_show = ['Run Date', 'Action', 'Symbol', 'Quantity', 'Price ($)', 'Amount ($)']
print(df[cols_to_show].head(50).to_string())

print("\n" + "=" * 80)
print("EXAMPLES OF EACH ACTION TYPE:")
print("=" * 80)
for action in df['Action'].unique():
    if pd.notna(action):
        print(f"\n--- {action} ---")
        samples = df[df['Action'] == action].head(2)
        print(samples[cols_to_show].to_string())
