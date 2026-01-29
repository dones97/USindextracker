import pandas as pd

df = pd.read_excel('History_for_Account_Z33464548.xlsx')
print("=" * 80)
print("UNIQUE ACTION TYPES:")
print("=" * 80)
print(df['Action'].value_counts())

print("\n" + "=" * 80)
print("SAMPLE TRANSACTIONS BY ACTION TYPE:")
print("=" * 80)

# Show samples of each action type
for action in df['Action'].unique():
    if pd.notna(action):
        print(f"\n--- {action} ---")
        sample = df[df['Action'] == action].head(3)
        for idx, row in sample.iterrows():
            print(f"  Date: {row['Run Date']}, Symbol: {row['Symbol']}, Qty: {row['Quantity']}, Price: {row['Price ($)']}, Amount: {row['Amount ($)']}")

print("\n" + "=" * 80)
print("COLUMNS IN THE FILE:")
print("=" * 80)
for col in df.columns:
    print(f"  - {col}")
