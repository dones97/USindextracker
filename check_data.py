import pandas as pd

df = pd.read_excel('History_for_Account_Z33464548.xlsx')
print("Columns:")
for col in df.columns:
    print(f"  - {col}")

print("\nFirst 10 rows:")
print(df.head(10))

print("\nData types:")
print(df.dtypes)

print("\nSample Action values:")
print(df['Action'].head(20))
