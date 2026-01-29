import pandas as pd
import shutil
import os
import time

original_file = r'c:\Users\dones\OneDrive\Documents\Investments\USindextracker\History_for_Account_Z33464548.xlsx'
temp_file = r'c:\Users\dones\OneDrive\Documents\Investments\USindextracker\temp_history_analysis_2.xlsx'

try:
    shutil.copy2(original_file, temp_file)
    time.sleep(1)
    df = pd.read_excel(temp_file)
    print("ALL UNIQUE ACTIONS:")
    print(df['Action'].unique().tolist())
except Exception as e:
    print(f"Error: {e}")
finally:
    if os.path.exists(temp_file):
        os.remove(temp_file)
