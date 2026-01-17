import pandas as pd
import os

file_path = '/Users/dmjeong/innercircle/project1/data/store_data.xlsx'

print(f"Loading data from {file_path}...")
try:
    df = pd.read_excel(file_path)
    print("\n[1] Data Info:")
    df.info()
    
    print("\n[2] First 5 rows:")
    print(df.head())
    
    print("\n[3] Null Counts:")
    print(df.isnull().sum())
    
    print("\n[4] Describe (Numerical):")
    print(df.describe())
    
    print("\n[5] Describe (Object):")
    print(df.describe(include=['O']))

except Exception as e:
    print(f"Error loading file: {e}")
