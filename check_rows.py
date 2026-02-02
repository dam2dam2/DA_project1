import pandas as pd
import glob
import os

def check_duplicates():
    files = sorted(glob.glob('/Users/dmjeong/innercircle/DA_project1/data/origin_data/*.xlsx'))
    sample_order = 'YMM250910-00000009'
    
    print(f"Checking Order: {sample_order}")
    all_data = []
    for f in files:
        df = pd.read_excel(f)
        matches = df[df['주문번호'] == sample_order]
        if not matches.empty:
            print(f"\n--- Found in File: {os.path.basename(f)} ---")
            print(matches[['주문번호', '상품명', '주문수량', '결제금액(상품별)']])
            all_data.append(matches)
    
    if all_data:
        merged = pd.concat(all_data)
        print("\n--- Merged View ---")
        print(merged[['주문번호', '상품명', '주문수량', '결제금액(상품별)']])

if __name__ == "__main__":
    check_duplicates()
