import pandas as pd
import numpy as np

def test_load():
    file_path = '/Users/dmjeong/innercircle/DA_project1/data/preprocessed_data.csv'
    df = pd.read_csv(file_path)
    
    print("Initial Column Types:")
    print(df.dtypes)
    
    numeric_cols = ['실결제 금액', '결제금액', '판매단가', '주문수량', '취소수량', '재구매 횟수', '무게(kg)']
    for col in numeric_cols:
        if col in df.columns:
            # 숫자가 아닌 값(콤마 등) 처리 후 수치형으로 변환
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    
    # --- 중복 합산 방지 및 정확한 매출 계산 ---
    if '판매단가' in df.columns and '주문수량' in df.columns and '취소수량' in df.columns:
        df['item_revenue'] = df['판매단가'] * (df['주문수량'] - df['취소수량'])
    else:
        df['item_revenue'] = df['결제금액'] # fallback
        
    print("\nTypes After Processing:")
    print(df[['판매단가', '주문수량', '취소수량', 'item_revenue']].dtypes)
    
    print("\nFirst 5 item_revenue values:")
    print(df['item_revenue'].head())
    
    total_sales = df['item_revenue'].sum()
    print(f"\nTotal Sales: {total_sales}")
    print(f"Type of Total Sales: {type(total_sales)}")

if __name__ == "__main__":
    test_load()
