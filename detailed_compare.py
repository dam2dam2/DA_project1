import pandas as pd
import numpy as np

def compare_csvs(old_path, new_path):
    print(f"Comparing:\nOld: {old_path}\nNew: {new_path}\n")
    
    old_df = pd.read_csv(old_path)
    new_df = pd.read_csv(new_path)
    
    # 1. Basic Stats
    print("--- 1. Basic Statistics ---")
    print(f"Old Rows: {len(old_df)}, Columns: {len(old_df.columns)}")
    print(f"New Rows: {len(new_df)}, Columns: {len(new_df.columns)}")
    print(f"Difference: {len(new_df) - len(old_df)} rows")
    
    # 2. Columns
    print("\n--- 2. Column Comparison ---")
    old_cols = set(old_df.columns)
    new_cols = set(new_df.columns)
    print(f"Common Columns: {len(old_cols & new_cols)}")
    if old_cols - new_cols:
        print(f"Only in OLD: {old_cols - new_cols}")
    if new_cols - old_cols:
        print(f"Only in NEW: {new_cols - old_cols}")
        
    # 3. Key Metrics (Sum comparison)
    print("\n--- 3. Numerical Metrics Check ---")
    def clean_and_sum(df, col):
        if col not in df.columns: return 0
        return pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce').sum()

    for col in ['결제금액', '실결제 금액', '판매단가', '주문수량', '취소수량']:
        old_sum = clean_and_sum(old_df, col)
        new_sum = clean_and_sum(new_df, col)
        print(f"{col}: Old={old_sum:,.0f} | New={new_sum:,.0f} | Diff={new_sum-old_sum:,.0f}")

    # 4. Key Metric (Unique Orders)
    print("\n--- 4. Order Level Check ---")
    old_orders = set(old_df['주문번호'].unique())
    new_orders = set(new_df['주문번호'].unique())
    print(f"Unique Orders in OLD: {len(old_orders)}")
    print(f"Unique Orders in NEW: {len(new_orders)}")
    print(f"New order IDs not in OLD: {len(new_orders - old_orders)}")
    print(f"Old order IDs not in NEW: {len(old_orders - new_orders)}")
    
    # 5. Analyzing why counts differ (Nested items)
    print("\n--- 5. Discrepancy Detail (Top 5 Examples) ---")
    old_counts = old_df.groupby('주문번호').size()
    new_counts = new_df.groupby('주문번호').size()
    
    merged_counts = pd.DataFrame({'old': old_counts, 'new': new_counts}).fillna(0)
    merged_counts['diff'] = merged_counts['new'] - merged_counts['old']
    diff_orders = merged_counts[merged_counts['diff'] != 0].sort_values('diff', ascending=False)
    
    print(f"Orders with different item counts: {len(diff_orders)}")
    print(diff_orders.head(10))

    # Example of a specific order
    if not diff_orders.empty:
        sample_id = diff_orders.index[0]
        print(f"\nExample Order [{sample_id}] Detail:")
        print("OLD Data Items:")
        print(old_df[old_df['주문번호'] == sample_id][['주문번호', '상품명', '고객선택옵션', '결제금액']])
        print("\nNEW Data Items:")
        print(new_df[new_df['주문번호'] == sample_id][['주문번호', '상품명', '고객선택옵션', '결제금액']])

if __name__ == "__main__":
    compare_csvs(
        '/Users/dmjeong/innercircle/DA_project1/data/preprocessed_data.csv',
        '/Users/dmjeong/innercircle/DA_project1/data/preprocessed_data_new.csv'
    )
