import pandas as pd
import numpy as np
import re
import hashlib
import os

def preprocess_data(input_file, output_dir):
    print(f"Loading data from {input_file}...")
    df = pd.read_excel(input_file)
    
    # --- 1. Basic Cleaning ---
    # Date conversion
    df['order_datetime'] = pd.to_datetime(df['주문일'])
    df['order_date'] = df['order_datetime'].dt.date
    df['order_hour'] = df['order_datetime'].dt.hour
    
    # Numeric conversion
    numeric_cols = ['주문수량', '결제금액(상품별)', '결제금액(통합)', '공급가', '포인트 사용금액(통합)', '쿠폰 사용금액(통합)', '부분취소금액(통합)', '주문취소 금액(상품별)']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
    df['delivery_ready_datetime'] = pd.to_datetime(df['배송준비 처리일'], errors='coerce')

    # --- 2. Product Name Parsing (Regex) ---
    def parse_product_name(name):
        # Split by ▶ to get option part
        parts = name.split('▶')
        base_name = parts[0].strip()
        option_part = parts[1] if len(parts) > 1 else name
        
        # Weight
        weight_match = re.search(r'(\d+(\.\d+)?)\s*(kg|g|KG|G)', option_part)
        weight_kg = float(weight_match.group(1)) if weight_match else 0.0
        if weight_match and weight_match.group(3).lower() == 'g':
            weight_kg /= 1000.0
            
        # Size
        size_match = re.search(r'(소과|중과|대과|로얄과|중대과)', option_part)
        option_type = size_match.group(0) if size_match else "일반"
        
        # Count (수)
        count_match = re.search(r'(\d+~?\d*)\s*수', option_part)
        size_range = count_match.group(0) if count_match else "미기재"
        
        # Category
        category = "기타"
        if "감귤" in base_name or "귤" in base_name: category = "감귤"
        elif "황금향" in base_name: category = "황금향"
        elif "레드향" in base_name: category = "레드향"
        elif "천혜향" in base_name: category = "천혜향"
        
        # Harvest Type
        harvest = "노지" if "노지" in base_name else ("하우스" if "하우스" in base_name else "미기재")
        
        # Promotion
        is_promotion = 1 if re.search(r'★.*?추가.*?★|1\+1', name) else 0
        
        return pd.Series([base_name, option_type, weight_kg, size_range, category, harvest, is_promotion])

    product_fields = df['상품명'].apply(parse_product_name)
    product_fields.columns = ['clean_product_name', 'option_type', 'weight_kg', 'size_range', 'product_category', 'harvest_type', 'is_promotion']
    df = pd.concat([df, product_fields], axis=1)

    # --- 3. Customer Hashing & Region ---
    def hash_id(val):
        if pd.isna(val): return "unknown"
        return hashlib.sha256(str(val).encode()).hexdigest()[:12]
    
    df['customer_id'] = df['주문자연락처'].apply(hash_id)
    
    def extract_region(address):
        if pd.isna(address): return "미기재", "미기재"
        parts = address.split()
        r1 = parts[0] if len(parts) > 0 else "미기재"
        r2 = parts[1] if len(parts) > 1 else "미기재"
        return r1, r2
    
    regions = df['주소'].apply(lambda x: pd.Series(extract_region(x)))
    df['region_1'], df['region_2'] = regions[0], regions[1]

    # --- 4. Derived Columns ---
    df['price_per_kg'] = np.where(df['weight_kg'] > 0, df['결제금액(상품별)'] / df['weight_kg'], 0)
    df['margin'] = df['결제금액(상품별)'] - df['공급가']
    df['margin_rate'] = np.where(df['결제금액(상품별)'] > 0, df['margin'] / df['결제금액(상품별)'], 0)
    
    df['weekday'] = df['order_datetime'].dt.day_name()
    df['is_weekend'] = df['order_datetime'].dt.weekday.isin([5, 6]).astype(int)
    
    def get_time_slot(hour):
        if 5 <= hour < 12: return "아침"
        elif 12 <= hour < 17: return "점심"
        elif 17 <= hour < 21: return "저녁"
        else: return "야간"
    df['time_slot'] = df['order_hour'].apply(get_time_slot)
    
    df['is_high_price'] = (df['결제금액(상품별)'] >= 50000).astype(int)

    # --- 5. Table Splitting ---
    # Orders
    orders = df[['주문번호', 'order_datetime', 'order_date', 'order_hour', 'weekday', 'time_slot', '주문경로', '결제방법', '결제금액(통합)', '부분취소금액(통합)', '포인트 사용금액(통합)', '쿠폰 사용금액(통합)', 'delivery_ready_datetime', 'is_weekend', 'customer_id']].drop_duplicates('주문번호')
    orders.columns = ['order_id', 'order_datetime', 'order_date', 'order_hour', 'weekday', 'time_slot', 'order_channel', 'payment_method', 'total_payment_amount', 'total_cancel_amount', 'point_used', 'coupon_used', 'delivery_ready_datetime', 'is_weekend', 'customer_id']
    
    # Order_Items
    items = df[['주문번호', '상품코드', 'clean_product_name', 'option_type', 'weight_kg', '주문수량', '결제금액(상품별)', '공급가', 'margin', 'margin_rate', 'is_promotion', 'price_per_kg']]
    # Note: unit_price = 결제금액(상품별) / 주문수량, supply_price = 공급가 / 주문수량
    items['unit_price'] = np.where(items['주문수량'] > 0, items['결제금액(상품별)'] / items['주문수량'], 0)
    items['supply_price'] = np.where(items['주문수량'] > 0, items['공급가'] / items['주문수량'], 0)
    
    items = items[['주문번호', '상품코드', 'clean_product_name', 'option_type', 'weight_kg', '주문수량', '결제금액(상품별)', 'unit_price', 'supply_price', 'margin', 'margin_rate', 'is_promotion', 'price_per_kg']]
    items.columns = ['order_id', 'product_code', 'product_name', 'option_type', 'weight_kg', 'quantity', 'item_payment_amount', 'unit_price', 'supply_price', 'margin', 'margin_rate', 'is_promotion', 'price_per_kg']
    
    # Customers
    customers = df[['customer_id', '회원구분', 'region_1', 'region_2']].drop_duplicates('customer_id')
    customers.columns = ['customer_id', 'member_type', 'region_1', 'region_2']
    
    # Products
    products = df[['상품코드', 'clean_product_name', 'product_category', 'is_promotion']].drop_duplicates('상품코드')
    products.columns = ['product_code', 'base_product_name', 'category', 'is_event_product']


    # --- 6. Save Outputs ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    orders.to_csv(os.path.join(output_dir, 'orders.csv'), index=False, encoding='utf-8-sig')
    items.to_csv(os.path.join(output_dir, 'order_items.csv'), index=False, encoding='utf-8-sig')
    customers.to_csv(os.path.join(output_dir, 'customers.csv'), index=False, encoding='utf-8-sig')
    products.to_csv(os.path.join(output_dir, 'products.csv'), index=False, encoding='utf-8-sig')
    
    print(f"Preprocessed files saved to {output_dir}")
    return orders, items, customers, products

if __name__ == "__main__":
    input_xlsx = '/Users/dmjeong/innercircle/DA_project1/data/store_data.xlsx'
    output_path = '/Users/dmjeong/innercircle/DA_project1/preprocessed'
    preprocess_data(input_xlsx, output_path)
