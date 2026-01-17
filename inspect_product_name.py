import pandas as pd
import os
import re

# 현재 스크립트 위치 기준 상대 경로
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'data', 'store_data.xlsx')

print(f"Loading {file_path}...")
df = pd.read_excel(file_path)

print("\n[Sample '상품명' Data]")
# 삼각형이 포함된 샘플 찾기
triangle_samples = df[df['상품명'].str.contains('▶|▷', regex=True)]['상품명'].head(10)
print("--- With Triangle ---")
for s in triangle_samples:
    print(s)

print("\n--- With Quantity Pattern (e.g., (2개)) ---")
qty_samples = df[df['상품명'].str.contains(r'\(\d+개\)', regex=True)]['상품명'].head(10)
for s in qty_samples:
    print(s)

# 유니코드 확인
if not triangle_samples.empty:
    sample = triangle_samples.iloc[0]
    print(f"\nSample String: {sample}")
    print(f"Unicode Chars: {[ord(c) for c in sample]}")
