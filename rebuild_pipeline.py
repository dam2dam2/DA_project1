import pandas as pd
import numpy as np
import os
import glob
import re
import hashlib

def generate_uid(row):
    try:
        val = str(row['판매단가']).replace(',', '')
        unit_price = str(int(float(val)))
    except:
        unit_price = "0"
    text = f"{str(row['상품코드'])}{str(row['상품명'])}{unit_price}"
    return "U" + hashlib.md5(text.encode()).hexdigest().upper()[:5]

def anonymize_phone(phone_series):
    unique_phones = sorted(phone_series.unique().tolist())
    phone_map = {phone: f"ANON_{i+1:05d}" for i, phone in enumerate(unique_phones)}
    return phone_series.map(phone_map)

def get_next_version(directory, base_name="preprocessed_data"):
    files = glob.glob(os.path.join(directory, f"{base_name}_*.csv"))
    if not files:
        return 1
    
    versions = []
    for f in files:
        match = re.search(rf"{base_name}_(\d+)\.csv$", f)
        if match:
            versions.append(int(match.group(1)))
    
    return max(versions) + 1 if versions else 1

def get_latest_version_path(directory, base_name="preprocessed_data"):
    files = glob.glob(os.path.join(directory, f"{base_name}_*.csv"))
    if not files:
        # Fallback to the original preprocessed_data.csv if exists
        fallback = os.path.join(directory, "preprocessed_data.csv")
        return fallback if os.path.exists(fallback) else None
    
    versions = []
    for f in files:
        match = re.search(rf"{base_name}_(\d+)\.csv$", f)
        if match:
            versions.append((int(match.group(1)), f))
    
    if versions:
        latest_ver, latest_path = max(versions, key=lambda x: x[0])
        return latest_path
    return None

def rebuild_pipeline():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ORIGIN_DIR = os.path.join(BASE_DIR, "data", "origin_data")
    DATA_DIR = os.path.join(BASE_DIR, "data")
    
    next_ver = get_next_version(DATA_DIR)
    PREPROCESSED_PATH = os.path.join(DATA_DIR, f"preprocessed_data_{next_ver}.csv")
    LATEST_OLD_PATH = get_latest_version_path(DATA_DIR)

    print(f"1. 원본 데이터 수집 및 통합 시작: {ORIGIN_DIR}")
    excel_files = sorted(glob.glob(os.path.join(ORIGIN_DIR, "*.xlsx")))
    
    if not excel_files:
        print("Error: 원본 엑셀 파일을 찾을 수 없습니다. origin_data 폴더를 확인해주세요.")
        return

    all_dfs = []
    for f in excel_files:
        print(f"   - {os.path.basename(f)} 읽는 중...")
        all_dfs.append(pd.read_excel(f))
    
    raw_df = pd.concat(all_dfs, ignore_index=True)
    
    # 중복 제거
    if '주문상품고유번호' in raw_df.columns:
        before_len = len(raw_df)
        raw_df = raw_df.drop_duplicates(subset=['주문상품고유번호'], keep='first')
        after_len = len(raw_df)
        if before_len > after_len:
            print(f"   - 중복 데이터 제거됨: {before_len - after_len}행 (주문상품고유번호 기준)")

    print("2. 익명화 및 기본 정제 진행 중...")
    raw_df['주문자연락처'] = anonymize_phone(raw_df['주문자연락처'].astype(str))
    raw_df['수령인연락처'] = anonymize_phone(raw_df['수령인연락처'].astype(str))
    
    # 옵션코드 생성
    df_opt = raw_df[['상품코드', '고객선택옵션']].drop_duplicates().copy()
    df_opt = df_opt.sort_values(['상품코드', '고객선택옵션'])
    df_opt['option_idx'] = df_opt.groupby('상품코드').cumcount() + 1
    df_opt['옵션코드'] = df_opt['상품코드'].astype(str) + "_" + df_opt['option_idx'].astype(str)
    raw_df = pd.merge(raw_df, df_opt[['상품코드', '고객선택옵션', '옵션코드']], on=['상품코드', '고객선택옵션'], how='left')

    mapping = {
        '결제금액(상품별)': '결제금액',
        '주문취소 금액(상품별)': '주문취소 금액',
        '상품금액(옵션포함)': '판매단가',
        '공급가': '공급단가'
    }
    df = raw_df.rename(columns=mapping).copy()

    # 숫자형 변환
    numeric_cols = ['결제금액', '주문취소 금액', '판매단가', '공급단가', '주문수량', '취소수량']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0).astype(int)

    df['실결제 금액'] = df['결제금액'] - df['주문취소 금액']
    df['주문-취소 수량'] = df['주문수량'] - df['취소수량']
    df['취소여부'] = df['취소수량'].apply(lambda x: 'Y' if x > 0 else 'N')
    
    # UID 생성
    df['UID'] = df.apply(generate_uid, axis=1)

    # 3. 상세 전처리 및 파생 필드
    print("3. 상세 전처리 및 파생 필드 생성 중...")
    df['주문일_dt'] = pd.to_datetime(df['주문일'])
    df['주문날짜'] = df['주문일_dt'].dt.date
    # 전체 기간에 대한 재구매 횟수 재계산 (Fresh Update)
    repurchase_map = df.groupby('주문자연락처')['주문날짜'].nunique().to_dict()
    df['재구매 횟수'] = df['주문자연락처'].map(lambda x: max(0, repurchase_map.get(str(x), 1) - 1))

    # 지역 분석
    def extract_sido(row):
        address, postcode = str(row['주소']), str(row['우편번호']).zfill(5)
        prefix = postcode[:2]
        sido_map = {'01':'서울','02':'서울','03':'서울','04':'서울','05':'서울','06':'서울','07':'서울','08':'서울','09':'서울','10':'경기','11':'경기','12':'경기','13':'경기','14':'경기','15':'경기','16':'경기','17':'경기','18':'경기','19':'경기','20':'경기','21':'인천','22':'인천','23':'인천','24':'강원','25':'강원','26':'강원','27':'충북','28':'충북','29':'충북','30':'세종','31':'충남','32':'충남','33':'충남','34':'대전','35':'대전','36':'경북','37':'경북','38':'경북','39':'경북','40':'경북','41':'대구','42':'대구','43':'대구','44':'울산','45':'울산','46':'부산','47':'부산','48':'부산','49':'부산','50':'경남','51':'경남','52':'경남','53':'경남','54':'전북','55':'전북','56':'전북','57':'전남','58':'전남','59':'전남','60':'전남','61':'광주','62':'광주','63':'제주'}
        formal = {'서울':'서울특별시','부산':'부산광역시','대구':'대구광역시','인천':'인천광역시','광주':'광주광역시','대전':'대전광역시','울산':'울산광역시','세종':'세종특별자치시','경기':'경기도','강원':'강원특별자치도','충북':'충청북도','충남':'충청남도','전북':'전북특별자치도','전남':'전라남도','경북':'경상북도','경남':'경상남도','제주':'제주특별자치도'}
        res = sido_map.get(prefix, next((s for s in sido_map.values() if s in address), "기타"))
        return res, formal.get(res, res)

    df[['광역지역', '광역지역(정식)']] = df.apply(lambda r: pd.Series(extract_sido(r)), axis=1)

    # 텍스트 분석
    def parse_text(row):
        txt = f"{str(row['상품명'])} {str(row['고객선택옵션'])}"
        cat = "감귤" if any(k in txt for k in ["감귤", "조생", "타이벡", "귤"]) else ("황금향" if "황금향" in txt else ("고구마" if "고구마" in txt else "기타"))
        sub = "타이벡" if "타이벡" in txt else ("조생" if "조생" in txt else ("하우스" if "하우스" in txt else "일반"))
        size = "기타"
        if "소과" in txt and "중소과" in txt: size = "소과, 중소과"
        elif "중대과" in txt and "대과" in txt: size = "중대과, 대과"
        elif "소과" in txt: size = "소과"
        elif "혼합" in txt: size = "혼합"
        elif "로얄과" in txt: size = "로얄과"
        elif "대과" in txt: size = "대과"
        elif "중과" in txt: size = "중과"

        gift = "선물세트" if any(k in txt for k in ["선물세트", "선물용"]) else "가정용"
        ev = "Y" if any(k in txt for k in ["이벤트", "1+1", "보장"]) else "N"
        grade = "프리미엄" if any(k in txt for k in ["프리미엄", "명품", "고당도", "타이벡"]) else "일반"

        w_match = re.findall(r'(\d+(\.\d+)?)\s*(kg|KG)', txt)
        w = sum(float(m[0]) for m in w_match) if w_match else 0.0
        grp = "<3kg" if 0 < w < 3 else ("3-5kg" if w <= 5 else ("5-10kg" if w <= 10 else ">10kg")) if w > 0 else "미분류"
        purpose = "선물" if str(row['주문자명']).replace(' ','') != str(row['수령인명']).replace(' ','') or "선물" in txt else "개인소비"
        
        # '고객선택옵션(타입제거)' 복구 (원본에 없을 경우를 대비해 고객선택옵션 사용)
        opt_clean = str(row.get('고객선택옵션(타입제거)', row['고객선택옵션']))

        return pd.Series([sub, cat, size, gift, w, grp, ev, grade, purpose, opt_clean])

    df[['감귤 세부', '품종', '과수 크기', '선물세트_여부', '무게(kg)', '무게 구분', '이벤트 여부', '상품성등급_그룹', '목적', '고객선택옵션(타입제거)']] = df.apply(parse_text, axis=1)

    def classify_price(a):
        return "1만원 이하" if a <= 10000 else ("1~3만원" if a <= 30000 else ("3~5만원" if a <= 50000 else ("5~10만원" if a <= 100000 else "10만원 초반")))
    df['가격대'] = df['실결제 금액'].apply(classify_price)

    # 4. 타입 정리 및 문자열 전환 (콤마 추가)
    processed_df = df.copy() # 원본 보존 (비교용)
    for col in ['결제금액', '실결제 금액', '판매단가', '공급단가']:
        df[col] = df[col].apply(lambda x: f"{int(x):,}")

    target_columns = [
        'UID', '주문번호', '주문일', '상품코드', '옵션코드', '상품명', '고객선택옵션', '주문수량', '취소수량', '주문-취소 수량',
        '결제금액', '주문취소 금액', '실결제 금액', '판매단가', '공급단가', '주문경로', '주문자명', '셀러명', '결제방법',
        '배송준비 처리일', '입금일', '주문자연락처', '주소', '입금자명', '수령인명', '수령인연락처', '회원구분', '우편번호',
        '고객선택옵션(타입제거)', '취소여부', '목적', '재구매 횟수', '광역지역', '광역지역(정식)',
        '감귤 세부', '품종', '과수 크기', '선물세트_여부', '무게(kg)', '무게 구분', '이벤트 여부', '상품성등급_그룹', '가격대'
    ]
    
    final_df = df.reindex(columns=target_columns).fillna("")
    print(f"4. 최종 결과 저장: {PREPROCESSED_PATH}")
    final_df.to_csv(PREPROCESSED_PATH, index=False, encoding='utf-8-sig')
    
    # 상세 보고
    print("\n" + "="*50)
    print(f"📊 전처리 작업 결과 보고 (Version {next_ver})")
    print("="*50)
    print(f"- 생성 파일: {os.path.basename(PREPROCESSED_PATH)}")
    print(f"- 전체 데이터 행 수: {len(final_df):,}행")
    
    if LATEST_OLD_PATH:
        print(f"- 이전 파일 대비 비교: {os.path.basename(LATEST_OLD_PATH)}")
        old_df = pd.read_csv(LATEST_OLD_PATH)
        row_diff = len(final_df) - len(old_df)
        print(f"  * 추가된 행 수: {row_diff:+,}행")
        
        # 금액 비교 (콤마 제거 후 계산)
        def get_sum(d, col):
            return pd.to_numeric(d[col].astype(str).str.replace(',', ''), errors='coerce').sum()
        
        old_sales = get_sum(old_df, '실결제 금액')
        new_sales = get_sum(final_df, '실결제 금액')
        print(f"  * 총 실결제 금액 변화: {old_sales:,.0f}원 -> {new_sales:,.0f}원 ({new_sales-old_sales:+,.0f}원)")
        
        # 신규 주문 확인
        old_order_ids = set(old_df['주문번호'].unique())
        new_order_ids = set(final_df['주문번호'].unique())
        added_orders = new_order_ids - old_order_ids
        if added_orders:
            print(f"  * 신규 유입 주문 수: {len(added_orders)}건")
    else:
        print("- 최초 전처리 파일 생성입니다.")
    
    print("="*50)
    print("✨ 파이프라인 처리가 완료되었습니다.")

if __name__ == "__main__":
    rebuild_pipeline()
