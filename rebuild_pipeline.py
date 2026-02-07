import pandas as pd
import numpy as np
import os
import glob
import re
import hashlib
import json

def generate_uid(row):
    try:
        val = str(row['판매단가']).replace(',', '')
        unit_price = str(int(float(val)))
    except:
        unit_price = "0"
    
    # UID = hash(상품코드 + 상품명 + 고객선택옵션 + 단가)
    opt = str(row['고객선택옵션']) if pd.notna(row['고객선택옵션']) else ""
    text = f"{str(row['상품코드'])}{str(row['상품명'])}{opt}{unit_price}"
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

def load_manifest(directory):
    path = os.path.join(directory, "processed_files_manifest.json")
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return set(json.load(f))
        except:
            return set()
    return set()

def save_manifest(directory, manifest_set):
    path = os.path.join(directory, "processed_files_manifest.json")
    with open(path, 'w') as f:
        json.dump(sorted(list(manifest_set)), f, indent=4)

def rebuild_pipeline():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ORIGIN_DIR = os.path.join(BASE_DIR, "data", "origin_data")
    DATA_DIR = os.path.join(BASE_DIR, "data")
    
    LATEST_OLD_PATH = os.path.join(DATA_DIR, "preprocessed_data.csv")
    
    # 1. 기존 데이터 및 매니페스트 확인
    existing_df = pd.DataFrame()
    if os.path.exists(LATEST_OLD_PATH):
        print(f"   - 기존 데이터 로드 중: {os.path.basename(LATEST_OLD_PATH)}")
        existing_df = pd.read_csv(LATEST_OLD_PATH)
    
    processed_files = load_manifest(DATA_DIR)
    
    # 2. 신규 파일 탐색
    all_excel_files = glob.glob(os.path.join(ORIGIN_DIR, "*.xlsx"))
    new_files = [f for f in all_excel_files if os.path.basename(f) not in processed_files]
    
    if not new_files:
        print("💡 추가된 신규 데이터 파일이 없습니다. 작업을 종료합니다.")
        return

    print(f"1. 신규 파일 가공 시작: {len(new_files)}개 파일")
    new_dfs = []
    for f in sorted(new_files):
        print(f"   - {os.path.basename(f)} 읽는 중...")
        new_dfs.append(pd.read_excel(f))
    
    raw_new = pd.concat(new_dfs, ignore_index=True)
    
    # 중복 제거 (신규 데이터 내에서만)
    if '주문상품고유번호' in raw_new.columns:
        raw_new = raw_new.drop_duplicates(subset=['주문상품고유번호'], keep='first')

    print("2. 신규 데이터 익명화 및 전처리 진행 중...")
    if '주문자연락처' in raw_new.columns:
        raw_new['주문자연락처'] = anonymize_phone(raw_new['주문자연락처'].astype(str))
    if '수령인연락처' in raw_new.columns:
        raw_new['수령인연락처'] = anonymize_phone(raw_new['수령인연락처'].astype(str))
    
    # 옵션코드 생성 (신규 데이터 기준)
    if '상품코드' in raw_new.columns and '고객선택옵션' in raw_new.columns:
        df_opt = raw_new[['상품코드', '고객선택옵션']].drop_duplicates().copy()
        df_opt['option_idx'] = df_opt.groupby('상품코드').cumcount() + 1
        df_opt['옵션코드'] = df_opt['상품코드'].astype(str) + "_" + df_opt['option_idx'].astype(str)
        raw_new = pd.merge(raw_new, df_opt[['상품코드', '고객선택옵션', '옵션코드']], on=['상품코드', '고객선택옵션'], how='left')

    mapping = {
        '결제금액(상품별)': '결제금액',
        '주문취소 금액(상품별)': '주문취소 금액',
        '상품금액(옵션포함)': '판매단가',
        '공급가': '공급단가'
    }
    df = raw_new.rename(columns=mapping).copy()

    # 숫자형 변환
    numeric_cols = ['결제금액', '주문취소 금액', '판매단가', '공급단가', '주문수량', '취소수량']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0).astype(float)

    # 기본 계산
    if '판매단가' in df.columns:
        df['판매단가'] = df.apply(lambda r: r['판매단가'] / r['주문수량'] if r['주문수량'] > 0 else r['판매단가'], axis=1)
    df['실결제 금액'] = df.get('결제금액', 0) - df.get('주문취소 금액', 0)
    df['주문-취소 수량'] = df.get('주문수량', 0) - df.get('취소수량', 0)
    df['취소여부'] = df.get('취소수량', 0).apply(lambda x: 'Y' if x > 0 else 'N')
    df['UID'] = df.apply(generate_uid, axis=1)

    # 3. 상세 분석 필드 생성
    if '주문일' in df.columns:
        df['주문일_dt'] = pd.to_datetime(df['주문일'])
        df['주문날짜'] = df['주문일_dt'].dt.date
    
    # 지역 분석
    def extract_region(row):
        address = str(row.get('주소', '')).strip()
        postcode = str(row.get('우편번호', '')).zfill(5)
        prefix = postcode[:2]
        
        sido_map = {
            '01':'서울','02':'서울','03':'서울','04':'서울','05':'서울','06':'서울','07':'서울','08':'서울','09':'서울',
            '10':'경기','11':'경기','12':'경기','13':'경기','14':'경기','15':'경기','16':'경기','17':'경기','18':'경기','19':'경기','20':'경기',
            '21':'인천','22':'인천','23':'인천','24':'강원','25':'강원','26':'강원','27':'충북','28':'충북','29':'충북','30':'세종',
            '31':'충남','32':'충남','33':'충남','34':'대전','35':'대전','36':'경북','37':'경북','38':'경북','39':'경북','40':'경북',
            '41':'대구','42':'대구','43':'대구','44':'울산','45':'울산','46':'부산','47':'부산','48':'부산','49':'부산',
            '50':'경남','51':'경남','52':'경남','53':'경남','54':'전북','55':'전북','56':'전북','57':'전남','58':'전남','59':'전남','60':'전남',
            '61':'광주','62':'광주','63':'제주'
        }
        formal_map = {
            '서울':'서울특별시','부산':'부산광역시','대구':'대구광역시','인천':'인천광역시','광주':'광주광역시','대전':'대전광역시','울산':'울산광역시','세종':'세종특별자치시',
            '경기':'경기도','강원':'강원특별자치도','충북':'충청북도','충남':'충청남도','전북':'전북특별자치도','전남':'전라남도','경북':'경상북도','경남':'경상남도','제주':'제주특별자치도'
        }
        
        sido = sido_map.get(prefix, next((s for s in sido_map.values() if s in address), "기타"))
        sido_formal = formal_map.get(sido, sido)
        
        # 시군구 추출 고도화
        parts = address.split()
        sigungu = "미분류"
        
        if len(parts) > 0:
            # Sido 키워드 목록
            sido_keywords = ['서울', '부산', '대구', '인천', '광주', '대전', '울산', '세종', '경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주']
            formal_keywords = list(formal_map.values())
            all_sido_kws = sido_keywords + formal_keywords + ["전북특별자치도", "강원특별자치도", "제주특별자치도"]
            
            # 첫 번째 단어가 Sido인지 확인
            is_first_sido = any(parts[0].startswith(kw) for kw in all_sido_kws)
            
            start_idx = 1 if is_first_sido else 0
            
            if len(parts) > start_idx:
                candidate = parts[start_idx]
                # '시', '군', '구'로 끝날 때만 시군구로 인정
                if candidate.endswith(('시', '군', '구')):
                    # '수원시 팔달구'처럼 시+구 형태인 경우 합침
                    if candidate.endswith('시') and len(parts) > start_idx + 1:
                        if parts[start_idx+1].endswith('구'):
                            candidate += " " + parts[start_idx+1]
                    sigungu = candidate
        
        if sido == "세종": sigungu = "세종시"
            
        return sido, sido_formal, sigungu

    df[['광역지역', '광역지역(정식)', '시군구']] = df.apply(lambda r: pd.Series(extract_region(r)), axis=1)

    # 텍스트 분석
    def parse_text(row):
        txt = f"{str(row.get('상품명', ''))} {str(row.get('고객선택옵션', ''))}"
        
        # 품종(cat) 세분화 로직
        if "한라봉" in txt: cat = "한라봉"
        elif "레드향" in txt: cat = "레드향"
        elif "천혜향" in txt: cat = "천혜향"
        elif "황금향" in txt: cat = "황금향"
        elif "카라향" in txt: cat = "카라향"
        elif any(k in txt for k in ["청귤", "풋귤"]): cat = "청귤"
        elif any(k in txt for k in ["감귤", "조생", "타이벡", "귤"]): cat = "감귤"
        elif "당근" in txt: cat = "당근"
        elif "고구마" in txt: cat = "고구마"
        elif "비트" in txt: cat = "비트"
        elif "콜라비" in txt: cat = "콜라비"
        else: cat = "기타"

        sub = "타이벡" if "타이벡" in txt else ("조생" if "조생" in txt else ("하우스" if "하우스" in txt else "일반"))
        size = "기타"
        if "소과" in txt and "중소과" in txt: size = "소과, 중소과"
        elif "중대과" in txt and "대과" in txt: size = "중대과, 대과"
        elif "소과" in txt: size = "소과"; 
        elif "혼합" in txt: size = "혼합"; 
        elif "로얄과" in txt: size = "로얄과"; 
        elif "대과" in txt: size = "대과"; 
        elif "중과" in txt: size = "중과";
        gift = "선물세트" if any(k in txt for k in ["선물세트", "선물용"]) else "가정용"
        ev = "Y" if any(k in txt for k in ["이벤트", "1+1", "보장"]) else "N"
        # 프리미엄 등급 신규 기준 적용
        is_premium = False
        if gift == "선물세트":
            is_premium = True
        elif cat == "감귤" and size == "로얄과":
            is_premium = True
        elif cat in ["황금향", "한라봉", "레드향", "천혜향"] and any(s in size for s in ["중과", "중대과", "대과"]):
            is_premium = True
        
        grade = "프리미엄" if is_premium else "일반"
        w_match = re.findall(r'(\d+(\.\d+)?)\s*(kg|KG)', txt)
        w = sum(float(m[0]) for m in w_match) if w_match else 0.0
        grp = "<3kg" if 0 < w < 3 else ("3-5kg" if w <= 5 else ("5-10kg" if w <= 10 else ">10kg")) if w > 0 else "미분류"
        purpose = "선물" if str(row.get('주문자명', '')).replace(' ','') != str(row.get('수령인명', '')).replace(' ','') or "선물" in txt else "개인소비"
        opt_clean = str(row.get('고객선택옵션(타입제거)', row.get('고객선택옵션', '')))
        return pd.Series([sub, cat, size, gift, w, grp, ev, grade, purpose, opt_clean])

    df[['감귤 세부', '품종', '과수 크기', '선물세트_여부', '무게(kg)', '무게 구분', '이벤트 여부', '상품성등급_그룹', '목적', '고객선택옵션(타입제거)']] = df.apply(parse_text, axis=1)
    df['가격대'] = df['실결제 금액'].apply(lambda a: "1만원 이하" if a <= 10000 else ("1~3만원" if a <= 30000 else ("3~5만원" if a <= 50000 else ("5~10만원" if a <= 100000 else "10만원 초반"))))

    # 4. 재구매 횟수 산출 (히스토리 참고)
    print("4. 히스토리 기반 재구매 횟수 산출 중...")
    if not existing_df.empty:
        existing_df['주문일_temp'] = pd.to_datetime(existing_df['주문일']).dt.date
        history_map = existing_df.groupby('주문자연락처')['주문일_temp'].unique().to_dict()
        history_map = {k: set(v) for k, v in history_map.items()}
    else:
        history_map = {}

    new_history = df.groupby('주문자연락처')['주문날짜'].unique().to_dict()
    new_history = {k: set(v) for k, v in new_history.items()}

    def calc_repurchase(row):
        contact = str(row['주문자연락처'])
        order_date = row['주문날짜']
        past_dates = history_map.get(contact, set())
        current_new_dates = new_history.get(contact, set())
        combined_dates = past_dates.union(current_new_dates)
        return max(0, len(combined_dates) - 1)

    df['재구매 횟수'] = df.apply(calc_repurchase, axis=1)

    # 5. 최종 병합 및 저장
    print("5. 데이터 병합 및 저장 중...")
    target_columns = [
        'UID', '주문번호', '주문일', '상품코드', '옵션코드', '상품명', '고객선택옵션', '주문수량', '취소수량', '주문-취소 수량',
        '결제금액', '주문취소 금액', '실결제 금액', '판매단가', '공급단가', '주문경로', '주문자명', '셀러명', '결제방법',
        '배송준비 처리일', '입금일', '주문자연락처', '주소', '입금자명', '수령인명', '수령인연락처', '회원구분', '우편번호',
        '고객선택옵션(타입제거)', '취소여부', '목적', '재구매 횟수', '광역지역', '광역지역(정식)', '시군구',
        '감귤 세부', '품종', '과수 크기', '선물세트_여부', '무게(kg)', '무게 구분', '이벤트 여부', '상품성등급_그룹', '가격대'
    ]
    
    new_final = df.reindex(columns=target_columns).fillna("")
    
    for col in ['결제금액', '실결제 금액', '판매단가', '공급단가']:
        new_final[col] = new_final[col].apply(lambda x: f"{int(x):,}")

    if not existing_df.empty:
        if '주문일_temp' in existing_df.columns:
            existing_df = existing_df.drop(columns=['주문일_temp'])
        final_df = pd.concat([existing_df.reindex(columns=target_columns), new_final], ignore_index=True)
    else:
        final_df = new_final

    final_df.to_csv(LATEST_OLD_PATH, index=False, encoding='utf-8-sig')
    
    # 버전 파일로도 저장
    next_ver = get_next_version(DATA_DIR)
    ver_path = os.path.join(DATA_DIR, f"preprocessed_data_{next_ver}.csv")
    final_df.to_csv(ver_path, index=False, encoding='utf-8-sig')

    # 매니페스트 업데이트
    for f in new_files:
        processed_files.add(os.path.basename(f))
    save_manifest(DATA_DIR, processed_files)
    
    print(f"✅ 처리가 완료되었습니다. {len(new_final):,}행이 추가되었습니다.")
    print(f"   - 최종 파일: {LATEST_OLD_PATH}")
    print(f"   - 백업 버전: {os.path.basename(ver_path)}")

if __name__ == "__main__":
    rebuild_pipeline()
