import pandas as pd
import numpy as np
import os
import re

def run_pipeline(raw_path, output_path):
    print(f"1. 원본 데이터 로드 중: {raw_path}")
    raw_df = pd.read_excel(raw_path)

    # 2. 칼럼 매핑 및 초기 정규화
    print("2. 칼럼 매핑 및 정식화 중...")
    mapping = {
        '결제금액(상품별)': '결제금액',
        '주문취소 금액(상품별)': '주문취소 금액',
        '단가': '판매단가',
        '공급가': '공급단가'
    }
    df = raw_df.rename(columns=mapping).copy()
    
    # 기초 필드 계산 및 숫자 전처리
    for col in ['결제금액', '주문취소 금액', '판매단가', '공급단가', '주문수량', '취소수량']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0).astype(int)

    df['실결제 금액'] = df['결제금액'] - df['주문취소 금액']
    df['주문-취소 수량'] = df['주문수량'] - df['취소수량']
    df['취소여부'] = df['취소수량'].apply(lambda x: 'Y' if x > 0 else 'N')
    
    # 3. 재구매 횟수 계산
    print("3. 재구매 횟수 계산 중...")
    df['주문일_dt'] = pd.to_datetime(df['주문일'])
    df['주문날짜'] = df['주문일_dt'].dt.date
    repurchase_map = df.groupby('주문자연락처')['주문날짜'].nunique().to_dict()
    df['재구매 횟수'] = df['주문자연락처'].map(lambda x: max(0, repurchase_map.get(x, 1) - 1))

    # 4. 지역 정보 보완
    print("4. 지역 정보 추출 중...")
    sido_map = {
        '서울': '서울', '부산': '부산', '대구': '대구', '인천': '인천', '광주': '광주', '대전': '대전', '울산': '울산',
        '세종': '세종특별자치시', '경기': '경기', '강원': '강원특별자치도', '충북': '충북', '충남': '충남', 
        '전북': '전북특별자치도', '전남': '전남', '경북': '경북', '경남': '경남', '제주': '제주특별자치도'
    }
    formal_sido_map = {
        '서울': '서울특별시', '부산': '부산광역시', '대구': '대구광역시', '인천': '인천광역시', '광주': '광주광역시', '대전': '대전광역시', '울산': '울산광역시',
        '세종': '세종특별자치시', '경기': '경기도', '강원': '강원특별자치도', '충북': '충청북도', '충남': '충청남도', 
        '전북': '전북특별자치도', '전남': '전라남도', '경북': '경상북도', '경남': '경상남도', '제주': '제주특별자치도'
    }
    postcode_map = {
        '01': '서울', '02': '서울', '03': '서울', '04': '서울', '05': '서울', '06': '서울', '07': '서울', '08': '서울', '09': '서울',
        '10': '경기', '11': '경기', '12': '경기', '13': '경기', '14': '경기', '15': '경기', '16': '경기', '17': '경기', '18': '경기', '19': '경기', '20': '경기',
        '21': '인천', '22': '인천', '23': '인천', '24': '강원', '25': '강원', '26': '강원', '27': '충북', '28': '충북', '29': '충북', '30': '세종',
        '31': '충남', '32': '충남', '33': '충남', '34': '대전', '35': '대전', '36': '경북', '37': '경북', '38': '경북', '39': '경북', '40': '경북',
        '41': '대구', '42': '대구', '43': '대구', '44': '울산', '45': '울산', '46': '부산', '47': '부산', '48': '부산', '49': '부산',
        '50': '경남', '51': '경남', '52': '경남', '53': '경남', '54': '전북', '55': '전북', '56': '전북', '57': '전남', '58': '전남', '59': '전남', '60': '전남',
        '61': '광주', '62': '광주', '63': '제주'
    }

    def extract_sido_logic(row):
        address = str(row['주소'])
        postcode = str(row['우편번호']).zfill(5)
        prefix = postcode[:2]
        if prefix in postcode_map: return postcode_map[prefix]
        for short, full in sido_map.items():
            if address.startswith(short) or address.startswith(full): return short
        for short, full in sido_map.items():
            if short in address or full in address: return short
        return ""

    df['광역지역'] = df.apply(extract_sido_logic, axis=1)
    df['광역지역(정식)'] = df['광역지역'].map(lambda x: formal_sido_map.get(x, x))

    # 5. 텍스트 분석 기반 파생 칼럼 생성
    print("5. 텍스트 분석 기반 파생 칼럼 생성 중...")
    
    def parse_derived_fields(row):
        full_text = f"{str(row['상품명'])} {str(row['고객선택옵션'])}"
        
        # 1) 품종 추출
        category_list = []
        if any(k in full_text for k in ["감귤", "조생", "타이벡", "귤"]): category_list.append("감귤")
        if "황금향" in full_text: category_list.append("황금향")
        if "고구마" in full_text: category_list.append("고구마")
        if "한라봉" in full_text: category_list.append("한라봉")
        if "레드향" in full_text: category_list.append("레드향")
        if "천혜향" in full_text: category_list.append("천혜향")
        if "당근" in full_text: category_list.append("당근")
        if "딸기" in full_text: category_list.append("딸기")
        category = ", ".join(category_list) if category_list else "기타"
        
        # 2) 감귤 세부
        sub_category = "일반"
        if "감귤" in category:
            if "타이벡" in full_text: sub_category = "타이벡"
            elif "유라실생" in full_text or "유라" in full_text: sub_category = "유라실생"
            elif "조생" in full_text: sub_category = "조생"
            elif "하우스" in full_text: sub_category = "하우스"
            elif "노지" in full_text: sub_category = "노지"
            elif "고당도" in full_text: sub_category = "고당도"
        
        # 3) 과수 크기
        size = "기타"
        if "소과" in full_text and "중소과" in full_text: size = "소과, 중소과"
        elif "중대과" in full_text and "대과" in full_text: size = "중대과, 대과"
        elif "소과" in full_text: size = "소과"
        elif "혼합" in full_text: size = "혼합"
        elif "로얄과" in full_text: size = "로얄과"
        elif "대과" in full_text: size = "대과"
        elif "중과" in full_text: size = "중과"
        
        # 4) 무게(kg) 추출
        weight_text = full_text.split('(')[0]
        weight_matches = re.findall(r'(\d+(\.\d+)?)\s*(kg|KG)', weight_text)
        weight_kg = sum(float(m[0]) for m in weight_matches) if weight_matches else 0.0
        
        if weight_kg == 0:
            all_matches = re.findall(r'(\d+(\.\d+)?)\s*(kg|KG)', full_text)
            weight_kg = float(all_matches[0][0]) if all_matches else 0.0
        
        # 5) 무게 구분
        weight_group = "미분류"
        if weight_kg > 0:
            if weight_kg < 3: weight_group = "<3kg"
            elif weight_kg <= 5: weight_group = "3-5kg"
            elif weight_kg <= 10: weight_group = "5-10kg"
            else: weight_group = ">10kg"
            
        # 6) 선물세트 여부 (키워드 기반)
        is_gift_set = "가정용"
        if "선물세트" in full_text or "선물용" in full_text:
            is_gift_set = "선물세트"
        elif "가정용" in full_text:
            is_gift_set = "가정용"
            
        # 7) 목적 (주문자 vs 수령인 기반)
        orderer = str(row['주문자명']).replace(' ', '')
        receiver = str(row['수령인명']).replace(' ', '')
        purpose = "선물" if orderer != receiver else "개인소비"
        if "선물" in full_text: purpose = "선물"
            
        # 8) 이벤트 여부
        is_event = "Y" if any(keyword in full_text for keyword in ["이벤트", "1+1", "보장", "★"]) else "N"
            
        # 9) 상품성등급
        grade = "프리미엄" if any(keyword in full_text for keyword in ["프리미엄", "명품", "고당도", "킹댕즈", "타이벡"]) else "일반"
            
        return pd.Series([sub_category, category, size, is_gift_set, weight_kg, weight_group, is_event, grade, purpose])

    derived_cols = ['감귤 세부', '품종', '과수 크기', '선물세트_여부', '무게(kg)', '무게 구분', '이벤트 여부', '상품성등급_그룹', '목적']
    df[derived_cols] = df.apply(parse_derived_fields, axis=1)

    # 6. 가격대 분류
    def classify_price(amount):
        if amount <= 10000: return "1만원 이하"
        elif amount <= 30000: return "1~3만원"
        elif amount <= 50000: return "3~5만원"
        elif amount <= 100000: return "5~10만원"
        else: return "10만원 초반"
    
    df['가격대'] = df['실결제 금액'].apply(classify_price)

    # 7. 최종 데이터 정규화 및 저장
    float_cols = ['무게(kg)']
    int_cols = ['결제금액', '주문취소 금액', '실결제 금액', '판매단가', '공급단가', '재구매 횟수', '주문수량', '취소수량', '주문-취소 수량']
    
    for col in int_cols:
        df[col] = df[col].fillna(0).astype(int)
    for col in float_cols:
        df[col] = df[col].fillna(0.0).astype(float)

    target_columns = [
        'UID', '주문번호', '주문일', '상품코드', '옵션코드', '상품명', '고객선택옵션', '주문수량', '취소수량', '주문-취소 수량',
        '결제금액', '주문취소 금액', '실결제 금액', '판매단가', '공급단가', '주문경로', '주문자명', '셀러명', '결제방법',
        '배송준비 처리일', '입금일', '주문자연락처', '주소', '입금자명', '수령인명', '수령인연락처', '회원구분', '우편번호',
        '고객선택옵션(타입제거)', '취소여부', '목적', '재구매 횟수', '광역지역', '광역지역(정식)',
        '감귤 세부', '품종', '과수 크기', '선물세트_여부', '무게(kg)', '무게 구분', '이벤트 여부', '상품성등급_그룹', '가격대'
    ]
    
    for col in target_columns:
        if col not in df.columns: df[col] = ""

    df = df[target_columns]
    
    print(f"6. 최종 결과 저장 중: {output_path}")
    df.to_csv(output_path, index=False)
    print("파이프라인 실행이 성공적으로 완료되었습니다!")

if __name__ == "__main__":
    raw_file = "/Users/dmjeong/innercircle/DA_project1/data/rawdata.xlsx"
    processed_file = "/Users/dmjeong/innercircle/DA_project1/data/preprocessed_data.csv"
    run_pipeline(raw_file, processed_file)
