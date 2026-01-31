import pandas as pd
import os
import re

def process_data(file_path):
    print(f"데이터를 불러오는 중: {file_path}")
    df = pd.read_csv(file_path)

    # 1. 재구매 횟수 계산
    # '주문일'을 날짜 형식으로 변환
    df['주문일_dt'] = pd.to_datetime(df['주문일'])
    df['주문날짜'] = df['주문일_dt'].dt.date

    # 주문자연락처별로 고유 주문날짜 수 계산
    repurchase_map = df.groupby('주문자연락처')['주문날짜'].nunique().to_dict()
    
    # 재구매 횟수 = (고유 날짜 수) - 1 (최소 0)
    df['재구매 횟수'] = df['주문자연락처'].map(lambda x: max(0, repurchase_map.get(x, 1) - 1))

    # 2. 지역 정보 보완 (광역지역)
    # 한국 광역지자체 리스트
    sido_list = [
        '서울특별시', '부산광역시', '대구광역시', '인천광역시', '광주광역시', '대전광역시', '울산광역시', 
        '세종특별자치시', '경기도', '강원특별자치도', '강원도', '충청북도', '충청남도', '전라북도', 
        '전북특별자치도', '전라남도', '경상북도', '경상남도', '제주특별자치도'
    ]
    # 축약어 매핑
    sido_map = {
        '서울': '서울', '부산': '부산', '대구': '대구', '인천': '인천', '광주': '광주', '대전': '대전', '울산': '울산',
        '세종': '세종특별자치시', '경기': '경기', '강원': '강원특별자치도', '충북': '충북', '충남': '충남', 
        '전북': '전북특별자치도', '전남': '전남', '경북': '경북', '경남': '경남', '제주': '제주특별자치도'
    }

    # 시도별 우편번호 앞 2자리 매핑
    postcode_map = {
        '01': '서울', '02': '서울', '03': '서울', '04': '서울', '05': '서울', '06': '서울', '07': '서울', '08': '서울', '09': '서울',
        '10': '경기', '11': '경기', '12': '경기', '13': '경기', '14': '경기', '15': '경기', '16': '경기', '17': '경기', '18': '경기', '19': '경기', '20': '경기',
        '21': '인천', '22': '인천', '23': '인천',
        '24': '강원', '25': '강원', '26': '강원',
        '27': '충북', '28': '충북', '29': '충북',
        '30': '세종',
        '31': '충남', '32': '충남', '33': '충남',
        '34': '대전', '35': '대전',
        '36': '경북', '37': '경북', '38': '경북', '39': '경북', '40': '경북',
        '41': '대구', '42': '대구', '43': '대구',
        '44': '울산', '45': '울산',
        '46': '부산', '47': '부산', '48': '부산', '49': '부산',
        '50': '경남', '51': '경남', '52': '경남', '53': '경남',
        '54': '전북', '55': '전북', '56': '전북',
        '57': '전남', '58': '전남', '59': '전남', '60': '전남',
        '61': '광주', '62': '광주',
        '63': '제주'
    }

    def extract_sido(row):
        address = str(row['주소'])
        postcode = str(row['우편번호']).zfill(5)
        
        # 1. 우편번호 우선 체크 (앞 2자리)
        prefix = postcode[:2]
        if prefix in postcode_map:
            return postcode_map[prefix]
            
        # 2. 주소 텍스트 기반 체크 (우편번호가 유효하지 않을 경우)
        for short, full in sido_map.items():
            if address.startswith(short) or address.startswith(full):
                return short
        
        for short, full in sido_map.items():
            if short in address or full in address:
                return short
        return ""

    # 광역지역이 비어 있거나 'nan'인 경우, 또는 유효하지 않은 값인 경우 보완
    # 유효한 광역지자체 명칭 패턴 (부분 일치 허용)
    valid_sidos = set(sido_map.keys()) | set(sido_list)
    
    def is_invalid_sido(sido):
        if pd.isna(sido) or str(sido).strip() == "" or str(sido).lower() == 'nan':
            return True
        # 주소 형태(도로명, 건물번호 등)가 들어있는지 대략적으로 체크
        if any(char.isdigit() for char in str(sido)):
            return True
        # 표준 리스트에 없는 경우
        if sido not in valid_sidos:
            return True
        return False

    mask = df['광역지역'].apply(is_invalid_sido)
    print(f"- 보완이 필요한 행 수: {mask.sum()}")
    
    df.loc[mask, '광역지역'] = df[mask].apply(extract_sido, axis=1)

    # 3. 광역지역(정식) 명칭 열 추가
    formal_sido_map = {
        '서울': '서울특별시', '부산': '부산광역시', '대구': '대구광역시', '인천': '인천광역시', '광주': '광주광역시', '대전': '대전광역시', '울산': '울산광역시',
        '세종': '세종특별자치시', '경기': '경기도', '강원': '강원특별자치도', '충북': '충청북도', '충남': '충청남도', 
        '전북': '전북특별자치도', '전남': '전라남도', '경북': '경상북도', '경남': '경상남도', '제주': '제주특별자치도'
    }
    df['광역지역(정식)'] = df['광역지역'].map(lambda x: formal_sido_map.get(x, x))
    
    # 열 순서 조정 (광역지역 바로 뒤에 배치)
    cols = list(df.columns)
    if '광역지역' in cols and '광역지역(정식)' in cols:
        idx = cols.index('광역지역')
        cols.remove('광역지역(정식)')
        cols.insert(idx + 1, '광역지역(정식)')
        df = df[cols]

    # 임시 컬럼 제거
    df = df.drop(columns=['주문일_dt', '주문날짜'])

    # 결과 저장
    df.to_csv(file_path, index=False)
    print(f"처리 완료 및 저장됨: {file_path}")
    
    # 통계 출력
    repurchase_customers = (df['재구매 횟수'] > 0).sum()
    print(f"- 재구매 이력이 있는 주문 건수: {repurchase_customers}")
    print(f"- 광역지역 정보가 보완된 건수: {mask.sum()}")
    print("- 광역지역(정식) 열이 추가되었습니다.")

if __name__ == "__main__":
    target_file = "/Users/dmjeong/innercircle/DA_project1/data/preprocessed_data.csv"
    if os.path.exists(target_file):
        process_data(target_file)
    else:
        print(f"파일을 찾을 수 없습니다: {target_file}")
