# 이커머스 주문 데이터 전처리 결과 보고서

본 보고서는 `store_data.xlsx` 원본 데이터를 `preprocessing_request.md`의 요구사항에 따라 전처리한 결과에 대해 설명합니다.

---

## 1. 전처리된 테이블 구조 및 컬럼 정의

### 1.1 `orders.csv` (주문 단위)
주문 및 결제 전반에 관한 정보를 담고 있습니다.
- `order_id`: 주문번호
- `order_datetime`: 주문 일시 (YYYY-MM-DD HH:MM:SS)
- `order_date`: 주문 날짜
- `order_hour`: 주문 시간 (0-23)
- `order_channel`: 주문 경로
- `payment_method`: 결제 방법
- `total_payment_amount`: 총 결제 금액
- `supply_amount`: 총 공급가
- `coupon_amount`: 사용 쿠폰 금액
- `point_amount`: 사용 포인트 금액
- `seller_name`: 셀러명
- `weekday`: 요일 (English)
- `is_weekend`: 주말 여부 (0: 평일, 1: 주말)
- `time_slot`: 시간대 구분 (아침/점심/저녁/야간)
- `channel_group`: 마케팅 채널 그룹 (검색/SNS/기타)

### 1.2 `order_items.csv` (상품 단위)
주문 내 개별 상품 및 옵션 상세 정보를 담고 있습니다.
- `order_id`: 주문번호 (FK)
- `product_code`: 상품 코드
- `product_name`: 정제된 상품 기본명
- `option_type`: 상품 옵션 (소과/중과/로얄과 등)
- `weight_kg`: 중량 (kg 단위, 숫자형)
- `size_range`: 크기/수량 범위 (예: 50~55수)
- `quantity`: 주문 수량
- `item_payment_amount`: 상품별 결제 금액
- `is_promotion`: 프로모션 포함 여부 (0/1)
- `promotion_detail`: 프로모션 상세 내용
- `price_per_kg`: kg당 가격 (파생변수)
- `margin`: 마진 금액 (결제금액 - 공급가)
- `margin_rate`: 마진율
- `is_high_price`: 고가 상품 여부 (5만원 이상)

### 1.3 `customers.csv` (고객 단위)
고객별 행동 및 속성 정보를 담고 있습니다.
- `customer_id`: 연락처 기반 SHA-256 해시 ID
- `member_type`: 회원/비회원 구분
- `region_1`: 주소 기반 시/도
- `region_2`: 주소 기반 시/군/구
- `first_order_date`: 해당 고객의 첫 구매일

### 1.4 `products.csv` (상품 마스터)
상품별 카테고리 및 재배 방식 정보를 담고 있습니다.
- `product_code`: 상품 코드 (PK)
- `product_category`: 상품 카테고리 (감귤/황금향 등)
- `harvest_type`: 재배 방식 (노지/하우스)
- `base_weight_kg`: 대표 중량

---

## 2. 데이터 예시 (전/후)

### 전 (Raw)
- **주문번호**: YMM251017-00000198
- **상품명**: 제주 산지직송 하우스감귤 2kg ... ▶ 소과 2kg (25~33수) ★2개 구매 시 0.5kg 추가 발송★ (2개)

### 후 (Preprocessed - Items)
- **product_name**: 제주 산지직송 하우스감귤 2kg ...
- **option_type**: 소과
- **weight_kg**: 2.0
- **size_range**: 25~33수
- **is_promotion**: 1
- **promotion_detail**: 2개 구매 시 0.5kg 추가 발송

---

## 3. 분석 가능 시나리오 (5가지)

1. **채널별 객단가(AOV) 비교**: `orders`의 `channel_group`별로 `total_payment_amount`의 평균을 비교하여 마케팅 효율성 파악.
2. **중량별 가격 민감도 분석**: `order_items`의 `weight_kg`에 따른 `price_per_kg` 트렌드를 분석하여 소비자가 선호하는 최적 가격대 도출.
3. **재구매 고객 행동 분석**: `customers`의 `first_order_date`와 `orders`의 `order_date`를 결합하여 첫 구매 후 재구매까지의 리드타임 분석.
4. **프로모션 기여도 분석**: `order_items`의 `is_promotion` 여부에 따른 판매량(`quantity`) 및 총 매출 차이 검정.
5. **지역별 선호 카테고리 분석**: `customers`의 `region_1`과 `products`의 `product_category`를 결합하여 지역별 수요 특성 파악.

---

## 4. GA4 / BigQuery 연계 시 활용 포인트

- **GA4 User ID 매핑**: `customer_id` (해시값)를 GA4의 `user_id`로 전송하여 웹 상의 비식별 행동 데이터와 실제 주문 데이터를 결합 분석 가능.
- **맞춤 측정기준(Custom Dimensions)**: `option_type`, `weight_kg`, `channel_group` 등을 BigQuery에서 GA4 이벤트 파라미터와 조인하여 상세 상품 속성별 전환율 분석.
- **RFM 세그멘테이션**: `customers`와 `orders` 데이터를 활용해 BigQuery ML 또는 SQL로 고객 등급을 분류하고 이를 다시 마케팅 액션에 활용.
