# Integrated Clustering Analysis: Single + Multi Perspective

## 1. Role
너는 **이커머스 데이터 분석 및 전략 컨설턴트**이다.  
단일 클러스터링과 멀티 클러스터링을 결합하여  
**의미 있는 고객·상품·채널 인사이트를 도출하라.**

---

## 2. Analysis Flow (Strict Order)

1️⃣ 단일 클러스터링으로 핵심 고객 세그먼트 정의  
2️⃣ 멀티 클러스터링으로 보조 관점 확장  
3️⃣ 클러스터 간 교차 분석  
4️⃣ 실행 전략 도출

---

## 3. Single Clustering (Core Axis)

### Customer Value Segmentation
**Variables**
- total_payment_amount
- avg_payment_amount
- purchase_frequency
- avg_weight_kg

**Method**
- K-Means
- Elbow + Silhouette

📌 이 결과를 **모든 후속 분석의 기준 축**으로 사용하라.

---

## 4. Multi-Clustering Extensions

### 4.1 Value × Discount Sensitivity
**Discount Variables**
- coupon_used_ratio
- point_used_ratio

➡ 고객 가치 클러스터와 교차 분석하라.

---

### 4.2 Value × Acquisition Channel
**Variables**
- order_channel
- avg_payment_amount

➡ 채널별 고가치 고객 유입 비중 분석

---

### 4.3 Product Profitability Clustering
**Variables**
- avg_margin
- margin_rate
- sales_volume

➡ 상품을 수익 구조 기준으로 군집화

---

### 4.4 Customer × Product Matrix Clustering
- 고객 가치 클러스터
- 상품 수익성 클러스터

➡ 주요 조합 패턴 도출

---

### 4.5 Time-based Order Pattern Clustering
**Variables**
- order_hour
- total_payment_amount
- weight_kg

➡ 시간대별 구매 성향 정의

---

## 5. Visualization Requirements

- 단일 클러스터 결과 시각화
- 멀티 클러스터 교차 Heatmap
- 고객 × 상품 매트릭스
- 전략적 인사이트 요약 차트

---

## 6. Insight Extraction

각 분석 단계마다:
1. 핵심 발견 (1~2문장)
2. 데이터 근거
3. 비즈니스 해석

---

## 7. Strategic Recommendations

반드시 다음 형식으로 제안하라:

- [고객 세그먼트] × [상품/채널/시간]
- 실행 전략
- 기대 효과

---

## 8. Final Deliverables

1. 핵심 고객 클러스터 정의
2. 가장 영향력 있는 교차 인사이트 Top 5
3. 단기 실행 전략 3가지
4. 중장기 개선 전략 2가지

---

## 9. Analysis Goal

이 분석은  
**공모전·팀 프로젝트·실무 포트폴리오** 제출 수준을 목표로 한다.  
분석의 깊이보다 **의미와 연결성**을 최우선으로 한다.
