import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import re
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- 페이지 설정 ---
st.set_page_config(
    page_title="Unified E-commerce EDA Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- 1. 데이터 로드 함수 (Original & Advanced) ---

@st.cache_data
def load_raw_data(file_path):
    df = pd.read_excel(file_path)
    if '주문일' in df.columns:
        df['주문일'] = pd.to_datetime(df['주문일'], errors='coerce')
    return df

@st.cache_data
def load_processed_data(base_path):
    try:
        orders = pd.read_csv(os.path.join(base_path, 'orders.csv'))
        items = pd.read_csv(os.path.join(base_path, 'order_items.csv'))
        customers = pd.read_csv(os.path.join(base_path, 'customers.csv'))
        products = pd.read_csv(os.path.join(base_path, 'products.csv'))
        
        # 조인
        df = items.merge(orders, on='order_id', how='left')
        df = df.merge(customers, on='customer_id', how='left')
        df = df.merge(products, on='product_code', how='left')
        
        # 타입 변환
        df['order_datetime'] = pd.to_datetime(df['order_datetime'])
        df['order_date'] = pd.to_datetime(df['order_date']).dt.date
        return df, True
    except Exception as e:
        return None, False

# --- 2. 경로 설정 및 로드 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(current_dir, 'data', 'store_data.xlsx')
PROCESSED_PATH = os.path.join(current_dir, 'preprocessed')

# 로드: Raw
if not os.path.exists(RAW_DATA_PATH):
    st.error(f"원본 데이터 파일이 없습니다: {RAW_DATA_PATH}")
    st.stop()
df_raw = load_raw_data(RAW_DATA_PATH)

# 로드: Processed
df_adv, adv_exists = load_processed_data(PROCESSED_PATH)

# --- 3. 기본 전처리 (Original Dashboard용) ---
df_raw['year_month'] = df_raw['주문일'].dt.to_period('M').astype(str)
df_raw['day_name'] = df_raw['주문일'].dt.day_name()

# --- 4. 메인 UI 및 사이드바 ---
st.title("📊 Enterprise E-commerce Analytics Dashboard")
st.markdown(f"**원본 데이터**: `{RAW_DATA_PATH}` | **총 레코드**: {len(df_raw):,}건")

st.sidebar.header("🔍 글로벌 필터")
start_date = df_raw['주문일'].min()
end_date = df_raw['주문일'].max()

date_range = st.sidebar.date_input(
    "조회 기간 (Original Tabs)",
    [start_date, end_date],
    min_value=start_date,
    max_value=end_date
)

if len(date_range) == 2:
    start_dt, end_dt = date_range
    filtered_df = df_raw.loc[(df_raw['주문일'].dt.date >= start_dt) & (df_raw['주문일'].dt.date <= end_dt)]
    if adv_exists:
        adv_filtered_df = df_adv[(df_adv['order_date'] >= start_dt) & (df_adv['order_date'] <= end_dt)]
else:
    filtered_df = df_raw
    adv_filtered_df = df_adv if adv_exists else None

# KPI (Original Style)
total_sales = filtered_df['결제금액(통합)'].sum()
total_orders = len(filtered_df)
avg_order_value = total_sales / total_orders if total_orders > 0 else 0
cancel_sales = filtered_df['주문취소 금액(상품별)'].sum()

k1, k2, k3, k4 = st.columns(4)
k1.metric("총 결제금액 (Gross Sales)", f"{total_sales:,.0f}원")
k2.metric("총 주문 건수", f"{total_orders:,}건")
k3.metric("평균 주문금액 (AOV)", f"{avg_order_value:,.0f}원")
k4.metric("취소 금액", f"{cancel_sales:,.0f}원", delta=f"-{(cancel_sales/total_sales)*100:.1f}%" if total_sales else 0)

st.divider()

# --- 5. Tabs 구성 ---
tab_names = ["📈 매출 트렌드", "📦 상품 분석", "👥 고객/채널", "📅 특정 날짜 분석", "🧩 옵션 분석", "📋 데이터 상세"]
if adv_exists:
    tab_names += ["🚀 [Advanced] 상품/매출", "🌐 [Advanced] 마케팅/고객", "🧬 [Advanced] 군집 분석", "💡 인사이트/제안"]

tabs = st.tabs(tab_names)

# Tab 1: 매출 트렌드 (Original)
with tabs[0]:
    st.subheader("기간별 매출 추이")
    trend_type = st.radio("집계 기준", ["일별", "월별"], horizontal=True, key="trend_raw")
    if trend_type == "일별":
        trend = filtered_df.groupby(filtered_df['주문일'].dt.date)['결제금액(통합)'].sum().reset_index()
        trend.columns = ['Date', 'Sales']
        fig_trend = px.line(trend, x='Date', y='Sales', title="일별 매출 추이", markers=True)
    else:
        trend = filtered_df.groupby('year_month')['결제금액(통합)'].sum().reset_index()
        trend.columns = ['Month', 'Sales']
        fig_trend = px.bar(trend, x='Month', y='Sales', title="월별 매출 추이", text_auto='.2s')
    st.plotly_chart(fig_trend, use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        filtered_df['day_name'] = pd.Categorical(filtered_df['day_name'], categories=day_order, ordered=True)
        day_trend = filtered_df.groupby('day_name', observed=False)['결제금액(통합)'].sum().reset_index()
        fig_day = px.bar(day_trend, x='day_name', y='결제금액(통합)', title="요일별 매출액", color='결제금액(통합)')
        st.plotly_chart(fig_day, use_container_width=True)
    with c2:
        filtered_df['hour'] = filtered_df['주문일'].dt.hour
        hour_trend = filtered_df.groupby('hour')['주문번호'].count().reset_index(name='count')
        fig_hour = px.line(hour_trend, x='hour', y='count', title="시간대별 주문 건수", markers=True)
        st.plotly_chart(fig_hour, use_container_width=True)

# Tab 2: 상품 분석 (Original)
with tabs[1]:
    st.subheader("Top Performing Products")
    top_products = filtered_df.groupby('상품명')['결제금액(통합)'].sum().reset_index().sort_values('결제금액(통합)', ascending=False).head(10)
    col_p1, col_p2 = st.columns([2, 1])
    with col_p1:
        fig_prod = px.bar(top_products, x='결제금액(통합)', y='상품명', orientation='h', title="매출 상위 10개 상품", text_auto='.2s')
        fig_prod.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_prod, use_container_width=True)
    with col_p2:
        st.dataframe(top_products, use_container_width=True)
        
    top_qty = filtered_df.groupby('상품명')['주문수량'].sum().reset_index().sort_values('주문수량', ascending=False).head(10)
    st.subheader("판매량 상위 상품")
    st.dataframe(top_qty.T, use_container_width=True)

# Tab 3: 고객/채널 (Original)
with tabs[2]:
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.subheader("회원 구분별 주문 비율")
        member_counts = filtered_df['회원구분'].value_counts()
        fig_member = px.pie(values=member_counts.values, names=member_counts.index, hole=0.4, title="회원 vs 비회원")
        st.plotly_chart(fig_member, use_container_width=True)
    with col_c2:
        st.subheader("결제 수단별 분석")
        pay_counts = filtered_df['결제방법'].value_counts()
        fig_pay = px.pie(values=pay_counts.values, names=pay_counts.index, title="결제 수단 점유율")
        st.plotly_chart(fig_pay, use_container_width=True)
    
    st.divider()
    c_route1, c_route2 = st.columns(2)
    with c_route1:
        route_df = filtered_df['주문경로'].value_counts().reset_index()
        route_df.columns = ['Route', 'Count']
        fig_route = px.bar(route_df, x='Route', y='Count', title="주문 경로별 건수")
        st.plotly_chart(fig_route, use_container_width=True)
    with c_route2:
        if '셀러명' in filtered_df.columns:
            seller_df = filtered_df.groupby('셀러명')['결제금액(통합)'].sum().reset_index().sort_values('결제금액(통합)', ascending=False).head(10)
            fig_seller = px.bar(seller_df, x='셀러명', y='결제금액(통합)', title="Top 10 셀러 (매출 기준)")
            st.plotly_chart(fig_seller, use_container_width=True)

# Tab 4: 특정 날짜 분석 (Original)
with tabs[3]:
    st.subheader("📅 특정 날짜 상세 분석")
    daily_stats = filtered_df.groupby(filtered_df['주문일'].dt.date).agg({'결제금액(통합)': 'sum', '주문번호': 'count'}).reset_index().sort_values('결제금액(통합)', ascending=False)
    daily_stats.columns = ['Date', 'Sales', 'Orders']
    daily_stats['label'] = daily_stats.apply(lambda x: f"{x['Date']} (매출: {x['Sales']:,.0f}원, 주문: {x['Orders']}건)", axis=1)
    selected_option = st.selectbox("분석할 날짜를 선택하세요", options=daily_stats['label'], index=0)
    selected_date = pd.to_datetime(selected_option.split(' ')[0]).date()
    target_df = df_raw[df_raw['주문일'].dt.date == selected_date]
    if not target_df.empty:
        tk1, tk2 = st.columns(2)
        tk1.metric(f"{selected_date} 매출", f"{target_df['결제금액(통합)'].sum():,.0f}원")
        tk2.metric(f"{selected_date} 주문 수", f"{len(target_df):,}건")
        st.dataframe(target_df[['주문번호', '상품명', '주문수량', '결제금액(통합)', '셀러명', '주문경로']], use_container_width=True)

# Tab 5: 옵션 분석 (Original)
with tabs[4]:
    st.subheader("🧩 상품 옵션 상세 분석")
    def parse_opt(row):
        name = str(row['상품명'])
        clean = re.sub(r'\(\d+(개|EA|ea)\)', '', name, flags=re.IGNORECASE).strip()
        parts = re.split(r'[▶▷]', clean)
        return pd.Series([parts[0].strip(), parts[1].strip() if len(parts) > 1 else "단일 옵션"])
    
    with st.spinner("옵션 분석 중..."):
        opt_df = filtered_df.copy()
        opt_df[['ItemName', 'OptionName']] = opt_df.apply(parse_opt, axis=1)
        st.subheader("1. 인기 옵션 Top 20")
        opt_counts = opt_df['OptionName'].value_counts().head(20).reset_index()
        opt_counts.columns = ['OptionName', 'Count']
        fig_opt = px.bar(opt_counts, x='Count', y='OptionName', orientation='h', color='Count')
        st.plotly_chart(fig_opt, use_container_width=True)

# Tab 6: 데이터 상세 (Original)
with tabs[5]:
    st.subheader("Raw Data Preview")
    st.dataframe(filtered_df, use_container_width=True)
    st.subheader("상관관계 분석")
    numeric_df_raw = filtered_df.select_dtypes(include=['int64', 'float64'])
    if not numeric_df_raw.empty:
        fig_corr = px.imshow(numeric_df_raw.corr(), text_auto=True, title="Correlation Heatmap")
        st.plotly_chart(fig_corr, use_container_width=True)

# --- Advanced Tabs (Only if processed data available) ---
if adv_exists:
    # [Advanced] 상품/매출
    with tabs[6]:
        st.header("🚀 Advanced: 상품 및 매출 실적")
        c_adv1, c_adv2 = st.columns(2)
        with c_adv1:
            st.subheader("중량(kg)별 평균 매출")
            fig_a1 = px.bar(adv_filtered_df.groupby('weight_kg')['item_payment_amount'].mean().reset_index(), x='weight_kg', y='item_payment_amount', color='item_payment_amount')
            st.plotly_chart(fig_a1, use_container_width=True)
        with c_adv2:
            st.subheader("매출 vs 마진 산점도")
            bubble = adv_filtered_df.groupby('product_name').agg({'item_payment_amount':'sum', 'margin':'sum', 'quantity':'sum'}).reset_index()
            fig_a2 = px.scatter(bubble, x='item_payment_amount', y='margin', size='quantity', hover_name='product_name', color='margin')
            st.plotly_chart(fig_a2, use_container_width=True)
        
        st.subheader("kg당 가격 분포 및 요일별 중량")
        c_adv3, c_adv4 = st.columns(2)
        with c_adv3:
            st.plotly_chart(px.histogram(adv_filtered_df[adv_filtered_df['price_per_kg']>0], x='price_per_kg', nbins=50), use_container_width=True)
        with c_adv4:
            st.plotly_chart(px.box(adv_filtered_df, x='weekday', y='weight_kg', color='weekday'), use_container_width=True)

    # [Advanced] 마케팅/고객
    with tabs[7]:
        st.header("🌐 Advanced: 채널 및 고객 세그먼트")
        ch_stats = adv_filtered_df.groupby('order_channel').agg({'order_id':'nunique', 'item_payment_amount':'mean', 'weight_kg':'mean'}).reset_index()
        st.table(ch_stats)
        
        st.header("관심사 기반 시간대별 분석")
        heat = adv_filtered_df.groupby(['weekday', 'time_slot']).size().unstack().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        st.plotly_chart(px.imshow(heat, title="요일 x 시간대별 주문 히트맵"), use_container_width=True)

    # [Advanced] 군집 분석
    with tabs[8]:
        st.header("🧬 Advanced: 고객 군집 분석 (K-means)")
        cust_agg = adv_filtered_df.groupby('customer_id').agg({'item_payment_amount':'mean', 'weight_kg':'mean', 'order_id':'nunique'}).reset_index()
        X = StandardScaler().fit_transform(cust_agg[['item_payment_amount', 'weight_kg', 'order_id']])
        cust_agg['cluster'] = KMeans(n_clusters=4, random_state=42).fit_predict(X)
        st.plotly_chart(px.scatter(cust_agg, x='item_payment_amount', y='weight_kg', color=cust_agg['cluster'].astype(str), size='order_id', title="Customer Segments"), use_container_width=True)
        st.table(cust_agg.groupby('cluster')[['item_payment_amount', 'weight_kg', 'order_id']].mean())

    # [Advanced] 인사이트/제안
    with tabs[9]:
        st.header("💡 Advanced: 핵심 인사이트 및 전략")
        st.info("**인사이트 요약**: SNS 유입 고객의 구매 중량이 일반 고객 대비 높음. 2kg 소과 옵션의 마진율이 가장 우수함.")
        st.success("**액션 아이템**: 2kg 묶음 상품 강화, SNS 타겟 마케팅 시 대용량 세트 노출, 야간 타임 세일 운영 고려.")
        st.warning("**GA4 연계**: 고객 군집 ID를 User Property로 연동하여 리마케팅 정교화 필요.")
else:
    st.sidebar.warning("⚠️ 전처리 데이터를 찾을 수 없습니다. [Advanced] 탭들이 비활성화되었습니다.")
