import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# --- 페이지 설정 ---
st.set_page_config(
    page_title="Store Data EDA Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- 데이터 로드 ---
@st.cache_data
def load_data(file_path):
    df = pd.read_excel(file_path)
    # 날짜 변환
    if '주문일' in df.columns:
        df['주문일'] = pd.to_datetime(df['주문일'], errors='coerce')
    return df

# 현재 스크립트의 디렉토리를 기준으로 상대 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(current_dir, 'data', 'store_data.xlsx')

if not os.path.exists(DATA_PATH):
    st.error(f"데이터 파일이 없습니다: {DATA_PATH}")
    st.stop()

df = load_data(DATA_PATH)

# --- 전처리 ---
df['year_month'] = df['주문일'].dt.to_period('M').astype(str)
df['day_name'] = df['주문일'].dt.day_name()
# 결제금액(상품별)이 실 매출로 추정됨 (취소 제외 필요 여부 확인, 우선 전체 매출로 봄)
# 주문취소 금액이 있으므로, 순매출 = 결제금액(상품별) - 주문취소 금액(상품별) ?
# 일반적인 e-commerce 데이터를 가정하여 '결제금액(상품별)'을 기준으로 분석하되, 취소 내역도 별도 분석.

# --- 메인 UI ---
st.title("📊 Store Data Exploratory Data Analysis")
st.markdown(f"**데이터 소스**: `{DATA_PATH}` | **총 레코드**: {len(df):,}건")

# 사이드바
st.sidebar.header("설정")
start_date = df['주문일'].min()
end_date = df['주문일'].max()

date_range = st.sidebar.date_input(
    "조회 기간",
    [start_date, end_date],
    min_value=start_date,
    max_value=end_date
)

if len(date_range) == 2:
    start_dt, end_dt = date_range
    mask = (df['주문일'].dt.date >= start_dt) & (df['주문일'].dt.date <= end_dt)
    filtered_df = df.loc[mask]
else:
    filtered_df = df

# KPI
total_sales = filtered_df['결제금액(통합)'].sum() # 통합 금액 사용
total_orders = len(filtered_df)
avg_order_value = total_sales / total_orders if total_orders > 0 else 0
cancel_sales = filtered_df['주문취소 금액(상품별)'].sum()
net_sales = total_sales - cancel_sales

k1, k2, k3, k4 = st.columns(4)
k1.metric("총 결제금액 (Gross Sales)", f"{total_sales:,.0f}원")
k2.metric("총 주문 건수", f"{total_orders:,}건")
k3.metric("평균 주문금액 (AOV)", f"{avg_order_value:,.0f}원")
k4.metric("취소 금액", f"{cancel_sales:,.0f}원", delta=f"-{(cancel_sales/total_sales)*100:.1f}%" if total_sales else 0)

st.divider()

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 매출 트렌드", "🛒 상품/카테고리 분석", "👥 고객/채널 분석", "📊 데이터 상세", "📅 특정 날짜 분석"])

# Tab 1: 매출 트렌드
with tab1:
    st.subheader("기간별 매출 추이")
    trend_type = st.radio("집계 기준", ["일별", "월별"], horizontal=True)
    
    if trend_type == "일별":
        trend = filtered_df.groupby(filtered_df['주문일'].dt.date)['결제금액(통합)'].sum().reset_index()
        trend.columns = ['Date', 'Sales']
        fig_trend = px.line(trend, x='Date', y='Sales', title="일별 매출 추이", markers=True)
    else:
        trend = filtered_df.groupby('year_month')['결제금액(통합)'].sum().reset_index()
        trend.columns = ['Month', 'Sales']
        fig_trend = px.bar(trend, x='Month', y='Sales', title="월별 매출 추이", text_auto='.2s')
        
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # 요일별 분석
    st.subheader("요일별 주문 패턴")
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    filtered_df['day_name'] = pd.Categorical(filtered_df['day_name'], categories=day_order, ordered=True)
    day_trend = filtered_df.groupby('day_name')['결제금액(통합)'].sum().reset_index()
    
    c1, c2 = st.columns(2)
    with c1:
        fig_day = px.bar(day_trend, x='day_name', y='결제금액(통합)', title="요일별 매출액", color='결제금액(통합)')
        st.plotly_chart(fig_day, use_container_width=True)
    with c2:
        # 시간대별 분석 (시간 정보가 있다면)
        # 엑셀 데이터상 시간 정보가 있는지 확인 필요 (datetime이면 있음)
        # datetime으로 변환했으므로 시각 추출
        filtered_df['hour'] = filtered_df['주문일'].dt.hour
        hour_trend = filtered_df.groupby('hour')['주문번호'].count().reset_index(name='count')
        fig_hour = px.line(hour_trend, x='hour', y='count', title="시간대별 주문 건수", markers=True)
        st.plotly_chart(fig_hour, use_container_width=True)

# Tab 2: 상품 분석
with tab2:
    st.subheader("Top Performing Products")
    
    # 상품별 매출 Top 10
    top_products = filtered_df.groupby('상품명')['결제금액(통합)'].sum().reset_index().sort_values('결제금액(통합)', ascending=False).head(10)
    
    col_p1, col_p2 = st.columns([2, 1])
    with col_p1:
        fig_prod = px.bar(top_products, x='결제금액(통합)', y='상품명', orientation='h', title="매출 상위 10개 상품", text_auto='.2s')
        fig_prod.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_prod, use_container_width=True)
    with col_p2:
        st.dataframe(top_products, use_container_width=True)
        
    # 상품별 판매량 Top 10
    top_qty = filtered_df.groupby('상품명')['주문수량'].sum().reset_index().sort_values('주문수량', ascending=False).head(10)
    st.subheader("판매량 상위 상품")
    st.dataframe(top_qty.T, use_container_width=True)

# Tab 3: 고객/채널 분석
with tab3:
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
    st.subheader("주문 경로 및 셀러 분석")
    
    c_route1, c_route2 = st.columns(2)
    with c_route1:
        # 주문경로 시각화
        route_df = filtered_df['주문경로'].value_counts().reset_index()
        route_df.columns = ['Route', 'Count']
        fig_route = px.bar(route_df, x='Route', y='Count', title="주문 경로별 건수")
        st.plotly_chart(fig_route, use_container_width=True)
        
    with c_route2:
        # 셀러별 매출
        if '셀러명' in filtered_df.columns:
            seller_df = filtered_df.groupby('셀러명')['결제금액(통합)'].sum().reset_index().sort_values('결제금액(통합)', ascending=False).head(10)
            fig_seller = px.bar(seller_df, x='셀러명', y='결제금액(통합)', title="Top 10 셀러 (매출 기준)")
            st.plotly_chart(fig_seller, use_container_width=True)

# Tab 4: 데이터 상세
with tab4:
    st.subheader("Raw Data Preview")
    st.dataframe(filtered_df, use_container_width=True)
    
    st.subheader("상관관계 분석 (수치형 변수)")
    numeric_df = filtered_df.select_dtypes(include=['int64', 'float64'])
    corr = numeric_df.corr()
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Correlation Heatmap")
    st.plotly_chart(fig_corr, use_container_width=True)

# Tab 5: 특정 날짜 상세 분석 (Peak Day Analysis)
with tab5:
    st.subheader("📅 특정 날짜 상세 분석 (Peak Day Deep Dive)")
    st.markdown("매출이 유독 높거나 관심 있는 **특정 날짜**를 선택하여, 해당 일자의 **효자 상품**과 **주요 셀러**를 분석합니다.")
    
    # 날짜별 매출 데이터 생성
    daily_stats = filtered_df.groupby(filtered_df['주문일'].dt.date).agg({
        '결제금액(통합)': 'sum', 
        '주문번호': 'count'
    }).reset_index().sort_values('결제금액(통합)', ascending=False)
    
    daily_stats.columns = ['Date', 'Sales', 'Orders']
    
    # 선택 옵션 생성 (예: "2025-10-17 (매출: 12,000,000원, 주문: 150건)")
    daily_stats['label'] = daily_stats.apply(
        lambda x: f"{x['Date']} (매출: {x['Sales']:,.0f}원, 주문: {x['Orders']}건)", axis=1
    )
    
    # 날짜 선택 (Selectbox 사용 - 검색 가능)
    # 기본값: 매출 1위 날짜
    selected_option = st.selectbox(
        "분석할 날짜를 선택하세요 (날짜 또는 매출로 검색 가능)", 
        options=daily_stats['label'],
        index=0
    )
    
    # 선택된 라벨에서 날짜 추출
    selected_date_str = selected_option.split(' ')[0]
    selected_date = pd.to_datetime(selected_date_str).date()
    
    # 해당 날짜 데이터 필터링
    target_df = df[df['주문일'].dt.date == selected_date]
    
    if not target_df.empty:
        # Day KPI
        day_sales = target_df['결제금액(통합)'].sum()
        day_orders = len(target_df)
        
        c_kpi1, c_kpi2 = st.columns(2)
        c_kpi1.metric(f"{selected_date} 매출", f"{day_sales:,.0f}원")
        c_kpi2.metric(f"{selected_date} 주문 수", f"{day_orders:,}건")
        
        st.divider()
        
        # 시각화: 상품 & 셀러
        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            st.subheader("🏆 당일 판매량 Top 10 상품")
            # 금액 기준 vs 수량 기준 (여기선 금액 기준)
            day_top_prod = target_df.groupby('상품명')['결제금액(통합)'].sum().reset_index().sort_values('결제금액(통합)', ascending=False).head(10)
            
            fig_day_prod = px.bar(day_top_prod, x='결제금액(통합)', y='상품명', orientation='h', 
                                  title=f"{selected_date} 상품별 매출", text_auto='.2s', color='결제금액(통합)')
            fig_day_prod.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_day_prod, use_container_width=True)
            
        with col_d2:
            st.subheader("🥇 당일 매출 Top 10 셀러")
            if '셀러명' in target_df.columns:
                day_top_seller = target_df.groupby('셀러명')['결제금액(통합)'].sum().reset_index().sort_values('결제금액(통합)', ascending=False).head(10)
                
                fig_day_seller = px.pie(day_top_seller, values='결제금액(통합)', names='셀러명', 
                                        title=f"{selected_date} 셀러 매출 비중", hole=0.3)
                st.plotly_chart(fig_day_seller, use_container_width=True)
            else:
                st.info("셀러명 정보가 없습니다.")
        
        st.subheader("📋 당일 주문 상세 내역")
        st.dataframe(target_df[['주문번호', '상품명', '주문수량', '결제금액(통합)', '셀러명', '주문경로']].sort_values('결제금액(통합)', ascending=False), 
                     use_container_width=True)
        
    else:
        st.warning("선택한 날짜에 주문 데이터가 없습니다.")
