import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- 페이지 설정 ---
st.set_page_config(
    page_title="Refined E-commerce Unified Dashboard",
    page_icon="🍊",
    layout="wide"
)

# --- 1. 데이터 로드 ---
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    # 날짜 변환
    if '주문일' in df.columns:
        df['주문일'] = pd.to_datetime(df['주문일'])
        df['date'] = df['주문일'].dt.date
        df['month'] = df['주문일'].dt.to_period('M').astype(str)
        df['day_name'] = df['주문일'].dt.day_name()
    return df

current_dir = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(current_dir, 'data', 'preprocessed_data.csv')

if not os.path.exists(FILE_PATH):
    st.error(f"전처리 데이터 파일이 없습니다: {FILE_PATH}")
    st.stop()

df = load_data(FILE_PATH)

# --- 2. 사이드바 및 필터 ---
st.sidebar.title("🔍 분석 필터")
st.sidebar.markdown("---")

# 날짜 필터
min_date = df['date'].min()
max_date = df['date'].max()
date_range = st.sidebar.date_input("조회 기간", [min_date, max_date], min_value=min_date, max_value=max_date)

# 카테고리(품종) 필터
variety_list = sorted(df['품종'].unique().tolist())
selected_variety = st.sidebar.multiselect("품종 선택", variety_list, default=variety_list)

# 데이터 필터링 적용
if len(date_range) == 2:
    start_dt, end_dt = date_range
    mask = (df['date'] >= start_dt) & (df['date'] <= end_dt) & (df['품종'].isin(selected_variety))
    filtered_df = df.loc[mask]
else:
    filtered_df = df[df['품종'].isin(selected_variety)]

# --- 3. 헤더 및 KPI ---
st.title("🍊 Unified E-commerce Analytics Dashboard")
st.markdown(f"파이프라인을 통해 정제된 데이터를 기반으로 한 **고도화 분석 대시보드**입니다.")

k1, k2, k3, k4 = st.columns(4)
total_sales = filtered_df['실결제 금액'].sum()
total_orders = filtered_df['주문번호'].nunique()
avg_order_value = total_sales / total_orders if total_orders > 0 else 0
cancel_rate = (filtered_df['취소여부'] == 'Y').mean() * 100

with k1:
    st.metric("총 실결제 금액", f"{total_sales:,.0f}원")
with k2:
    st.metric("총 주문 건수", f"{total_orders:,}건")
with k3:
    st.metric("평균 주문 단가(AOV)", f"{avg_order_value:,.0f}원")
with k4:
    st.metric("주문 취소율", f"{cancel_rate:.1f}%", delta_color="inverse")

st.divider()

# --- 4. Tabs 구성 ---
tabs = st.tabs(["📈 매출 및 성과", "📦 품종 및 상품 분석", "⚖️ 무게/가격 분포", "🧬 고객 군집 분석(Clustering)", "� 셀러별 심층 분석", "�🌐 지역별 분석", "📋 데이터 탐색기"])

# Tab 1: 매출 및 성과
with tabs[0]:
    st.subheader("매출 트렌드 분석")
    t1, t2 = st.columns([2, 1])
    
    with t1:
        trend_agg = filtered_df.groupby('date')['실결제 금액'].sum().reset_index()
        fig_trend = px.line(trend_agg, x='date', y='실결제 금액', title="일별 매출 추이", 
                            line_shape="spline", render_mode="svg")
        fig_trend.update_traces(line_color='#FF8C00', fill='tozeroy')
        st.plotly_chart(fig_trend, use_container_width=True)
        
    with t2:
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_agg = filtered_df.groupby('day_name')['실결제 금액'].sum().reindex(day_order).reset_index()
        fig_day = px.bar(day_agg, x='day_name', y='실결제 금액', color='실결제 금액',
                         title="요일별 매출 비중", color_continuous_scale='Oranges')
        st.plotly_chart(fig_day, use_container_width=True)

    st.markdown("---")
    st.subheader("주문 경로 및 방법")
    c1, c2 = st.columns(2)
    with c1:
        path_agg = filtered_df['주문경로'].value_counts()
        st.plotly_chart(px.pie(values=path_agg.values, names=path_agg.index, hole=0.5, title="주문 경로 점유율"), use_container_width=True)
    with c2:
        method_agg = filtered_df['결제방법'].value_counts()
        st.plotly_chart(px.pie(values=method_agg.values, names=method_agg.index, title="결제 방법 점유율"), use_container_width=True)

# Tab 2: 품종 및 상품 분석
with tabs[1]:
    st.subheader("품종 및 상품 포트폴리오")
    p1, p2 = st.columns(2)
    
    with p1:
        variety_sales = filtered_df.groupby('품종')['실결제 금액'].sum().sort_values(ascending=False).reset_index()
        fig_var = px.bar(variety_sales, x='실결제 금액', y='품종', orientation='h', title="품종별 매출 순위",
                         color='실결제 금액', color_continuous_scale='Viridis')
        st.plotly_chart(fig_var, use_container_width=True)
        
    with p2:
        size_agg = filtered_df['과수 크기'].value_counts()
        st.plotly_chart(px.pie(values=size_agg.values, names=size_agg.index, title="과수 크기별 선호도"), use_container_width=True)

    st.subheader("Top 10 상품 리스트 (실결제 기준)")
    top_items = filtered_df.groupby('상품명')['실결제 금액'].sum().sort_values(ascending=False).head(10).reset_index()
    st.table(top_items)

# Tab 3: 무게/가격 분포
with tabs[2]:
    st.subheader("중량 및 가격대 분포 분석")
    w1, w2 = st.columns(2)
    
    with w1:
        fig_w = px.histogram(filtered_df, x='무게(kg)', nbins=20, title="주문 중량(kg) 분포",
                             color_discrete_sequence=['#4B0082'])
        st.plotly_chart(fig_w, use_container_width=True)
        
    with w2:
        price_order = ["1만원 이하", "1~3만원", "3~5만원", "5~10만원", "10만원 초반"]
        price_agg = filtered_df['가격대'].value_counts().reindex(price_order).reset_index()
        fig_p = px.bar(price_agg, x='가격대', y='count', title="가격대별 주문 건수", 
                       color='가격대', color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_p, use_container_width=True)

    st.markdown("---")
    st.subheader("목적별 주문 특성 (개인소비 vs 선물)")
    purpose_agg = filtered_df.groupby('목적').agg({'실결제 금액':'mean', '무게(kg)':'mean', '주문번호':'count'}).reset_index()
    purpose_agg.columns = ['목적', '평균 결제액', '평균 중량(kg)', '주문 건수']
    st.dataframe(purpose_agg.style.format({'평균 결제액': '{:,.0f}원', '평균 중량(kg)': '{:.2f}kg'}), use_container_width=True)

# Tab 4: 고객 군집 분석
with tabs[3]:
    st.header("🧬 고객 가치 세그먼테이션 (Clustering)")
    st.markdown("고객별 총 결제금액, 재구매 횟수, 평균 구매 중량을 기반으로 고객을 분류합니다.")
    
    # 군집 분석용 데이터 준비
    cust_data = filtered_df.groupby('UID').agg({
        '실결제 금액': 'sum',
        '재구매 횟수': 'max',
        '무게(kg)': 'mean'
    }).reset_index()
    cust_data.columns = ['UID', 'total_spent', 'max_repurchase', 'avg_weight']
    
    if len(cust_data) >= 4:
        # 스케일링
        features = ['total_spent', 'max_repurchase', 'avg_weight']
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(cust_data[features])
        
        # K-Means
        n_clusters = st.slider("군집 수(K) 선택", 2, 6, 4)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cust_data['cluster'] = kmeans.fit_predict(scaled_features)
        
        g1, g2 = st.columns([2, 1])
        with g1:
            fig_cluster = px.scatter_3d(cust_data, x='total_spent', y='max_repurchase', z='avg_weight',
                                        color='cluster', title="3D 고객 세그먼트 시각화",
                                        labels={'total_spent':'총 지출', 'max_repurchase':'재구매 횟수', 'avg_weight':'평균 중량'},
                                        opacity=0.7)
            st.plotly_chart(fig_cluster, use_container_width=True)
            
        with g2:
            cluster_summary = cust_data.groupby('cluster')[features].mean().reset_index()
            st.write("**군집별 평균 지표**")
            st.dataframe(cluster_summary.style.background_gradient(cmap='Blues'))
            
        st.info("💡 **군집 해석 팁**: 지출과 재구매가 모두 높은 군집은 '충성 고객', 재구매는 낮지만 지출이 높은 군집은 '대량 구매 신규 고객'으로 해석할 수 있습니다.")
    else:
        st.warning("군집 분석을 위한 데이터가 충분하지 않습니다.")

# Tab 5: 셀러별 심층 분석
with tabs[4]:
    st.header("🏪 셀러별 심층 성과 분석")
    
    # 셀러 선택 필터 (탭 내부용)
    seller_list = sorted(filtered_df['셀러명'].unique().tolist())
    selected_seller = st.selectbox("분석할 셀러를 선택하세요", seller_list, index=0)
    
    s_df = filtered_df[filtered_df['셀러명'] == selected_seller]
    
    sk1, sk2, sk3, sk4 = st.columns(4)
    s_total_sales = s_df['실결제 금액'].sum()
    s_total_orders = s_df['주문번호'].nunique()
    s_avg_payment = s_total_sales / s_total_orders if s_total_orders > 0 else 0
    s_repurchase_rate = (s_df['재구매 횟수'] > 0).mean() * 100
    
    sk1.metric("셀러 총 매출", f"{s_total_sales:,.0f}원")
    sk2.metric("총 주문 건수", f"{s_total_orders:,}건")
    sk3.metric("평균 주문 단가", f"{s_avg_payment:,.0f}원")
    sk4.metric("고객 재구매율", f"{s_repurchase_rate:.1f}%")
    
    st.divider()
    
    # 1. 시계열 분석 및 주문 경로
    sc1, sc2 = st.columns(2)
    with sc1:
        s_trend = s_df.groupby('date')['실결제 금액'].sum().reset_index()
        fig_s_trend = px.line(s_trend, x='date', y='실결제 금액', title=f"[{selected_seller}] 매출 트렌드")
        fig_s_trend.update_traces(line_color='#FF4B4B')
        st.plotly_chart(fig_s_trend, use_container_width=True)
    with sc2:
        s_path = s_df['주문경로'].value_counts().reset_index()
        fig_s_path = px.pie(s_path, values='count', names='주문경로', hole=0.4, title=f"[{selected_seller}] 주문경로 점유율")
        st.plotly_chart(fig_s_path, use_container_width=True)
        
    # 2. 지역 및 품종 분포
    sc3, sc4 = st.columns(2)
    with sc3:
        s_region = s_df['광역지역(정식)'].value_counts().reset_index().head(10)
        fig_s_region = px.bar(s_region, x='count', y='광역지역(정식)', orientation='h', 
                             title=f"[{selected_seller}] 주요 판매 지역 (Top 10)", color='count', color_continuous_scale='Reds')
        st.plotly_chart(fig_s_region, use_container_width=True)
    with sc4:
        s_variety = s_df['품종'].value_counts().reset_index()
        fig_s_variety = px.bar(s_variety, x='품종', y='count', title=f"[{selected_seller}] 취급 품종 분포", color='품종')
        st.plotly_chart(fig_s_variety, use_container_width=True)

    st.divider()
    st.subheader(f"💡 {selected_seller} 셀러 전략 제언")
    
    # 간단한 로직 기반 제언
    top_path = s_path.iloc[0]['주문경로'] if not s_path.empty else "N/A"
    if s_repurchase_rate > 30:
        st.success(f"✅ **충성도 높음**: 재구매율이 {s_repurchase_rate:.1f}%로 매우 높습니다. 단골 고객 대상 감사 이벤트를 추천합니다.")
    else:
        st.info(f"ℹ️ **신규 유입 중심**: 현재 {top_path}를 통한 유입이 가장 많습니다. 초기 구매 고객을 단골로 전환하기 위한 첫 구매 혜택 강화가 필요합니다.")

# Tab 6: 지역별 분석
with tabs[5]:
    st.subheader("🌐 광역지자체별 성과")
    region_agg = filtered_df.groupby('광역지역(정식)').agg({'실결제 금액':'sum', '주문번호':'count'}).reset_index().sort_values('실결제 금액', ascending=False)
    
    r1, r2 = st.columns([2, 1])
    with r1:
        fig_region = px.bar(region_agg, x='광역지역(정식)', y='실결제 금액', color='실결제 금액',
                            title="지역별 총 매출액", color_continuous_scale='Tealgrn')
        st.plotly_chart(fig_region, use_container_width=True)
    with r2:
        st.write("**지역별 매출 상세**")
        st.dataframe(region_agg, use_container_width=True)

# Tab 7: 데이터 탐색기
with tabs[6]:
    st.subheader("상세 데이터 테이블")
    st.dataframe(filtered_df, use_container_width=True)
    
    st.subheader("수치형 칼럼 상관관계")
    numeric_cols = filtered_df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 1:
        corr = filtered_df[numeric_cols].corr()
        st.plotly_chart(px.imshow(corr, text_auto=True, title="Correlation Heatmap", color_continuous_scale='RdBu_r'))
