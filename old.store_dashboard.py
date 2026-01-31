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
        st.header("🚀 Advanced: 상품 및 매출 실적 (전체 데이터)")
        c_adv1, c_adv2 = st.columns(2)
        with c_adv1:
            st.subheader("📦 중량(kg)별 평균 매출")
            # 여기서는 이상치 제어 안함 (전체 데이터)
            weight_avg = adv_filtered_df.groupby('weight_kg')['item_payment_amount'].mean().reset_index().sort_values('weight_kg')
            fig_a1 = px.bar(weight_avg, x='weight_kg', y='item_payment_amount', 
                            color='item_payment_amount', labels={'weight_kg': '중량 (kg)', 'item_payment_amount': '평균 매출(원)'},
                            text_auto=True)
            st.plotly_chart(fig_a1, use_container_width=True)
        with c_adv2:
            st.subheader("💰 매출 vs 마진 산점도")
            bubble = adv_filtered_df.groupby('product_name').agg({'item_payment_amount':'sum', 'margin':'sum', 'quantity':'sum'}).reset_index()
            fig_a2 = px.scatter(bubble, x='item_payment_amount', y='margin', size='quantity', 
                                hover_name='product_name', color='margin',
                                labels={'item_payment_amount': '총 매출(원)', 'margin': '총 마진(원)', 'quantity': '판매 수량'})
            st.plotly_chart(fig_a2, use_container_width=True)
        
        st.divider()
        st.header("⚖️ kg당 가격 분포 및 요일별 중량 (이상치 제외)")
        
        # 분포 분석용 데이터 정제 (이상치 20kg 초과 제외)
        dist_df = adv_filtered_df[(adv_filtered_df['weight_kg'] > 0) & (adv_filtered_df['weight_kg'] <= 20)].copy()
        
        # 세분화 필터
        st.subheader("🔍 세분화 분석 필터")
        seg_col1, seg_col2 = st.columns(2)
        with seg_col1:
            seg_target = st.selectbox("세분화 기준", ["전체", "category", "member_type", "region_1"], key="seg_target")
        with seg_col2:
            st.info("카테고리, 회원 유형, 지역별로 분포를 나누어 볼 수 있습니다.")

        c_adv3, c_adv4 = st.columns(2)
        with c_adv3:
            st.subheader("kg당 가격(Price per KG) 분포")
            if seg_target == "전체":
                fig_dist3 = px.histogram(dist_df, x='price_per_kg', nbins=30, labels={'price_per_kg': 'kg당 가격(원)'})
            else:
                fig_dist3 = px.histogram(dist_df, x='price_per_kg', color=seg_target, barmode='overlay', nbins=30, labels={'price_per_kg': 'kg당 가격(원)'})
            st.plotly_chart(fig_dist3, use_container_width=True)
            
        with c_adv4:
            st.subheader("요일별 주문 중량 분포")
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            if seg_target == "전체":
                fig_dist4 = px.box(dist_df, x='weekday', y='weight_kg', color='weekday', 
                                   category_orders={"weekday": day_order}, labels={'weekday': '요일', 'weight_kg': '중량 (kg)'})
            else:
                fig_dist4 = px.box(dist_df, x='weekday', y='weight_kg', color=seg_target,
                                   category_orders={"weekday": day_order}, labels={'weekday': '요일', 'weight_kg': '중량 (kg)'})
            st.plotly_chart(fig_dist4, use_container_width=True)

    # [Advanced] 마케팅/고객
    with tabs[7]:
        st.header("🌐 Advanced: 채널 및 고객 세그먼트")
        ch_stats = adv_filtered_df.groupby('order_channel').agg({'order_id':'nunique', 'item_payment_amount':'mean', 'weight_kg':'mean'}).reset_index()
        st.table(ch_stats)
        
        st.header("관심사 기반 시간대별 분석")
        heat = adv_filtered_df.groupby(['weekday', 'time_slot']).size().unstack().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        st.plotly_chart(px.imshow(heat, title="요일 x 시간대별 주문 히트맵"), use_container_width=True)

    # 5. 통합 군집 분석 (Integrated Clustering Analysis)
    with tabs[8]:
        st.header("🧬 Advanced: 통합 군집 분석 (Integrated Perspective)")
        st.markdown("`clustering.md` 지침에 따라 **고객 가치**를 핵심 축으로 상호 연결된 분석을 수행합니다.")
        
        # --- [Step 0] 공통 유틸리티 및 PCA ---
        from sklearn.decomposition import PCA
        
        def run_clustering_integrated(data, features, n_clusters=4, title="Cluster Plot", show_2d=False):
            if data.empty: return None
            df_cl = data.dropna(subset=features).copy()
            if len(df_cl) < n_clusters: return None
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_cl[features])
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            df_cl['cluster'] = kmeans.fit_predict(X_scaled)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                # 2D vs 3D 토글 처리
                if show_2d and len(features) > 2:
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_scaled)
                    df_cl['pca1'] = X_pca[:, 0]
                    df_cl['pca2'] = X_pca[:, 1]
                    fig = px.scatter(df_cl, x='pca1', y='pca2', color='cluster', 
                                     title=f"{title} (PCA 2D Projection)", opacity=0.7)
                elif len(features) >= 3:
                    fig = px.scatter_3d(df_cl, x=features[0], y=features[1], z=features[2], 
                                        color='cluster', title=title, opacity=0.7)
                else:
                    fig = px.scatter(df_cl, x=features[0], y=features[1], 
                                     color='cluster', title=title)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("**군집별 평균 지표**")
                summary = df_cl.groupby('cluster')[features].mean()
                st.dataframe(summary.style.format("{:.2f}").background_gradient(cmap='YlGnBu'))
            return df_cl

        # --- [Step 1] Core Axis: Customer Value Segmentation ---
        st.subheader("1️⃣ 핵심 고객 세그먼트 (Core Axis)")
        cust_core = adv_filtered_df.groupby('customer_id').agg({
            'total_payment_amount': 'sum',
            'item_payment_amount': 'mean',
            'order_id': 'nunique',
            'weight_kg': 'mean'
        }).reset_index()
        cust_core.columns = ['customer_id', 'total_payment', 'avg_payment', 'frequency', 'avg_weight']
        
        # 시각화 모드 선택
        viz_mode = st.radio("시각화 모드", ["3D Perspective", "2D PCA Projection"], horizontal=True, key="core_viz")
        
        df_cust_clustered = run_clustering_integrated(
            cust_core, ['total_payment', 'avg_payment', 'frequency', 'avg_weight'], 
            n_clusters=4, title="Customer Value Clusters", show_2d=(viz_mode=="2D PCA Projection")
        )
        
        if df_cust_clustered is not None:
            # --- [Step 2] Multi-Clustering Extensions ---
            st.divider()
            st.subheader("2️⃣ 멀티 관점 확장 및 교차 분석")
            
            ext_tab1, ext_tab2, ext_tab3 = st.tabs(["🎫 할인 민감도", "🌐 획득 채널", "🕒 주문 패턴"])
            
            with ext_tab1:
                st.markdown("**고객 가치 x 할인 반응도 교차 분석**")
                # 할인 데이터 매핑
                orders_sens = adv_filtered_df.groupby('order_id').first().reset_index()
                orders_sens['coupon_ratio'] = orders_sens['coupon_used'] / orders_sens['total_payment_amount'].replace(0, 1)
                orders_sens['point_ratio'] = orders_sens['point_used'] / orders_sens['total_payment_amount'].replace(0, 1)
                cust_sens = orders_sens.groupby('customer_id').agg({'coupon_ratio':'mean', 'point_ratio':'mean'}).reset_index()
                
                # 가치 군집 정보 병합
                merged_sens = pd.merge(cust_sens, df_cust_clustered[['customer_id', 'cluster']], on='customer_id')
                fig_sens = px.box(merged_sens, x='cluster', y='coupon_ratio', color='cluster', points="all", title="고객 군집별 쿠폰 사용 비중")
                st.plotly_chart(fig_sens, use_container_width=True)
                st.info("💡 **인사이트**: 고가치 고객(VIP) 군집의 할인 민감도가 낮다면 프리미엄 서비스 집중, 높다면 VIP 전용 쿠폰 전략 유효")

            with ext_tab2:
                st.markdown("**고객 가치 x 획득 채널 교차 분석**")
                cust_chan = adv_filtered_df.groupby(['customer_id', 'order_channel']).size().reset_index(name='count')
                merged_chan = pd.merge(cust_chan, df_cust_clustered[['customer_id', 'cluster']], on='customer_id')
                chan_dist = merged_chan.groupby(['cluster', 'order_channel']).size().unstack(fill_value=0)
                chan_dist_norm = chan_dist.div(chan_dist.sum(axis=1), axis=0) * 100
                st.plotly_chart(px.bar(chan_dist_norm.reset_index().melt(id_vars='cluster'), x='cluster', y='value', color='order_channel', 
                                       title="군집별 유입 채널 비중 (%)", barmode='stack'), use_container_width=True)

            with ext_tab3:
                st.markdown("**고객 가치 x 시간대별 구매 패턴**")
                merged_time = pd.merge(adv_filtered_df, df_cust_clustered[['customer_id', 'cluster']], on='customer_id')
                time_heat = merged_time.groupby(['cluster', 'time_slot']).size().unstack(fill_value=0)
                st.plotly_chart(px.imshow(time_heat, title="고객 군집 x 유입 시간대 Heatmap", text_auto=True), use_container_width=True)

            # --- [Step 3] Product Profitability & Matrix ---
            st.divider()
            st.subheader("3️⃣ 상품 수익성 및 고객-상품 매트릭스")
            
            prod_agg = adv_filtered_df.groupby('product_name').agg({
                'margin': 'mean',
                'margin_rate': 'mean',
                'quantity': 'sum'
            }).reset_index()
            prod_agg.columns = ['product_name', 'avg_margin', 'margin_rate', 'sales_volume']
            df_prod_clustered = run_clustering_integrated(prod_agg, ['avg_margin', 'margin_rate', 'sales_volume'], n_clusters=4, title="Product Profitability Clusters")
            
            if df_prod_clustered is not None:
                st.markdown("**고객 군집 × 상품 군집 매트릭스**")
                # 고객-상품 구매 관계 생성
                cust_prod_rel = pd.merge(adv_filtered_df[['customer_id', 'product_name']], df_cust_clustered[['customer_id', 'cluster']], on='customer_id')
                cust_prod_rel = pd.merge(cust_prod_rel, df_prod_clustered[['product_name', 'cluster']], on='product_name', suffixes=('_cust', '_prod'))
                
                matrix = cust_prod_rel.groupby(['cluster_cust', 'cluster_prod']).size().unstack(fill_value=0)
                st.plotly_chart(px.imshow(matrix, labels=dict(x="상품 군집", y="고객 군집"), title="Customer x Product Matrix (구매 건수)", text_auto=True), use_container_width=True)

            # --- [Final] Strategic Recommendations ---
            st.divider()
            st.subheader("🚀 4️⃣ 전략적 실행 제언 (Strategic Recommendations)")
            
            rec_data = [
                {"세그먼트 조합": "VIP 고객 x 고마진 상품", "실행 전략": "프리미엄 멤버십 전용 큐레이션 및 선결제 혜택", "기대 효과": "LTV 극대화 및 고마진 상품 매출 비중 확대"},
                {"세그먼트 조합": "신규/일반 고객 x 베스트셀러", "실행 전략": "첫 구매 감사 쿠폰 및 연관 상품 추천(Cross-sell)", "기대 효과": "재구매율(Retention) 향상 및 충성 고객 전환"},
                {"세그먼트 조합": "할인 민감군 x 이벤트 상품", "실행 전략": "타임 세일 및 한정 수량 프로모션 타겟팅", "기대 효과": "재고 순환 속도 가속화 및 집객력 강화"},
                {"세그먼트 조합": "야간 구매군 x 모바일 채널", "실행 전략": "야간 전용 앱 푸시 및 모바일 전용 할인권", "기대 효과": "특정 시간대 점유율 확보 및 앱 활성 지표 개선"}
            ]
            st.table(pd.DataFrame(rec_data))

    # [Advanced] 인사이트/제안
    with tabs[9]:
        st.header("💡 Advanced: 핵심 인사이트 및 전략")
        st.info("**인사이트 요약**: SNS 유입 고객의 구매 중량이 일반 고객 대비 높음. 2kg 소과 옵션의 마진율이 가장 우수함.")
        st.success("**액션 아이템**: 2kg 묶음 상품 강화, SNS 타겟 마케팅 시 대용량 세트 노출, 야간 타임 세일 운영 고려.")
        st.warning("**GA4 연계**: 고객 군집 ID를 User Property로 연동하여 리마케팅 정교화 필요.")
else:
    st.sidebar.warning("⚠️ 전처리 데이터를 찾을 수 없습니다. [Advanced] 탭들이 비활성화되었습니다.")
