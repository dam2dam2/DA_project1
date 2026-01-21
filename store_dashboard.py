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

    # 5. 클러스터링 (Multi-Clustering Expansion)
    with tabs[8]:
        st.header("🧬 Advanced: 다차원 군집 분석 (Multi-Clustering)")
        st.markdown("`clustering.md` 지침에 따라 10가지 상이한 관점의 군집 분석을 제공합니다.")
        
        cluster_option = st.selectbox(
            "📍 분석 유형 선택",
            [
                "3.1 고객 가치 군집 (Customer Value)",
                "3.2 고객 행동 군집 (Customer Behavior)",
                "3.3 가격 민감도 군집 (Price Sensitivity)",
                "3.4 상품 수익성 군집 (Product Profitability)",
                "3.5 중량/옵션 기반 상품 군집 (Weight & Option)",
                "3.6 프로모션 효율 군집 (Promotion Effectiveness)",
                "3.7 주문 패턴 군집 (Order Pattern)",
                "3.8 채널 기반 주문 군집 (Channel-Based)",
                "3.9 지역 소비 군집 (Regional Consumption)",
                "3.10 리드타임 군집 (Delivery Lead-Time)"
            ]
        )
        
        st.divider()

        # 공통 클러스터링 함수
        def run_clustering(data, features, n_clusters=4, title="Cluster Plot"):
            if data.empty:
                st.warning("데이터가 부족하여 분석을 수행할 수 없습니다.")
                return None
            
            # 결측치 처리
            df_cl = data.dropna(subset=features)
            if len(df_cl) < n_clusters:
                st.warning("데이터 레코드가 군집 수보다 적습니다.")
                return None

            # Scaling
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_cl[features])
            
            # K-means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            df_cl['cluster'] = kmeans.fit_predict(X_scaled)
            
            # 시각화
            col1, col2 = st.columns([2, 1])
            with col1:
                if len(features) >= 3:
                    fig = px.scatter_3d(df_cl, x=features[0], y=features[1], z=features[2], 
                                        color=df_cl['cluster'].astype(str), title=title, opacity=0.7)
                else:
                    fig = px.scatter(df_cl, x=features[0], y=features[1], 
                                     color=df_cl['cluster'].astype(str), title=title)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("**군집별 평균 지표**")
                summary = df_cl.groupby('cluster')[features].mean()
                st.dataframe(summary.style.format("{:.2f}").background_gradient(cmap='YlGnBu'))
                
            return df_cl

        # 3.1 고객 가치 군집
        if cluster_option.startswith("3.1"):
            st.subheader("👥 고객 가치 군집 분석 (Customer Value)")
            cust_val = adv_filtered_df.groupby('customer_id').agg({
                'item_payment_amount': ['mean', 'sum'],
                'weight_kg': 'mean'
            }).reset_index()
            cust_val.columns = ['customer_id', 'avg_payment', 'total_payment', 'avg_weight']
            run_clustering(cust_val, ['avg_payment', 'total_payment', 'avg_weight'], title="Value Clusters (Avg Pay x Total Pay x Avg Weight)")
            st.info("**Persona**: VIP(고단가/대량), 일반(평균), 이탈위험(저단가/소량)")

        # 3.2 고객 행동 군집
        elif cluster_option.startswith("3.2"):
            st.subheader("🕒 고객 행동 군집 분석 (Customer Behavior)")
            cust_beh = adv_filtered_df.groupby('customer_id').agg({
                'order_id': 'nunique',
                'order_hour': 'mean',
                'order_channel': 'nunique'
            }).reset_index()
            cust_beh.columns = ['customer_id', 'frequency', 'avg_hour', 'channel_count']
            run_clustering(cust_beh, ['frequency', 'avg_hour', 'channel_count'], title="Behavior Clusters (Freq x Hour x Channel)")
            st.info("**Insight**: 특정 시간대(예: 야간) 집중 구매군과 다채널 이용 충성 고객군 식별")

        # 3.3 가격 민감도 군집
        elif cluster_option.startswith("3.3"):
            st.subheader("🎫 가격 민감도 군집 분석 (Price Sensitivity)")
            # 주문단위 쿠폰/포인트 비율 계산
            orders_subset = adv_filtered_df.groupby('order_id').first().reset_index()
            orders_subset['coupon_ratio'] = orders_subset['coupon_used'] / orders_subset['total_payment_amount'].replace(0, 1)
            orders_subset['point_ratio'] = orders_subset['point_used'] / orders_subset['total_payment_amount'].replace(0, 1)
            
            cust_sens = orders_subset.groupby('customer_id').agg({
                'coupon_ratio': 'mean',
                'point_ratio': 'mean',
                'total_payment_amount': 'mean'
            }).reset_index()
            cust_sens.columns = ['customer_id', 'coupon_ratio', 'point_ratio', 'avg_payment']
            run_clustering(cust_sens, ['coupon_ratio', 'point_ratio', 'avg_payment'], title="Sensitivity Clusters")
            st.success("**Strategy**: 할인 반응도가 높은 군집에는 전용 쿠폰 발송, 저민감 군집에는 프리미엄 상품 추천")

        # 3.4 상품 수익성 군집
        elif cluster_option.startswith("3.4"):
            st.subheader("💰 상품 수익성 군집 분석 (Product Profitability)")
            prod_prof = adv_filtered_df.groupby('product_name').agg({
                'unit_price': 'mean',
                'margin': 'mean',
                'margin_rate': 'mean',
                'quantity': 'sum'
            }).reset_index()
            prod_prof.columns = ['product_name', 'avg_price', 'avg_margin', 'avg_margin_rate', 'total_qty']
            run_clustering(prod_prof, ['avg_price', 'avg_margin', 'total_qty'], title="Profitability Clusters")
            st.warning("**Action**: 고마진/저판매 상품의 노출 확대, 저마진/고판매 상품의 운영 효율화")

        # 3.5 중량/옵션 기반 상품 군집
        elif cluster_option.startswith("3.5"):
            st.subheader("⚖️ 중량 및 옵션 기반 상품 군집 (Weight & Option)")
            # option_type encoding
            prod_opt = adv_filtered_df.groupby('product_name').agg({
                'weight_kg': 'mean',
                'price_per_kg': 'mean',
                'option_type': 'first'
            }).reset_index()
            prod_opt['opt_code'] = pd.factorize(prod_opt['option_type'])[0]
            run_clustering(prod_opt, ['weight_kg', 'price_per_kg', 'opt_code'], title="Option-Weight Clusters")

        # 3.6 프로모션 효율 군집 (임시: is_promotion 여부에 따른 상품별 매출 변화)
        elif cluster_option.startswith("3.6"):
            st.subheader("📢 프로모션 효율 군집 분석 (Promotion Effectiveness)")
            prod_promo = adv_filtered_df.groupby(['product_name', 'is_promotion']).agg({
                'quantity': 'sum',
                'item_payment_amount': 'sum'
            }).unstack(fill_value=0).reset_index()
            # 복잡한 변수 생성 생략하고 단순 판매량/금액으로 군집
            prod_promo.columns = ['product_name', 'qty_no_promo', 'qty_promo', 'sales_no_promo', 'sales_promo']
            run_clustering(prod_promo, ['qty_promo', 'sales_promo', 'qty_no_promo'], title="Promo Response Clusters")

        # 3.7 주문 패턴 군집
        elif cluster_option.startswith("3.7"):
            st.subheader("📅 주문 패턴 군집 분석 (Order Pattern)")
            ord_pat = adv_filtered_df.groupby('order_id').agg({
                'order_hour': 'first',
                'total_payment_amount': 'first',
                'quantity': 'sum'
            }).reset_index()
            run_clustering(ord_pat, ['order_hour', 'total_payment_amount', 'quantity'], title="Order Patterns")

        # 3.8 채널 기반 주문 군집
        elif cluster_option.startswith("3.8"):
            st.subheader("🌐 채널 기반 주문 군집 분석 (Channel-Based)")
            adv_filtered_df['chan_code'] = pd.factorize(adv_filtered_df['order_channel'])[0]
            chan_clus = adv_filtered_df.groupby('order_id').agg({
                'chan_code': 'first',
                'total_payment_amount': 'first',
                'weight_kg': 'mean'
            }).reset_index()
            run_clustering(chan_clus, ['chan_code', 'total_payment_amount', 'weight_kg'], title="Channel-Weight Clusters")

        # 3.9 지역 소비 군집
        elif cluster_option.startswith("3.9"):
            st.subheader("📍 지역 소비 군집 분석 (Regional Consumption)")
            adv_filtered_df['reg_code'] = pd.factorize(adv_filtered_df['region_1'])[0]
            reg_clus = adv_filtered_df.groupby('customer_id').agg({
                'reg_code': 'first',
                'weight_kg': 'mean',
                'item_payment_amount': 'mean'
            }).reset_index()
            run_clustering(reg_clus, ['reg_code', 'weight_kg', 'item_payment_amount'], title="Regional Segments")

        # 3.10 리드타임 군집
        elif cluster_option.startswith("3.10"):
            st.subheader("⏱️ 리드타임 및 운영 군집 (Delivery Lead-Time)")
            # 리드타임 계산: 배송준비 - 주문일
            time_df = adv_filtered_df.groupby('order_id').first().reset_index()
            if 'delivery_ready_datetime' in time_df.columns:
                time_df['lead_time_hrs'] = (pd.to_datetime(time_df['delivery_ready_datetime']) - pd.to_datetime(time_df['order_datetime'])).dt.total_seconds() / 3600
                time_df = time_df[time_df['lead_time_hrs'] >= 0].fillna(0)
                run_clustering(time_df, ['lead_time_hrs', 'total_payment_amount'], title="Lead-Time vs Amount")
                st.info("**Note**: 리드타임이 긴 주문의 고객 불만 관리 및 운영 최적화 지표로 활용")
            else:
                st.error("리드타임 분석을 위한 날짜 데이터가 부족합니다.")

    # [Advanced] 인사이트/제안
    with tabs[9]:
        st.header("💡 Advanced: 핵심 인사이트 및 전략")
        st.info("**인사이트 요약**: SNS 유입 고객의 구매 중량이 일반 고객 대비 높음. 2kg 소과 옵션의 마진율이 가장 우수함.")
        st.success("**액션 아이템**: 2kg 묶음 상품 강화, SNS 타겟 마케팅 시 대용량 세트 노출, 야간 타임 세일 운영 고려.")
        st.warning("**GA4 연계**: 고객 군집 ID를 User Property로 연동하여 리마케팅 정교화 필요.")
else:
    st.sidebar.warning("⚠️ 전처리 데이터를 찾을 수 없습니다. [Advanced] 탭들이 비활성화되었습니다.")
