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
    
    # 수치형 변환 및 전처리 (콤마 등 문자열 처리 포함)
    numeric_cols = ['실결제 금액', '결제금액', '판매단가', '주문수량', '취소수량', '재구매 횟수', '무게(kg)']
    for col in numeric_cols:
        if col in df.columns:
            # 문자열인 경우 콤마 제거 후 숫자로 변환
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    
    # --- 정확한 매출 계산 (품목별 실 매출액) ---
    # 사용자가 확인한 총액(306,819,910)은 '판매단가 * (주문수량 - 취소수량)'의 합계와 일치함
    if '판매단가' in df.columns and '주문수량' in df.columns and '취소수량' in df.columns:
        df['item_revenue'] = df['판매단가'] * (df['주문수량'] - df['취소수량'])
    else:
        df['item_revenue'] = df['결제금액'] # fallback
        
    # 날짜 변환
    if '주문일' in df.columns:
        df['주문일'] = pd.to_datetime(df['주문일'])
        df['date'] = df['주문일'].dt.date
        df['month'] = df['주문일'].dt.to_period('M').astype(str)
        df['day_name'] = df['주문일'].dt.day_name()
        df['hour'] = df['주문일'].dt.hour
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
variety_list = sorted([str(x) for x in df['품종'].unique() if pd.notna(x)])
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

# TypeError 방지를 위해 확실하게 수치형으로 변환 후 연산
# 총 매출액은 주문번호 기준 중복을 제거한 '실결제 금액'의 합계 (306,819,910원에 맞춤)
total_sales = pd.to_numeric(filtered_df.drop_duplicates('주문번호')['실결제 금액'].sum(), errors='coerce')
if pd.isna(total_sales): total_sales = 0

total_orders = filtered_df['주문번호'].nunique()
avg_order_value = total_sales / total_orders if total_orders > 0 else 0
cancel_rate = (filtered_df['취소여부'] == 'Y').mean() * 100

with k1:
    st.metric("총 결제금액", f"{total_sales:,.0f}원")
with k2:
    st.metric("총 주문 건수", f"{total_orders:,}건")
with k3:
    st.metric("평균 주문 단가(AOV)", f"{avg_order_value:,.0f}원")
with k4:
    st.metric("주문 취소율", f"{cancel_rate:.1f}%", delta_color="inverse")

st.divider()

# --- 4. Tabs 구성 ---
tabs = st.tabs(["📈 매출 및 성과", "📦 품종 및 상품 분석", "⚖️ 무게/가격 분포", "🧬 고객 군집 분석(Clustering)", "🏪 셀러별 심층 분석", "📊 셀러 통합 비교", "🌐 지역별 분석", "💡 가설 검증", "📋 데이터 탐색기"])

# Tab 1: 매출 및 성과
with tabs[0]:
    if not filtered_df.empty:
        st.subheader("매출 트렌드 분석")
        t1, t2 = st.columns([2, 1])
        
        with t1:
            trend_agg = filtered_df.groupby('date')['item_revenue'].sum().reset_index()
            fig_trend = px.line(trend_agg, x='date', y='item_revenue', title="일별 매출 추이")
            fig_trend.update_traces(line_color='#FF8C00', fill='tozeroy')
            st.plotly_chart(fig_trend, use_container_width=True)
            
        with t2:
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_agg = filtered_df.groupby('day_name')['item_revenue'].sum().reindex(day_order).fillna(0).reset_index()
            fig_day = px.bar(day_agg, x='day_name', y='item_revenue', color='item_revenue',
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
    else:
        st.warning("선택된 조건에 해당하는 데이터가 없습니다.")

# Tab 2: 품종 및 상품 분석
with tabs[1]:
    if not filtered_df.empty:
        st.subheader("품종 및 상품 포트폴리오")
        p1, p2 = st.columns(2)
        
        with p1:
            variety_sales = filtered_df.groupby('품종')['item_revenue'].sum().sort_values(ascending=False).reset_index()
            fig_var = px.bar(variety_sales, x='item_revenue', y='품종', orientation='h', title="품종별 매출 순위",
                             color='item_revenue', color_continuous_scale='Viridis')
            st.plotly_chart(fig_var, use_container_width=True)
            
        with p2:
            size_agg = filtered_df['과수 크기'].value_counts()
            st.plotly_chart(px.pie(values=size_agg.values, names=size_agg.index, title="과수 크기별 선호도"), use_container_width=True)

        st.subheader("Top 10 상품 리스트 (결제금액 기준)")
        top_items = filtered_df.groupby('상품명')['item_revenue'].sum().sort_values(ascending=False).head(10).reset_index()
        st.table(top_items)
    else:
        st.warning("데이터가 없습니다.")

# Tab 3: 무게/가격 분포
with tabs[2]:
    if not filtered_df.empty:
        st.subheader("중량 및 가격대 분포 분석")
        w1, w2 = st.columns(2)
        
        with w1:
            fig_w = px.histogram(filtered_df, x='무게(kg)', nbins=20, title="주문 중량(kg) 분포",
                                 color_discrete_sequence=['#4B0082'])
            st.plotly_chart(fig_w, use_container_width=True)
            
        with w2:
            price_order = ["1만원 이하", "1~3만원", "3~5만원", "5~10만원", "10만원 초반"]
            price_agg = filtered_df['가격대'].value_counts().reindex(price_order).fillna(0).reset_index()
            fig_p = px.bar(price_agg, x='가격대', y='count', title="가격대별 주문 건수", 
                           color='가격대', color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_p, use_container_width=True)

        st.markdown("---")
        st.subheader("목적별 주문 특성 (개인소비 vs 선물)")
        purpose_agg = filtered_df.groupby('목적').agg({'item_revenue':'mean', '무게(kg)':'mean', '주문번호':'count'}).reset_index()
        purpose_agg.columns = ['목적', '평균 결제액', '평균 중량(kg)', '주문 건수']
        st.dataframe(purpose_agg.style.format({'평균 결제액': '{:,.0f}원', '평균 중량(kg)': '{:.2f}kg'}), use_container_width=True)
    else:
        st.warning("데이터가 없습니다.")

# Tab 4: 고객 군집 분석
# Tab 4: 다차원 군집 분석
with tabs[3]:
    st.header("🧬 다차원 군집 분석 (Multi-Clustering)")
    if not filtered_df.empty:
        scenario = st.selectbox("분석 시나리오를 선택하세요", 
                               ["지역별 성과", "셀러별 역량", "시간대별 패턴", "가격/중량별 특성", "고객 가치 세그먼트"])
        
        # 시나리오별 데이터 집계
        if scenario == "지역별 성과":
            agg_df = filtered_df.groupby('광역지역').agg({
                'item_revenue': 'sum',
                '주문번호': 'nunique'
            }).reset_index()
            agg_df.columns = ['ID', 'total_sales', 'order_count']
            features = ['total_sales', 'order_count']
            labels = {'total_sales':'총 매출액', 'order_count':'주문 건수'}
            info_txt = "지역별 매출액과 주문 규모를 기준으로 지역군을 분류합니다."
            
        elif scenario == "셀러별 역량":
            agg_df = filtered_df.groupby('셀러명').agg({
                'item_revenue': 'sum',
                '재구매 횟수': 'mean'
            }).reset_index()
            agg_df.columns = ['ID', 'total_sales', 'avg_repurchase']
            features = ['total_sales', 'avg_repurchase']
            labels = {'total_sales':'총 매출액', 'avg_repurchase':'평균 재구매 횟수'}
            info_txt = "셀러별 매출 규모와 고객 유지력(재구매)을 기준으로 핵심 셀러군을 가려냅니다."
            
        elif scenario == "시간대별 패턴":
            agg_df = filtered_df.groupby('hour').agg({
                'item_revenue': 'sum',
                '주문번호': 'nunique'
            }).reset_index()
            agg_df.columns = ['ID', 'total_sales', 'order_count']
            features = ['total_sales', 'order_count']
            labels = {'total_sales':'시간대별 총 매출', 'order_count':'시간대별 주문수'}
            info_txt = "시간대별 주문 집중도와 매출 기여도를 분석하여 피크 타임군을 식별합니다."
            
        elif scenario == "가격/중량별 특성":
            # 상품(UID) 기준으로 분석
            agg_df = filtered_df.groupby('UID').agg({
                '판매단가': 'mean',
                '무게(kg)': 'mean'
            }).reset_index()
            agg_df.columns = ['ID', 'avg_price', 'avg_weight']
            features = ['avg_price', 'avg_weight']
            labels = {'avg_price':'평균 판매가', 'avg_weight':'평균 중량(kg)'}
            info_txt = "상품의 가격대와 중량을 기준으로 상품군(가성비팩, 프리미엄 선물 등)을 세분화합니다."
            
        else: # 고객 가치 세그먼트
            agg_df = filtered_df.groupby('주문자연락처').agg({
                'item_revenue': 'sum',
                '재구매 횟수': 'max'
            }).reset_index()
            agg_df.columns = ['ID', 'total_spent', 'max_repurchase']
            features = ['total_spent', 'max_repurchase']
            labels = {'total_spent':'총 지출액', 'max_repurchase':'재구매 횟수'}
            info_txt = "고객별 지출력과 재방문 충성도를 기준으로 고객군을 분류합니다."

        if len(agg_df) >= 3:
            st.markdown(f"**{scenario} 분석**: {info_txt}")
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(agg_df[features].fillna(0))
            
            n_clusters = st.slider(f"{scenario} 군집 수 선택", 2, 6, 3 if len(agg_df) > 5 else 2)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            agg_df['cluster'] = kmeans.fit_predict(scaled_features)
            
            c1, c2 = st.columns([1.5, 1])
            with c1:
                # 2차원 산점도
                fig_2d = px.scatter(agg_df, x=features[0], y=features[1], color='cluster',
                                   hover_data=['ID'], title=f"[{scenario}] 군집 시각화",
                                   labels=labels, color_continuous_scale='Viridis')
                fig_2d.update_traces(marker=dict(size=12, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
                st.plotly_chart(fig_2d, use_container_width=True)
            
            with c2:
                # 군집별 요약 표
                cluster_sum = agg_df.groupby('cluster')[features].mean().reset_index()
                st.write("**군집별 평균 지표**")
                # 컬럼명 가독성 개선
                cluster_sum.columns = ['군집'] + [labels[f] for f in features]
                st.dataframe(cluster_sum.style.background_gradient(cmap='Greens'), use_container_width=True)
                
            st.info(f"💡 **분석 가이드**: 우측 상단으로 갈수록 {labels[features[0]]}와 {labels[features[1]]}가 모두 높은 핵심 군집을 의미합니다.")
        else:
            st.warning(f"분석을 위한 데이터 포인트가 부족합니다. (현재 {len(agg_df)}개, 최소 3개 필요)")
    else:
        st.warning("데이터가 없습니다.")

# Tab 5: 셀러별 심층 분석
with tabs[4]:
    st.header("🏪 셀러별 심층 성과 분석")
    if not filtered_df.empty:
        seller_list = sorted([str(x) for x in filtered_df['셀러명'].unique() if pd.notna(x)])
        selected_seller = st.selectbox("분석할 셀러를 선택하세요", seller_list, index=0)
        s_df = filtered_df[filtered_df['셀러명'] == selected_seller]
        
        if not s_df.empty:
            sk1, sk2, sk3, sk4 = st.columns(4)
            s_total_sales = s_df['item_revenue'].sum()
            s_total_orders = s_df['주문번호'].nunique()
            s_avg_payment = s_total_sales / s_total_orders if s_total_orders > 0 else 0
            s_repurchase_rate = (s_df['재구매 횟수'] > 0).mean() * 100
            
            sk1.metric("셀러 총 매출", f"{s_total_sales:,.0f}원")
            sk2.metric("총 주문 건수", f"{s_total_orders:,}건")
            sk3.metric("평균 주문 단가", f"{s_avg_payment:,.0f}원")
            sk4.metric("고객 재구매율", f"{s_repurchase_rate:.1f}%" if not np.isnan(s_repurchase_rate) else "0.0%")
            
            st.divider()
            sc1, sc2 = st.columns(2)
            with sc1:
                s_trend = s_df.groupby('date')['item_revenue'].sum().reset_index()
                if not s_trend.empty:
                    fig_s_trend = px.line(s_trend, x='date', y='item_revenue', title=f"[{selected_seller}] 매출 트렌드")
                    fig_s_trend.update_traces(line_color='#FF4B4B')
                    st.plotly_chart(fig_s_trend, use_container_width=True)
            with sc2:
                s_path_counts = s_df['주문경로'].value_counts().reset_index()
                if not s_path_counts.empty:
                    fig_s_path = px.pie(s_path_counts, values='count', names='주문경로', hole=0.4, title=f"[{selected_seller}] 주문경로 점유율")
                    st.plotly_chart(fig_s_path, use_container_width=True)
            
            sc3, sc4 = st.columns(2)
            with sc3:
                s_region = s_df['광역지역(정식)'].value_counts().reset_index().head(10)
                if not s_region.empty:
                    fig_s_region = px.bar(s_region, x='count', y='광역지역(정식)', orientation='h', 
                                         title=f"[{selected_seller}] 주요 판매 지역 (Top 10)", color='count', color_continuous_scale='Reds')
                    st.plotly_chart(fig_s_region, use_container_width=True)
            with sc4:
                s_variety = s_df['품종'].value_counts().reset_index()
                if not s_variety.empty:
                    fig_s_variety = px.bar(s_variety, x='품종', y='count', title=f"[{selected_seller}] 취급 품종 분포", color='품종')
                    st.plotly_chart(fig_s_variety, use_container_width=True)

            st.divider()
            st.subheader(f"💡 {selected_seller} 셀러 전략 제언")
            s_path_top = s_path_counts.iloc[0]['주문경로'] if not s_path_counts.empty else "N/A"
            if s_repurchase_rate > 30:
                st.success(f"✅ **충성도 높음**: 재구매율이 {s_repurchase_rate:.1f}%로 매우 높습니다. 단골 고객 대상 감사 이벤트를 추천합니다.")
            else:
                st.info(f"ℹ️ **신규 유입 중심**: 현재 {s_path_top}를 통한 유입이 가장 많습니다. 초기 구매 고객을 단골로 전환하기 위한 첫 구매 혜택 강화가 필요합니다.")
        else:
            st.warning("해당 셀러의 데이터가 없습니다.")
    else:
        st.warning("데이터가 없습니다.")

# Tab 6: 셀러 통합 비교
with tabs[5]:
    st.header("📊 셀러별 통합 비교 분석")
    if not filtered_df.empty:
        # 셀러별 지표 집계
        seller_perf = filtered_df.groupby('셀러명').agg({
            'item_revenue': 'sum',
            '주문번호': 'nunique',
            '재구매 횟수': 'mean',
            '무게(kg)': 'mean'
        }).reset_index()
        
        seller_perf.columns = ['셀러명', '총 매출액', '주문 건수', '평균 재구매 횟수', '평균 중량(kg)']
        seller_perf['평균 주문단가(AOV)'] = seller_perf['총 매출액'] / seller_perf['주문 건수']
        seller_perf = seller_perf.sort_values('총 매출액', ascending=False)

        # 1. 상위 셀러 매출 비교
        st.subheader("🏆 상위 셀러 매출 현황")
        top_n = st.slider("표시할 셀러 수", 5, 20, 10)
        fig_multi_sales = px.bar(seller_perf.head(top_n), x='총 매출액', y='셀러명', orientation='h',
                                 title=f"매출 상위 {top_n}개 셀러", color='총 매출액',
                                 color_continuous_scale='Sunset')
        st.plotly_chart(fig_multi_sales, use_container_width=True)

        st.divider()
        
        # 2. 성과 매트릭스 (버러 차트)
        st.subheader("📈 셀러 성과 매트릭스")
        st.markdown("주문 건수 대비 매출액을 비교하며, 버블 크기는 평균 주문단가(AOV)를 나타냅니다.")
        fig_bubble = px.scatter(seller_perf, x='주문 건수', y='총 매출액', size='평균 주문단가(AOV)', 
                                color='셀러명', hover_name='셀러명',
                                title="셀러별 매출 vs 주문건수 vs AOV",
                                labels={'주문 건수': '총 주문 건수', '총 매출액': '총 결제금액'})
        st.plotly_chart(fig_bubble, use_container_width=True)

        st.divider()

        # 3. 셀러별 주문 경로 및 상품 분석
        st.subheader("📊 셀러별 주문 경로 및 상품 포트폴리오")
        c1, c2 = st.columns(2)
        
        with c1:
            # 상위 N개 셀러의 주문경로 비중
            top_sellers = seller_perf.head(top_n)['셀러명']
            path_df = filtered_df[filtered_df['셀러명'].isin(top_sellers)]
            path_agg = path_df.groupby(['셀러명', '주문경로']).size().reset_index(name='주문건수')
            
            fig_path_multi = px.bar(path_agg, x='주문건수', y='셀러명', color='주문경로',
                                   title=f"상위 {top_n}개 셀러 주문경로 비중",
                                   orientation='h', barmode='stack',
                                   color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_path_multi, use_container_width=True)
            
        with c2:
            # 셀러별 매출 상위 상품 (Treemap)
            product_agg = path_df.groupby(['셀러명', '상품명'])['item_revenue'].sum().reset_index()
            # 각 셀러별 Top 5 상품만 추출
            product_agg = product_agg.sort_values(['셀러명', 'item_revenue'], ascending=[True, False])
            product_agg = product_agg.groupby('셀러명').head(5)
            
            fig_tree = px.treemap(product_agg, path=['셀러명', '상품명'], values='item_revenue',
                                  title=f"상위 {top_n}개 셀러별 주요 판매 상품 (Top 5)",
                                  color='item_revenue', color_continuous_scale='RdYlGn')
            st.plotly_chart(fig_tree, use_container_width=True)

        st.divider()

        # 4. 셀러별 지역 판매 분포 (Phase 3)
        st.subheader("🌐 셀러별 주요 판매 지역 분포")
        region_df = filtered_df[filtered_df['셀러명'].isin(top_sellers)]
        region_agg = region_df.groupby(['셀러명', '광역지역(정식)'])['item_revenue'].sum().reset_index()
        
        fig_region_multi = px.bar(region_agg, x='item_revenue', y='셀러명', color='광역지역(정식)',
                                 title=f"상위 {top_n}개 셀러의 지역별 매출 비중",
                                 orientation='h', barmode='stack',
                                 color_discrete_sequence=px.colors.qualitative.T10)
        st.plotly_chart(fig_region_multi, use_container_width=True)

        st.divider()

        # 5. 셀러 생애주기 및 성장성 분석 (Phase 3)
        st.subheader("📈 셀러 생애주기 및 플랫폼 성장성")
        st.markdown("셀러의 유입(신규)과 유지(기존)에 따른 매출 변화 및 활성 셀러 수를 분석합니다.")
        
        # 전체 데이터 기준 셀러별 첫 주문일 계산 (생애주기 분석용)
        # filtered_df는 조회 기간 내 데이터이므로, df(전체)를 사용하여 '진짜' 첫 주문일을 파악해야 함
        seller_first_order = df.groupby('셀러명')['주문일'].min().reset_index()
        seller_first_order.columns = ['셀러명', 'first_order_date']
        
        # filtered_df와 병합하여 주문 시점 기준 신규/기존 구분
        lifecycle_df = filtered_df.merge(seller_first_order, on='셀러명')
        lifecycle_df['is_new_seller'] = lifecycle_df['주문일'].dt.to_period('M') == lifecycle_df['first_order_date'].dt.to_period('M')
        lifecycle_df['seller_type'] = lifecycle_df['is_new_seller'].map({True: '신규 셀러', False: '기존 셀러'})
        
        lc1, lc2 = st.columns(2)
        with lc1:
            # 월별 활성 셀러 수 트렌드
            active_sellers_trend = filtered_df.groupby('month')['셀러명'].nunique().reset_index()
            fig_active_trend = px.line(active_sellers_trend, x='month', y='셀러명', 
                                      title="월별 활성 셀러 수 추이", markers=True)
            fig_active_trend.update_traces(line_color='#2E8B57')
            st.plotly_chart(fig_active_trend, use_container_width=True)
            
        with lc2:
            # 신규 vs 기존 셀러 매출 기여도
            cohort_revenue = lifecycle_df.groupby(['month', 'seller_type'])['item_revenue'].sum().reset_index()
            fig_cohort = px.area(cohort_revenue, x='month', y='item_revenue', color='seller_type',
                                 title="신규 vs 기존 셀러 매출 기여도",
                                 color_discrete_map={'신규 셀러': '#FFA07A', '기존 셀러': '#4682B4'})
            st.plotly_chart(fig_cohort, use_container_width=True)

        st.divider()

        # 6. 셀러 상세 비교 테이블
        st.subheader("📑 셀러별 주요 지표 상세")
        st.dataframe(seller_perf.style.format({
            '총 매출액': '{:,.0f}원',
            '주문 건수': '{:,}건',
            '평균 재구매 횟수': '{:.2f}회',
            '평균 중량(kg)': '{:.2f}kg',
            '평균 주문단가(AOV)': '{:,.0f}원'
        }).background_gradient(subset=['총 매출액', '주문 건수'], cmap='YlGnBu'), use_container_width=True)
    else:
        st.warning("데이터가 없습니다.")

# Tab 8: 가설 검증 (Hypothesis Verification)
with tabs[7]:
    st.header("💡 비즈니스 가설 검증 (Hypothesis Verification)")
    if not filtered_df.empty:
        # 가설 선택
        hypotheses = [
            "1. 경기도권 매출은 특정 셀러의 영향인가?",
            "2. 이벤트 상품은 실제 구매량이 더 높은가?",
            "3. 선물 목적일 때 더 프리미엄 옵션을 선택하는가?",
            "4. 특정 셀러에게 재구매가 편중되어 있는가?",
            "5. 키워드별(1+1, 초고당도 등) 판매 효율 비교",
            "6. 전체 매출 감소와 셀러 이탈의 상관관계",
            "7. 서울 지역은 소량(적은 무게) 구매 비중이 높은가?"
        ]
        selected_hypo = st.selectbox("검증할 가설을 선택하세요", hypotheses)
        st.divider()

        if selected_hypo.startswith("1."):
            st.subheader("📍 경기도권 매출과 셀러의 상관성")
            gg_df = filtered_df[filtered_df['광역지역'] == '경기']
            if not gg_df.empty:
                gg_seller = gg_df.groupby('셀러명')['item_revenue'].sum().reset_index().sort_values('item_revenue', ascending=False)
                c1, c2 = st.columns([1.5, 1])
                with c1:
                    fig_gg = px.pie(gg_seller.head(10), values='item_revenue', names='셀러명', title="경기도 내 매출 상위 10개 셀러 비중")
                    st.plotly_chart(fig_gg, use_container_width=True)
                with c2:
                    st.write("**경기도 매출 기여도 상위 셀러**")
                    st.dataframe(gg_seller.head(10), use_container_width=True)
                st.info("💡 **인사이트**: 상위 셀러의 비중이 압도적이라면 경기도 매출은 지역적 특성보다 특정 셀러의 마케팅 영향이 큽니다.")
            else: st.warning("경기도 데이터가 없습니다.")

        elif selected_hypo.startswith("2."):
            st.subheader("🎁 이벤트 상품의 구매 유도 효과")
            ev_agg = filtered_df.groupby('이벤트 여부').agg({
                '주문수량': 'mean',
                'item_revenue': 'mean',
                '주문번호': 'count'
            }).reset_index()
            ev_agg.columns = ['이벤트 여부', '평균 주문수량', '평균 객단가', '총 주문건수']
            
            c1, c2 = st.columns(2)
            with c1:
                fig_ev1 = px.bar(ev_agg, x='이벤트 여부', y='평균 주문수량', color='이벤트 여부', title="이벤트 유무별 평균 주문 수량")
                st.plotly_chart(fig_ev1, use_container_width=True)
            with c2:
                fig_ev2 = px.bar(ev_agg, x='이벤트 여부', y='평균 객단가', color='이벤트 여부', title="이벤트 유무별 평균 객단가")
                st.plotly_chart(fig_ev2, use_container_width=True)
            st.info("💡 **인사이트**: 이벤트 상품의 평균 주문수량이 일반 상품보다 유의미하게 높다면 이벤트의 볼륨업 효과가 증명된 것입니다.")

        elif selected_hypo.startswith("3."):
            st.subheader("💝 선물 목적 방문자의 프리미엄 선호도")
            # 등급 및 가격대 교차 분석
            gift_df = filtered_df.groupby(['목적', '상품성등급_그룹']).size().unstack(fill_value=0).reset_index()
            fig_gift = px.bar(gift_df, x='목적', y=gift_df.columns[1:], title="주문 목적별 상품 등급 선택 비중", barmode='group')
            st.plotly_chart(fig_gift, use_container_width=True)
            
            st.write("**목적별 평균 결제 단가 비교**")
            gift_price = filtered_df.groupby('목적')['판매단가'].mean().reset_index()
            st.dataframe(gift_price.style.format({'판매단가': '{:,.0f}원'}), use_container_width=True)
            st.info("💡 **인사이트**: '선물' 목적 주문에서 '프리미엄' 등급 비중이 높거나 평균 단가가 높다면 가설이 성립합니다.")

        elif selected_hypo.startswith("4."):
            st.subheader("🔄 셀러별 고객 충성도(재구매) 랭킹")
            seller_loyalty = filtered_df.groupby('셀러명').agg({
                '주문번호': 'nunique',
                '재구매 횟수': lambda x: (x > 0).sum()
            }).reset_index()
            seller_loyalty['재구매 발생률(%)'] = (seller_loyalty['재구매 횟수'] / seller_loyalty['주문번호']) * 100
            seller_loyalty = seller_loyalty.sort_values('재구매 발생률(%)', ascending=False).head(15)
            
            fig_loyalty = px.bar(seller_loyalty, x='셀러명', y='재구매 발생률(%)', color='재구매 발생률(%)', title="재구매 발생 빈도가 높은 상위 15개 셀러")
            st.plotly_chart(fig_loyalty, use_container_width=True)
            st.info("💡 **인사이트**: 특정 셀러의 재구매율이 독보적으로 높다면 해당 셀러의 상품 품질이나 서비스가 우수함을 의미합니다.")

        elif selected_hypo.startswith("5."):
            st.subheader("🔍 상품명 핵심 키워드별 판매 성과")
            keywords = ["1+1", "초고당도", "과즙", "명품", "가정용", "산지직송", "실속"]
            kw_results = []
            for kw in keywords:
                mask = filtered_df['상품명'].str.contains(kw, na=False)
                kw_df = filtered_df[mask]
                if not kw_df.empty:
                    kw_results.append({
                        '키워드': kw,
                        '평균 매출액': kw_df['item_revenue'].mean(),
                        '주문 건수': kw_df['주문번호'].nunique()
                    })
            kw_perf = pd.DataFrame(kw_results).sort_values('평균 매출액', ascending=False)
            
            c1, c2 = st.columns(2)
            with c1:
                fig_kw1 = px.bar(kw_perf, x='키워드', y='평균 매출액', title="키워드별 평균 매출액(AOV)")
                st.plotly_chart(fig_kw1, use_container_width=True)
            with c2:
                fig_kw2 = px.bar(kw_perf, x='키워드', y='주문 건수', title="키워드별 총 주문 건수")
                st.plotly_chart(fig_kw2, use_container_width=True)
            st.info("💡 **인사이트**: 어떤 키워드가 더 많은 클릭(주문)을 유도하고, 어떤 키워드가 더 높은 단가의 구매를 유도하는지 비교할 수 있습니다.")

        elif selected_hypo.startswith("6."):
            st.subheader("📉 전체 매출 추이와 셀러 수의 상관성")
            monthly_trend = filtered_df.groupby('month').agg({
                'item_revenue': 'sum',
                '셀러명': 'nunique'
            }).reset_index()
            
            fig_dual = go.Figure()
            fig_dual.add_trace(go.Scatter(x=monthly_trend['month'], y=monthly_trend['item_revenue'], name='총 매출액', line=dict(color='firebrick', width=4)))
            fig_dual.add_trace(go.Bar(x=monthly_trend['month'], y=monthly_trend['셀러명'], name='활성 셀러 수', yaxis='y2', opacity=0.3))
            
            fig_dual.update_layout(
                title="매출액 vs 활성 셀러 수 추이",
                yaxis=dict(title="매출액"),
                yaxis2=dict(title="셀러 수", overlaying='y', side='right')
            )
            st.plotly_chart(fig_dual, use_container_width=True)
            st.info("💡 **인사이트**: 매출 하락기에 셀러 수도 함께 줄어든다면 셀러 이탈이 매출 감소의 핵심 원인일 가능성이 높습니다.")

        elif selected_hypo.startswith("7."):
            st.subheader("🏢 서울 지역의 소량 구매 특성")
            seoul_vs_others = filtered_df.copy()
            seoul_vs_others['지역구분'] = seoul_vs_others['광역지역'].apply(lambda x: '서울' if x == '서울' else '기타')
            
            fig_box = px.box(seoul_vs_others, x='지역구분', y='무게(kg)', color='지역구분', title="서울 vs 기타 지역 구매 중량(kg) 분포")
            st.plotly_chart(fig_box, use_container_width=True)
            
            st.write("**지역별 평균 구매 중량 상세**")
            weight_avg = seoul_vs_others.groupby('지역구분')['무게(kg)'].mean().reset_index()
            st.dataframe(weight_avg.style.format({'무게(kg)': '{:.2f}kg'}), use_container_width=True)
            st.info("💡 **인사이트**: 서울 지역의 평균 무게가 유의미하게 낮거나 3kg 미만 비중이 높다면 1~2인 가구 타겟 전략이 필요합니다.")
    else:
        st.warning("데이터가 없습니다.")

# Tab 9: 데이터 탐색기
with tabs[8]:
    st.subheader("상세 데이터 테이블")
    st.dataframe(filtered_df, use_container_width=True)
    if not filtered_df.empty:
        st.subheader("수치형 칼럼 상관관계")
        numeric_cols = filtered_df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 1:
            corr = filtered_df[numeric_cols].corr()
            st.plotly_chart(px.imshow(corr, text_auto=True, title="Correlation Heatmap", color_continuous_scale='RdBu_r'))
