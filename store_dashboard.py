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
            st.subheader("📍 경기도권 매출과 셀러의 입지 계수(LQ) 분석")
            st.markdown("""
            **입지 계수(LQ, Location Quotient)**: 특정 셀러가 특정 지역에 얼마나 특화되어 있는지를 나타내는 지표입니다.
            - **LQ > 1**: 해당 셀러가 전체 지역보다 경기도에서 상대적으로 더 높은 경쟁력을 가짐 (특화됨)
            - **LQ < 1**: 해당 셀러의 경기도 매출 비중이 전체 평균보다 낮음
            """)
            
            # LQ 계산 로직
            # 1. 전체 셀러의 경기도 매출 비중 (기준점)
            total_sales_all = filtered_df['item_revenue'].sum()
            total_gg_sales = filtered_df[filtered_df['광역지역'] == '경기']['item_revenue'].sum()
            base_ratio = total_gg_sales / total_sales_all if total_sales_all > 0 else 0
            
            # 2. 셀러별 지표 계산
            seller_region = filtered_df.groupby('셀러명').agg({
                'item_revenue': 'sum'
            }).reset_index()
            seller_gg = filtered_df[filtered_df['광역지역'] == '경기'].groupby('셀러명').agg({
                'item_revenue': 'sum'
            }).reset_index()
            seller_gg.columns = ['셀러명', 'gg_sales']
            
            lq_df = pd.merge(seller_region, seller_gg, on='셀러명', how='left').fillna(0)
            lq_df['gg_ratio'] = lq_df['gg_sales'] / lq_df['item_revenue']
            lq_df['LQ'] = lq_df['gg_ratio'] / base_ratio if base_ratio > 0 else 0
            
            # 매출액순으로 정렬하여 상위 분석
            lq_top = lq_df.sort_values('item_revenue', ascending=False).head(15)
            
            c1, c2 = st.columns([1.5, 1])
            with c1:
                fig_lq = px.bar(lq_top, x='셀러명', y='LQ', color='LQ',
                               title="상위 매출 셀러들의 경기도 입지 계수(LQ)",
                               color_continuous_scale='RdYlGn', range_color=[0, 2])
                fig_lq.add_hline(y=1.0, line_dash="dash", line_color="black", annotation_text="전체 평균 (LQ=1.0)")
                st.plotly_chart(fig_lq, use_container_width=True)
            with c2:
                st.write("**셀러별 지역 의존도 상세**")
                display_lq = lq_top[['셀러명', 'item_revenue', 'gg_ratio', 'LQ']].copy()
                display_lq.columns = ['셀러명', '총 매출', '경기 매출 비중', '입지계수(LQ)']
                st.dataframe(display_lq.style.format({'총 매출': '{:,.0f}원', '경기 매출 비중': '{:.1%}', '입지계수(LQ)': '{:.2f}'}), use_container_width=True)
            
            st.info(f"💡 **인사이트**: 현재 경기도 전체 매출 비중은 **{base_ratio:.1%}**입니다. LQ가 1.5 이상인 셀러는 경기도 고객들에게 특별히 선택받는 셀러이며, 이들이 많을수록 '경기도 매출은 특정 셀러가 주도한다'는 가설이 강화됩니다.")

        elif selected_hypo.startswith("2."):
            st.subheader("🎁 이벤트 상품의 매출 상승 지수(Lift Index)")
            ev_agg = filtered_df.groupby('이벤트 여부').agg({
                '주문수량': 'mean',
                'item_revenue': 'mean',
                '주문번호': 'nunique'
            }).reset_index()
            
            # Lift 계산 (N 대비 Y의 배수)
            try:
                non_ev = ev_agg[ev_agg['이벤트 여부'] == 'N'].iloc[0]
                is_ev = ev_agg[ev_agg['이벤트 여부'] == 'Y'].iloc[0]
                lift_qty = is_ev['주문수량'] / non_ev['주문수량']
                lift_rev = is_ev['item_revenue'] / non_ev['item_revenue']
            except:
                lift_qty, lift_rev = 0, 0
            
            l1, l2 = st.columns(2)
            l1.metric("주문수량 상승 지수", f"{lift_qty:.2f}배", help="일반 상품 대비 이벤트 상품의 평균 주문수량 배수")
            l2.metric("결제금액 상승 지수", f"{lift_rev:.2f}배", help="일반 상품 대비 이벤트 상품의 평균 결제금액 배수")
            
            st.write("**품종별 이벤트 효과 차이**")
            var_ev = filtered_df.groupby(['품종', '이벤트 여부'])['item_revenue'].mean().unstack().reset_index()
            var_ev['Lift'] = var_ev['Y'] / var_ev['N']
            fig_var_ev = px.bar(var_ev.sort_values('Lift', ascending=False), x='품종', y='Lift', title="품종별 이벤트 매출 상승 지수")
            st.plotly_chart(fig_var_ev, use_container_width=True)
            st.info("💡 **인사이트**: 상승 지수가 1.0보다 높을수록 이벤트의 '객단가 높이기' 효과가 실존함을 의미합니다.")

        elif selected_hypo.startswith("3."):
            st.subheader("💝 선물 목적 구매자의 고가 옵션 선택 편향")
            # 가격대별 비중 분석
            price_order = ["1만원 이하", "1~3만원", "3~5만원", "5~10만원", "10만원 초반"]
            bias_df = filtered_df.groupby(['목적', '가격대']).size().unstack(fill_value=0)
            bias_ratio = bias_df.div(bias_df.sum(axis=1), axis=0)
            bias_ratio = bias_ratio.reindex(columns=price_order).reset_index()
            
            fig_bias = px.bar(bias_ratio, x='목적', y=price_order, title="구매 목적별 가격대 선택 비중 (%)", barmode='group')
            st.plotly_chart(fig_bias, use_container_width=True)
            
            # 로얄과/프리미엄 선택 확률 비교
            premium_prob = filtered_df.groupby('목적')['상품성등급_그룹'].apply(lambda x: (x == '프리미엄').mean()).reset_index()
            premium_prob.columns = ['목적', '프리미엄 선택 확률']
            st.write("**프리미엄 등급 선택 확률 비교**")
            st.table(premium_prob.style.format({'프리미엄 선택 확률': '{:.2%}'}))
            st.info("💡 **인사이트**: '선물' 목적 시 프리미엄 선택 확률이 '개인소비'보다 유의미하게 높다면 고객은 선물 시 더 비싼 옵션을 기꺼이 수용함을 뜻합니다.")

        elif selected_hypo.startswith("4."):
            st.subheader("🔄 셀러별 재구매 유지력(Retention) 심화")
            # 셀러별로 2회 이상 주문한 고객의 수 / 전체 고객의 수
            retention_df = filtered_df.groupby('셀러명').agg({
                '주문자연락처': ['nunique', lambda x: x.duplicated().sum()]
            }).reset_index()
            retention_df.columns = ['셀러명', 'total_customers', 'returning_customers']
            retention_df['Retention_Rate(%)'] = (retention_df['returning_customers'] / retention_df['total_customers']) * 100
            retention_df = retention_df[retention_df['total_customers'] >= 10].sort_values('Retention_Rate(%)', ascending=False).head(15)
            
            fig_ret = px.scatter(retention_df, x='total_customers', y='Retention_Rate(%)', size='total_customers',
                                text='셀러명', title="셀러별 규모 대비 재구매 유지율 (최소 고객 10명 이상)",
                                labels={'total_customers':'전체 고객 수', 'Retention_Rate(%)':'재구매 고객 비중(%)'})
            st.plotly_chart(fig_ret, use_container_width=True)
            st.info("💡 **인사이트**: 오른쪽 상단에 위치한 셀러는 규모와 충성도를 모두 잡은 핵심 셀러입니다.")

        elif selected_hypo.startswith("5."):
            st.subheader("🔍 키워드별 매출 기여 및 프리미엄 지수")
            keywords = ["1+1", "초고당도", "꿀", "명품", "가정용", "산지직송", "실속"]
            kw_list = []
            avg_base_price = filtered_df['판매단가'].mean()
            
            for kw in keywords:
                kw_df = filtered_df[filtered_df['상품명'].str.contains(kw, na=False)]
                if not kw_df.empty:
                    kw_list.append({
                        '키워드': kw,
                        '건수': len(kw_df),
                        '평균단가': kw_df['판매단가'].mean(),
                        '가격 프리미엄': kw_df['판매단가'].mean() / avg_base_price
                    })
            kw_advanced = pd.DataFrame(kw_list).sort_values('가격 프리미엄', ascending=False)
            
            fig_kw_adv = px.scatter(kw_advanced, x='건수', y='가격 프리미엄', text='키워드', size='건수',
                                   title="키워드별 노출 빈도 vs 가격 프리미엄 배수",
                                   labels={'가격 프리미엄':'전체 평균 단가 대비 배수'})
            fig_kw_adv.add_hline(y=1.0, line_dash="dash")
            st.plotly_chart(fig_kw_adv, use_container_width=True)
            st.info("💡 **인사이트**: '가격 프리미엄'이 1.0보다 높은 키워드는 해당 단어를 썼을 때 고객이 더 높은 가격을 지불할 의사가 있음을 시사합니다.")

        elif selected_hypo.startswith("6."):
            st.subheader("📉 매출 하락 원인 분석: 셀러 이탈 vs 객단가 하락")
            m_agg = filtered_df.groupby('month').agg({
                'item_revenue': 'sum',
                '셀러명': 'nunique',
                '주문번호': 'nunique'
            }).reset_index()
            m_agg['temp_revenue_per_seller'] = m_agg['item_revenue'] / m_agg['셀러명']
            
            fig_churn = go.Figure()
            fig_churn.add_trace(go.Bar(x=m_agg['month'], y=m_agg['셀러명'], name='활성 셀러 수'))
            fig_churn.add_trace(go.Scatter(x=m_agg['month'], y=m_agg['temp_revenue_per_seller'], name='셀러당 평균 매출', yaxis='y2'))
            
            fig_churn.update_layout(title="월별 활성 셀러 수와 셀러당 평균 기여도",
                                   yaxis=dict(title="셀러 수"),
                                   yaxis2=dict(title="셀러당 매출", overlaying='y', side='right'))
            st.plotly_chart(fig_churn, use_container_width=True)
            st.info("💡 **인사이트**: 셀러 수는 유지되는데 셀러당 매출이 줄어드는지, 혹은 셀러 수 자체가 줄어드는지 구분하여 하락 원인을 진단합니다.")

        else: # 7. 서울 소량 구매
            st.subheader("🏢 지역별 소량(3kg 이하) 주문 비중 비교")
            weight_mask = filtered_df['무게区分'] == '<3kg'
            
            region_weight = filtered_df.groupby('광역지역').apply(lambda x: (x['무게区分'] == '<3kg').mean()).reset_index()
            region_weight.columns = ['광역지역', '소량주문 비중']
            region_weight = region_weight.sort_values('소량주문 비중', ascending=False)
            
            fig_rw = px.bar(region_weight, x='광역지역', y='소량주문 비중', color='소량주문 비중',
                           title="지역별 3kg 이하 소량 주문 건수 비중 (%)",
                           color_continuous_scale='Blues')
            fig_rw.add_hline(y=region_weight['소량주문 비중'].mean(), line_dash="dash", annotation_text="전 지역 평균")
            st.plotly_chart(fig_rw, use_container_width=True)
            st.info("💡 **인사이트**: 서울의 소량 주문 비중이 전체 평균보다 월등히 높다면 '1~2인 가구의 소량 주문' 가설이 설득력을 얻습니다.")
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
