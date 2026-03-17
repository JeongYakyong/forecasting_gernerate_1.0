import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import pvlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go 
import plotly.express as px 
from datetime import datetime, timedelta
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
import warnings
import os
custom_header_css = """
<style>
    /* 상단 헤더 배경색을 부드러운 연두색으로 변경 */
    header[data-testid="stHeader"] {
        background-color: #e0f8e0 !important;
    }
    
    /* 헤더 왼쪽 상단에 고정 제목 추가 */
    header[data-testid="stHeader"]::before {
        content: "🌱 제주 신재생에너지 예측 대시보드";
        position: absolute;
        left: 50px;            /* 왼쪽 여백 */
        top: 15px;             /* 위쪽 여백 */
        font-size: 20px;       /* 글자 크기 */
        font-weight: 800;      /* 글자 굵기 */
        color: #2c3e50;        /* 텍스트 색상 (짙은 남색 계열) */
        z-index: 9999;
    }
</style>
"""

# HTML/CSS 적용 (이미 작성하셨겠지만 참고용으로 남겨둡니다)
st.markdown(custom_header_css, unsafe_allow_html=True)

warnings.filterwarnings("ignore", message=".*torch.classes.*")
logging.getLogger("torch").setLevel(logging.ERROR)

# 사용자 정의 모듈
from utils.db_manager import JejuEnergyDB
from utils.data_pipeline import (add_capacity_features, 
daily_historical_update, daily_forecast_and_predict, daily_historical_kpx, daily_historical_kma, daily_historical_kpx_smp,
run_model_prediction, prepare_model_input, daily_forecast_kpx, daily_forecast_kma)

# ==========================================
# 공통 헬퍼 함수
# ==========================================
EDA_ONLY_COLUMNS = {'HVDC_Total', 'LNG_Gen', 'Oil_Gen'}
PREDICTION_OUTPUT_COLUMNS = {'est_Solar_Utilization', 'est_Wind_Utilization'}

def check_data_status(df, key_columns=None):
    """
    데이터프레임의 무결성을 검사하는 함수.
    EDA_ONLY_COLUMNS에 포함된 컬럼은 결측치 검사에서 제외됩니다.
    """
    if df.empty:
        return {
            "status": "Empty", 
            "missing_timestamps": 0, 
            "nan_counts": {},
            "incomplete_rows": 0,
            "incomplete_details": {}
        }
    
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)
    
    # 주요 컬럼 결정: 지정 없으면 전체 숫자 컬럼 (EDA 참고용 제외)
    if key_columns is None:
        key_columns = [
            c for c in df.select_dtypes(include=[np.number]).columns.tolist()
            if c not in EDA_ONLY_COLUMNS
        ]
    key_columns = [c for c in key_columns if c in df.columns]
    
    # 1. 시계열 누락
    expected_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='1h')
    missing_timestamps = expected_range.difference(df.index)
    
    # 2. 전체 컬럼 결측치 (EDA 참고용 제외)
    check_cols = [c for c in df.columns if c not in EDA_ONLY_COLUMNS]
    nan_counts = df[check_cols].isna().sum()
    nan_counts = nan_counts[nan_counts > 0].to_dict()
    
    # 3. 주요 컬럼 기준 불완전 행
    if key_columns:
        key_df = df[key_columns]
        incomplete_mask = key_df.isna().any(axis=1)
        incomplete_rows = int(incomplete_mask.sum())
        incomplete_details = key_df.isna().sum()
        incomplete_details = incomplete_details[incomplete_details > 0].to_dict()
    else:
        incomplete_rows = 0
        incomplete_details = {}
    
    has_problem = len(missing_timestamps) > 0 or nan_counts or incomplete_rows > 0
    
    return {
        "status": "Warning" if has_problem else "Good",
        "missing_timestamps": len(missing_timestamps),
        "missing_dates": missing_timestamps.tolist()[:5],
        "nan_counts": nan_counts,
        "incomplete_rows": incomplete_rows,
        "incomplete_details": incomplete_details,
        "key_columns_checked": key_columns
    }

def date_range_selector(key_prefix, allow_future_days=0, default_option="1주"):
    """
    버튼 한 줄로 큰 기간을 선택 → 슬라이더로 세부 구간 조절하는 공통 헬퍼 함수.
    
    Parameters:
        key_prefix (str): 위젯 key 충돌 방지용 접두사
        allow_future_days (int): 미래 날짜 허용 일수 (0이면 오늘까지만)
        default_option (str): 초기 선택값
        
    Returns:
        (start_date, end_date): datetime.date 튜플
    """
    today = datetime.now().date()
    max_date = today + timedelta(days=allow_future_days)
    
    state_key = f"date_range_{key_prefix}"
    if state_key not in st.session_state:
        st.session_state[state_key] = default_option
    
    options = {
        "하루": 1,
        "1주": 7,
        "2주": 14,
        "30일": 30,
        "90일": 90,
        "1년": 365,
    }
    
    # 1단계: 버튼으로 큰 범위 선택
    cols = st.columns(len(options) + 1)
    
    for i, (label, _) in enumerate(options.items()):
        is_active = st.session_state[state_key] == label
        button_type = "primary" if is_active else "secondary"
        if cols[i].button(label, key=f"{key_prefix}_btn_{label}", type=button_type, width="stretch"):
            st.session_state[state_key] = label
            st.rerun()
    
    is_custom = st.session_state[state_key] == "기간선택"
    custom_type = "primary" if is_custom else "secondary"
    if cols[-1].button("기간선택", key=f"{key_prefix}_btn_custom", type=custom_type, width="stretch"):
        st.session_state[state_key] = "기간선택"
        st.rerun()
    
    current_selection = st.session_state[state_key]
    
    # 2단계: 선택된 범위 내에서 슬라이더로 세부 구간 조절
    if current_selection == "기간선택":
        date_range = st.date_input(
            "달력에서 직접 선택",
            [today - timedelta(days=7), today],
            max_value=max_date,
            key=f"{key_prefix}_custom_date"
        )
        if len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date, end_date = date_range[0], date_range[0]
    else:
        days_back = options[current_selection]
        range_start = today - timedelta(days=days_back)
        range_end = today if allow_future_days == 0 else min(today + timedelta(days=allow_future_days), max_date)
        
        # 하루짜리는 슬라이더가 의미 없으므로 바로 반환
        if days_back <= 1:
            return range_start, range_end
        
        # 슬라이더로 구간 세부 조절
        start_date, end_date = st.slider(
            "📅 구간 조절",
            min_value=range_start,
            max_value=range_end,
            value=(range_start, range_end),
            format="YYYY-MM-DD",
            key=f"{key_prefix}_slider",
            label_visibility="collapsed"
        )
    
    return start_date, end_date

# ==========================================
# 전역 색상 팔레트
# ==========================================
COLORS = {
    'solar_real': 'darkorange',
    'solar_est': 'orange',
    'wind_real': 'darkblue',
    'wind_est': 'skyblue',
    'demand': 'gray',
    'net_demand': 'red',
    'renew_total': 'green',
    'smp': 'purple',
    'error': 'red',
}

def merge_actual_and_forecast(db, start_str, end_str):
    """
    실측+예보 머지 후 발전량 계산까지 완료된 DataFrame 반환.
    Option E의 Tab1, Tab2, Tab3에서 반복되던 로직을 통합.
    """
    hist_df = db.get_historical(start_str, end_str)
    fore_df = db.get_forecast(start_str, end_str)
    
    if hist_df.empty or fore_df.empty:
        return pd.DataFrame()
    
    merged = pd.merge(hist_df, fore_df, left_index=True, right_index=True, how='inner', suffixes=('', '_fore'))
    
    if merged.empty:
        return pd.DataFrame()
    
    cap_solar_col = 'Solar_Capacity_Est_fore' if 'Solar_Capacity_Est_fore' in merged.columns else 'Solar_Capacity_Est'
    cap_wind_col = 'Wind_Capacity_Est_fore' if 'Wind_Capacity_Est_fore' in merged.columns else 'Wind_Capacity_Est'
    
    merged['est_solar_gen'] = merged['est_Solar_Utilization'] * merged[cap_solar_col]
    merged['est_wind_gen'] = merged['est_Wind_Utilization'] * merged[cap_wind_col]
    
    return merged

def plot_actual_vs_pred(df, date_title, radio_key):
    """실제 vs 예측 비교 차트. Option E의 Tab1, Tab2에서 재사용."""
    source_choice = st.radio("발전원 선택:", ["태양광 (Solar)", "풍력 (Wind)"], horizontal=True, key=radio_key)
    
    fig = go.Figure()
    if source_choice == "태양광 (Solar)":
        real_col, est_col = 'real_solar_gen', 'est_solar_gen'
        color_real, color_est = COLORS['solar_real'], COLORS['solar_est']
    else:
        real_col, est_col = 'real_wind_gen', 'est_wind_gen'
        color_real, color_est = COLORS['wind_real'], COLORS['wind_est']
    
    fig.add_trace(go.Scatter(x=df.index, y=df[real_col], name="실제 발전량", line=dict(color=color_real, width=3)))
    fig.add_trace(go.Scatter(x=df.index, y=df[est_col], name="예측 발전량", line=dict(color=color_est, width=3, dash='dash')))
    
    df_plot = df.copy()
    df_plot['error'] = df_plot[est_col] - df_plot[real_col]
    fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['error'], name="오차", marker_color=COLORS['error'], opacity=0.3))
    
    fig.update_layout(
        title=f"{date_title} {source_choice} 실제 vs 예측 비교",
        hovermode="x unified",
        yaxis_title="발전량 (MW)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        dragmode='zoom',
        yaxis=dict(fixedrange=True)
    )
    fig.update_traces(hovertemplate='%{y:,.1f}')
    st.plotly_chart(fig, width="stretch")

# ==========================================
# 페이지 설정
# ==========================================
st.set_page_config(page_title="제주 신재생에너지 예측 대시보드", layout="wide")

# ==========================================
# 모델 클래스 정의
# ==========================================

class InstanceNormalization(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.affine = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode='norm', mean=None, std=None):
        if mode == 'norm':
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            return (x - self.mean) / self.std * self.affine + self.bias
        elif mode == 'denorm':
            return (x - self.bias) / self.affine * std + mean
        return x

class Patch_Weather_Attention(nn.Module):
    def __init__(self, patch_input_dim, hidden_dim):
        super().__init__()
        self.W_Q = nn.Sequential(nn.Linear(patch_input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim))
        self.W_K = nn.Sequential(nn.Linear(patch_input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim))
        self.scale_factor = 1.0 / (hidden_dim ** 0.5)

    def forward(self, future_weather_patch, past_weather_patches, transformer_output):
        Q = self.W_Q(future_weather_patch).unsqueeze(1)
        K = self.W_K(past_weather_patches)
        score = torch.bmm(Q, K.transpose(1, 2)) * self.scale_factor
        attn_weights = F.softmax(score, dim=-1)
        context = torch.bmm(attn_weights, transformer_output)
        return context.squeeze(1), attn_weights

class PatchTST_Weather_Model(nn.Module):
    def __init__(self, num_features, seq_len=336, pred_len=24, patch_len=24, stride=12, d_model=128, num_heads=4, num_layers=2, d_ff=256, dropout=0.2):
        super(PatchTST_Weather_Model, self).__init__()

        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_patches = (self.seq_len - self.patch_len) // self.stride + 1

        self.inst_norm = InstanceNormalization(num_features)

        patch_input_dim = self.patch_len * num_features
        self.patch_embedding = nn.Linear(patch_input_dim, self.d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, self.d_model))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=num_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.num_weather_feats = num_features - 1
        weather_patch_dim = self.patch_len * self.num_weather_feats

        self.weather_attn = Patch_Weather_Attention(
            patch_input_dim=weather_patch_dim,
            hidden_dim=self.d_model
        )

        future_weather_flat_dim = self.pred_len * self.num_weather_feats
        self.regressor = nn.Sequential(
            nn.Linear(self.d_model + future_weather_flat_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(256, self.pred_len)
        )

    def forward(self, batch, device='cpu'):
        p_num = batch['past_numeric'].to(device)
        p_y = batch['past_y'].to(device)
        f_num = batch['future_numeric'].to(device)
        B = p_num.shape[0]

        x_past = torch.cat([p_num, p_y], dim=-1)
        x_past = self.inst_norm(x_past, mode='norm')

        x_patches = x_past.unfold(dimension=1, size=self.patch_len, step=self.stride)
        x_patches = x_patches.permute(0, 1, 3, 2).reshape(B, self.num_patches, -1)

        enc_out = self.patch_embedding(x_patches) + self.pos_embedding
        enc_out = self.transformer_encoder(self.dropout(enc_out))

        future_weather_flat = f_num.reshape(B, -1)

        x_past_weather = x_past[..., :-1]
        w_patches = x_past_weather.unfold(1, self.patch_len, self.stride)
        w_patches = w_patches.permute(0, 1, 3, 2).reshape(B, self.num_patches, -1)

        context, _ = self.weather_attn(future_weather_flat, w_patches, enc_out)

        total_input = torch.cat([context, future_weather_flat], dim=1)
        prediction = self.regressor(total_input)

        return prediction

# ==========================================
# DB 및 모델 초기화
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database", "jeju_energy.db")

@st.cache_resource
def get_db():
    return JejuEnergyDB(DB_PATH)
    
@st.cache_resource
def load_assets():
    print("[1/6] load_assets 시작!")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("[2/6] 메타데이터 및 스케일러 로딩 중...")
    metadata = joblib.load('models/metadata.pkl')
    scaler_solar = joblib.load('models/robust_scaler_solar.pkl')    
    scaler_wind = joblib.load('models/robust_scaler_wind.pkl')
    
    scalers = {'solar': scaler_solar, 'wind': scaler_wind}
    
    seq_len = metadata['SEQ_LEN']
    pred_len = metadata['PRED_LEN']
    
    print("[3/6] 모델 초기화 중...")
    
    solar_model = PatchTST_Weather_Model(
        num_features=len(metadata['features_solar']),
        seq_len=seq_len, pred_len=pred_len,
        patch_len=24, stride=12, d_model=512, num_layers=4, d_ff=1024
    ).to(device)
    
    wind_model = PatchTST_Weather_Model(
        num_features=len(metadata['features_wind']),
        seq_len=seq_len, pred_len=pred_len,
        patch_len=24, stride=24, d_model=128, num_layers=3, d_ff=512
    ).to(device)

    print("[4/6] 태양광 모델 가중치 로딩 중...")
    solar_model.load_state_dict(torch.load('models/best_patchtst_solar_model.pth', map_location=device))
    
    print("[5/6] 풍력 모델 가중치 로딩 중...")
    wind_model.load_state_dict(torch.load('models/best_patchtst_wind_model.pth', map_location=device))
    
    solar_model.eval()
    wind_model.eval()
    
    print("[6/6] load_assets 완료!")
    return solar_model, wind_model, scalers, metadata, device

db = get_db()
assets = load_assets()

# ==========================================
# 사이드바 메뉴
# ==========================================

st.sidebar.title("✔️ Side Bar")
menu = st.sidebar.radio("메뉴 선택:", [
    "Option A : DB 관리",
    "Option B : 데이터 분석 (EDA)",
    "Option C : 발전량 예측",
    "Option D : 예측 결과 시각화",
    "Option E : 예측 정확도 검증",
    "Option F : 시스템 안내"     # ← 추가
])
# ==========================================
# Option A : DB 관리
# ==========================================
if menu == "Option A : DB 관리":
    st.title("🗂️ DB 관리 및 Data Status")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Data Status", "API 데이터 수집", "데이터 조회", "CSV 업로드"])
    
    # --- Tab 1: Data Status ---
    with tab1:
        header_col1, header_col2 = st.columns([8, 2])
        with header_col1:
            st.subheader("DB 상태 점검 (전체 기간)")
        with header_col2:
            if st.button("🔄 새로고침", help="최신 데이터베이스 정보를 다시 불러옵니다.", width="stretch", key="refresh_db_button1"):
                st.rerun()

        with st.spinner("전체 실측 데이터를 불러오고 무결성을 검사하는 중입니다..."):
            full_hist_df = db.get_historical()
        
        if not full_hist_df.empty:
            st.write("### 📊 데이터 저장 현황")
            
            if not pd.api.types.is_datetime64_any_dtype(full_hist_df.index):
                full_hist_df.index = pd.to_datetime(full_hist_df.index)
                
            min_date = full_hist_df.index.min()
            max_date = full_hist_df.index.max()
            
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            max_date_only = max_date.replace(hour=0, minute=0, second=0, microsecond=0)
            gap_days = (today - max_date_only).days
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("총 데이터 수", f"{len(full_hist_df):,} 행")
            col2.metric("시작 날짜", min_date.strftime('%Y-%m-%d'))
            col3.metric("최근 날짜", max_date.strftime('%Y-%m-%d'))
            today_midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            check_df = full_hist_df[full_hist_df.index < today_midnight]
            status_info = check_data_status(check_df)

            if gap_days > 0:
                col4.metric(
                    label="업데이트 필요 (오늘 기준)", 
                    value=f"{gap_days}일 분량", 
                    delta=f"{gap_days}일 지연됨", 
                    delta_color="inverse"
                )
            elif gap_days == 0 and status_info['status'] == "Good":
                col4.metric(
                    label="업데이트 필요 (오늘 기준)", 
                    value="최신 상태", 
                    delta="결측 없음", 
                    delta_color="normal"
                )
            elif gap_days == 0:
                col4.metric(
                    label="업데이트 필요 (오늘 기준)", 
                    value="최신 상태", 
                    delta=f"결측 {status_info['incomplete_rows']}건", 
                    delta_color="inverse"
                )
            else:
                col4.metric("업데이트 필요", "미래 데이터 포함", f"+{abs(gap_days)}일")
                        
            # 불완전 행 요약 (주요 컬럼 기준)
            if status_info['incomplete_rows'] > 0:
                st.warning(f"⚠️ timestamp는 존재하지만 주요 컬럼 값이 비어있는 행: **{status_info['incomplete_rows']}건**")
            
            if status_info['missing_timestamps'] > 0:
                st.warning(f"⚠️ 시계열 누락 (timestamp 자체가 빠진 시간대): **{status_info['missing_timestamps']}건**")
            
            if status_info['status'] == "Good":
                st.success("✅ 모든 주요 컬럼의 데이터가 빈틈없이 채워져 있습니다!")
            
            with st.expander("🔍 전체 컬럼별 결측치 상세 확인", expanded=False):
                st.write("각 컬럼별로 비어있는(Null, NaN) 데이터의 개수를 보여줍니다.")
                
                missing_info = check_df.isnull().sum().reset_index()
                missing_info.columns = ["컬럼명", "결측치 개수"]
                
                st.dataframe(missing_info, width="stretch", hide_index=True)
                total_missing = missing_info["결측치 개수"].sum()

                if total_missing > 0:
                    if st.button("✨ 결측치 자동 보간 (최대 3건 제한) 및 DB 적용", help="최대 3개의 연속된 결측치까지만 시간 비례로 채웁니다."):
                        with st.spinner("결측치를 보간하고 DB에 저장하는 중입니다..."):
                            try:
                                interpolated_df = check_df.interpolate(method='time', limit=3, limit_direction='forward')
                                interpolated_df.index = interpolated_df.index.strftime('%Y-%m-%d %H:%M:%S')
                                db.save_historical(interpolated_df)
                                st.success("🎉 결측치 보간 및 DB 업데이트가 완료되었습니다!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"보간 처리 중 오류가 발생했습니다: {e}")
                                    
            with st.expander("👀 결측치가 발생한 시간대 직접 확인하기", expanded=False):
                st.info("어느 시간대(timestamp)의 데이터가 비어있는지 확인해 보세요.")
                missing_rows = full_hist_df[full_hist_df.isna().any(axis=1)]
                st.dataframe(missing_rows, width="stretch")
        else:
            st.error("데이터베이스가 비어있습니다. [API 데이터 수집] 탭에서 데이터를 수집해 주세요.")
            
    # --- Tab 2: API 데이터 수집 ---
    with tab2:
        st.subheader("API를 통한 데이터 수집")
        st.info("Data Status에서 확인한 결측 구간을 지정하여 데이터를 채워 넣으세요.")
        today = datetime.now().date()
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### 📈 실측 데이터 (Historical)")
            st.caption("최대 수집가능 기간은 30일입니다.")
            hist_start = st.date_input("시작일", today - timedelta(days=7), key='h_start')
            hist_end = st.date_input("종료일", today, key='h_end')
            
            invalid_date = False
            if hist_start > hist_end:
                st.error("시작일이 종료일보다 늦을 수 없습니다.")
                invalid_date = True
            elif (hist_end - hist_start).days > 30:
                st.error("실측 데이터 조회는 최대 30일까지만 가능합니다.")
                invalid_date = True
            elif hist_end > today or hist_start > today:
                st.warning("⚠️ 현재시간 이후의 데이터는 정상적으로 수집되지 않을 수 있습니다.")
            
            if st.button("실측 데이터 수집", width="stretch", disabled=invalid_date):
                with st.spinner("모든 실측 데이터를 수집하고 있습니다..."):
                    try:
                        daily_historical_kpx(hist_start.strftime("%Y-%m-%d"), hist_end.strftime("%Y-%m-%d"))
                        daily_historical_kma(hist_start.strftime("%Y-%m-%d"), hist_end.strftime("%Y-%m-%d"))
                        daily_historical_kpx_smp(hist_start.strftime("%Y-%m-%d"), hist_end.strftime("%Y-%m-%d"))
                        st.success("실측 데이터 수집 완료!")
                    except Exception as e:
                        st.error(f"API 호출 실패: {e}")
            
            with st.expander("🛠️ 개별 API 수집"):
                st.caption("필요한 특정 데이터만 개별적으로 수집할 수 있습니다.")
                
                h_start_str = hist_start.strftime("%Y-%m-%d")
                h_end_str = hist_end.strftime("%Y-%m-%d")
                
                if st.button("KPX 발전량 수집", key="btn_kpx_hist", disabled=invalid_date):
                    with st.spinner("KPX 발전량 데이터를 수집 중입니다..."):
                        try:
                            daily_historical_kpx(h_start_str, h_end_str) 
                            st.success("KPX 발전량 데이터 수집 완료!")
                        except Exception as e:
                            st.error(f"KPX 발전량 API 호출 실패: {e}")
                        
                if st.button("KPX SMP 수집", key="btn_kpx_smp", disabled=invalid_date):
                    with st.spinner("KPX SMP 데이터를 수집 중입니다..."):
                        try:
                            daily_historical_kpx_smp(h_start_str, h_end_str)
                            st.success("KPX SMP 데이터 수집 완료!")
                        except Exception as e:
                            st.error(f"KPX SMP API 호출 실패: {e}")
                        
                if st.button("KMA 기상 데이터 수집", key="btn_kma_hist", disabled=invalid_date):
                    with st.spinner("KMA 기상 데이터를 수집 중입니다..."):
                        try:
                            daily_historical_kma(h_start_str, h_end_str)
                            st.success("KMA 기상 데이터 수집 완료!")
                        except Exception as e:
                            st.error(f"KMA 기상 API 호출 실패: {e}")
                        
        with col2:
            st.write("### 🌤️ Forecast 데이터 (예보)")
            st.caption("주의: 과거 3일 전 ~ 미래 1일 후 (최대 30일 간격)")
            
            fore_start = st.date_input("시작일", today - timedelta(days=1), key='f_start')
            fore_end = st.date_input("종료일", today + timedelta(days=1), key='f_end')
            invalid_fore_date = False
            
            if fore_start > fore_end:
                st.error("시작일이 종료일보다 늦을 수 없습니다.")
                invalid_fore_date = True
            elif fore_start < today - timedelta(days=3):
                limit_past = (today - timedelta(days=3)).strftime("%Y-%m-%d")
                st.error(f"예보는 과거 3일 전({limit_past})까지만 조회 가능합니다.")
                invalid_fore_date = True
            elif fore_end > today + timedelta(days=1):
                limit_future = (today + timedelta(days=1)).strftime("%Y-%m-%d")
                st.error(f"예보는 내일({limit_future})까지만 조회 가능합니다.")
                invalid_fore_date = True
            elif (fore_end - fore_start).days > 30:
                st.error("예보 데이터 조회는 최대 30일까지만 가능합니다.")
                invalid_fore_date = True
                
            if st.button("Forecast 데이터 수집", width="stretch", disabled=invalid_fore_date):
                with st.spinner("모든 예보 데이터를 수집하고 있습니다..."):
                    try:
                        daily_forecast_and_predict(fore_start.strftime("%Y-%m-%d"), fore_end.strftime("%Y-%m-%d"))
                        st.success("Forecast 데이터 수집 완료!")
                    except Exception as e:
                        st.error(f"Forecast API 호출 실패: {e}")
            
            with st.expander("🛠️ 개별 API 수집"):
                st.caption("필요한 특정 예보 데이터만 개별적으로 수집할 수 있습니다.")
                
                f_start_str = fore_start.strftime("%Y-%m-%d")
                f_end_str = fore_end.strftime("%Y-%m-%d")
                
                if st.button("KPX 발전량 Forecast 수집", key="btn_kpx_fore_ind", disabled=invalid_fore_date):
                    with st.spinner("KPX Forecast 데이터를 수집 중입니다..."):
                        try:
                            daily_forecast_kpx(f_start_str, f_end_str)
                            st.success("KPX Forecast 수집 완료!")
                        except Exception as e:
                            st.error(f"KPX Forecast API 호출 실패: {e}")
                        
                if st.button("KMA 기상 Forecast 수집", key="btn_kma_fore_ind", disabled=invalid_fore_date):
                    with st.spinner("KMA 기상 Forecast 데이터를 수집 중입니다..."):
                        try:
                            daily_forecast_kma(f_start_str, f_end_str)
                            st.success("KMA Forecast 수집 완료!")
                        except Exception as e:
                            st.error(f"KMA Forecast API 호출 실패: {e}")
        
        st.markdown("---")
        st.info("💡 Forecast 자료는 전날 23시에 업로드됩니다. 매일 자정 이후에 수집하시는 것을 추천드립니다.")

    # --- Tab 3: 데이터 조회 ---
    with tab3:
        st.subheader("데이터 조회")
        st.caption("LNG, HVDC, 기력 발전량은 실시간 업데이트가 불가능합니다. 전력거래소 CSV 별도 다운로드 필요.")
        
        ctrl_col1, ctrl_col2 = st.columns([5, 1])
        with ctrl_col1:
            table_choice = st.radio("조회할 테이블:", ["실측 데이터 (Historical)", "Forecast 데이터"], horizontal=True, label_visibility="collapsed")
        with ctrl_col2:
            if st.button("🔄 새로고침", width="stretch", key="refresh_tab3"):
                st.rerun()
        
        future_days = 2 if table_choice == "Forecast 데이터" else 0
        start_date, end_date = date_range_selector("db_table", allow_future_days=future_days, default_option="1주")
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        if table_choice == "실측 데이터 (Historical)":
            df = db.get_historical(start_str, end_str)
        else:
            df = db.get_forecast(start_str, end_str)
        
        if not df.empty:
            st.caption(f"📅 {start_str} ~ {end_str}  |  총 {len(df):,}행")
            st.dataframe(df, width="stretch")
        else:
            st.warning(f"조회하신 기간({start_str} ~ {end_str})에 해당하는 데이터가 없습니다.")

    # --- Tab 4: CSV 업로드 ---
    with tab4:
        st.subheader("과거 CSV 파일 일괄 적재 (백업/복구용)")
        st.info("💡 초기 셋팅을 하거나 DB가 손실되었을 때, 과거 CSV 파일을 올려서 한 번에 복구할 수 있습니다.")
        
        uploaded_file = st.file_uploader("과거 데이터 CSV 파일을 올려주세요", type=['csv'])
        
        if uploaded_file is not None:
            preview_df = pd.read_csv(uploaded_file)
            st.write("### 📊 업로드된 파일 결측치 분석")
            missing_preview = preview_df.isnull().sum().reset_index()
            missing_preview.columns = ["컬럼명", "결측치 개수"]
            
            if missing_preview['결측치 개수'].sum() > 0:
                st.warning("⚠️ 업로드된 파일에 결측치가 존재합니다. 아래 표를 확인하세요.")
            else:
                st.success("✅ 결측치가 없는 깔끔한 데이터입니다!")
                
            st.dataframe(missing_preview, width="stretch", hide_index=True)
            
            if st.button("DB에 적재하기", type="primary"):
                with st.spinner("데이터를 분석하고 DB에 기록하는 중입니다..."):
                    try:
                        df = preview_df.copy()
                        
                        if 'timestamp' not in df.columns:
                            st.error("CSV 파일에 'timestamp' 컬럼이 없습니다. 파일 형식을 확인해주세요!")
                        else:
                            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                            df = df.set_index('timestamp')
                            
                            df = add_capacity_features(df)
                            saved_rows = db.save_historical(df)
                            status_info = check_data_status(df)
                            
                            st.success(f"🎉 '{uploaded_file.name}' 파일 업로드 및 DB 적재 완료! (총 {saved_rows:,}행)")
                            st.info("추가/갱신된 파생 변수: Solar_Capacity_Est, Wind_Capacity_Est, Solar_Utilization, Wind_Utilization")
                            
                            if status_info["status"] == "Good":
                                st.success("✅ 업로드된 데이터의 시계열이 촘촘하고 결측치가 없습니다.")
                            else:
                                st.warning(f"⚠️ 업로드된 데이터에 {status_info['missing_timestamps']}개의 누락된 시간이 있거나 결측치가 포함되어 있습니다.")
                                
                    except Exception as e:
                        st.error(f"데이터 적재 중 오류가 발생했습니다: {e}")

# ==========================================
# Option B : 데이터 분석 (EDA)
# ==========================================
elif menu == "Option B : 데이터 분석 (EDA)":
    st.title("🔍 데이터 분석 (EDA)")
    
    # 1단계: 피처 선택 (상단 expander)
    with st.expander("🎯 분석할 피처 선택", expanded=True):
        # 일단 전체 컬럼 목록을 가져오기 위해 최근 데이터를 살짝 조회
        sample_df = db.get_historical(
            (datetime.now().date() - timedelta(days=7)).strftime('%Y-%m-%d'),
            datetime.now().date().strftime('%Y-%m-%d')
        )
        
        if sample_df.empty:
            st.warning("DB에 데이터가 없습니다. [Option A : DB 관리]에서 데이터를 먼저 수집해 주세요.")
            selected_features = []
        else:
            if 'real_demand' in sample_df.columns and 'real_renew_gen' in sample_df.columns:
                sample_df['real_net_demand'] = sample_df['real_demand'] - sample_df['real_renew_gen']
            
            numeric_cols = sample_df.select_dtypes(include=[np.number]).columns.tolist()
            
            selected_features = st.multiselect(
                "분석할 피처를 선택하세요", 
                options=numeric_cols, 
                default=['real_demand', 'real_solar_gen', 'real_wind_gen', 'real_renew_gen', 'real_net_demand'] if 'real_demand' 
                in numeric_cols else numeric_cols[:2],
                label_visibility="collapsed"
            )

    # 메인 컨텐츠: 탭 + 차트
    if not selected_features:
        st.info("👆 위에서 분석할 피처를 하나 이상 선택해 주세요.")
    else:
        tab1, tab2, tab3, tab4= st.tabs([
            "📈 시계열 데이터", 
            "📊 통계 요약", 
            "🔥 상관관계 히트맵", 
            "🌌 산점도",
        ])
        
        # 2단계: 기간 선택 (하단 expander) — 탭 아래에 배치
        with st.expander("📅 조회 기간 설정", expanded=True):
            start_date, end_date = date_range_selector("eda", allow_future_days=0, default_option="1주")
        
        # 기간이 정해진 후 DB 조회
        df = db.get_historical(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        if df.empty:
            st.warning(f"선택하신 기간({start_date} ~ {end_date})에 해당하는 데이터가 없습니다.")
        else:
            if 'real_demand' in df.columns and 'real_renew_gen' in df.columns:
                df['real_net_demand'] = df['real_demand'] - df['real_renew_gen']
            
            # 선택된 피처 중 실제 df에 존재하는 것만 필터
            valid_features = [f for f in selected_features if f in df.columns]
            if not valid_features:
                st.warning("선택한 피처가 조회된 데이터에 존재하지 않습니다.")
            else:
                analysis_df = df[valid_features].copy()
                
                with tab1:                
                    fig = px.line(
                        analysis_df, 
                        x=analysis_df.index, 
                        y=valid_features,
                        title="시계열 데이터 분석"
                    )
                    fig.update_layout(
                        hovermode="x unified",
                        legend_title_text="선택된 피처",
                        xaxis_title="시간",
                        yaxis_title="수치",
                        dragmode='zoom',
                        yaxis=dict(fixedrange=True)
                    )
                    fig.update_traces(hovertemplate='%{y:,.2f}') 
                    st.plotly_chart(fig, width="stretch")
                    st.caption("💡 **Tip:** 차트를 드래그하면 X축만 확대됩니다. 더블클릭으로 원래 범위로 복귀합니다.")

                with tab2:
                    st.subheader(f"통계 요약 ({start_date} ~ {end_date})")
                    st.dataframe(analysis_df.describe(), width="stretch")
                    
                with tab3:
                    if len(valid_features) < 2:
                        st.warning("상관관계를 분석하려면 최소 2개 이상의 피처를 선택해야 합니다.")
                    else:
                        corr_matrix = analysis_df.corr()
                        fig_corr = px.imshow(
                            corr_matrix,
                            text_auto=".2f",
                            aspect="auto",
                            color_continuous_scale="RdBu_r",
                            zmin=-1, zmax=1,
                            title="상관관계 행렬"
                        )
                        fig_corr.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig_corr, width="stretch")

                with tab4:
                    if len(valid_features) < 2:
                        st.warning("산점도를 그리려면 위에서 최소 2개 이상의 피처를 선택해야 합니다.")
                    else:
                        sc_col1, sc_col2 = st.columns(2)
                        with sc_col1:
                            x_axis = st.selectbox("X축 피처 선택:", options=valid_features, index=0)
                        with sc_col2:
                            y_axis = st.selectbox("Y축 피처 선택:", options=valid_features, index=1 if len(valid_features) > 1 else 0)
                        
                        fig_scatter = px.scatter(
                            analysis_df,
                            x=x_axis,
                            y=y_axis,
                            opacity=0.5,
                            marginal_x="histogram",
                            marginal_y="histogram",
                            title=f"{x_axis} vs {y_axis} 산점도"
                        )
                        st.plotly_chart(fig_scatter, width="stretch")
                        st.info("20-24년도 자료에는 발전량 자료가 포함되어 있어 상관관계 분석 가능 합니다.")

# ==========================================
# Option C : 발전량 예측
# ==========================================
elif menu == "Option C : 발전량 예측":
    st.title("🔮 발전량 예측 및 DB 저장")
    st.write("선택한 날짜의 예보 데이터를 바탕으로 태양광 및 풍력 발전 가동률을 예측하고 저장합니다.")
    
    default_pred_date = st.session_state.get('last_predicted_date', datetime.now().date())
    target_date = st.date_input("예측 대상 날짜", default_pred_date)
    
    # ==========================================
    # 날짜 선택 즉시: 데이터 상태 사전 점검
    # ==========================================
    st.markdown("---")
    st.write("### 🔍 입력 데이터 상태 점검")
    st.caption(f"모델 추론에 필요한 데이터: 과거 실측 336시간 + 미래 예보 24시간 (대상일: {target_date})")
    
    # 과거 실측 데이터 범위: target_date 기준 14일 전 ~ 전일 23시
    past_end = f"{target_date - timedelta(days=1)} 23:00:00"
    past_start = f"{target_date - timedelta(days=14)} 00:00:00"
    #past_end = f"{target_date} 00:00:00"
    #past_start = (target_date - timedelta(days=14)).strftime('%Y-%m-%d')
    past_df = db.get_historical(past_start, past_end)
    
    # 미래 예보 데이터 범위: target_date 00시 ~ 23시
    future_start = f"{target_date} 00:00:00"
    future_end = f"{target_date} 23:00:00"
    future_df = db.get_forecast(future_start, future_end)
    
    past_hours = len(past_df) if not past_df.empty else 0
    future_hours = len(future_df) if not future_df.empty else 0
    
    # 결측치 검사
    #past_missing = int(past_df.drop(columns=EDA_ONLY_COLUMNS, errors='ignore').isna().sum().sum()) if not past_df.empty else 0
    #future_missing = int(future_df.drop(columns=EDA_ONLY_COLUMNS, errors='ignore').isna().sum().sum()) if not future_df.empty else 0

    EXCLUDE_FROM_CHECK = EDA_ONLY_COLUMNS | PREDICTION_OUTPUT_COLUMNS
    # → {'HVDC_Total', 'LNG_Gen', 'Oil_Gen', 'est_Solar_Utilization', 'est_Wind_Utilization'}

    past_missing = (
        int(past_df.drop(columns=EXCLUDE_FROM_CHECK, errors='ignore').isna().sum().sum())
        if not past_df.empty else 0
    )

    future_missing = (
        int(future_df.drop(columns=EXCLUDE_FROM_CHECK, errors='ignore').isna().sum().sum())
        if not future_df.empty else 0
    )


    col1, col2, col3, col4 = st.columns(4)
    
    # 과거 실측 상태
    past_ok = past_hours >= 336 and past_missing == 0
    col1.metric(
        "과거 실측 (Historical)", 
        f"{past_hours} / 336시간",
        "정상" if past_ok else "부족",
        delta_color="normal" if past_ok else "inverse"
    )
    
    # 미래 예보 상태
    future_ok = future_hours >= 24 and future_missing == 0
    col2.metric(
        "미래 예보 (Forecast)", 
        f"{future_hours} / 24시간",
        "정상" if future_ok else "부족",
        delta_color="normal" if future_ok else "inverse"
    )
    
    # 실측 결측
    col3.metric(
        "실측 결측치", 
        f"{past_missing}건",
        "없음" if past_missing == 0 else "보간 필요",
        delta_color="normal" if past_missing == 0 else "inverse"
    )
    
    # 예보 결측
    col4.metric(
        "예보 결측치", 
        f"{future_missing}건",
        "없음" if future_missing == 0 else "API 재수집 필요",
        delta_color="normal" if future_missing == 0 else "inverse"
    )
    
    # 문제가 있을 때 구체적 안내
    if not past_ok or not future_ok:
        
        with st.expander("⚠️ 부족한 데이터 상세 확인", expanded=True):
            if past_hours < 336:
                st.warning(f"📈 **실측 데이터 부족**: {past_hours}시간 수집됨 (필요: 336시간). [Option A : DB 관리 → API 데이터 수집]에서 실측 데이터를 수집해 주세요.")
            # 과거 실측
            if past_missing > 0 and not past_df.empty:
                missing_cols = past_df.drop(columns=EXCLUDE_FROM_CHECK, errors='ignore').isna().sum()
                missing_cols = missing_cols[missing_cols > 0]
                if not missing_cols.empty:
                    st.caption(" " + ", ".join([f"{col}: {cnt}건" for col, cnt in missing_cols.items()]))
            
            if future_hours < 24:
                st.warning(f"🌤️ **Forecast 데이터 부족**: {future_hours}시간 수집됨 (필요: 24시간). [Option A : DB 관리 → API 데이터 수집]에서 예보를 수집해 주세요.")
            # 미래 예보
            if future_missing > 0 and not future_df.empty:
                missing_cols_f = future_df.drop(columns=EXCLUDE_FROM_CHECK, errors='ignore').isna().sum()
                missing_cols_f = missing_cols_f[missing_cols_f > 0]
                if not missing_cols_f.empty:
                    st.caption(" " + ", ".join([f"{col}: {cnt}건" for col, cnt in missing_cols_f.items()]))

    # ==========================================
    # 예측 실행 버튼
    # ==========================================
    st.markdown("---")

    if st.button("🚀 예측 실행", type="primary", width="stretch"):
        with st.spinner(f"{target_date} 예측을 진행 중입니다... (데이터 검증 및 모델 추론)"):
            
            success, message, input_info = run_model_prediction(
                target_date.strftime('%Y-%m-%d'), db, assets
            )

            if success:
                st.session_state['last_predicted_date'] = target_date
                st.success(message)
                
                st.write(f"### 📈 {target_date} 최종 예측 결과")
                res_df = db.get_forecast(f"{target_date} 00:00:00", f"{target_date} 23:00:00")
                
                if not res_df.empty:
                    fig = px.line(
                        res_df, 
                        x=res_df.index, 
                        y=['est_Solar_Utilization', 'est_Wind_Utilization'],
                        title="예측 가동률",
                        labels={"value": "가동률 (0~1)", "timestamp": "시간", "variable": "발전원"},
                        color_discrete_map={
                            "est_Solar_Utilization": "orange",
                            "est_Wind_Utilization": "skyblue"
                        }
                    )
                    fig.update_layout(hovermode="x unified")
                    fig.update_traces(hovertemplate='%{y:,.3f}')
                    st.plotly_chart(fig, width="stretch")
                
                st.info("👉 예측 결과를 시각화하려면 왼쪽 메뉴의 **Option D : 예측 결과 시각화**로 이동하세요.")
            else:
                st.error(f"예측 실패: {message}")
# ==========================================
# Option D : 예측 결과 시각화 (실측 오버레이 통합 버전)
# ==========================================
elif menu == "Option D : 예측 결과 시각화":
    
    # --- 헤더 + 실시간 KPX 수집 버튼 ---
    header_col1, header_col2 = st.columns([7, 3])
    with header_col1:
        st.title("📈 예측 결과 및 Net Demand 분석")
    with header_col2:
        st.write("")
        if st.button("⚡ 실시간 KPX 데이터 수집", width="stretch", type="primary"):
            with st.spinner("최신 실측 데이터를 수집하고 있습니다..."):
                try:
                    today_str = datetime.now().strftime("%Y-%m-%d")
                    yesterday_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
                    daily_historical_kpx(yesterday_str, today_str)
                    st.success("실측 데이터 수집 완료!")
                    st.rerun()
                except Exception as e:
                    st.error(f"실측 데이터 수집 실패: {e}")
    
    st.caption("모델이 예측한 가동률을 바탕으로 실제 발전량과 순부하(Net Demand)를 시각화합니다. 실측 데이터가 있으면 자동으로 오버레이됩니다.")
    st.markdown("---")
    
    # 💡 session_state에서 마지막 예측 날짜를 기본값으로 사용
    default_vis_date = st.session_state.get('last_predicted_date', datetime.now().date())
    target_date = st.date_input("조회할 예측 날짜 선택", default_vis_date)
    
    start_str = f"{target_date} 00:00:00"
    end_str = f"{target_date} 23:00:00"
    df_res = db.get_forecast(start_str, end_str)
    
    if df_res.empty or 'est_Solar_Utilization' not in df_res.columns or df_res['est_Solar_Utilization'].isnull().all():
        st.warning(f"{target_date}의 예측 데이터가 없습니다. [Option C : 발전량 예측]에서 먼저 예측을 실행해 주세요.")
    else:
        df = df_res.copy()
        
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)
            
        df['est_solar_gen'] = df['est_Solar_Utilization'] * df['Solar_Capacity_Est']
        df['est_wind_gen'] = df['est_Wind_Utilization'] * df['Wind_Capacity_Est']
        df['est_renew_total'] = df['est_solar_gen'] + df['est_wind_gen']
        df['est_net_demand'] = df['est_demand'] - df['est_renew_total']
        
        smp_col = 'smp_jeju' if 'smp_jeju' in df.columns else ('est_smp_jeju' if 'est_smp_jeju' in df.columns else None)
        
        # --- 실측 데이터 병합 시도 (historical 테이블에서 직접 조회) ---
        hist_df = db.get_historical(start_str, f"{target_date} 23:59:59")
        has_actual = False
        
        if not hist_df.empty:
            if not pd.api.types.is_datetime64_any_dtype(hist_df.index):
                hist_df.index = pd.to_datetime(hist_df.index)
            
            # 실측 컬럼을 df에 직접 매핑
            actual_cols = ['real_solar_gen', 'real_wind_gen', 'real_demand', 'real_renew_gen']
            for col in actual_cols:
                if col in hist_df.columns:
                    df[col] = hist_df[col].reindex(df.index)
            
            # 실측 값이 하나라도 있는지 확인 (None/NaN이 아닌 실제 숫자)
            if 'real_solar_gen' in df.columns and df['real_solar_gen'].notna().any():
                has_actual = True
                # 실측 합계 및 순부하 계산 (NaN은 NaN으로 유지, 0으로 채우지 않음)
                df['real_renew_total'] = df['real_solar_gen'] + df['real_wind_gen']
                df['real_net_demand'] = df['real_demand'] - df['real_renew_total']
        
        # session_state로 설정값 초기 유지
        if 'vis_selected_vars' not in st.session_state:
            st.session_state['vis_selected_vars'] = ['est_demand', 'est_net_demand', 'est_solar_gen', 'est_wind_gen']
        if 'vis_warn_low' not in st.session_state:
            st.session_state['vis_warn_low'] = 250
        if 'vis_warn_high' not in st.session_state:
            st.session_state['vis_warn_high'] = 750
        if 'vis_smp_low' not in st.session_state:
            st.session_state['vis_smp_low'] = 10
        if 'vis_warn_min_enabled' not in st.session_state:
            st.session_state['vis_warn_min_enabled'] = False
        if 'vis_warn_min' not in st.session_state:
            st.session_state['vis_warn_min'] = 150
        if 'vis_warn_max_enabled' not in st.session_state:
            st.session_state['vis_warn_max_enabled'] = False
        if 'vis_warn_max' not in st.session_state:
            st.session_state['vis_warn_max'] = 900
        # 📌 수정포인트: 실측 오버레이 체크박스 상태 초기화
        if 'vis_show_actual' not in st.session_state:
            st.session_state['vis_show_actual'] = False
        
        plot_options = {
            'est_demand': '총 전력수요 예측 (est_demand)',
            'est_net_demand': '순부하 예측 (est_net_demand)',
            'est_solar_gen': '태양광 발전량 예측 (est_solar_gen)',
            'est_wind_gen': '풍력 발전량 예측 (est_wind_gen)',
            'est_renew_total': '총 재생에너지 발전량 (est_renew_total)'
        }
        if smp_col:
            plot_options[smp_col] = f'제주 SMP 가격 ({smp_col})'
        
        tab_chart, tab_settings, tab_table = st.tabs(["📈 시각화", "⚙️ 데이터 선택 / 경고 설정", "📋 데이터 테이블"])
        
        # --- Tab 2: 설정 ---
        with tab_settings:
            st.subheader("데이터 선택")
            
            st.markdown("#### 📈 시각화할 예측 데이터 선택")
            selected_vars = st.multiselect(
                "예측 항목을 선택하세요:",
                options=list(plot_options.keys()),
                format_func=lambda x: plot_options[x],
                default=st.session_state.get('vis_selected_vars', []),
                key='vis_multiselect'
            )
            st.session_state['vis_selected_vars'] = selected_vars
        
            st.divider() 
        
            st.markdown("#### 📊 시각화할 실측 데이터 선택")
            if has_actual:
                actual_label_map = {
                    'real_demand': '수요 실측',
                    'real_solar_gen': '태양광 실측',
                    'real_wind_gen': '풍력 실측',
                    'real_renew_total': '재생E 실측 합계',
                    'real_net_demand': '순부하 실측',
                }
                
                available_actual = {col: label for col, label in actual_label_map.items() 
                                    if col in df.columns and df[col].notna().any()}
        
                if available_actual:
                    default_actual = st.session_state.get('vis_actual_cols', list(available_actual.keys()))
                    default_actual = [c for c in default_actual if c in available_actual]
                    
                    # 📌 수정포인트: key가 'vis_actual_cols' 이므로 수동 저장 코드를 제거했습니다.
                    st.multiselect(
                        "실측 항목을 선택하세요:",
                        options=list(available_actual.keys()),
                        format_func=lambda x: available_actual[x],
                        default=default_actual,
                        key='vis_actual_cols'
                    )
                else:
                    st.info("시각화 가능한 실측 데이터가 데이터프레임에 없습니다.")
            else:
                st.warning("현재 로드된 데이터에 실측 정보가 포함되어 있지 않습니다.")
        
            st.markdown("---")
            st.subheader("경고 기준 설정")
            st.caption("est_net_demand 기준으로 위험 구간을 음영 처리합니다.")
            
            warn_col1, warn_col2, warn_col3 = st.columns(3)
            with warn_col1:
                warning_threshold = st.number_input("🔴 저발전 경고 (MW)", value=st.session_state['vis_warn_low'], step=10, key='vis_warn_low_input')
                st.session_state['vis_warn_low'] = warning_threshold
            with warn_col2:
                warning_threshold2 = st.number_input("🔵 고발전 경고 (MW)", value=st.session_state['vis_warn_high'], step=10, key='vis_warn_high_input')
                st.session_state['vis_warn_high'] = warning_threshold2
            with warn_col3:
                if smp_col:
                    smp_threshold = st.number_input("🟡 SMP 하한 (원) — 저발전 경고에 OR 통합", value=st.session_state['vis_smp_low'], step=10, key='vis_smp_low_input')
                    st.session_state['vis_smp_low'] = smp_threshold
                else:
                    st.info("SMP 데이터가 없습니다.")
            
            st.caption("💡 저발전 경고: est_net_demand < 저발전 임계값 **또는** SMP < SMP 하한일 때 발동됩니다.")
            
            st.markdown("---")
            st.subheader("추가 경고 (선택)")
            st.caption("필요한 경우에만 활성화하세요.")
            
            extra_col1, extra_col2 = st.columns(2)
            with extra_col1:
                warn_min_on = st.checkbox("🟣 최저발전 경고 활성화", value=st.session_state['vis_warn_min_enabled'], key='vis_warn_min_cb')
                st.session_state['vis_warn_min_enabled'] = warn_min_on
                if warn_min_on:
                    warn_min_val = st.number_input("최저발전 임계값 (MW)", value=st.session_state['vis_warn_min'], step=10, key='vis_warn_min_input')
                    st.session_state['vis_warn_min'] = warn_min_val
            with extra_col2:
                warn_max_on = st.checkbox("🟤 최대발전 경고 활성화", value=st.session_state['vis_warn_max_enabled'], key='vis_warn_max_cb')
                st.session_state['vis_warn_max_enabled'] = warn_max_on
                if warn_max_on:
                    warn_max_val = st.number_input("최대발전 임계값 (MW)", value=st.session_state['vis_warn_max'], step=10, key='vis_warn_max_input')
                    st.session_state['vis_warn_max'] = warn_max_val
            
            st.info("💡 설정을 변경한 후 [📈 시각화] 탭으로 돌아가면 반영됩니다")
        
        # 설정값 로드
        selected_vars = st.session_state['vis_selected_vars']
        warning_threshold = st.session_state['vis_warn_low']
        warning_threshold2 = st.session_state['vis_warn_high']
        smp_threshold = st.session_state['vis_smp_low'] if smp_col else 0
        
        # --- Tab 1: 시각화 차트 ---
        with tab_chart:
            if not selected_vars:
                st.info("👉 [⚙️ 데이터 선택 / 경고 설정] 탭에서 시각화할 데이터를 선택해 주세요.")
            else:
                # 설정 탭에서 선택된 실측 항목 가져오기
                selected_actual_cols = st.session_state.get('vis_actual_cols', [])
                
                # 📌 수정포인트: vis_show_actual 세션 상태를 직접 읽어서 오버레이 활성화 여부를 결정합니다.
                show_actual = st.session_state.get('vis_show_actual', True) if has_actual else False
                overlay_active = show_actual and len(selected_actual_cols) > 0
                
                # --- 차트 생성 ---
                fig = go.Figure()
                
                # (이하 COLORS 딕셔너리가 외부에 정의되어 있다고 가정합니다)
                colors = {
                    'est_demand': COLORS['demand'] if 'COLORS' in globals() else 'black', 
                    'est_net_demand': COLORS['net_demand'] if 'COLORS' in globals() else 'blue', 
                    'est_solar_gen': COLORS['solar_est'] if 'COLORS' in globals() else 'orange', 
                    'est_wind_gen': COLORS['wind_est'] if 'COLORS' in globals() else 'green',
                    'est_renew_total': COLORS['renew_total'] if 'COLORS' in globals() else 'cyan'
                }
                if smp_col: colors[smp_col] = COLORS['smp'] if 'COLORS' in globals() else 'red'
                
                actual_map = {
                    'est_solar_gen': 'real_solar_gen',
                    'est_wind_gen': 'real_wind_gen',
                    'est_renew_total': 'real_renew_total',
                    'est_net_demand': 'real_net_demand',
                    'est_demand': 'real_demand',
                }
                
                for var in selected_vars:
                    # --- 예측 트레이스 ---
                    line_style = dict(
                        color=colors.get(var, 'black'),
                        width=2 if var != 'est_net_demand' else 4,
                        dash='dot' if overlay_active else 'solid'
                    )
                    if var in ['est_solar_gen', 'est_wind_gen'] and not overlay_active:
                        line_style['dash'] = 'dash'
                        
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df[var],
                        mode='lines+markers',
                        name=plot_options[var],
                        line=line_style,
                        hovertemplate='%{y:,.1f}',
                        legendgroup=var
                    ))
                    
                    # --- 실측 트레이스 ---
                    if overlay_active:
                        actual_col = actual_map.get(var)
                        if actual_col and actual_col in selected_actual_cols and actual_col in df.columns:
                            actual_name_map = {
                                'real_solar_gen': '태양광 실측',
                                'real_wind_gen': '풍력 실측',
                                'real_renew_total': '재생E 실측 합계',
                                'real_net_demand': '순부하 실측',
                                'real_demand': '수요 실측',
                            }
                            fig.add_trace(go.Scatter(
                                x=df.index, y=df[actual_col],
                                mode='lines+markers',
                                name=actual_name_map.get(actual_col, actual_col),
                                line=dict(
                                    color=colors.get(var, 'black'),
                                    width=3 if var != 'est_net_demand' else 5,
                                    dash='solid'
                                ),
                                marker=dict(size=6),
                                hovertemplate='%{y:,.1f}',
                                legendgroup=var
                            ))
                
                # --- 현재 시각 세로선 ---
                now = datetime.now()
                target_start = datetime.combine(target_date, datetime.min.time())
                target_end = datetime.combine(target_date, datetime.max.time())
                # 변경 후: datetime을 문자열로 변환

                if target_start <= now <= target_end:
                    now_str = now.strftime('%Y-%m-%d %H:%M:%S')
                    
                    # 세로선
                    fig.add_shape(
                        type="line",
                        x0=now_str, x1=now_str,
                        y0=0, y1=1,
                        yref="paper",
                        line=dict(color="red", width=2, dash="dash")
                    )

                if target_start <= now <= target_end:
                    now_str = now.strftime('%Y-%m-%d %H:%M:%S')
                    
                    # 세로선 추가
                    fig.add_vline(x=now_str, line_width=1, line_dash="solid", line_color="black")
                    
                    # 텍스트 라벨 추가
                    fig.add_annotation(
                        x=now_str,
                        y=-0.02,
                        yref="paper",
                        text="현재",
                        showarrow=False,
                        font=dict(size=10, color="black")
                    )
                # --- 위험 구간 음영 ---
                def draw_danger_zones(condition_series, fill_color, annotation_text=None, show_legend_label=None, layer_pos="below", fill_opacity=0.15):
                    if condition_series.any():
                        danger_df = df[condition_series].copy()
                        danger_df['group'] = (condition_series != condition_series.shift()).cumsum()
                        danger_df['temp_time'] = danger_df.index 
                        
                        danger_zones = danger_df.groupby('group').agg(
                            start=('temp_time', 'min'), 
                            end=('temp_time', 'max')
                        )
                        
                        for _, row in danger_zones.iterrows():
                            start_time = row['start'] - timedelta(hours=1)
                            end_time = row['end'] + timedelta(hours=1)
                            fig.add_vrect(
                                x0=start_time, x1=end_time,
                                fillcolor=fill_color, opacity=fill_opacity, # 투명도 적용
                                layer=layer_pos, line_width=0,              # 레이어 위치 적용
                                annotation_text=annotation_text, 
                                annotation_position="top left" if annotation_text else None
                            )
                        
                        if show_legend_label:
                            fig.add_trace(go.Scatter(
                                x=[None], y=[None], mode='markers',
                                marker=dict(size=10, color=fill_color, symbol='square'),
                                name=show_legend_label, showlegend=True
                            ))
                # 저발전 경고
                # --- (1) 일반 저발전/고발전 경고 (배경에 옅게 깔림) ---
                low_gen_condition = df['est_net_demand'] < warning_threshold
                if smp_col and smp_col in df.columns:
                    low_gen_condition = low_gen_condition | (df[smp_col] < smp_threshold)
                
                draw_danger_zones(low_gen_condition, "red", "저발전 경고🚨", layer_pos="below", fill_opacity=0.15)
                draw_danger_zones(df['est_net_demand'] > warning_threshold2, "blue", "고발전 경고🚨", layer_pos="below", fill_opacity=0.15)
                
                # --- (2) 최저/최대발전 경고 (전면에 진하게 표시됨) ---
                # 💡 layer_pos="above" 로 설정하여 그래프와 다른 음영 위로 올라오게 합니다. 투명도(0.3)도 조금 올렸습니다.
                if st.session_state.get('vis_warn_min_enabled', False):
                    draw_danger_zones(
                        df['est_net_demand'] < st.session_state['vis_warn_min'], 
                        "purple", 
                        annotation_text=" ", 
                        show_legend_label="최저발전구간",
                        layer_pos="above", 
                        fill_opacity=0.3
                    )
                if st.session_state.get('vis_warn_max_enabled', False):
                    draw_danger_zones(
                        df['est_net_demand'] > st.session_state['vis_warn_max'], 
                        "brown", 
                        annotation_text=" ", 
                        show_legend_label="최대발전구간",
                        layer_pos="above", 
                        fill_opacity=0.3
                    )
                fig.update_layout(
                    title=f"{target_date} 전력수급 및 재생에너지 예측 결과" + (" (실측 오버레이)" if overlay_active else ""),
                    xaxis_title="시간",
                    yaxis_title="발전량 / 전력량 (MW)",
                    hovermode="x unified", 
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    dragmode='zoom',
                    yaxis=dict(fixedrange=True)
                )
                
                st.plotly_chart(fig, width="stretch")
                
                # 📌 수정포인트: 체크박스의 key를 'vis_show_actual'로 할당하여, 위젯과 세션 상태가 자연스럽게 동기화되도록 수정했습니다. 수동 할당 코드는 삭제했습니다.
                if has_actual:
                    st.checkbox(
                        "📊 실측 데이터 오버레이",
                        key='vis_show_actual' 
                    )
                else:
                    st.info("ℹ️ 실측 데이터가 없습니다. [⚡ 실시간 KPX 데이터 수집] 버튼으로 최신 데이터를 수집하세요.")
        
        # --- Tab 3: 데이터 테이블 ---
        with tab_table:
            st.subheader(f"{target_date} 상세 데이터")
            display_cols = ['est_demand', 'est_solar_gen', 'est_wind_gen', 'est_renew_total', 'est_net_demand']
            
            if has_actual:
                actual_display = ['real_solar_gen', 'real_wind_gen', 'real_renew_total', 'real_net_demand']
                if 'real_demand' in df.columns:
                    actual_display.insert(0, 'real_demand')
                display_cols += [c for c in actual_display if c in df.columns]
            
            if smp_col: display_cols.append(smp_col)
            display_cols = [c for c in display_cols if c in df.columns]
            
            def highlight_warnings(row):
                styles = [''] * len(row)
                if 'est_net_demand' in row.index and row['est_net_demand'] < warning_threshold:
                    idx = row.index.get_loc('est_net_demand')
                    styles[idx] = 'background-color: #ffcccc'
                if smp_col and smp_col in row.index and row[smp_col] < smp_threshold:
                    idx = row.index.get_loc(smp_col)
                    styles[idx] = 'background-color: #ffe5b4'
                return styles

            st.dataframe(df[display_cols].style.apply(highlight_warnings, axis=1), width="stretch")
            
# ==========================================
# Option E : 예측 정확도 검증 (실시간 비교 탭 제거)
# ==========================================
elif menu == "Option E : 예측 정확도 검증":
    
    header_col1, header_col2 = st.columns([7, 3])
    with header_col1:
        st.title("✅ 예측 정확도 검증")
    with header_col2:
        st.write("")
        if st.button("⚡ 실시간 KPX 데이터 수집", width="stretch", type="primary"):
            with st.spinner("최신 실측 데이터를 수집하고 있습니다..."):
                try:
                    today_str = datetime.now().strftime("%Y-%m-%d")
                    yesterday_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
                    daily_historical_kpx(yesterday_str, today_str)
                    st.success("수집 완료!")
                    st.rerun()
                except Exception as e:
                    st.error(f"실측 데이터 수집 실패: {e}")
    
    st.caption("예측 모델의 결과와 실제 발전량을 비교하여 정확도를 평가합니다. 실시간 비교는 [Option D : 예측 결과 시각화]에서 확인할 수 있습니다.")
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["📊 일간 비교", "📈 기간별 정확도 평가"])
    
    # --- Tab 1: 일간 비교 ---
    with tab1:
        st.subheader("일간 실제 vs 예측 비교")
        
        default_val_date = st.session_state.get('last_predicted_date', datetime.now().date() - timedelta(days=1))
        target_date = st.date_input("비교할 날짜를 선택하세요", default_val_date, key="val_daily_date")
        
        val_df = merge_actual_and_forecast(db, f"{target_date} 00:00:00", f"{target_date} 23:59:59")
        
        if val_df.empty:
            st.warning(f"{target_date}의 실측 데이터 또는 예측 데이터가 부족합니다.")
        else:
            plot_actual_vs_pred(val_df, str(target_date), radio_key="daily_radio")

    # --- Tab 2: 기간별 정확도 ---
    with tab2:
        st.subheader("기간별 모델 정확도 평가")
        
        w_start, w_end = date_range_selector("val_weekly", allow_future_days=0, default_option="1주")
        
        w_val_df = merge_actual_and_forecast(db, w_start.strftime("%Y-%m-%d"), w_end.strftime("%Y-%m-%d 23:59:59"))
        
        if w_val_df.empty:
            st.warning("선택한 기간의 데이터가 부족합니다.")
        else:
            w_val_df = w_val_df.dropna(subset=['real_solar_gen', 'est_solar_gen', 'real_wind_gen', 'est_wind_gen'])
            
            if len(w_val_df) > 0:
                st.write("### 🎯 평가지표")
                st.caption("💡 **RMSE**: 큰 오차에 패널티 부여 / **MAE**: 실제 발전량과 평균적으로 몇 MW 차이가 나는지 직관적으로 보여줌")
                
                solar_rmse = np.sqrt(mean_squared_error(w_val_df['real_solar_gen'], w_val_df['est_solar_gen']))
                solar_mae = mean_absolute_error(w_val_df['real_solar_gen'], w_val_df['est_solar_gen'])
                wind_rmse = np.sqrt(mean_squared_error(w_val_df['real_wind_gen'], w_val_df['est_wind_gen']))
                wind_mae = mean_absolute_error(w_val_df['real_wind_gen'], w_val_df['est_wind_gen'])
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("태양광 RMSE", f"{solar_rmse:.2f} MW")
                m2.metric("태양광 MAE", f"{solar_mae:.2f} MW")
                m3.metric("풍력 RMSE", f"{wind_rmse:.2f} MW")
                m4.metric("풍력 MAE", f"{wind_mae:.2f} MW")
                
                st.markdown("---")
                
                st.write("### 📉 실제 vs 예측 추이")
                st.caption("전체 기간 동안의 모델 예측이 실제 트렌드를 잘 따라가는지 확인합니다.")
                
                fig_w_line = px.line(
                    w_val_df, 
                    x=w_val_df.index, 
                    y=['real_solar_gen', 'est_solar_gen', 'real_wind_gen', 'est_wind_gen'],
                    labels={"value": "발전량 (MW)", "index": "시간", "variable": "항목"}
                )
                fig_w_line.update_layout(
                    hovermode="x unified",
                    dragmode='zoom',
                    yaxis=dict(fixedrange=True)
                )
                st.plotly_chart(fig_w_line, width="stretch")
                
                st.markdown("---")
                
                st.write("### 🌌 실제값 vs 예측값 산점도")
                st.caption("점이 대각선(y=x)에 가깝게 모여 있을수록 정확한 예측입니다.")
                
                sc_col1, sc_col2 = st.columns(2)
                with sc_col1:
                    fig_s = px.scatter(w_val_df, x='real_solar_gen', y='est_solar_gen', opacity=0.5, title="태양광 (Solar)", color_discrete_sequence=[COLORS['solar_est']])
                    fig_s.add_shape(type="line", line=dict(dash="dash", color="gray"), x0=0, y0=0, x1=w_val_df['real_solar_gen'].max(), y1=w_val_df['real_solar_gen'].max())
                    st.plotly_chart(fig_s, width="stretch")
                    
                with sc_col2:
                    fig_w = px.scatter(w_val_df, x='real_wind_gen', y='est_wind_gen', opacity=0.5, title="풍력 (Wind)", color_discrete_sequence=[COLORS['wind_est']])
                    fig_w.add_shape(type="line", line=dict(dash="dash", color="gray"), x0=0, y0=0, x1=w_val_df['real_wind_gen'].max(), y1=w_val_df['real_wind_gen'].max())
                    st.plotly_chart(fig_w, width="stretch")
                    
# ==========================================
# Option F : 시스템 안내 (README)
# ==========================================
elif menu == "Option F : 시스템 안내":
    st.title("📖 시스템 안내")
    st.caption("제주 재생에너지 및 전력 순부하 예측 대시보드의 사용법과 구조를 안내합니다.")

    tab_disclaimer, tab_overview, tab_menu, tab_model, tab_setup = st.tabs([
        "⚠️ 면책 고지",
        "🌟 시스템 개요", 
        "📋 메뉴별 사용법", 
        "🤖 예측 모델 안내",
        "🚀 설치 및 환경 설정"
    ])
    
    # ─── Tab 0: 면책 고지 (Disclaimer) ───
    with tab_disclaimer:
        st.subheader("⚠️ 면책 고지 (Disclaimer)")
        
        st.error(
            "**본 대시보드는 참고용 보조 도구이며, 운영에 관한 최종 의사결정의 근거로 사용할 수 없습니다.**"
        )
        
        st.markdown("---")
        
        st.write("#### 1. 시스템의 목적 및 용도")
        st.write(
            "본 시스템은 제주 지역의 재생에너지 발전량 및 전력 순부하를 **예측·분석하기 위한 참고용 도구**입니다.\n\n"
            "제공되는 모든 예측 결과, 시각화, 통계 수치는 실제 계통 운영, 급전 지시, "
            "설비 투자, 전력 거래 등 **실무 의사결정을 대체하지 않습니다.**"
        )
        
        st.write("#### 2. 예측 정확도의 한계")
        st.write(
            "본 시스템에 탑재된 AI 예측 모델(PatchTST)은 과거 데이터를 기반으로 학습된 통계적 모델입니다.\n\n"
            "다음과 같은 상황에서 예측 정확도가 크게 저하될 수 있습니다."
        )
        st.write(
            "- **출력제어, 발전 유지 등 전력거래소의 의사결정**\n"
            "- 전력 계통의 구조적 변화 (정기점검, 송전선 차단 등)\n"
            "- 급격한 기상 변화 (태풍, 집중호우, 폭설 등 이상 기후)\n"
            "- 학습 데이터에 포함되지 않은 계절적 패턴이나 설비 변경\n"
            "- 입력 데이터(실측/예보)의 결측, 지연, 오류\n"
        )
        
        st.write("#### 3. 데이터 출처 및 신뢰성")
        st.write(
            "본 시스템은 전력거래소(KPX) 및 기상청(KMA)의 공개 API를 통해 데이터를 수집합니다.\n\n"
            "해당 기관의 API 장애, 데이터 지연, 형식 변경, 수치 정정 등으로 인해 "
            "수집된 데이터가 불완전하거나 부정확할 수 있습니다.\n\n"
            "본 시스템은 **외부 데이터의 정확성을 보증하지 않습니다.**"
        )
        
        st.write("#### 4. 면책 조항")
        st.write(
            "본 시스템은 다음 사항에 대하여 어떠한 법적 책임도 지지 않습니다."
        )
        st.write(
            "- 예측 결과의 부정확성으로 인해 발생한 직접적·간접적 손실\n"
            "- 시스템 장애, 데이터 유실, API 중단 등으로 인한 서비스 불가\n"
            "- 본 시스템의 출력을 근거로 한 의사결정에 따른 재정적·운영적 손해\n"
            "- 제3자 서비스(KPX, KMA API 등)의 변경 또는 중단으로 인한 영향"
        )
        
        st.write("#### 5. 사용자의 책임")
        st.write(
            "본 시스템을 사용하는 모든 사용자는 위 내용을 충분히 이해하고 동의한 것으로 간주합니다.\n\n"
            "예측 결과를 실무에 활용할 경우, 반드시 **자체 검증 절차를 거치고 "
            "전문 인력의 판단을 병행**하여야 합니다.\n\n"
            "본 시스템의 사용으로 인해 발생하는 모든 결과에 대한 책임은 사용자 본인에게 있습니다."
        )
        
        st.markdown("---")
        st.caption("본 고지는 시스템 최초 배포일 기준으로 작성되었으며, 사전 통보 없이 변경될 수 있습니다.")
        st.caption("작성자 : 김범준")
    # ─── Tab 1: 시스템 개요 ───
    with tab_overview:
        st.subheader("제주 재생에너지 및 전력 순부하 예측 대시보드")
        st.write(
            "기상청(KMA) 예보 데이터와 전력거래소(KPX) 데이터를 활용하여, **제주 지역의 태양광/풍력 발전 가동률을 예측**하고\n\n"
            "**전력 순부하(Net Demand) 및 경제성 지표(SMP)**를 모니터링하는 AI 대시보드입니다."
        )
        
        st.markdown("---")
        
        st.write("### 핵심 워크플로우")
        st.write(
            "이 시스템은 **데이터 수집 → 분석 → 예측 → 시각화 → 검증**의 순환 구조로 설계되었습니다. "
            "각 단계는 사이드바 메뉴(Option A~E)에 대응합니다."
        )
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.markdown("**Option A : DB 관리**\n\n`데이터 수집`")
        col2.markdown("**B. EDA**\n\n`탐색·분석`")
        col3.markdown("**C. 예측**\n\n`모델 추론`")
        col4.markdown("**D. 시각화**\n\n`결과 확인`")
        col5.markdown("**E. 검증**\n\n`정확도 평가`")
        
        st.markdown("---")
        
        st.write("### 기술 스택")
        col_fe, col_be, col_ai, col_data = st.columns(4)
        col_fe.markdown("**Frontend**\n\nStreamlit, Plotly")
        col_be.markdown("**Backend**\n\nPython, SQLite3")
        col_ai.markdown("**AI/ML**\n\nPyTorch (PatchTST)\n\nscikit-learn")
        col_data.markdown("**Data**\n\nPandas, NumPy\n\npvlib")
        
        st.markdown("---")

        st.write("### 데이터 출처")
        st.caption("실측 발전량 및 전력수요, SMP 가격은 **KPX(전력거래소)** API에서, 기상 관측/예보 데이터는 **KMA(기상청)** API에서 수집합니다.")
            
        st.write(
            "**KPX(전력거래소)**\n\n"
            "- [한국전력거래소_계통한계가격 및 수요예측(하루전 발전계획용)](https://www.data.go.kr/data/15131225/openapi.do)\n\n"
            "- [대국민 전력수급현황 공유 시스템](https://openapi.kpx.or.kr/smp_day_avg.do)\n\n"
            "**KMA(기상청)**\n\n"
            "- [기상청_지상(종관, ASOS) 일자료 조회서비스](https://www.data.go.kr/data/15057210/openapi.do)\n\n"
            "- [한국형수치예보모델(KIM) 자료 조회](https://apihub.kma.go.kr/)"
        )

    # ─── Tab 2: 메뉴별 사용법 ───
    with tab_menu:
        st.subheader("메뉴별 사용법")
        st.caption("각 메뉴를 펼쳐서 상세 사용법을 확인하세요.")
        
        with st.expander("**Option A : DB 관리**", expanded=False):
            st.write("#### Data Status")
            st.write(
                "DB에 저장된 전체 실측 데이터의 무결성을 점검합니다.\n\n"
                "시계열 누락(빠진 시간대), 컬럼별 결측치, 주요 컬럼의 불완전 행을 3가지 관점에서 검사하며, API로 채워지지 않는 결측이라면 \n\n"
                "**시간 비례 보간(최대 3건 연속)**을 적용할 수 있습니다."
            )
            st.write("#### API 데이터 수집")
            st.write(
                "시작일/종료일을 지정하여 KPX(발전량, SMP) 및 KMA(기상) 데이터를 수집합니다.\n\n"
                "실측 데이터는 최대 30일, Forecast 데이터는 과거 3일 ~ 미래 1일 범위로 수집 가능합니다."
            )
            st.info("💡 Forecast 자료는 전날 23시에 업로드됩니다. 매일 자정 이후 수집을 권장합니다.")
            st.write("#### 데이터 조회")
            st.write("실측/Forecast 테이블을 기간별로 조회할 수 있습니다.")
            st.write("#### CSV 업로드")
            st.write(
                "초기 셋팅이나 DB 복구 시, 과거 CSV 파일을 일괄 적재합니다. "
                "업로드 시 `timestamp` 컬럼이 반드시 포함되어야 하며, \n\n"
                "파생 변수(Solar_Capacity_Est 등)가 자동 계산됩니다."
            )
            st.caption("⚠️ LNG, HVDC, 기력 발전량은 API 실시간 수집이 불가합니다. 전력거래소 CSV를 별도 다운로드해 주세요.")
            st.caption("LNG, HVDC, 기력 발전량 데이터는 현재 20.01.01 - 24.12.31 까지 준비되어 있습니다.")
        
        with st.expander("**Option B : 데이터 분석 (EDA)**", expanded=False):
            st.write(
                "DB에 저장된 실측 데이터를 대상으로 탐색적 데이터 분석을 수행합니다.\n\n "
                "상단에서 분석할 피처를 선택하고, 하단에서 조회 기간을 설정하면 "
                "4개 탭에서 각각 다른 시각화를 확인할 수 있습니다."
            )
            st.write(
                "- **시계열 데이터**: 선택한 피처의 시간 흐름 그래프 (드래그로 X축 확대 가능)\n"
                "- **통계 요약**: describe() 기반의 기초 통계량\n"
                "- **상관관계 히트맵**: 피처 간 상관계수 행렬 (최소 2개 피처 필요)\n"
                "- **산점도**: X/Y축 피처를 선택하여 분포 확인 (히스토그램 포함)"
            )
        
        with st.expander("**Option C : 발전량 예측**", expanded=False):
            st.write("#### 예측 흐름")
            st.write(
                "1. 예측 대상 날짜를 선택합니다.\n"
                "2. 시스템이 자동으로 입력 데이터 상태를 점검합니다 (과거 실측 336시간 + 미래 예보 24시간).\n"
                "3. 데이터가 충분하면 [예측 실행] 버튼으로 모델을 구동합니다.\n"
                "4. 예측된 가동률이 Forecast 테이블에 저장됩니다."
            )
            st.warning(
                "⚠️ 과거 실측 336시간 + 미래 예보 24시간이 모두 채워져 있어야 예측이 가능합니다.\n\n"
                "부족하면 상세 안내에 따라 [Option A : DB 관리]에서 데이터를 보충해 주세요."
            )
        
        with st.expander("**Option D : 예측 결과 시각화**", expanded=False):
            st.write(
                "Option C에서 예측한 결과를 바탕으로 발전량, 순부하(Net Demand), SMP 등을 시각화합니다."
            )
            st.write("#### 경고 구간 설정")
            st.write(
                "[데이터 선택 / 경고 설정] 탭에서 경고 임계값을 조절할 수 있습니다. "
                "est_net_demand 기준으로 저발전/고발전 구간이 차트에 음영 처리됩니다."
            )
            st.write(
                "- **저발전 경고**: LNG 발전량이 지나치게 적은 구간 (기본 250MW 이하)\n"
                "- **고발전 경고**: LNG 발전량이 지나치게 높은 구간 (기본 750MW 이상)\n"
                "- **SMP 하한 경고**: 제주 SMP가 설정값 이하로 떨어지는 구간\n"
                "- **추가 경고**: 최저발전/최대발전 임계값을 별도 활성화 가능"
            )
        
        with st.expander("**Option E : 예측 정확도 검증**", expanded=False):
            st.write(
                "모델의 예측 결과와 실제 발전량을 비교하여 정확도를 평가합니다."
            )
            st.write(
                "- **실시간 비교**: 어제~오늘 범위의 최신 데이터로 바로 비교 (우측 상단 [최신 실측 데이터 수집] 버튼 활용)\n"
                "- **일간 비교**: 특정 날짜를 선택하여 24시간 단위로 실제 vs 예측 비교\n"
                "- **기간별 정확도 평가**: 선택한 기간 동안의 RMSE/MAE 산출 및 산점도(y=x 대각선 기준) 확인"
            )
            st.write("#### 평가지표 해석")
            st.write(
                "- **RMSE** (Root Mean Squared Error): 큰 오차에 패널티를 부여하여 극단적 예측 실패를 감지\n"
                "- **MAE** (Mean Absolute Error): 실제 발전량과 평균적으로 몇 MW 차이가 나는지 직관적으로 표현"
            )
    
    # ─── Tab 3: 예측 모델 안내 ───
    with tab_model:
        st.subheader("예측 모델 구조")
        
        st.write("#### PatchTST + Weather Attention")
        st.write(
            "본 시스템의 예측 모델은 **PatchTST**(Patch Time Series Transformer) 아키텍처에 "
            "**Weather Attention** 메커니즘을 결합한 구조입니다.\n\n"
            "시계열 데이터를 패치(patch) 단위로 분할하여 Transformer Encoder에 입력하고, "
            "미래 예보와 과거 기상 간의 어텐션을 통해 발전량을 예측합니다."
        )
        
        st.markdown("---")
        
        col_input, col_output = st.columns(2)
        with col_input:
            st.write("#### 입력 데이터")
            st.write(
                "- **과거 실측 336시간** (14일): 발전량 + 기상 관측치\n"
                "- **미래 예보 24시간** (1일): 기상 예보치"
            )
        with col_output:
            st.write("#### 출력 데이터")
            st.write(
                "- **태양광 가동률** (est_Solar_Utilization): 0~1 범위\n"
                "- **풍력 가동률** (est_Wind_Utilization): 0~1 범위\n"
                "- 가동률 × 설비용량 = 예측 발전량(MW)"
            )
        
        st.markdown("---")
        
        st.write("#### 모델 상세 스펙")
        
        spec_col1, spec_col2 = st.columns(2)
        with spec_col1:
            st.write("**태양광 모델**")
            st.write(
                "- d_model: 512\n"
                "- num_layers: 4\n"
                "- d_ff: 1024\n"
                "- patch_len: 24, stride: 12"
            )
        with spec_col2:
            st.write("**풍력 모델**")
            st.write(
                "- d_model: 128\n"
                "- num_layers: 3\n"
                "- d_ff: 512\n"
                "- patch_len: 24, stride: 24"
            )
        
        st.caption("두 모델 모두 Scaler는 Robust Scaler를 사용하였습니다.")
        
        st.markdown("---")
        
        st.write("#### 핵심 구성 요소")
        
        with st.expander("Patch Embedding + Positional Encoding"):
            st.write(
                "입력 시계열(336시간)을 patch_len 크기의 패치로 분할합니다. \n\n"
                "각 패치는 Linear 레이어를 통해 d_model 차원으로 임베딩되고, "
                "학습 가능한 Positional Encoding이 더해져 시간 순서 정보를 유지합니다."
            )
        
        with st.expander("Transformer Encoder"):
            st.write(
                "Multi-Head Self-Attention과 Feed-Forward Network로 구성된 Encoder Layer를 "
                "여러 층 쌓아 패치 간의 시간적 의존성을 학습합니다.\n\n "
                "norm_first=True (Pre-Norm) 구조를 사용하여 학습 안정성을 확보했습니다."
            )
        
        with st.expander("Weather Attention (핵심 차별점)"):
            st.write(
                "미래 기상 예보(24시간)를 Query로, 과거 기상 패치를 Key로 사용하여 "
                "\"미래 날씨와 가장 유사했던 과거 구간\"을 찾아냅니다. \n\n"
                "해당 구간의 Transformer 출력(발전 패턴)을 가중 합산하여 "
                "미래 발전량 예측의 컨텍스트로 활용합니다."
            )
        
        with st.expander("Regressor (최종 예측)"):
            st.write(
                "Weather Attention의 출력과 미래 기상 예보 벡터를 결합하여 "
                "2-Layer MLP(LeakyReLU + Dropout)로 최종 24시간 가동률을 출력합니다."
            )
    
    # ─── Tab 4: 실행 및 환경 설정 ───
    with tab_setup:
        st.subheader("실행 및 환경 설정")
        
        st.write("#### 1. 실행")
        st.write("내부 서버 구동의 어려움으로 외부 서버로 접속하고 있습니다.")
        st.write("접속링크는 매번 달라질 수 있습니다.")
           
        st.markdown("---")
        
        st.write("#### 2. 프로젝트 구조")
        st.code("""jeju_energy_project/
├── app.py                          # 메인 Streamlit 실행 파일
├── requirements.txt                # 패키지 의존성
├── .env                            # API Key (보안, git 미추적)
├── database/
│   └── jeju_energy.db              # SQLite 데이터베이스
├── models/
│   ├── best_patchtst_solar_model.pth   # 태양광 모델 가중치
│   ├── best_patchtst_wind_model.pth    # 풍력 모델 가중치
│   ├── metadata.pkl                    # 모델 메타데이터
│   └── robust_scaler_*.pkl             # 스케일러
└── utils/
    ├── api_fetchers.py             # KMA/KPX API 수집 모듈
    ├── data_pipeline.py            # 전처리 및 추론 파이프라인
    └── db_manager.py               # DB 연결 및 쿼리 관리""", language="text")
        
        st.markdown("---")
        
        st.write("#### 3. 초기 데이터 셋업 순서")
        st.write(
            "처음 실행 시 DB가 비어있으므로 아래 순서로 데이터를 채워야 합니다."
        )
        st.write(
            "1. **[Option A : DB 관리 → CSV 업로드]** 에서 과거 CSV 파일을 적재하거나\n"
            "2. **[Option A : DB 관리 → API 데이터 수집]** 에서 실측 데이터(최소 14일분)를 수집\n"
            "3. **[Option A : DB 관리 → API 데이터 수집]** 에서 Forecast 데이터(대상일)를 수집\n"
            "4. **[Option A : DB 관리 → Data Status]** 에서 결측치 확인 및 보간\n"
            "5. **[Option C : 발전량 예측]** 에서 예측 실행"
        )
        st.info("💡 이후 운영 시에는 매일 자정 이후 Forecast를 수집하고 예측을 실행하는 루틴을 권장합니다.")
