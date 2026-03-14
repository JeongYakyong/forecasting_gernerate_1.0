import builtins, sys
_orig = builtins.print
builtins.print = lambda *a, **kw: _orig(*a, **{**kw, "flush": True})

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

warnings.filterwarnings("ignore", message=".*torch.classes.*")
logging.getLogger("torch").setLevel(logging.ERROR)

# 사용자 정의 모듈 (기존 파일들)
from utils.db_manager import JejuEnergyDB
from utils.data_pipeline import (add_capacity_features, 
daily_historical_update, daily_forecast_and_predict, daily_historical_kpx, daily_historical_kma, daily_historical_kpx_smp,
run_model_prediction, prepare_model_input, daily_forecast_kpx, daily_forecast_kma)

# 필요한 함수 정리 (무결성 검사)

def check_data_status(df):
    """
    데이터프레임의 결측치(NaN) 및 시계열 연속성(누락된 시간)을 확인하는 함수
    """
    if df.empty:
        return {"status": "Empty", "missing_timestamps": 0, "nan_counts": {}}
    
    # 1. NaN 값 확인
    nan_counts = df.isna().sum()
    nan_counts = nan_counts[nan_counts > 0].to_dict()
    
    # 2. 시계열 누락 확인 (1시간 간격 기준)
    # 인덱스가 문자열이라면 datetime으로 변환
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)
        
    # 시작 시간부터 끝 시간까지 1시간 간격의 완벽한 타임라인 생성
    expected_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='1H')
    
    # 실제 인덱스와 비교하여 누락된 시간 찾기
    missing_timestamps = expected_range.difference(df.index)
    
    return {
        "status": "Warning" if len(missing_timestamps) > 0 or nan_counts else "Good",
        "missing_timestamps": len(missing_timestamps),
        "missing_dates": missing_timestamps.tolist()[:5], # 너무 많으면 5개만 보여줌
        "nan_counts": nan_counts
    }

# 1. 페이지 설정 (최상단 배치)
st.set_page_config(page_title="Jeju Energy Management System", layout="wide")

# 2. 모델 클래스 정의

# [Sub-Module 1] RevIN
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

# [Sub-Module 2] Patch-wise Weather Attention
class Patch_Weather_Attention(nn.Module):
    def __init__(self, patch_input_dim, hidden_dim):
        super().__init__()
        self.W_Q = nn.Sequential(nn.Linear(patch_input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim))
        self.W_K = nn.Sequential(nn.Linear(patch_input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim))
        self.scale_factor = 1.0 / (hidden_dim ** 0.5)

    def forward(self, future_weather_patch, past_weather_patches, transformer_output):
        # future_weather_patch: (B, future_dim)
        Q = self.W_Q(future_weather_patch).unsqueeze(1)
        K = self.W_K(past_weather_patches)
        score = torch.bmm(Q, K.transpose(1, 2)) * self.scale_factor
        attn_weights = F.softmax(score, dim=-1)
        context = torch.bmm(attn_weights, transformer_output)
        return context.squeeze(1), attn_weights

# [Main Model]
class PatchTST_Weather_Model(nn.Module):
    # 💡 하드코딩되었던 변수들을 기본값 파라미터로 뺐습니다.
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
        # 💡 DEVICE 전역 변수 대신 인자로 받도록 수정
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

# 3. 리소스 로드 함수
@st.cache_resource
def get_db():
    return JejuEnergyDB("database/jeju_energy.db")

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
    
    print("[3/6] 모델 초기화 중... (태양광/풍력 독립 세팅)")
    
    # ☀️ 태양광 모델 세팅 (Solar)
    solar_model = PatchTST_Weather_Model(
        num_features = len(metadata['features_solar']),
        seq_len=seq_len, 
        pred_len=pred_len,
        patch_len=24,      # 기본값
        stride=12,         # 태양광은 stride 12
        d_model=512,       # 태양광 d_model
        num_layers=4,      # 태양광 레이어 수
        d_ff=1024          # 태양광 d_ff
    ).to(device)
    
    # 🌬️ 풍력 모델 세팅 (Wind)
    wind_model = PatchTST_Weather_Model(
        num_features = len(metadata['features_wind']),
        seq_len=seq_len, 
        pred_len=pred_len,
        patch_len=24,      # 기본값
        stride=24,         # 💡 에러 로그 확인: 풍력은 stride가 24입니다!
        d_model=128,       # 💡 에러 로그 확인: 풍력 d_model은 256
        num_layers=3,      # 풍력 레이어 수 # 3짜리도 있음!
        d_ff=512           # 💡 에러 로그 확인: 풍력 d_ff는 512
    ).to(device)

    print("[4/6] 태양광 모델 가중치(.pth) 로딩 중...")
    solar_model.load_state_dict(torch.load('models/best_patchtst_solar_model.pth', map_location=device))
    
    print("[5/6] 풍력 모델 가중치(.pth) 로딩 중...")
    wind_model.load_state_dict(torch.load('models/best_patchtst_wind_model.pth', map_location=device))
    
    solar_model.eval()
    wind_model.eval()
    
    print("[6/6] load_assets 완료! ")
    print("메타데이터의 태양광 피처 순서:", metadata['features_solar'])
    print("메타데이터의 풍력 피처 순서:", metadata['features_wind'])
    return solar_model, wind_model, scalers, metadata, device


# 전역 객체 초기화
db = get_db()
assets = load_assets()

# 4. 사이드바 메뉴
st.sidebar.title("System Menu")
menu = st.sidebar.radio("Go to:", ["Option A: DB Management", "Option B: EDA", "Option C: Prediction",
                                   "Option D: Visualization", "Option E: Validation"])

# Option A: DB Management
if menu == "Option A: DB Management":
    st.title("DB Management & Data Status")
    
    # 요청하신 4개의 탭으로 명확하게 분리
    tab1, tab2, tab3, tab4 = st.tabs(["Data Status", "API Update", "Data Table", "CSV Upload"])
    
    # ==========================================
    # Tab 1: Data Status
    # ==========================================
    with tab1:
        header_col1, header_col2 = st.columns([8, 2])
        with header_col1:
            st.subheader("Database Health Check (전체 기간)")
        with header_col2:
            # st.rerun()을 사용하여 F5 대신 Streamlit 앱만 새로고침 되도록 처리
            if st.button("🔄 DB 새로고침", help="최신 데이터베이스 정보를 다시 불러옵니다.", width='stretch', key="refresh_db_button1"):
                st.rerun()

        with st.spinner("전체 실측 데이터를 불러오고 무결성을 검사하는 중입니다..."):
            # 인자를 비워두어 전체 데이터를 조회
            full_hist_df = db.get_historical()
        
        if not full_hist_df.empty:
            st.write("### 📊 데이터 저장 현황")
            
            # 인덱스를 datetime으로 변환
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
            
            if gap_days > 0:
                col4.metric(
                    label="업데이트 필요 (오늘 기준)", 
                    value=f"{gap_days}일 분량", 
                    delta=f"{gap_days}일 지연됨", 
                    delta_color="inverse"
                )
            elif gap_days == 0:
                col4.metric(
                    label="업데이트 필요 (오늘 기준)", 
                    value="최신 상태 ", 
                    delta="완벽합니다", 
                    delta_color="normal"
                )
            else:
                col4.metric("업데이트 필요", "미래 데이터 포함", f"+{abs(gap_days)}일")
                        
            # [추가 기능] 컬럼별 결측치 확인 및 보간(Interpolate)
            with st.expander("🔍 전체 컬럼 목록 및 결측치 확인", expanded=False):
                st.write("각 컬럼별로 비어있는(Null, NaN) 데이터의 개수를 보여줍니다.")

                # ==========================================
                # 💡 [핵심 추가] 어제 23시까지의 확정 데이터만 잘라내서 검사합니다.
                # 오늘 자정(00:00:00)보다 작은 인덱스(시간)만 가져옵니다.
                today_midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                check_df = full_hist_df[full_hist_df.index < today_midnight]
                # ==========================================
                
                # 결측치 계산 및 데이터프레임 변환
                missing_info = check_df.isnull().sum().reset_index()
                missing_info.columns = ["컬럼명 (Columns)", "결측치 개수 (Missing Values)"]
                
                # 시각적으로 편하게 볼 수 있도록 테이블 형태로 출력
                st.dataframe(missing_info, width='stretch', hide_index=True)
                total_missing = missing_info["결측치 개수 (Missing Values)"].sum()

                if total_missing > 0:
                    st.warning(f"⚠️ 총 {total_missing}개의 결측치가 발견되었습니다.")
                    # ==========================================
                    if st.button("✨ 결측치 자동 보간 (최대 3건 제한) 및 DB 적용", help="최대 3개의 연속된 결측치까지만 시간 비례로 채웁니다. 너무 긴 결측 구간은 왜곡 방지를 위해 채우지 않습니다."):
                        with st.spinner("결측치를 보간하고 DB에 저장하는 중입니다..."):
                            try:
                                # 1. 시계열 보간 적용 
                                # limit=3: 최대 3개의 연속된 결측치만 채움
                                # limit_direction='forward': 시간 순서대로(위에서 아래로) 채움
                                interpolated_df = check_df.interpolate(
                                    method='time', 
                                    limit=3, 
                                    limit_direction='forward'
                                )
                                interpolated_df.index = interpolated_df.index.strftime('%Y-%m-%d %H:%M:%S')
                                # 2. DB에 다시 저장 (UPSERT 방식으로 설정되어 있다면 기존 데이터를 덮어씁니다)
                                db.save_historical(interpolated_df)
                                
                                st.success("🎉 결측치 보간 및 DB 업데이트가 완료되었습니다! 화면을 새로고침합니다.")
                                st.rerun() # 업데이트 된 결과를 화면에 바로 반영
                            except Exception as e:
                                st.error(f"보간 처리 중 오류가 발생했습니다: {e}")
                else:
                    st.success("✅ 현재 DB에 결측치가 없습니다! 아주 깔끔한 상태입니다.")
                                    
                # ==========================================
                # 💡 [추가된 디버깅 기능] 결측치가 있는 행만 직접 확인하기!
                # ==========================================
            with st.expander("👀 결측치가 발생한 시간대 직접 확인하기", expanded=False):
                st.info("어느 시간대(timestamp)의 데이터가 비어있는지 확인해 보세요.")
                # 데이터프레임에서 하나라도 NaN(결측치)이 있는 행(row)만 필터링합니다.
                missing_rows = full_hist_df[full_hist_df.isna().any(axis=1)]
                st.dataframe(missing_rows, width='stretch')
        else:
            st.error("데이터베이스가 비어있습니다. [API Update] 탭에서 데이터를 수집해 주세요.")
            
    # ==========================================
    # Tab 2: API Update
    # ==========================================
    with tab2:
        st.subheader("Update Database via API")
        st.info("Data Status에서 확인한 결측 구간을 지정하여 데이터를 채워 넣으세요.")
        today = datetime.now().date()
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### 📈 실측 데이터 (Historical)")
            st.caption("최대 수집가능 기간은 30일입니다.")
            hist_start = st.date_input("시작일 (Historical)", today - timedelta(days=7), key='h_start')
            hist_end = st.date_input("종료일 (Historical)", today, key='h_end')
            
            # ==========================================
            # [날짜 제한 검증 로직] - 프론트엔드로 이동
            # ==========================================
            invalid_date = False
            if hist_start > hist_end:
                st.error("시작일이 종료일보다 늦을 수 없습니다.")
                invalid_date = True
            elif hist_end > today or hist_start > today:
                st.error(f"오늘({today.strftime('%Y-%m-%d')}) 이후의 미래 날짜는 조회할 수 없습니다.")
                invalid_date = True
            elif (hist_end - hist_start).days > 30:
                st.error("실측 데이터 조회는 최대 30일까지만 가능합니다.")
                invalid_date = True
            
            # 통합 업데이트 버튼
            if st.button("Update Historical", width='stretch', disabled=invalid_date):
                with st.spinner("모든 실측 데이터를 수집하고 있습니다..."):
                    # 3개 함수를 순차적으로 모두 호출하거나, 기존 통합 함수를 사용
                    daily_historical_kpx(hist_start.strftime("%Y-%m-%d"), hist_end.strftime("%Y-%m-%d"))
                    daily_historical_kma(hist_start.strftime("%Y-%m-%d"), hist_end.strftime("%Y-%m-%d"))
                    daily_historical_kpx_smp(hist_start.strftime("%Y-%m-%d"), hist_end.strftime("%Y-%m-%d"))
                    st.success("Historical data 업데이트 완료! (Data Status 탭을 눌러 확인하세요)")
            
            # 개별 API 업데이트 Expander
            with st.expander("🛠️ 개별 API 수집"):
                st.caption("필요한 특정 데이터만 개별적으로 수집할 수 있습니다.")
                
                h_start_str = hist_start.strftime("%Y-%m-%d")
                h_end_str = hist_end.strftime("%Y-%m-%d")
                
                # 버튼들도 invalid_date가 True면 눌리지 않게 처리하면 좋습니다.
                if st.button("KPX 발전량 수집", key="btn_kpx_hist", disabled=invalid_date):
                    with st.spinner("KPX 발전량 데이터를 수집 중입니다..."):
                        daily_historical_kpx(h_start_str, h_end_str) 
                        st.success("KPX 발전량 데이터 수집 완료!")
                        
                if st.button("KPX SMP 수집", key="btn_kpx_smp", disabled=invalid_date):
                    with st.spinner("KPX SMP 데이터를 수집 중입니다..."):
                        daily_historical_kpx_smp(h_start_str, h_end_str)
                        st.success("KPX SMP 데이터 수집 완료!")
                        
                if st.button("KMA 날씨 수집", key="btn_kma_hist", disabled=invalid_date):
                    with st.spinner("KMA 날씨 데이터를 수집 중입니다..."):
                        daily_historical_kma(h_start_str, h_end_str)
                        st.success("KMA 날씨 데이터 수집 완료!")
                        
        with col2:
            st.write("### 🌤️ 예보 데이터 (Forecast)")
            st.caption("주의: 과거 3일 전 ~ 미래 1일 후 (최대 30일 간격)")
            
            fore_start = st.date_input("시작일 (Forecast)", today - timedelta(days=1), key='f_start')
            fore_end = st.date_input("종료일 (Forecast)", today + timedelta(days=1), key='f_end')
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
                
            # 통합 업데이트 버튼
            if st.button("Update Forecast (All)", width='stretch', disabled=invalid_fore_date):
                with st.spinner("모든 예보 데이터를 수집하고 있습니다..."):
                    # 병합 로직이 들어있는 통합 백엔드 함수 호출
                    daily_forecast_and_predict(fore_start.strftime("%Y-%m-%d"), fore_end.strftime("%Y-%m-%d"))
                    st.success("Forecast 통합 업데이트 완료! (Data Status 탭에서 확인하세요)")
            
            # 개별 API 업데이트 Expander
            with st.expander("🛠️ 개별 API 수집"):
                st.caption("필요한 특정 예보 데이터만 개별적으로 수집할 수 있습니다.")
                
                f_start_str = fore_start.strftime("%Y-%m-%d")
                f_end_str = fore_end.strftime("%Y-%m-%d")
                
                if st.button("KPX 발전량 예보 수집", key="btn_kpx_fore_ind", disabled=invalid_fore_date):
                    with st.spinner("⚡ KPX 예보 데이터를 수집 중입니다..."):
                        daily_forecast_kpx(f_start_str, f_end_str)
                        st.success("KPX Forecast 업데이트 완료!")
                        
                if st.button("KMA 날씨 예보 수집", key="btn_kma_fore_ind", disabled=invalid_fore_date):
                    with st.spinner("🌤️ KMA 날씨 예보 데이터를 수집 중입니다..."):
                        daily_forecast_kma(f_start_str, f_end_str)
                        st.success("KMA Forecast 업데이트 완료!")
    # ==========================================
    # Tab 3: Data Table
    # ==========================================
    with tab3:
        # [추가 기능] 우측 상단에 작은 DB 새로고침 버튼 배치
        header_col1, header_col2 = st.columns([8, 2])
        with header_col1:
            st.subheader("Data Table Preview")
            st.caption("LNG, HVDC, 기력 발전량은 실시간 업데이트가 불가능 합니다. 전력거래소 csv 별도 다운로드 필요.")
        with header_col2:
            # st.rerun()을 사용하여 F5 대신 Streamlit 앱만 새로고침 되도록 처리
            if st.button("🔄 DB 새로고침", help="최신 데이터베이스 정보를 다시 불러옵니다.", width='stretch'):
                st.rerun()

        table_choice = st.radio("조회할 테이블 선택:", ["Historical Data", "Forecast Data"], horizontal=True)
        
        st.write("### 📅 조회 기간 설정")
        today = datetime.now().date()
        
        if table_choice == "Historical Data":
            max_allowed_date = today
        else:
            max_allowed_date = today + timedelta(days=2) 
            
        col1, col2 = st.columns([1, 2])
        
        with col1:
            range_option = st.selectbox(
                "기간 간편 선택",
                ["하루", "최근 1주","최근 3주", "최근 3개월", "최근 6개월", "오늘 부터 2일전, 1일후", "직접 선택 (기타)"]
            )
        
        with col2:
            if range_option == "하루":
                start_date, end_date = today - timedelta(days=1), today
            elif range_option == "최근 1주":
                start_date, end_date = today - timedelta(days=7), today
            elif range_option == "최근 3주":
                start_date, end_date = today - timedelta(days=21), today
            elif range_option == "최근 3개월":
                start_date, end_date = today - timedelta(days=90), today
            elif range_option == "최근 6개월":
                start_date, end_date = today - timedelta(days=180), today
            elif range_option == "오늘 부터 2일전, 1일후":
                start_date, end_date = today - timedelta(days=2), today + timedelta(days=1)
            else:
                date_range = st.date_input(
                    "달력에서 직접 선택", 
                    [today - timedelta(days=7), today], 
                    max_value=max_allowed_date
                )
                if len(date_range) == 2:
                    start_date, end_date = date_range
                else:
                    start_date, end_date = date_range[0], date_range[0]
    
        st.markdown("---")
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        if table_choice == "Historical Data":
            df = db.get_historical(start_str, end_str)
            if not df.empty:
                st.dataframe(df, width='stretch')
            else:
                st.warning(f"조회하신 기간({start_str} ~ {end_str})에 해당하는 실측(Historical) 데이터가 없습니다.")
        else:
            fore_df = db.get_forecast(start_str, end_str)
            if not fore_df.empty:
                st.dataframe(fore_df, width='stretch')
            else:
                st.warning(f"조회하신 기간({start_str} ~ {end_str})에 해당하는 예보(Forecast) 데이터가 없습니다.")

    # ==========================================
    # Tab 4: CSV Upload
    # ==========================================
    with tab4:
        st.subheader("과거 CSV 파일 일괄 적재 (백업/복구용)")
        st.info("💡 초기 셋팅을 하거나 DB가 손실되었을 때, 과거 CSV 파일을 올려서 한 번에 복구할 수 있습니다.")
        
        uploaded_file = st.file_uploader("과거 데이터 CSV 파일을 올려주세요", type=['csv'])
        
        if uploaded_file is not None:
            # [추가 기능] 파일을 올리자마자 바로 읽어서 결측치 미리보기 제공
            preview_df = pd.read_csv(uploaded_file)
            st.write("### 📊 업로드된 파일 결측치 분석")
            missing_preview = preview_df.isnull().sum().reset_index()
            missing_preview.columns = ["컬럼명 (Columns)", "결측치 개수 (Missing Values)"]
            
            # 결측치가 하나라도 있는지 확인하여 메세지 표시
            if missing_preview['결측치 개수 (Missing Values)'].sum() > 0:
                st.warning("⚠️ 업로드된 파일에 결측치가 존재합니다. 아래 표를 확인하세요.")
            else:
                st.success("✅ 결측치가 없는 깔끔한 데이터입니다!")
                
            st.dataframe(missing_preview, width='stretch', hide_index=True)
            
            # 여기서부터 기존 로직 수행
            if st.button("DB에 적재하기", type="primary"):
                with st.spinner("데이터를 분석하고 DB에 기록하는 중입니다..."):
                    try:
                        # 이미 읽어온 preview_df를 재활용
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
                                st.warning(f"⚠️ 업로드된 데이터에 {status_info['missing_timestamps']}개의 누락된 시간이 있거나 결측치가 포함되어 있습니다. (Data Status 탭에서 확인하세요)")
                                
                    except Exception as e:
                        st.error(f"데이터 적재 중 오류가 발생했습니다: {e}")
# --- Option B: EDA ---
elif menu == "Option B: EDA":
    st.title("Exploratory Data Analysis (EDA)")
    
    # -----------------------------------------------------------
    # [공통 영역] 기간 및 피처 설정 (컴팩트 레이아웃 적용!)
    # -----------------------------------------------------------
    # Expander를 사용해 설정을 마친 후에는 접어둘 수 있게 만듭니다.
    with st.expander("⚙️ 데이터 분석 설정 (기간 및 피처 선택)", expanded=True):
        
        # 좌우 비율을 1:2로 나누어 기간과 피처 선택을 한 줄에 배치합니다.
        col_date, col_feature = st.columns([1, 2])
        
        with col_date:
            today = datetime.now().date()
            range_option = st.selectbox(
                "📅 조회 기간",
                [ "하루", "최근 1주", "최근 1개월", "최근 3개월", "최근 6개월", "최근 1년", "직접 선택 (기타)"]
            )
            
            if range_option == "하루":
                start_date, end_date = today - timedelta(days=1), today
            elif range_option == "최근 1주":
                start_date, end_date = today - timedelta(days=7), today
            elif range_option == "최근 1개월":
                start_date, end_date = today - timedelta(days=30), today
            elif range_option == "최근 3개월":
                start_date, end_date = today - timedelta(days=90), today
            elif range_option == "최근 6개월":
                start_date, end_date = today - timedelta(days=180), today
            elif range_option == "최근 1년":
                start_date, end_date = today - timedelta(days=365), today
            else:
                date_range = st.date_input(
                    "달력에서 직접 선택", 
                    [today - timedelta(days=7), today], 
                    max_value=today
                )
                if len(date_range) == 2:
                    start_date, end_date = date_range
                else:
                    start_date, end_date = date_range[0], date_range[0]

        # 💡 [핵심] 날짜가 정해진 직후에 DB에서 데이터를 불러옵니다.
        df = db.get_historical(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        with col_feature:
            if df.empty:
                st.warning(f"선택하신 기간({start_date} ~ {end_date})에 해당하는 데이터가 없습니다.")
                selected_features = [] # 데이터가 없으면 빈 리스트 처리
            else:
                if 'real_demand' in df.columns and 'real_renew_gen' in df.columns:
                    df['real_net_demand'] = df['real_demand'] - df['real_renew_gen']
                
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                # 피처 선택창을 기간 설정창 바로 옆(오른쪽)에 띄웁니다.
                selected_features = st.multiselect(
                    "🎯 분석할 피처", 
                    options=numeric_cols, 
                    default=['real_demand', 'real_solar_gen', 'real_wind_gen', 'real_renew_gen', 'real_net_demand'] if 'real_demand' 
                    in numeric_cols else numeric_cols[:2])

    # -----------------------------------------------------------
    # 메인 컨텐츠 영역 (설정이 완료되었을 때만 탭을 보여줌)
    # -----------------------------------------------------------
    if not df.empty:
        if not selected_features:
            st.info("👆 위 설정창에서 분석을 진행할 피처를 하나 이상 선택해 주세요.")
        else:
            analysis_df = df[selected_features].copy()
            
            tab1, tab2, tab3, tab4 = st.tabs([
                "📈 시계열 데이터", 
                "📊 통계 요약", 
                "🔥 상관관계 히트맵", 
                "🌌 산점도 (Scatter)"
            ])
            
            # [Tab 1: 시계열 데이터 시각화]
            with tab1:                
                fig = px.line(
                    analysis_df, 
                    x=analysis_df.index, 
                    y=selected_features,
                    title="Time Series Data Analysis"
                )
                fig.update_layout(
                    hovermode="x unified",
                    legend_title_text="선택된 피처",
                    xaxis_title="시간 (Time)",
                    yaxis_title="수치 (Value)"
                )
                # 이전 단계에서 개선했던 깔끔한 호버 텍스트 유지!
                fig.update_traces(hovertemplate='%{y:,.2f}') 
                
                st.plotly_chart(fig, width='stretch')
                st.caption("💡 **Tip:** 우측 범례(Legend) 항목을 **더블클릭**하여 단독으로 확대해서 볼 수 있습니다.")
            # [Tab 2: 통계 요약]
            with tab2:
                st.subheader(f"Data Summary ({start_date} ~ {end_date})")
                st.dataframe(analysis_df.describe(), width='stretch')
                
            # [Tab 3: 상관관계 히트맵]
            with tab3:
                if len(selected_features) < 2:
                    st.warning("상관관계를 분석하려면 최소 2개 이상의 피처를 선택해야 합니다.")
                else:
                    corr_matrix = analysis_df.corr()
                    fig_corr = px.imshow(
                        corr_matrix,
                        text_auto=".2f",
                        aspect="auto",
                        color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1,
                        title="Correlation Matrix"
                    )
                    fig_corr.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_corr, width='stretch')

            # [Tab 4: 산점도 (Scatter Plot)]
            with tab4:
                if len(selected_features) < 2:
                    st.warning("산점도를 그리려면 위에서 최소 2개 이상의 피처를 선택해야 합니다.")
                else:
                    sc_col1, sc_col2 = st.columns(2)
                    with sc_col1:
                        x_axis = st.selectbox("X축 피처 선택:", options=selected_features, index=0)
                    with sc_col2:
                        y_axis = st.selectbox("Y축 피처 선택:", options=selected_features, index=1 if len(selected_features) > 1 else 0)
                    
                    fig_scatter = px.scatter(
                        analysis_df,
                        x=x_axis,
                        y=y_axis,
                        opacity=0.5,
                        marginal_x="histogram",
                        marginal_y="histogram",
                        title=f"{x_axis} vs {y_axis} 산점도"
                    )
                    st.plotly_chart(fig_scatter, width='stretch')


# --- Option C: Prediction & DB Storage ---
elif menu == "Option C: Prediction":
    st.title("Model Prediction & Save to DB")
    st.write("선택한 날짜의 예보 데이터를 바탕으로 태양광 및 풍력 발전 가동률을 예측하고 저장합니다.")
    
    target_date = st.date_input("예측을 수행할 대상 날짜(Target Date)", datetime.now().date())
    
    if st.button("🚀 Run Prediction", type="primary", width='stretch'):
        with st.spinner(f"{target_date} 예측을 진행 중입니다... (데이터 검증 및 모델 추론)"):
            
            success, message, input_info = run_model_prediction(
                target_date.strftime('%Y-%m-%d'), db, assets
            )
            
            # 공통 변수 가져오기
            past_hrs = input_info.get('past_hours_found', 0) if input_info else 0
            fut_hrs = input_info.get('future_hours_found', 0) if input_info else 0
            missing_cnt = input_info.get('missing_values', 0) if input_info else 0
            if success:
                st.success(message)
                
                st.write("### 🔍 모델 입력 데이터 결측치 갯수")
                col1, col2, col3 = st.columns(3)
                
                col1.metric("과거 실측 데이터 (Past)", f"{past_hrs}시간", "정상")
                col2.metric("미래 예보 데이터 (Future)", f"{fut_hrs}시간", "정상")
                # 💡 "0건" 하드코딩 대신 변수 사용 (혹시 모를 상황 대비)
                col3.metric("결측치 (Missing Values)", f"{missing_cnt}건", "완벽함")
                
                st.markdown("---")
                
                # 예측 결과 시각화
                st.write(f"### 📈 {target_date} 최종 예측 결과")
                res_df = db.get_forecast(f"{target_date} 00:00:00", f"{target_date} 23:00:00")
                
                if not res_df.empty:
                    import plotly.express as px
                    
                    # 💡 Option D와 통일감을 주기 위해 color_discrete_map으로 태양광/풍력 고유 색상 지정!
                    fig = px.line(
                        res_df, 
                        x=res_df.index, 
                        y=['est_Solar_Utilization', 'est_Wind_Utilization'],
                        title="예측 발전 가동률 (Predicted Utilization)",
                        labels={"value": "가동률 (0~1)", "timestamp": "시간 (Time)", "variable": "발전원"},
                        color_discrete_map={
                            "est_Solar_Utilization": "orange",
                            "est_Wind_Utilization": "skyblue"
                        }
                    )
                    fig.update_layout(hovermode="x unified")
                    fig.update_traces(hovertemplate='%{y:,.3f}')
                    
                    st.plotly_chart(fig, width='stretch')
            else:
                st.error(f"예측 실패: {message}")
                
                if input_info:
                    st.write("### ⚠️ 모델 입력 데이터 상태 (실패 원인 분석)")
                    col1, col2, col3 = st.columns(3)
                    
                    col1.metric("과거 실측 (Past)", f"{past_hrs}시간", "정상" if past_hrs == 336 else "부족")
                    col2.metric("미래 예보 (Future)", f"{fut_hrs}시간", "정상" if fut_hrs == 24 else "부족")
                    
                    # 1. 결측치가 원인인 경우
                    if missing_cnt > 0:
                        col3.metric("결측치 (Missing Values)", f"{missing_cnt}건", "보간 필요", delta_color="inverse")
                        st.info("💡 **해결 방법:** 왼쪽 메뉴의 [Option A: DB Management] 탭으로 이동하여 '결측치 자동 보간' 기능을 사용하거나 API를 통해 데이터를 재수집하세요.")
                        
                    # 2. 데이터 길이 자체가 짧은 경우
                    elif past_hrs < 336 or fut_hrs < 24:
                        col3.metric("결측치", "-", "데이터 길이 부족", delta_color="off")
                        st.warning(f"현재 수집된 데이터: 총 {input_info.get('total_rows', 0)}시간 (모델 필요량: {input_info.get('expected_rows', 360)}시간)")
                        st.info("💡 **해결 방법:** 모델 추론을 위한 시계열 데이터가 모자랍니다. [Option A: DB Management] 탭에서 부족한 날짜의 데이터를 수집해 주세요.")



# --- Option D: Visualization ---
elif menu == "Option D: Visualization":
    st.title("📊 Prediction Results & Net Demand Analysis")
    st.write("모델이 예측한 가동률을 바탕으로 실제 발전량과 순부하(Net Demand) 및 경제성 지표를 시각화합니다.")
    
    # 1. 조회할 타겟 날짜 선택
    target_date = st.date_input("조회할 예측 날짜 선택", datetime.now().date())
    
    # DB에서 해당 날짜의 예보(Forecast) 데이터 불러오기 (00시 ~ 23시)
    start_str = f"{target_date} 00:00:00"
    end_str = f"{target_date} 23:00:00"
    df_res = db.get_forecast(start_str, end_str)
    
    if df_res.empty or 'est_Solar_Utilization' not in df_res.columns or df_res['est_Solar_Utilization'].isnull().all():
        st.warning(f"{target_date}의 예측 데이터가 없습니다. [Option C]에서 먼저 예측을 실행해 주세요.")
    else:
        df = df_res.copy()
        
        # 인덱스가 문자열이면 datetime으로 변환
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)
            
        # ==========================================
        # 발전량 및 순부하(Net Demand) 계산
        # ==========================================
        df['est_solar_gen'] = df['est_Solar_Utilization'] * df['Solar_Capacity_Est']
        df['est_wind_gen'] = df['est_Wind_Utilization'] * df['Wind_Capacity_Est']
        
        df['est_renew_total'] = df['est_solar_gen'] + df['est_wind_gen']
        df['est_net_demand'] = df['est_demand'] - df['est_renew_total']
        
        # SMP 컬럼 확인 (이름이 smp_jeju 또는 est_smp_jeju 일 수 있으므로 방어적 코드 작성)
        smp_col = 'smp_jeju' if 'smp_jeju' in df.columns else ('est_smp_jeju' if 'est_smp_jeju' in df.columns else None)
        
        # ==========================================
        # [UI 컨트롤: 플롯 옵션 및 경고 임계값 설정]
        # ==========================================
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            plot_options = {
                'est_demand': '총 전력수요 예측 (est_demand)',
                'est_net_demand': '순부하 예측 (est_net_demand)',
                'est_solar_gen': '태양광 발전량 예측 (est_solar_gen)',
                'est_wind_gen': '풍력 발전량 예측 (est_wind_gen)',
                'est_renew_total': '총 재생에너지 발전량 (est_renew_total)'
            }
            # SMP가 있다면 시각화 옵션에 추가
            if smp_col:
                plot_options[smp_col] = f'제주 SMP 가격 ({smp_col})'
            
            selected_vars = st.multiselect(
                "📈 시각화할 데이터를 선택하세요:",
                options=list(plot_options.keys()),
                format_func=lambda x: plot_options[x],
                default=['est_demand', 'est_net_demand', 'est_solar_gen', 'est_wind_gen']
            )
        with col2:
            # 💡 [UI 개선] 경고 기준 설정을 Expander로 숨겨서 공간 확보
            with st.expander("🚨 경고 기준 설정", expanded=False):
                st.caption("est_net_demand 기준")
                warning_threshold = st.number_input("🔴 저발전 경고 (MW)", value=290, step=10, help="순부하가 이 수치보다 낮으면 빨간색으로 표시됩니다.")
                warning_threshold2 = st.number_input("🔵 고발전 경고 (MW)", value=750, step=10, help="순부하가 이 수치보다 높으면 파란색으로 표시됩니다.")
                
                # SMP 컬럼이 존재할 때만 SMP 임계값 입력창 표시
                if smp_col:
                    smp_threshold = st.number_input("🟡 SMP 하한 경고 (원)", value=10, step=10, help="SMP가 이 수치보다 낮으면 노란색으로 표시됩니다.")    
            
        # ==========================================
        # [Plotly 시각화 및 음영 처리]
        # ==========================================
        if selected_vars:
            fig = go.Figure()
            
            colors = {
                'est_demand': 'gray', 
                'est_net_demand': 'red', 
                'est_solar_gen': 'orange', 
                'est_wind_gen': 'blue',
                'est_renew_total': 'green'
            }
            if smp_col: colors[smp_col] = 'purple'
            
            # 1. 선택된 변수들의 Line Chart 추가
            for var in selected_vars:
                # 💡 [핵심] 태양광과 풍력은 보조 지표이므로 점선(dash)으로 처리합니다.
                line_style = dict(color=colors.get(var, 'black'), width=2 if var != 'est_net_demand' else 4)
                if var in ['est_solar_gen', 'est_wind_gen']:
                    line_style['dash'] = 'dash'
                    
                # SMP는 단위(원)가 다르므로 보조 Y축(오른쪽)을 쓰면 좋지만, 일단 기본 축에 그립니다.
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[var],
                    mode='lines+markers',
                    name=plot_options[var],
                    line=line_style,
                    hovertemplate='%{y:,.1f}' # 마우스 오버 시 소수점 1자리 + 천단위 콤마로 깔끔하게!
                ))
            
            # 💡 [리팩토링] 경고 구간(음영)을 그리는 헬퍼 함수
            # 이 함수 덕분에 코드를 세 번 복붙할 필요가 없어졌습니다!
            def draw_danger_zones(condition_series, fill_color, annotation_text):
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
                            fillcolor=fill_color, opacity=0.15, # 투명도를 살짝 낮춰서 여러 색이 겹쳐도 예쁘게 보이게 함
                            layer="below", line_width=0,
                            annotation_text=annotation_text, annotation_position="top left"
                        )

            # 2. 헬퍼 함수를 이용해 3가지 경고 음영 칠하기
            draw_danger_zones(df['est_net_demand'] < warning_threshold, "red", "LNG발전량 적음 🚨")
            draw_danger_zones(df['est_net_demand'] > warning_threshold2, "blue", "LNG발전량 높음 🚨")
            
            if smp_col:
                draw_danger_zones(df[smp_col] < smp_threshold, "orange", "SMP 매우 낮음 📉")

            # 3. 차트 레이아웃 디자인 설정
            fig.update_layout(
                title=f"{target_date} 전력수급 및 재생에너지 예측 결과",
                xaxis_title="시간 (Time)",
                yaxis_title="발전량 / 전력량 (MW)",
                hovermode="x unified", 
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, width='stretch')
            
            # ==========================================
            # [데이터 테이블 표시]
            # ==========================================
            st.write("### 📋 상세 데이터 테이블")
            display_cols = ['est_demand', 'est_solar_gen', 'est_wind_gen', 'est_renew_total', 'est_net_demand']
            if smp_col: display_cols.append(smp_col)
            
            # 스타일 함수 정의 (Net Demand와 SMP 동시 강조)
            def highlight_warnings(row):
                styles = [''] * len(row)
                # Net Demand 빨간색 경고
                if row['est_net_demand'] < warning_threshold:
                    idx = row.index.get_loc('est_net_demand')
                    styles[idx] = 'background-color: #ffcccc'
                # SMP 주황색 경고
                if smp_col and row[smp_col] < smp_threshold:
                    idx = row.index.get_loc(smp_col)
                    styles[idx] = 'background-color: #ffe5b4'
                return styles

            st.dataframe(df[display_cols].style.apply(highlight_warnings, axis=1), width='stretch')
            
# --- Option E: Validation ---
elif menu == "Option E: Validation":
    st.title("✅ Model Validation & Accuracy")
    st.write("예측 모델의 결과와 실제 발전량을 비교하여 모델의 정확도를 평가합니다.")
    
    # ==========================================
    # 1. 실시간 실측 데이터 업데이트 트리거 (가볍게 수정!)
    # ==========================================
    st.markdown("---")
    header_col1, header_col2 = st.columns([7, 3])
    with header_col1:
        st.write("### 🔄 실시간 데이터 최신화")
        st.caption("어제부터 현재 시간까지의 KPX 실측 데이터만 가볍게 업데이트하여 실시간 오차를 확인합니다.")
    with header_col2:
        if st.button("⚡ 실시간 데이터 가져오기", width='stretch', type="primary"):
            with st.spinner("최신 실측 데이터를 수집하고 있습니다..."):
                today_str = datetime.now().strftime("%Y-%m-%d")
                yesterday_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
                
                # 💡 [핵심] 어제~오늘 데이터만 가볍게 호출!
                daily_historical_kpx(yesterday_str, today_str)
                st.success("업데이트 완료! [실시간 비교] 탭에서 결과를 확인하세요.")
                
    st.markdown("---")
    
    # ==========================================
    # 2. 탭 구성 (실시간 / 일간 / 주간)
    # ==========================================
    tab1, tab2, tab3 = st.tabs(["⚡ 실시간 비교 (Real-time)", "📊 일간 비교 (Daily)", "📈 주간 정확도 평가 (Weekly)"])
    
    # 공통 시각화 함수 (Tab 1, Tab 2에서 재사용)
    def plot_actual_vs_pred(df, date_title):
        source_choice = st.radio("발전원 선택:", ["태양광 (Solar)", "풍력 (Wind)"], horizontal=True, key=f"radio_{date_title}")
        
        fig = go.Figure()
        if source_choice == "태양광 (Solar)":
            real_col, est_col = 'real_solar_gen', 'est_solar_gen'
            color_real, color_est = "darkorange", "gold"
        else:
            real_col, est_col = 'real_wind_gen', 'est_wind_gen'
            color_real, color_est = "darkblue", "skyblue"
        
        # 실제값 라인 (실선)
        fig.add_trace(go.Scatter(x=df.index, y=df[real_col], name="실제 발전량 (Actual)", line=dict(color=color_real, width=3)))
        # 예측값 라인 (점선)
        fig.add_trace(go.Scatter(x=df.index, y=df[est_col], name="예측 발전량 (Predicted)", line=dict(color=color_est, width=3, dash='dash')))
        
        # 오차(Error) 영역 바 차트로 추가
        df['error'] = df[est_col] - df[real_col]
        fig.add_trace(go.Bar(x=df.index, y=df['error'], name="오차 (Error)", marker_color="red", opacity=0.3))
        
        fig.update_layout(
            title=f"{date_title} {source_choice} 실제 vs 예측 비교",
            hovermode="x unified",
            yaxis_title="발전량 (MW)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_traces(hovertemplate='%{y:,.1f}')
        st.plotly_chart(fig, width='stretch')

    # -----------------------------------------------------------
    # [Tab 1: 실시간 비교 (Real-time)]
    # -----------------------------------------------------------
    with tab1:
        st.subheader("Real-time Actual vs Predicted (Yesterday & Today)")
        st.caption("어제부터 현재 시간까지 들어온 가장 최근의 데이터를 바로 비교합니다.")
        st.caption("더 긴 기간의 데이터는 Option A에서 다운받으세요.")
        today_date = datetime.now().date()
        yesterday_date = today_date - timedelta(days=1)
        
        rt_start_str = f"{yesterday_date} 00:00:00"
        rt_end_str = f"{today_date} 23:59:59"
        
        rt_hist_df = db.get_historical(rt_start_str, rt_end_str)
        rt_fore_df = db.get_forecast(rt_start_str, rt_end_str)
        
        if rt_hist_df.empty or rt_fore_df.empty:
            st.warning("실시간 비교를 위한 데이터가 부족합니다. 위의 [실시간 데이터 가져오기] 버튼을 눌러주세요.")
        else:
            # suffixes를 명시적으로 지정하여 이름 충돌 방지
            rt_val_df = pd.merge(rt_hist_df, rt_fore_df, left_index=True, right_index=True, how='inner', suffixes=('', '_fore'))
            
            if rt_val_df.empty:
                st.warning("시간대가 일치하는 데이터가 없어 비교할 수 없습니다.")
            else:
                # Capacity 컬럼 이름이 겹치면 _fore 가 붙음. 방어적 코딩으로 처리.
                cap_solar_col = 'Solar_Capacity_Est_fore' if 'Solar_Capacity_Est_fore' in rt_val_df.columns else 'Solar_Capacity_Est'
                cap_wind_col = 'Wind_Capacity_Est_fore' if 'Wind_Capacity_Est_fore' in rt_val_df.columns else 'Wind_Capacity_Est'
                
                rt_val_df['est_solar_gen'] = rt_val_df['est_Solar_Utilization'] * rt_val_df[cap_solar_col]
                rt_val_df['est_wind_gen'] = rt_val_df['est_Wind_Utilization'] * rt_val_df[cap_wind_col]
                
                plot_actual_vs_pred(rt_val_df, "어제~오늘 실시간")

    # -----------------------------------------------------------
    # [Tab 2: 일간 비교 (Daily)]
    # -----------------------------------------------------------
    with tab2:
        st.subheader("Daily Actual vs Predicted")
        target_date = st.date_input("비교할 날짜를 선택하세요", datetime.now().date() - timedelta(days=1), key="val_daily_date")
        
        start_str = f"{target_date} 00:00:00"
        end_str = f"{target_date} 23:59:59"
        
        hist_df = db.get_historical(start_str, end_str)
        fore_df = db.get_forecast(start_str, end_str)
        
        if hist_df.empty or fore_df.empty:
            st.warning(f"{target_date}의 실측 데이터 또는 예측 데이터가 부족합니다.")
        else:
            val_df = pd.merge(hist_df, fore_df, left_index=True, right_index=True, how='inner', suffixes=('', '_fore'))
            
            if val_df.empty:
                st.warning("해당 날짜에 시간대가 일치하는 데이터가 없습니다.")
            else:
                cap_solar_col = 'Solar_Capacity_Est_fore' if 'Solar_Capacity_Est_fore' in val_df.columns else 'Solar_Capacity_Est'
                cap_wind_col = 'Wind_Capacity_Est_fore' if 'Wind_Capacity_Est_fore' in val_df.columns else 'Wind_Capacity_Est'
                
                val_df['est_solar_gen'] = val_df['est_Solar_Utilization'] * val_df[cap_solar_col]
                val_df['est_wind_gen'] = val_df['est_Wind_Utilization'] * val_df[cap_wind_col]
                
                plot_actual_vs_pred(val_df, str(target_date))

    # -----------------------------------------------------------
    # [Tab 3: 주간 정확도 (Weekly Metrics & Plot)]
    # -----------------------------------------------------------
    with tab3:
        st.subheader("Weekly Model Accuracy Metrics")
        
        today = datetime.now().date()
        week_start = today - timedelta(days=7)
        
        date_range = st.date_input("정확도를 평가할 기간을 선택하세요 (최대 14일 권장)", [week_start, today], max_value=today, key="val_weekly_date")
        
        if len(date_range) == 2:
            w_start, w_end = date_range
            
            w_hist_df = db.get_historical(w_start.strftime("%Y-%m-%d"), w_end.strftime("%Y-%m-%d 23:59:59"))
            w_fore_df = db.get_forecast(w_start.strftime("%Y-%m-%d"), w_end.strftime("%Y-%m-%d 23:59:59"))
            
            if w_hist_df.empty or w_fore_df.empty:
                st.warning("선택한 기간의 데이터가 부족합니다.")
            else:
                w_val_df = pd.merge(w_hist_df, w_fore_df, left_index=True, right_index=True, how='inner', suffixes=('', '_fore'))
                
                cap_solar_col = 'Solar_Capacity_Est_fore' if 'Solar_Capacity_Est_fore' in w_val_df.columns else 'Solar_Capacity_Est'
                cap_wind_col = 'Wind_Capacity_Est_fore' if 'Wind_Capacity_Est_fore' in w_val_df.columns else 'Wind_Capacity_Est'
                
                w_val_df['est_solar_gen'] = w_val_df['est_Solar_Utilization'] * w_val_df[cap_solar_col]
                w_val_df['est_wind_gen'] = w_val_df['est_Wind_Utilization'] * w_val_df[cap_wind_col]
                
                w_val_df = w_val_df.dropna(subset=['real_solar_gen', 'est_solar_gen', 'real_wind_gen', 'est_wind_gen'])
                
                if len(w_val_df) > 0:
                    st.write("### 🎯 평가지표 (Evaluation Metrics)")
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
                    
                    # 💡 [요구사항 추가] 주간 흐름을 한눈에 보는 라인 차트
                    st.write("### 📉 주간 실제 vs 예측 흐름 (Time Series)")
                    st.caption("전체 기간 동안의 모델 예측이 실제 트렌드를 잘 따라가는지 확인합니다.")
                    
                    fig_w_line = px.line(
                        w_val_df, 
                        x=w_val_df.index, 
                        y=['real_solar_gen', 'est_solar_gen', 'real_wind_gen', 'est_wind_gen'],
                        labels={"value": "발전량 (MW)", "index": "시간", "variable": "항목"}
                    )
                    fig_w_line.update_layout(hovermode="x unified")
                    st.plotly_chart(fig_w_line, width='stretch')
                    
                    st.markdown("---")
                    
                    # 에러 분포 산점도
                    st.write("### 🌌 실제값 vs 예측값 산점도 (Scatter Plot)")
                    st.caption("점이 대각선(y=x)에 가깝게 모여 있을수록 정확한 예측입니다.")
                    
                    sc_col1, sc_col2 = st.columns(2)
                    with sc_col1:
                        fig_s = px.scatter(w_val_df, x='real_solar_gen', y='est_solar_gen', opacity=0.5, title="태양광 (Solar)")
                        fig_s.add_shape(type="line", line=dict(dash="dash", color="gray"), x0=0, y0=0, x1=w_val_df['real_solar_gen'].max(), y1=w_val_df['real_solar_gen'].max())
                        st.plotly_chart(fig_s, width='stretch')
                        
                    with sc_col2:
                        fig_w = px.scatter(w_val_df, x='real_wind_gen', y='est_wind_gen', opacity=0.5, title="풍력 (Wind)", color_discrete_sequence=['skyblue'])
                        fig_w.add_shape(type="line", line=dict(dash="dash", color="gray"), x0=0, y0=0, x1=w_val_df['real_wind_gen'].max(), y1=w_val_df['real_wind_gen'].max())
                        st.plotly_chart(fig_w, width='stretch')
