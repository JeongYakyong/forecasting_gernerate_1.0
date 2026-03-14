from utils.db_manager import JejuEnergyDB
from utils.api_fetchers import (
    fetch_kpx_past,
    fetch_kpx_future,
    fetch_kpx_historical,
    fetch_kma_past_asos,
    fetch_kma_future_ncm
)
import pandas as pd
import pvlib
import numpy as np
import torch
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv


load_dotenv()
KMA_KEY = os.getenv("KMA_API_KEY")
KPX_KEY = os.getenv("KPX_API_KEY")

# ============================================================================
# 시나리오 1: 초기 데이터 수집 (5년치)
# ============================================================================

def add_capacity_features(df):
    df = df.copy()
    
    # 1. Capacity 추정 (Rolling Cummax 방식)
    if 'real_solar_gen' in df.columns:
        #df['Solar_Capacity_Est'] = df['real_solar_gen'].expanding().max()
        df['Solar_Capacity_Est'] = df['real_solar_gen'].rolling(720, min_periods=1).max()
        # 또는 720시간 윈도우 유지하려면:

    
    if 'real_wind_gen' in df.columns:
        #df['Wind_Capacity_Est'] = df['real_wind_gen'].expanding().max()
        df['Wind_Capacity_Est'] = df['real_wind_gen'].rolling(720, min_periods=1).max()
    # 2. Utilization 계산
    if 'real_solar_gen' in df.columns and 'Solar_Capacity_Est' in df.columns:
        df['Solar_Utilization'] = df['real_solar_gen'] / df['Solar_Capacity_Est']
        df['Solar_Utilization'] = df['Solar_Utilization'].fillna(0)
    
    if 'real_wind_gen' in df.columns and 'Wind_Capacity_Est' in df.columns:
        df['Wind_Utilization'] = df['real_wind_gen'] / df['Wind_Capacity_Est']
        df['Wind_Utilization'] = df['Wind_Utilization'].fillna(0)
    
    return df


# ============================================================================
# 시나리오 2: 실측 업데이트(사용자가 기간 설정)
# ============================================================================
def daily_historical_update(start_date, end_date):
    """
    실측 데이터 업데이트 (최대 30일 제한, 미래 날짜 제한, 독립적 API 호출)
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    # 시간 정보를 제외한 오늘 날짜 (0시 0분 0초 기준)
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    # ==========================================
    # [날짜 제한 검증 로직]
    # ==========================================
    # 1. 시작일이 종료일보다 늦은 경우 방지
    if start_dt > end_dt:
        print("[Error] 시작일이 종료일보다 늦을 수 없습니다.")
        return
        
    # 2. 미래 날짜 선택 제한 (오늘보다 앞선 날짜만 가능)
    if end_dt > today or start_dt > today:
        print(f"[Error] 오늘({today.strftime('%Y-%m-%d')}) 이후의 미래 날짜는 조회할 수 없습니다.")
        return
        
    # 3. 최대 30일 간격 제한
    if (end_dt - start_dt).days > 30:
        print("[Error] 실측 데이터 조회는 최대 30일까지만 가능합니다.")
        return
    # ==========================================
        
    print(f"일일 실측 업데이트 시작: {start_date} ~ {end_date}")
    
    db = JejuEnergyDB()
    
    # 1. API 각각 독립 호출
    kpx_data = pd.DataFrame()
    try:
        kpx_data = fetch_kpx_past(start_date, end_date)
    except Exception as e:
        print(f"[Fail] KPX 실측 API 호출 실패: {e}")

    asos_data = pd.DataFrame()
    try:
        asos_data = fetch_kma_past_asos(
            start_date.replace('-', ''), 
            end_date.replace('-', ''), 
            KMA_KEY
        )
    except Exception as e:
        print(f"[Fail] KMA 실측 기상 API 호출 실패: {e}")

    smp_data = pd.DataFrame()
    try:
        smp_data = fetch_kpx_historical(start_date, end_date, KPX_KEY)
    except Exception as e:
        print(f"[Fail] KPX SMP API 호출 실패: {e}")

    # 2. 병합 준비 (성공해서 데이터가 있는 것만 모음)
    df_list = [df for df in [kpx_data, asos_data, smp_data] if not df.empty]
    
    if not df_list:
        print("[Fail] 수집된 실측 데이터가 없어 업데이트를 종료합니다.")
        db.close()
        return

    try:
        # 3. 데이터 병합
        actual_df = pd.concat(df_list, axis=1)
        
        # 4. Capacity 및 Utilization 계산
        lookback_date = (start_dt - timedelta(days=30)).strftime("%Y-%m-%d")
        historical_for_calc = db.get_historical(lookback_date, end_date)
        
        if not historical_for_calc.empty:
            combined = pd.concat([historical_for_calc, actual_df])
            combined = combined[~combined.index.duplicated(keep='last')]
            
            combined['Solar_Capacity_Est'] = combined['real_solar_gen'].rolling(720, min_periods=1).max()
            combined['Wind_Capacity_Est'] = combined['real_wind_gen'].rolling(720, min_periods=1).max()
            
            actual_df = combined.loc[actual_df.index]
        else:
            if 'real_solar_gen' in actual_df.columns:
                actual_df['Solar_Capacity_Est'] = actual_df['real_solar_gen'].rolling(720, min_periods=1).max()
            if 'real_wind_gen' in actual_df.columns:
                actual_df['Wind_Capacity_Est'] = actual_df['real_wind_gen'].rolling(720, min_periods=1).max()
                
        if 'real_solar_gen' in actual_df.columns and 'Solar_Capacity_Est' in actual_df.columns:
            actual_df['Solar_Utilization'] = actual_df['real_solar_gen'] / actual_df['Solar_Capacity_Est']
        if 'real_wind_gen' in actual_df.columns and 'Wind_Capacity_Est' in actual_df.columns:
            actual_df['Wind_Utilization'] = actual_df['real_wind_gen'] / actual_df['Wind_Capacity_Est']
        
        # 5. DB 저장
        db.save_historical(actual_df)
        print(f"[OK] 실측 데이터 {len(actual_df)}행 업데이트 완료")
        
    except Exception as e:
        print(f"[Fail] 실측 데이터 병합 또는 저장 실패: {e}")
        
    db.close()

def daily_historical_kpx(start_date, end_date):
    """ KPX 실측 발전량 데이터 수집 및 파생변수 계산 """
    print(f"KPX 발전량 실측 업데이트 시작: {start_date} ~ {end_date}")
    db = JejuEnergyDB()
    
    try:
        actual_df = fetch_kpx_past(start_date, end_date)
        if actual_df.empty:
            print("[Fail] 수집된 KPX 데이터가 없습니다.")
            return

        # Capacity 및 Utilization 계산을 위해 과거 30일 데이터 불러오기
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        lookback_date = (start_dt - timedelta(days=30)).strftime("%Y-%m-%d")
        historical_for_calc = db.get_historical(lookback_date, end_date)
        
        if not historical_for_calc.empty:
            combined = pd.concat([historical_for_calc, actual_df])
            combined = combined[~combined.index.duplicated(keep='last')]
            
            if 'real_solar_gen' in combined.columns:
                combined['Solar_Capacity_Est'] = combined['real_solar_gen'].rolling(720, min_periods=1).max()
            if 'real_wind_gen' in combined.columns:
                combined['Wind_Capacity_Est'] = combined['real_wind_gen'].rolling(720, min_periods=1).max()
            
            actual_df = combined.loc[actual_df.index]
        else:
            if 'real_solar_gen' in actual_df.columns:
                actual_df['Solar_Capacity_Est'] = actual_df['real_solar_gen'].rolling(720, min_periods=1).max()
            if 'real_wind_gen' in actual_df.columns:
                actual_df['Wind_Capacity_Est'] = actual_df['real_wind_gen'].rolling(720, min_periods=1).max()
                
        if 'real_solar_gen' in actual_df.columns and 'Solar_Capacity_Est' in actual_df.columns:
            actual_df['Solar_Utilization'] = actual_df['real_solar_gen'] / actual_df['Solar_Capacity_Est']
        if 'real_wind_gen' in actual_df.columns and 'Wind_Capacity_Est' in actual_df.columns:
            actual_df['Wind_Utilization'] = actual_df['real_wind_gen'] / actual_df['Wind_Capacity_Est']
        
        db.save_historical(actual_df)
        print(f"[OK] KPX 발전량 데이터 {len(actual_df)}행 업데이트 완료")
        
    except Exception as e:
        print(f"[Fail] KPX 실측 데이터 처리 실패: {e}")
    finally:
        db.close()

def daily_historical_kma(start_date, end_date):
    """ KMA 종관기상관측(ASOS) 실측 데이터 수집 """
    print(f"KMA 기상 실측 업데이트 시작: {start_date} ~ {end_date}")
    db = JejuEnergyDB()
    
    try:
        asos_data = fetch_kma_past_asos(
            start_date.replace('-', ''), 
            end_date.replace('-', ''), 
            KMA_KEY  # 전역 변수나 설정에서 가져와야 함
        )
        if not asos_data.empty:
            db.save_historical(asos_data)
            print(f"[OK] KMA 기상 데이터 {len(asos_data)}행 업데이트 완료")
        else:
            print("[Fail] 수집된 KMA 데이터가 없습니다.")
            
    except Exception as e:
        print(f"[Fail] KMA 기상 데이터 처리 실패: {e}")
    finally:
        db.close()

def daily_historical_kpx_smp(start_date, end_date):
    """ KPX SMP 실측 가격 데이터 수집 """
    print(f"KPX SMP 실측 업데이트 시작: {start_date} ~ {end_date}")
    db = JejuEnergyDB()
    
    try:
        smp_data = fetch_kpx_historical(start_date, end_date, KPX_KEY) # 전역 변수 필요
        if not smp_data.empty:
            db.save_historical(smp_data)
            print(f"[OK] KPX SMP 데이터 {len(smp_data)}행 업데이트 완료")
        else:
            print("[Fail] 수집된 SMP 데이터가 없습니다.")
            
    except Exception as e:
        print(f"[Fail] KPX SMP 데이터 처리 실패: {e}")
    finally:
        db.close()
        

# ============================================================================
# 시나리오 3: 예측 정보 업데이트 (현재 기준 과거는 3일전, 미래는 1일 후 까지)
# ============================================================================
def daily_forecast_and_predict(start_date, end_date):
    """ 예보 데이터 통합 업데이트 (KPX + KMA 병합 처리) """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    print(f"🔄 통합 예보 업데이트 시작: {start_date} ~ {end_date}")
    db = JejuEnergyDB()
    current_dt = start_dt
    
    while current_dt <= end_dt:
        target_date = current_dt.strftime("%Y-%m-%d")
        print(f"[{target_date}] 데이터 수집 시작...")
        
        kpx_data, kma_data = None, None
        
        # 1. KPX 수집
        try:
            kpx_forecast = fetch_kpx_future(target_date, KPX_KEY)
            if not kpx_forecast.empty: kpx_data = kpx_forecast
        except Exception as e:
            print(f"  └── [Error] KPX API 실패: {e}")
            
        # 2. KMA 수집
        try:
            # 좌표 33.50, 126.53은 예시입니다.
            ncm_forecast = fetch_kma_future_ncm(33.50, 126.53, KMA_KEY, target_date)
            if not ncm_forecast.empty: kma_data = ncm_forecast
        except Exception as e:
            print(f"  └── [Error] KMA API 실패: {e}")
            
        # 3. 병합 및 저장
        if kpx_data is not None and kma_data is not None:
            combined_data = pd.merge(kpx_data, kma_data, left_index=True, right_index=True, how='outer')
            db.save_forecast(combined_data, auto_add_capacity=True)
            print(f"  └── ✅ KPX & KMA 병합 데이터 DB 저장 완료")
        elif kpx_data is not None:
            db.save_forecast(kpx_data, auto_add_capacity=True)
            print(f"  └── ✅ KPX 데이터만 DB 저장 완료")
        elif kma_data is not None:
            db.save_forecast(kma_data, auto_add_capacity=True)
            print(f"  └── ✅ KMA 데이터만 DB 저장 완료")
        else:
            print(f"  └── ❌ 저장할 데이터가 없습니다.")
            
        current_dt += timedelta(days=1)
        
    db.close()
    print("[OK] 전체 예보 통합 업데이트 완료")


def daily_forecast_kpx(start_date, end_date):
    """ KPX 예보 데이터만 단독 수집 """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    print(f"⚡ KPX 단독 예보 업데이트 시작: {start_date} ~ {end_date}")
    db = JejuEnergyDB()
    current_dt = start_dt
    
    while current_dt <= end_dt:
        target_date = current_dt.strftime("%Y-%m-%d")
        try:
            kpx_forecast = fetch_kpx_future(target_date, KPX_KEY)
            if not kpx_forecast.empty:
                db.save_forecast(kpx_forecast, auto_add_capacity=True)
                print(f"[{target_date}] ✅ KPX 단독 저장 완료")
            else:
                print(f"[{target_date}] ❌ KPX 데이터 없음")
        except Exception as e:
            print(f"[{target_date}] [Error] KPX API 실패: {e}")
            
        current_dt += timedelta(days=1)
    db.close()


def daily_forecast_kma(start_date, end_date):
    """ KMA 예보 데이터만 단독 수집 """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    print(f"🌤️ KMA 단독 예보 업데이트 시작: {start_date} ~ {end_date}")
    db = JejuEnergyDB()
    current_dt = start_dt
    
    while current_dt <= end_dt:
        target_date = current_dt.strftime("%Y-%m-%d")
        try:
            ncm_forecast = fetch_kma_future_ncm(33.50, 126.53, KMA_KEY, target_date)
            if not ncm_forecast.empty:
                db.save_forecast(ncm_forecast, auto_add_capacity=True)
                print(f"[{target_date}] ✅ KMA 단독 저장 완료")
            else:
                print(f"[{target_date}] ❌ KMA 데이터 없음")
        except Exception as e:
            print(f"[{target_date}] [Error] KMA API 실패: {e}")
            
        current_dt += timedelta(days=1)
    db.close()
    
# ============================================================================
# 유틸리티: 모델 입력 데이터 준비
# ============================================================================
def prepare_model_input(df):
    """
    모델 입력에 필요한 파생 변수(시간 주기성, 태양광/대기 일사량 등) 생성
    """
    if df.empty:
        return df
    
    df = df.copy()
    if df.index.name == 'timestamp':
        df = df.reset_index()
    
    # timestamp 파싱 (다양한 포맷에 대응하도록 errors='coerce' 사용)
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
    
    # 시간 피처
    df['Hour_sin'] = np.sin(2 * np.pi * df['timestamp_dt'].dt.hour / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['timestamp_dt'].dt.hour / 24)
    df['Year_sin'] = np.sin(2 * np.pi * df['timestamp_dt'].dt.dayofyear / 365)
    df['Year_cos'] = np.cos(2 * np.pi * df['timestamp_dt'].dt.dayofyear / 365)
    
    # 1. Extra Radiation 및 태양 고도각
    lat, lon = 33.3615, 126.5292
    times = pd.DatetimeIndex(df['timestamp_dt']).tz_localize('Asia/Seoul')

    df['Solar_Elevation'] = pvlib.solarposition.get_solarposition(times, lat, lon)['elevation'].values

    dni_extra = pvlib.irradiance.get_extra_radiation(times).values
    df['Extra_Radiation'] = dni_extra * np.sin(np.radians(df['Solar_Elevation']))
    df['Extra_Radiation'] = df['Extra_Radiation'].clip(lower=0)

    # 2. Solar Elevation 스케일링
    min_el = df['Solar_Elevation'].min()
    max_el = df['Solar_Elevation'].max()
    # 0으로 나누는 오류 방지
    if max_el != min_el:
        df['Solar_Elevation_scaled'] = (df['Solar_Elevation'] - min_el) / (max_el - min_el)
    else:
        df['Solar_Elevation_scaled'] = 0.0
    
    # smp_gap
    if 'smp_jeju' in df.columns and 'smp_land' in df.columns:
        df['smp_gap'] = df['smp_jeju'] - df['smp_land']
    
    df = df.set_index('timestamp')
    df = df.drop(columns=['timestamp_dt', 'Solar_Elevation'], errors='ignore')
    
    return df


def run_model_prediction(target_date, db, assets):
    solar_model, wind_model, scalers, metadata, device = assets 
    
    seq_len = metadata['SEQ_LEN']   # 과거 336
    pred_len = metadata['PRED_LEN'] # 미래 24
    total_len = seq_len + pred_len  # 총 360
    
    features_solar = metadata['features_solar']
    features_wind = metadata['features_wind']
    
    future_features_solar = [col for col in features_solar if 'Utilization' not in col]
    future_features_wind = [col for col in features_wind if 'Utilization' not in col]
    
    # DB 데이터 조회
    target_dt = datetime.strptime(target_date, "%Y-%m-%d")
    start_dt = target_dt - timedelta(hours=seq_len)
    start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    end_str = (target_dt.replace(hour=23)).strftime("%Y-%m-%d %H:%M:%S")
    
    df = db.get_historical_and_forecast(start_str, end_str)
    
    input_info = {
        "total_rows": len(df),
        "expected_rows": total_len,
        "missing_values": 0,
        "past_hours_found": 0,
        "future_hours_found": 0
    }
    
    if df.empty or len(df) < total_len:
        return False, f"[{target_date}] 데이터 길이가 부족합니다. (필요: {total_len}, 현재: {len(df)})", input_info
        
    df = prepare_model_input(df)
    
    # 1. 과거(336h)와 미래(24h) 분리
    past_df = df.iloc[:seq_len]
    future_df = df.iloc[seq_len:total_len]
    
    # 2. 모델이 "실제로 사용하는" 데이터만 콕 집어서 결측치 확인!
    # 통합 피처 리스트 (중복 제거)
    used_features = list(set(future_features_solar + future_features_wind))
    target_cols = ['Solar_Utilization', 'Wind_Utilization']
    
    # [핵심] 과거는 피처+정답값 모두 검사, 미래는 피처만 검사 (미래 정답값은 당연히 없으므로 제외)
    past_missing = past_df[used_features + target_cols].isnull().sum().sum()
    future_missing = future_df[used_features].isnull().sum().sum()
    real_missing_cnt = int(past_missing + future_missing)
    
    input_info["missing_values"] = real_missing_cnt
    input_info["past_hours_found"] = len(past_df)
    input_info["future_hours_found"] = len(future_df)
    
    # 3. 🚨 진짜 결측치가 있으면 단호하게 에러 반환! (보간 절대 안 함)
    if real_missing_cnt > 0:
        return False, f"모델 입력 데이터에 {real_missing_cnt}개의 실제 결측치가 존재합니다. [Option A]에서 데이터를 점검하세요.", input_info
        
    # 4. 검증을 무사히 통과했다면, 모델이 쓰지 않는 껍데기 컬럼(예: 미래 실측값)의 NaN만 
    # 파이토치 텐서 변환 시 에러가 나지 않도록 0으로 덮어줌 (데이터 왜곡 없음)
    df = df.fillna(0)
    
    # ==========================================
    # 태양광 / 풍력 독립 스케일링 (정상일 때만 진행)
    # ==========================================
    scaler_solar = scalers['solar']
    scaler_wind = scalers['wind']
    
    df_solar = df.copy()
    df_solar[future_features_solar] = scaler_solar.transform(df_solar[future_features_solar])
    
    df_wind = df.copy()
    df_wind[future_features_wind] = scaler_wind.transform(df_wind[future_features_wind])
    
    def create_batch_from_scaled(df_target, future_features_list, target_col):
        past_df = df_target.iloc[:seq_len]
        future_df = df_target.iloc[seq_len:total_len]
        
        past_numeric = past_df[future_features_list].values
        past_y = past_df[[target_col]].values 
        future_numeric = future_df[future_features_list].values
        
        batch = {
            'past_numeric': torch.FloatTensor(past_numeric).unsqueeze(0),
            'past_y': torch.FloatTensor(past_y).unsqueeze(0),            
            'future_numeric': torch.FloatTensor(future_numeric).unsqueeze(0)
        }
        return batch

    # ==========================================
    # 배치 생성 및 모델 추론
    # ==========================================
    try:
        # ☀️ 태양광 추론
        solar_batch = create_batch_from_scaled(df_solar, future_features_solar, 'Solar_Utilization')
        with torch.no_grad():
            pred_solar = solar_model(solar_batch, device=device).squeeze().cpu().numpy()
            
        # 🌬️ 풍력 추론
        wind_batch = create_batch_from_scaled(df_wind, future_features_wind, 'Wind_Utilization')
        with torch.no_grad():
            pred_wind = wind_model(wind_batch, device=device).squeeze().cpu().numpy()
            
    except Exception as e:
        return False, f"추론 중 에러 발생: {e}", input_info
        
    # 클리핑 (0 ~ 1 사이로 맞춤)
    pred_solar = np.clip(pred_solar, a_min=0.0, a_max=1.0)
    pred_wind = np.clip(pred_wind, a_min=0.0, a_max=1.0)
    
    # 결과 DB 저장
    target_timestamps = df.index[seq_len:total_len]
    pred_df = pd.DataFrame({
        'timestamp': target_timestamps,
        'est_Solar_Utilization': pred_solar,
        'est_Wind_Utilization': pred_wind
    })
    
    updated_rows = db.update_forecast_predictions(pred_df)
    
    return True, f"✅ [{target_date}] 태양광/풍력 예측 완료 및 {updated_rows}행 저장 성공!", input_info