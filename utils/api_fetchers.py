import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timedelta
import io


# ============================================================================
# 1. KPX API - 계통 데이터
# ============================================================================

def fetch_kpx_past(start_date, end_date):
    url = "https://openapi.kpx.or.kr/downloadChejuSukubCSV.do"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://openapi.kpx.or.kr/chejusukub.do"
    }
    payload = {
        'startDate': start_date,  # 하이픈 포함 그대로
        'endDate': end_date
    }
    
    try:
        resp = requests.post(url, data=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        
        df = pd.read_csv(io.StringIO(resp.text))
        df.columns = df.columns.str.strip()
        
        # 1시간 단위 필터링 (0000으로 끝나는 행)
        df = df[df['기준일시'].astype(str).str.endswith('0000')].copy()
        
        # timestamp 생성
        df['timestamp'] = pd.to_datetime(
            df['기준일시'].astype(str), 
            format='%Y%m%d%H%M%S'
        ).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # 컬럼명 변경
        df = df.rename(columns={
            '공급능력(MW)': 'supply_cap',
            '현재수요(MW)': 'real_demand',
            '신재생총합(MW)': 'real_renew_gen',
            '신재생태양광(MW)': 'real_solar_gen',
            '신재생풍력(MW)': 'real_wind_gen'
        })
        
        # 필요한 컬럼만 선택 및 숫자 변환
        result = df.set_index('timestamp')[
            ['supply_cap', 'real_demand', 'real_renew_gen', 
             'real_solar_gen', 'real_wind_gen']
        ].apply(pd.to_numeric, errors='coerce')
        
        print(f"  [KPX Past] {len(result)}행 수집")
        return result
        
    except Exception as e:
        print(f"  [KPX Past] 실패: {e}")
        return pd.DataFrame()

def fetch_kpx_future(target_date, service_key):
    url = 'https://apis.data.go.kr/B552115/SmpWithForecastDemand/getSmpWithForecastDemand'
    params = {
        'serviceKey': service_key, 
        'dataType': 'json', 
        'date': target_date.replace('-', ''), 
        'numOfRows': '100'
    }
    
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        
        items = resp.json()['response']['body']['items']['item']
        df = pd.DataFrame(items)
        
        if df.empty:
            print(f"  [KPX Future] {target_date} 데이터가 없습니다.")
            return pd.DataFrame()

        # 1. 제주 데이터와 육지 데이터를 각각 분리합니다.
        df_jeju = df[df['areaName'] == '제주'].copy()
        df_land = df[df['areaName'] == '육지'].copy()
        
        # 2. 필요한 컬럼만 남기고 이름 변경 (제주수요: jlfd -> est_demand)
        df_jeju = df_jeju[['date', 'hour', 'smp', 'jlfd']].rename(columns={'smp': 'smp_jeju', 'jlfd': 'est_demand'})
        df_land = df_land[['date', 'hour', 'smp']].rename(columns={'smp': 'smp_land'})
        
        # 3. 날짜와 시간을 기준으로 두 데이터를 가로로 예쁘게 합칩니다.
        df_merged = pd.merge(df_jeju, df_land, on=['date', 'hour'], how='outer')
        
        # 4. timestamp 생성 (hour 1~24 -> 00~23)
        df_merged['timestamp'] = (
            pd.to_datetime(df_merged['date'], format='%Y%m%d') + 
            pd.to_timedelta(df_merged['hour'].astype(int) - 1, unit='h')
        ).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # 5. 최종 데이터 정리 및 숫자형 변환
        
        result = df_merged[['timestamp', 'smp_jeju', 'smp_land', 'est_demand']].copy()
        result['smp_jeju'] = pd.to_numeric(result['smp_jeju'], errors='coerce')
        result['smp_land'] = pd.to_numeric(result['smp_land'], errors='coerce')
        result['est_demand'] = pd.to_numeric(result['est_demand'], errors='coerce')
        
        result = result.set_index('timestamp')
        
        return result
        
    except Exception as e:
        print(f"  [KPX Future] 실패: {e}")
        return pd.DataFrame()
    
def fetch_kpx_historical(start_date, end_date, service_key):
    all_smp = []
    
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    total_days = (end - current).days + 1
    print(f"  [KPX Historical] {total_days}일치 수집 시작...")
    
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        
        try:
            # 과거 날짜로 future API 호출
            df = fetch_kpx_future(date_str, service_key)
            
            if not df.empty and 'smp_jeju' in df.columns and 'smp_land' in df.columns and 'est_demand' in df.columns:
                all_smp.append(df[['est_demand', 'smp_jeju', 'smp_land']])
                print(f"    {date_str}: {len(df)}행 ✓")
            else:
                print(f"    {date_str}: 데이터 없음")
                
        except Exception as e:
            print(f"    {date_str}: 실패 ({e})")
        
        current += timedelta(days=1)
    
    if not all_smp:
        print(f"  [KPX SMP] 전체 데이터 없음")
        return pd.DataFrame()
    
    result = pd.concat(all_smp)
    print(f"  [KPX SMP] 총 {len(result)}행 수집 완료")
    return result


# ============================================================================
# 2. KMA API - 기상 데이터
# ============================================================================

def fetch_kma_past_asos(start_date, end_date, auth_key):
    url = "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm3.php"
    
    params = {
        "tm1": f"{start_date}0000",  
        "tm2": f"{end_date}2300",    
        "stn": "184",  # 제주
        "authKey": auth_key, 
        "help": "0"  # 🔥 help=0으로 변경 (컬럼명 헤더 제거)
    }
    
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        
        # 주석(#) 제거하고 데이터만 추출
        lines = [l for l in resp.text.split('\n') if l.strip() and not l.startswith('#')]
        
        if not lines:
            print("  [KMA ASOS] 데이터 없음")
            return pd.DataFrame()
        
        # 🔥 공백으로 split (고정폭 아님!)
        df_raw = pd.DataFrame([l.split() for l in lines])
        
        # 결측치 처리 함수
        def clean(val):
            try:
                v = float(val)
                return np.nan if v <= -9 else v
            except:
                return np.nan
        
        df = pd.DataFrame()
        
        # Timestamp (0번 컬럼)
        df['timestamp'] = pd.to_datetime(df_raw[0], format='%Y%m%d%H%M').dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # 기상 변수 (인덱스 기반)
        df['temp_c'] = df_raw[11].apply(clean)  # 기온
        df['humidity'] = df_raw[13].apply(clean)  # 습도
        
        # 구름 (0~10 → 0~1)
        df['total_cloud'] = df_raw[25].apply(clean) / 10
        df['midlow_cloud'] = df_raw[26].apply(clean) / 10
        
        # 풍향/풍속
        wind_spd = df_raw[3].apply(clean).fillna(0)
        wind_dir = df_raw[2].apply(clean).fillna(0) * 10  # 36방위 → 360도
        
        df['wind_spd'] = wind_spd.round(2)
        df['wd_sin'] = np.sin(np.radians(wind_dir)).round(4)
        df['wd_cos'] = np.cos(np.radians(wind_dir)).round(4)
        
        # 일사량 (MJ/m² → W/m²는 나중에 처리)
        solar_raw = df_raw[34].apply(clean).fillna(0)
        df['solar_rad'] = solar_raw.clip(lower=0)  # 음수 제거
        
        # 강수량
        df['rainfall'] = df_raw[15].apply(clean).fillna(0)
        
        # 적설량 (cm → m)
        df['snow_depth'] = df_raw[21].apply(clean).fillna(0) / 100
        
        result = df.set_index('timestamp')
        
        print(f"  [KMA ASOS] {len(result)}행 수집")
        return result
        
    except Exception as e:
        print(f"  [KMA ASOS] 실패: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def fetch_kma_future_ncm(lat, lon, auth_key, base_date_kst):
    # 변수 매핑
    VARN_MAP = {
        51: 'solar_rad',      # 일사량 W/m²
        25: 'temp_k',         # 기온 Kelvin
        37: 'total_cloud',    # 전운량 0~1
        35: 'mid_cloud',      # 중층운량 0~1
        34: 'low_cloud',      # 저층운량 0~1
        20: 'u_wind',         # 동서바람 m/s
        21: 'v_wind',         # 남북바람 m/s
        41: 'snow_depth',     # 적설량 m
        26: 'humidity',       # 습도 %
        65: 'rain_conv',      # 대류성 강수 mm
        66: 'rain_strat',     # 층상 강수 mm
    }
    
    def parse_raw_text_by_varn(raw_text):
        parsed_dict = {}
        lines = raw_text.strip().split('\n')
        for line in lines:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                varn_code = int(parts[2])
                value = float(parts[4])
                if varn_code in VARN_MAP:
                    parsed_dict[VARN_MAP[varn_code]] = value
            except (ValueError, IndexError):
                continue
        return parsed_dict
    
    # 전날 12UTC 사용
    target_dt = pd.to_datetime(base_date_kst)
    base_date = (target_dt - pd.Timedelta(days=1)).strftime('%Y%m%d')
    base_tmfc = base_date + "12"
    base_time_kst = target_dt - pd.Timedelta(hours=3)  # 전날 21시 KST
    
    url = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-kim_nc_pt_txt2"
    
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    
    rows = []
    print(f"  [KMA NCM] 수집 시작 - {base_time_kst} 이후 27시간 ")
    print(f" 실제로는 UTC {base_tmfc}, 한국시간보다 12시간 느리게 설정")
    
    for hour in range(28):  # 28시간 예보
        params = {
            'group': 'KIMG', 'nwp': 'NE57', 'data': 'U',
            'name': 'dswrsfc,t2m,tcld,mcld,lcld,u10m,v10m,snowd,rh2m,rainc_acc,rainl_acc',
            'tmfc': base_tmfc, 'hf': str(hour),
            'lat': str(lat), 'lon': str(lon),
            'disp': 'A', 'help': '0', 'authKey': auth_key
        }
        try:
            resp = session.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = parse_raw_text_by_varn(resp.text)
                if data:
                    data['hour_offset'] = hour
                    rows.append(data)
        except Exception as e:
            print(f"    [Fail] +{hour}h: {e}")
    
    session.close()
    
    if not rows:
        print(f"  [KMA NCM] 데이터 없음")
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    df['timestamp'] = base_time_kst + pd.to_timedelta(df['hour_offset'], unit='h')
    df = df.drop('hour_offset', axis=1)
    
    # 후처리
    if 'solar_rad' in df.columns:
        df['solar_rad'] = (df['solar_rad'] * 0.0036).round(2)  # W/m² → MJ/m²
    
    if 'temp_k' in df.columns:
        df['temp_c'] = (df['temp_k'] - 273.15).round(2)
        df = df.drop(columns=['temp_k'])
    
    if 'u_wind' in df.columns and 'v_wind' in df.columns:
        df['wind_spd'] = np.sqrt(df['u_wind']**2 + df['v_wind']**2).round(2)
        wind_dir = (270 - np.degrees(np.arctan2(df['v_wind'], df['u_wind']))) % 360
        df['wd_sin'] = np.sin(np.radians(wind_dir)).round(4)
        df['wd_cos'] = np.cos(np.radians(wind_dir)).round(4)
        df = df.drop(columns=['u_wind', 'v_wind'])
    
    if 'low_cloud' in df.columns and 'mid_cloud' in df.columns:
        df['midlow_cloud'] = df['low_cloud'] + df['mid_cloud'] * (1 - df['low_cloud'])
        df = df.drop(columns=['low_cloud', 'mid_cloud'])
    
    if 'rain_conv' in df.columns and 'rain_strat' in df.columns:
        df['rainfall'] = (df['rain_conv'].fillna(0) + df['rain_strat'].fillna(0)).round(2)
        df = df.drop(columns=['rain_conv', 'rain_strat'])
    
    # timestamp 포맷 변환
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    df = df.set_index('timestamp')
    
    print(f"  [KMA NCM] {len(df)}행 수집")
    return df