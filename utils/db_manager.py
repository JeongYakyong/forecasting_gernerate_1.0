import pandas as pd
import numpy as np
import sqlite3
import os
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


class JejuEnergyDB:
    """제주 에너지 데이터베이스 최종 버전"""
    def __init__(self, db_path="database/jeju_energy.db"):
        self.db_path = db_path
        
        # 1. 파일 경로에서 '폴더 이름(database)'만 쏙 뽑아냅니다.
        folder_path = os.path.dirname(self.db_path)
        
        # 2. 만약 그 폴더가 존재하지 않는다면? 에러 내지 말고 알아서 만들어라!
        if folder_path:  
            os.makedirs(folder_path, exist_ok=True)
            
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_tables()
        print(f"DB 연결: {db_path}")
    
    def _init_tables(self):
        """테이블 초기화"""
        cursor = self.conn.cursor()
        
        # 1. 실측 데이터 (Historical)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS historical_data (
                timestamp TEXT PRIMARY KEY,
                -- Raw Power Features
                supply_cap REAL,
                real_demand REAL,
                real_renew_gen REAL,
                real_solar_gen REAL,
                real_wind_gen REAL,
                smp_jeju REAL,
                smp_land REAL,
                est_demand REAL,
                -- Raw Weather Features
                temp_c REAL,
                rainfall REAL,
                wind_spd REAL,
                humidity REAL,
                solar_rad REAL,
                total_cloud REAL,
                midlow_cloud REAL,
                wd_sin REAL,
                wd_cos REAL,
                -- Stored Derived Features
                Solar_Capacity_Est REAL,
                Wind_Capacity_Est REAL,
                Solar_Utilization REAL,
                Wind_Utilization REAL,
                -- Metadata
                updated_at TEXT
            )
        """)
        
        # 2. 예보 데이터 (Forecast)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS forecast_data (
                timestamp TEXT PRIMARY KEY,
                -- Raw Forecast Features
                est_demand REAL,
                smp_jeju REAL,
                smp_land REAL,
                temp_c REAL,
                rainfall REAL,
                wind_spd REAL,
                humidity REAL,
                solar_rad REAL,
                total_cloud REAL,
                midlow_cloud REAL,
                wd_sin REAL,
                wd_cos REAL,
                -- Capacity (copied from latest historical)
                Solar_Capacity_Est REAL,
                Wind_Capacity_Est REAL,
                -- Prediction Results (model output)
                est_Solar_Utilization REAL,
                est_Wind_Utilization REAL,
                -- Metadata
                forecast_time TEXT,
                updated_at TEXT
            )
        """)
        
        self.conn.commit()
        print("테이블 초기화 완료")
    
    # ==========================================
    # 1. 실측 데이터 (Historical)
    # ==========================================
    def save_historical(self, df):
        """
        실측 데이터 저장 (진짜 UPSERT 적용)
        """
        if df.empty:
            print("빈 데이터프레임")
            return 0
        
        if df.index.name == 'timestamp':
            df = df.reset_index()
        
        df['updated_at'] = datetime.now().isoformat()
        
        # 전체 컬럼 (순서대로)
        all_cols = [
            'timestamp',
            'supply_cap', 'real_demand', 'real_renew_gen', 
            'real_solar_gen', 'real_wind_gen', 
            'smp_jeju', 'smp_land', 'est_demand',
            'temp_c', 'rainfall', 'wind_spd', 'humidity', 
            'solar_rad', 'total_cloud', 'midlow_cloud', 
            'wd_sin', 'wd_cos',
            'Solar_Capacity_Est', 'Wind_Capacity_Est',
            'Solar_Utilization', 'Wind_Utilization',
            'updated_at'
        ]
        
        df_to_save = df[[col for col in all_cols if col in df.columns]].copy()
        
        cursor = self.conn.cursor()
        for _, row in df_to_save.iterrows():
            placeholders = ', '.join(['?' for _ in row])
            columns = ', '.join(row.index)
            
            # 💡 [핵심 방어막] 결측치 완벽 처리
            import pandas as pd
            safe_row_values = tuple(None if pd.isna(x) else x for x in row) 
            
            update_cols = [col for col in row.index if col != 'timestamp']
            update_sql = ', '.join([f"{col}=COALESCE(excluded.{col}, historical_data.{col})" for col in update_cols])
            
            if update_sql:
                # 💡 [문법 수정] OR REPLACE 제거 -> 순수 INSERT INTO 사용
                query = f"""
                    INSERT INTO historical_data ({columns})
                    VALUES ({placeholders})
                    ON CONFLICT(timestamp) DO UPDATE SET 
                    {update_sql}
                """
            else:
                # 💡 [문법 수정] 업데이트할 게 없으면 OR IGNORE
                query = f"""
                    INSERT OR IGNORE INTO historical_data ({columns})
                    VALUES ({placeholders})
                """
                
            cursor.execute(query, safe_row_values)
        
        self.conn.commit()
        print(f"실측 데이터 {len(df_to_save):,}행 저장")
        return len(df_to_save)

    def get_historical(self, start_date=None, end_date=None, columns=None):
        """
        실측 데이터 조회
        
        Args:
            start_date: 'YYYY-MM-DD' or 'YYYY-MM-DD HHh'
            end_date: 'YYYY-MM-DD' or 'YYYY-MM-DD HHh'
            columns: 조회할 컬럼 리스트 (None이면 전체)
        
        Returns:
            DataFrame with timestamp index
        """
        if columns:
            cols = ', '.join(['timestamp'] + columns)
            query = f"SELECT {cols} FROM historical_data"
        else:
            query = "SELECT * FROM historical_data"
        
        conditions = []
        if start_date:
            conditions.append(f"timestamp >= '{start_date}'")
        if end_date:
            conditions.append(f"timestamp <= '{end_date}'")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY timestamp"
        
        df = pd.read_sql(query, self.conn)
        if not df.empty:
            df = df.set_index('timestamp')
            df = df.drop(columns=['updated_at'], errors='ignore')
        
        return df
    
    
    def get_latest_capacity(self):
        """
        가장 최근의 Capacity 값 조회 (Forecast에서 사용)
        
        Returns:
            dict: {'Solar_Capacity_Est': value, 'Wind_Capacity_Est': value}
        """
        query = """
            SELECT Solar_Capacity_Est, Wind_Capacity_Est 
            FROM historical_data 
            WHERE Solar_Capacity_Est IS NOT NULL 
            ORDER BY timestamp DESC 
            LIMIT 1
        """
        
        df = pd.read_sql(query, self.conn)
        
        if df.empty:
            print("Capacity 데이터 없음 - 기본값 사용")
            return {'Solar_Capacity_Est': None, 'Wind_Capacity_Est': None}
        
        return df.iloc[0].to_dict()
    
    # ==========================================
    # 2. 예보 데이터 (Forecast)
    # ==========================================
    def save_forecast(self, df, forecast_time=None, auto_add_capacity=True):
        """
        예보 데이터 저장 (진짜 UPSERT 적용)
        """
        if df.empty:
            print("빈 데이터프레임")
            return 0
        
        if df.index.name == 'timestamp':
            df = df.reset_index()
        
        if auto_add_capacity:
            if 'Solar_Capacity_Est' not in df.columns or 'Wind_Capacity_Est' not in df.columns:
                latest_cap = self.get_latest_capacity()
                df['Solar_Capacity_Est'] = latest_cap.get('Solar_Capacity_Est')
                df['Wind_Capacity_Est'] = latest_cap.get('Wind_Capacity_Est')
        
        if forecast_time is None:
            forecast_time = datetime.now().isoformat()
        
        df['forecast_time'] = forecast_time
        df['updated_at'] = datetime.now().isoformat()
        
        all_cols = [
            'timestamp',
            'est_demand', 'smp_jeju', 'smp_land', 'temp_c', 'rainfall', 'wind_spd', 
            'humidity', 'solar_rad', 'total_cloud', 'midlow_cloud',
            'wd_sin', 'wd_cos',
            'Solar_Capacity_Est', 'Wind_Capacity_Est',
            'est_Solar_Utilization', 'est_Wind_Utilization',
            'forecast_time', 'updated_at'
        ]
        
        df_to_save = df[[col for col in all_cols if col in df.columns]].copy()
        
        cursor = self.conn.cursor()
        for _, row in df_to_save.iterrows():
            placeholders = ', '.join(['?' for _ in row])
            columns = ', '.join(row.index)
            
            import pandas as pd
            safe_row_values = tuple(None if pd.isna(x) else x for x in row)
            
            update_cols = [col for col in row.index if col != 'timestamp']
            update_sql = ', '.join([f"{col}=COALESCE(excluded.{col}, forecast_data.{col})" for col in update_cols])
            
            if update_sql:
                # 💡 [문법 수정] OR REPLACE 제거 -> 테이블명 forecast_data 완벽 적용
                query = f"""
                    INSERT INTO forecast_data ({columns})
                    VALUES ({placeholders})
                    ON CONFLICT(timestamp) DO UPDATE SET 
                    {update_sql}
                """
            else:
                query = f"""
                    INSERT OR IGNORE INTO forecast_data ({columns})
                    VALUES ({placeholders})
                """
                
            cursor.execute(query, safe_row_values)
        
        self.conn.commit()
        print(f"예보 데이터 {len(df_to_save):,}행 저장 (예보시각: {forecast_time[:16]})")
        return len(df_to_save)
    
    def update_forecast_predictions(self, df_predictions):
        """
        예보 데이터에 예측 결과 업데이트
        
        Args:
            df_predictions: DataFrame with timestamp index
                필수 컬럼: est_Solar_Utilization, est_Wind_Utilization
        """
        if df_predictions.empty:
            print("빈 예측 결과")
            return 0
        
        if df_predictions.index.name == 'timestamp':
            df_predictions = df_predictions.reset_index()
        
        cursor = self.conn.cursor()
        updated = 0
        
        for _, row in df_predictions.iterrows():
            cursor.execute("""
                UPDATE forecast_data
                SET est_Solar_Utilization = ?, est_Wind_Utilization = ?, updated_at = ?
                WHERE timestamp = ?
            """, (
                row.get('est_Solar_Utilization'),
                row.get('est_Wind_Utilization'),
                datetime.now().isoformat(),
                row['timestamp']
            ))
            updated += cursor.rowcount
        
        self.conn.commit()
        print(f"예측 결과 {updated}행 업데이트")
        return updated
    
    def get_forecast(self, start_date=None, end_date=None):
        """예보 데이터 조회"""
        query = "SELECT * FROM forecast_data"
        conditions = []
        
        if start_date:
            conditions.append(f"timestamp >= '{start_date}'")
        if end_date:
            conditions.append(f"timestamp <= '{end_date}'")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY timestamp"
        
        df = pd.read_sql(query, self.conn)
        if not df.empty:
            df = df.set_index('timestamp')
            df = df.drop(columns=['forecast_time', 'updated_at'], errors='ignore')
        
        return df
    
    def get_historical_and_forecast(self, start_date_str, end_date_str):
        """
        [모델 예측용] 과거 실측 데이터와 미래 예보 데이터를 하나로 병합하여 반환
        - start_date_str: 과거 데이터 시작일 (예: '2026-02-18 00:00:00')
        - end_date_str: 예측 타겟 데이터 종료일 (예: '2026-03-04 23:00:00')
        """
        # 1. Historical 테이블에서 기간만큼 조회
        hist_df = self.get_historical(start_date=start_date_str, end_date=end_date_str)
        
        # 2. Forecast 테이블에서 기간만큼 조회
        fore_df = self.get_forecast(start_date=start_date_str, end_date=end_date_str)
        
        # 3. 인덱스 기준으로 병합
        # Forecast 데이터에 있는 값(예보 날씨, 예측 타겟 등)을 우선으로 하되,
        # 과거 구간은 Historical 데이터(실측값)로 채움
        
        # 두 데이터프레임의 모든 시간대(인덱스)를 합침
        all_index = hist_df.index.union(fore_df.index)
        
        # 빈 데이터프레임 생성 후 합치기
        combined_df = pd.DataFrame(index=all_index)
        
        # Historical 데이터를 먼저 넣고, Forecast 데이터로 없는 부분을 채우거나 업데이트(combine_first)
        # 중요: 모델 입력 시에는 '실측'을 과거로, '예보'를 미래로 사용
        if not hist_df.empty and not fore_df.empty:
            combined_df = fore_df.combine_first(hist_df)
        elif not hist_df.empty:
            combined_df = hist_df
        elif not fore_df.empty:
            combined_df = fore_df
            
        return combined_df
    
    def clear_old_forecasts(self, keep_hours=48):
        """
        오래된 예보 삭제
        
        Args:
            keep_hours: 보관 시간 (기본 48시간)
        """
        cutoff = (datetime.now() - timedelta(hours=keep_hours)).strftime('%Y-%m-%d %H:%M:%S')
        
        cursor = self.conn.cursor()
        cursor.execute(f"DELETE FROM forecast_data WHERE timestamp < '{cutoff}'")
        deleted = cursor.rowcount
        
        self.conn.commit()
        print(f"오래된 예보 {deleted}행 삭제 (기준: {cutoff} 이전)")
        return deleted
    
    # ==========================================
    # 3. 유틸리티
    # ==========================================
    
    def get_data_summary(self):
        """데이터베이스 현황 요약"""
        cursor = self.conn.cursor()
        
        print("\n" + "="*60)
        print("데이터베이스 현황")
        print("="*60)
        
        # Historical
        cursor.execute("SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM historical_data")
        count, min_ts, max_ts = cursor.fetchone()
        
        print(f"\n[실측 데이터 - historical_data]")
        print(f"  데이터 개수: {count:,}행")
        if min_ts and max_ts:
            print(f"  기간: {min_ts} ~ {max_ts}")
            
            # Capacity 통계
            cursor.execute("""
                SELECT 
                    AVG(Solar_Capacity_Est), AVG(Wind_Capacity_Est),
                    MAX(Solar_Capacity_Est), MAX(Wind_Capacity_Est)
                FROM historical_data
                WHERE Solar_Capacity_Est IS NOT NULL
            """)
            solar_avg, wind_avg, solar_max, wind_max = cursor.fetchone()
            if solar_avg:
                print(f"  Solar Capacity: 평균 {solar_avg:.1f} MW, 최대 {solar_max:.1f} MW")
                print(f"  Wind Capacity: 평균 {wind_avg:.1f} MW, 최대 {wind_max:.1f} MW")
        
        # Forecast
        cursor.execute("SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM forecast_data")
        count, min_ts, max_ts = cursor.fetchone()
        
        print(f"\n[예보 데이터 - forecast_data]")
        print(f"  데이터 개수: {count:,}행")
        if min_ts and max_ts:
            print(f"  기간: {min_ts} ~ {max_ts}")
            
            # 예측 완료 여부
            cursor.execute("""
                SELECT COUNT(*) 
                FROM forecast_data 
                WHERE est_Solar_Utilization IS NOT NULL
            """)
            predicted = cursor.fetchone()[0]
            print(f"  예측 완료: {predicted}/{count}행")
        
        print("="*60 + "\n")
    
    def cleanup_old_data(self, keep_years=5):
        """
        오래된 실측 데이터 삭제
        
        Args:
            keep_years: 보관 연수 (기본 5년)
        """
        cutoff = (datetime.now() - timedelta(days=keep_years*365)).strftime('%Y-%m-%d')
        
        cursor = self.conn.cursor()
        cursor.execute(f"DELETE FROM historical_data WHERE timestamp < '{cutoff}'")
        deleted = cursor.rowcount
        
        self.conn.commit()
        print(f"오래된 실측 데이터 {deleted:,}행 삭제 (기준: {cutoff} 이전)")
        return deleted
    
    def close(self):
        """DB 연결 종료"""
        self.conn.close()
        print("dB 연결 종료")
