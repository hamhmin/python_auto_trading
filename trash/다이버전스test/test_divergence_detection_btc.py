import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # 서버 환경에서도 이미지 생성이 가능하도록 설정
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv
from binance.client import Client

# .env 로드
load_dotenv()
client = Client(os.getenv('API_KEY'), os.getenv('SECRET_KEY'))

# ============================================================================
# 설정값
# ============================================================================
SYMBOL = "BTCUSDT"
TIMEFRAME = "15m"
RSI_PERIOD = 14
LOOKBACK_LEFT = 5
LOOKBACK_RIGHT = 10
RANGE_LOWER = 5
RANGE_UPPER = 60
TRADE_DURATION = 38  # 18번째 봉(약 4.5시간) 뒤에 종료

# ============================================================================
# 핵심 로직 함수
# ============================================================================

def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))

def find_pivot_high(series, left, right, idx):
    if idx - left < 0 or idx + right >= len(series): return False
    cv = series.iloc[idx]
    return all(series.iloc[idx-left:idx] < cv) and all(series.iloc[idx+1:idx+right+1] < cv)

def find_pivot_low(series, left, right, idx):
    if idx - left < 0 or idx + right >= len(series): return False
    cv = series.iloc[idx]
    return all(series.iloc[idx-left:idx] > cv) and all(series.iloc[idx+1:idx+right+1] > cv)

# ============================================================================
# 다이버전스 감지 및 수익률 계산
# ============================================================================

def detect_all_divergences(df):
    signals = []
    rsi = df['rsi']
    high = df['high']
    low = df['low']
    close = df['close']
    
    # 데이터 순회하며 모든 지점 체크
    for i in range(RANGE_UPPER + LOOKBACK_LEFT, len(df) - LOOKBACK_RIGHT):
        # 1. Bearish Divergence (하락 신호)
        if find_pivot_high(rsi, LOOKBACK_LEFT, LOOKBACK_RIGHT, i):
            for j in range(i - RANGE_LOWER, i - RANGE_UPPER, -1):
                if find_pivot_high(rsi, LOOKBACK_LEFT, LOOKBACK_RIGHT, j):
                    if high.iloc[i] > high.iloc[j] and rsi.iloc[i] < rsi.iloc[j]:
                        # 수익률 계산 (18봉 뒤)
                        exit_idx = i + TRADE_DURATION
                        roi, exit_p = None, None
                        if exit_idx < len(df):
                            exit_p = close.iloc[exit_idx]
                            roi = (close.iloc[i] - exit_p) / close.iloc[i] * 100 # 숏 수익
                        
                        signals.append({
                            'time': df['timestamp'].iloc[i],
                            'type': 'bearish',
                            'current_idx': i,
                            'past_idx': j,
                            'entry_price': close.iloc[i],
                            'exit_price': exit_p,
                            'return_pct': roi,
                            'curr_rsi': rsi.iloc[i],
                            'past_rsi': rsi.iloc[j],
                            'curr_high_low': high.iloc[i],
                            'past_high_low': high.iloc[j]
                        })
                        break

        # 2. Bullish Divergence (상승 신호)
        if find_pivot_low(rsi, LOOKBACK_LEFT, LOOKBACK_RIGHT, i):
            for j in range(i - RANGE_LOWER, i - RANGE_UPPER, -1):
                if find_pivot_low(rsi, LOOKBACK_LEFT, LOOKBACK_RIGHT, j):
                    if low.iloc[i] < low.iloc[j] and rsi.iloc[i] > rsi.iloc[j]:
                        exit_idx = i + TRADE_DURATION
                        roi, exit_p = None, None
                        if exit_idx < len(df):
                            exit_p = close.iloc[exit_idx]
                            roi = (exit_p - close.iloc[i]) / close.iloc[i] * 100 # 롱 수익
                        
                        signals.append({
                            'time': df['timestamp'].iloc[i],
                            'type': 'bullish',
                            'current_idx': i,
                            'past_idx': j,
                            'entry_price': close.iloc[i],
                            'exit_price': exit_p,
                            'return_pct': roi,
                            'curr_rsi': rsi.iloc[i],
                            'past_rsi': rsi.iloc[j],
                            'curr_high_low': low.iloc[i],
                            'past_high_low': low.iloc[j]
                        })
                        break
    return signals

# ============================================================================
# 시각화 및 파일 저장
# ============================================================================

def process_results(df, signals):
    # 1. CSV 저장
    result_df = pd.DataFrame(signals)
    file_prefix = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_name = f"divergence_report_{file_prefix}.csv"
    result_df.to_csv(csv_name, index=False, encoding='utf-8-sig')
    print(f"✅ CSV 저장 완료: {csv_name}")

    # 2. 이미지 생성
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    ax1.plot(df.index, df['close'], color='black', alpha=0.5, label='Price')
    ax2.plot(df.index, df['rsi'], color='blue', alpha=0.7, label='RSI')
    ax2.axhline(70, color='red', linestyle='--', alpha=0.3)
    ax2.axhline(30, color='green', linestyle='--', alpha=0.3)

    for sig in signals:
        idx, p_idx = sig['current_idx'], sig['past_idx']
        roi = sig['return_pct']
        roi_text = f"{roi:+.2f}%" if roi is not None else "N/A"
        color = 'red' if sig['type'] == 'bearish' else 'green'
        
        # 가격 차트 표시
        ax1.plot([p_idx, idx], [sig['past_high_low'], sig['curr_high_low']], color=color, linestyle='--', linewidth=2)
        ax1.scatter([p_idx, idx], [sig['past_high_low'], sig['curr_high_low']], color=color, s=40)
        
        # 수익률 텍스트 추가 (이미지 상단/하단)
        y_pos = sig['curr_high_low'] * (1.005 if sig['type'] == 'bearish' else 0.995)
        ax1.text(idx, y_pos, roi_text, color=color, fontsize=10, fontweight='bold', ha='center')

        # RSI 차트 표시
        ax2.plot([p_idx, idx], [sig['past_rsi'], sig['curr_rsi']], color=color, linestyle='--', linewidth=2)

    img_name = f"divergence_chart_{file_prefix}.png"
    plt.savefig(img_name, dpi=150, bbox_inches='tight')
    print(f"✅ 이미지 저장 완료: {img_name}")

# ============================================================================
# 메인 실행
# ============================================================================
def main():
    print(f"📊 {SYMBOL} 데이터 분석 시작...")
    klines = client.futures_klines(symbol=SYMBOL, interval=TIMEFRAME, limit=1000)
    df = pd.DataFrame(klines, columns=['timestamp','open','high','low','close','vol','ct','qv','tr','tb','tq','i'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df[['high','low','close']] = df[['high','low','close']].astype(float)
    
    df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
    signals = detect_all_divergences(df)
    
    if signals:
        process_results(df, signals)
        # 요약 정보 출력
        rois = [s['return_pct'] for s in signals if s['return_pct'] is not None]
        if rois:
            print(f"📈 평균 수익률: {np.mean(rois):+.2f}% / 승률: {(np.array(rois) > 0).mean()*100:.1f}%")
    else:
        print("📭 신호가 발견되지 않았습니다.")

if __name__ == "__main__":
    main()