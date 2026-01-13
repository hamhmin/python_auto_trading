import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv
from binance.client import Client

load_dotenv()
client = Client(os.getenv('API_KEY'), os.getenv('SECRET_KEY'))

# ============================================================================
# 설정값
# ============================================================================
SYMBOL = "BTCUSDT"
TIMEFRAME = "15m"
RSI_PERIOD = 14
LOOKBACK_LEFT = 2
LOOKBACK_RIGHT = 5
RANGE_LOWER = 5
RANGE_UPPER = 60
TRADE_DURATION = 38

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
    if idx - left < 0 or idx + right >= len(series): 
        return False
    cv = series.iloc[idx]
    return all(series.iloc[idx-left:idx] < cv) and all(series.iloc[idx+1:idx+right+1] < cv)

def find_pivot_low(series, left, right, idx):
    if idx - left < 0 or idx + right >= len(series): 
        return False
    cv = series.iloc[idx]
    return all(series.iloc[idx-left:idx] > cv) and all(series.iloc[idx+1:idx+right+1] > cv)

# ============================================================================
# 다이버전스 감지 - 🔧 올바른 진입 시점 적용
# ============================================================================

def detect_all_divergences(df):
    """
    다이버전스 감지 (올바른 진입 시점)
    - 피벗 확정 시점: i + LOOKBACK_RIGHT
    - 진입 시점: i + LOOKBACK_RIGHT
    """
    signals = []
    rsi = df['rsi']
    high = df['high']
    low = df['low']
    close = df['close']
    
    # 피벗 체크 범위 (봇/그리드와 동일)
    for i in range(LOOKBACK_LEFT, len(df) - LOOKBACK_RIGHT):
        
        # 1. Bearish Divergence
        if find_pivot_high(rsi, LOOKBACK_LEFT, LOOKBACK_RIGHT, i):
            # 과거 피벗 찾기
            for j in range(i - RANGE_LOWER, max(i - RANGE_UPPER, LOOKBACK_LEFT), -1):
                if find_pivot_high(rsi, LOOKBACK_LEFT, LOOKBACK_RIGHT, j):
                    if high.iloc[i] > high.iloc[j] and rsi.iloc[i] < rsi.iloc[j]:
                        
                        # 🔧 진입 시점: 피벗 확정 시점 (봇/그리드와 동일)
                        entry_idx = i + LOOKBACK_RIGHT
                        
                        if entry_idx >= len(df):
                            break
                        
                        # 청산 시점
                        exit_idx = entry_idx + TRADE_DURATION
                        roi, exit_p = None, None
                        
                        if exit_idx < len(df):
                            entry_p = close.iloc[entry_idx]
                            exit_p = close.iloc[exit_idx]
                            roi = (entry_p - exit_p) / entry_p * 100  # 숏 수익
                        
                        signals.append({
                            'time': df['timestamp'].iloc[entry_idx] if 'timestamp' in df.columns else entry_idx,
                            'type': 'bearish',
                            'pivot_idx': i,              # 피벗 인덱스
                            'entry_idx': entry_idx,      # 진입 인덱스 (확정 시점)
                            'past_idx': j,
                            'entry_price': close.iloc[entry_idx],
                            'exit_price': exit_p,
                            'return_pct': roi,
                            'curr_rsi': rsi.iloc[i],
                            'past_rsi': rsi.iloc[j],
                            'curr_high_low': high.iloc[i],
                            'past_high_low': high.iloc[j]
                        })
                        break

        # 2. Bullish Divergence
        if find_pivot_low(rsi, LOOKBACK_LEFT, LOOKBACK_RIGHT, i):
            for j in range(i - RANGE_LOWER, max(i - RANGE_UPPER, LOOKBACK_LEFT), -1):
                if find_pivot_low(rsi, LOOKBACK_LEFT, LOOKBACK_RIGHT, j):
                    if low.iloc[i] < low.iloc[j] and rsi.iloc[i] > rsi.iloc[j]:
                        
                        # 🔧 진입 시점: 피벗 확정 시점
                        entry_idx = i + LOOKBACK_RIGHT
                        
                        if entry_idx >= len(df):
                            break
                        
                        exit_idx = entry_idx + TRADE_DURATION
                        roi, exit_p = None, None
                        
                        if exit_idx < len(df):
                            entry_p = close.iloc[entry_idx]
                            exit_p = close.iloc[exit_idx]
                            roi = (exit_p - entry_p) / entry_p * 100  # 롱 수익
                        
                        signals.append({
                            'time': df['timestamp'].iloc[entry_idx] if 'timestamp' in df.columns else entry_idx,
                            'type': 'bullish',
                            'pivot_idx': i,
                            'entry_idx': entry_idx,
                            'past_idx': j,
                            'entry_price': close.iloc[entry_idx],
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
    result_df = pd.DataFrame(signals)
    file_prefix = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_name = f"divergence_report_{file_prefix}.csv"
    result_df.to_csv(csv_name, index=False, encoding='utf-8-sig')
    print(f"✅ CSV 저장 완료: {csv_name}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    ax1.plot(df.index, df['close'], color='black', alpha=0.5, label='Price')
    ax2.plot(df.index, df['rsi'], color='blue', alpha=0.7, label='RSI')
    ax2.axhline(70, color='red', linestyle='--', alpha=0.3)
    ax2.axhline(30, color='green', linestyle='--', alpha=0.3)

    for sig in signals:
        pivot_idx = sig['pivot_idx']
        entry_idx = sig['entry_idx']
        p_idx = sig['past_idx']
        roi = sig['return_pct']
        roi_text = f"{roi:+.2f}%" if roi is not None else "N/A"
        color = 'red' if sig['type'] == 'bearish' else 'green'
        
        # 피벗 연결선 (RSI 기준)
        ax1.plot([p_idx, pivot_idx], [sig['past_high_low'], sig['curr_high_low']], 
                color=color, linestyle='--', linewidth=2, alpha=0.7)
        ax1.scatter([p_idx, pivot_idx], [sig['past_high_low'], sig['curr_high_low']], 
                   color=color, s=40, alpha=0.7)
        
        # 진입 시점 강조
        ax1.axvline(entry_idx, color=color, linestyle=':', alpha=0.3)
        ax1.scatter([entry_idx], [sig['entry_price']], 
                   color=color, s=100, marker='*', edgecolors='black', linewidths=1, zorder=5)
        
        # 수익률 표시
        y_pos = sig['entry_price'] * (1.005 if sig['type'] == 'bearish' else 0.995)
        ax1.text(entry_idx, y_pos, roi_text, color=color, fontsize=10, fontweight='bold', ha='center')

        # RSI 차트
        ax2.plot([p_idx, pivot_idx], [sig['past_rsi'], sig['curr_rsi']], 
                color=color, linestyle='--', linewidth=2, alpha=0.7)
        ax2.axvline(entry_idx, color=color, linestyle=':', alpha=0.3)

    img_name = f"divergence_chart_{file_prefix}.png"
    plt.savefig(img_name, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 이미지 저장 완료: {img_name}")

# ============================================================================
# 메인 실행
# ============================================================================
def main():
    print(f"📊 {SYMBOL} 데이터 분석 시작...")
    print(f"⚙️  설정:")
    print(f"   LOOKBACK_RIGHT: {LOOKBACK_RIGHT}")
    print(f"   TRADE_DURATION: {TRADE_DURATION}")
    print(f"   진입 로직: 피벗 확정 시점 (피벗 + {LOOKBACK_RIGHT})")
    print()
    
    klines = client.futures_klines(symbol=SYMBOL, interval=TIMEFRAME, limit=1500)
    df = pd.DataFrame(klines, columns=['timestamp','open','high','low','close','vol','ct','qv','tr','tb','tq','i'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df[['high','low','close']] = df[['high','low','close']].astype(float)
    
    df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
    df = df.dropna().reset_index(drop=True)
    
    print(f"✅ 데이터 로드: {len(df)}개 캔들")
    
    signals = detect_all_divergences(df)
    
    print(f"\n📊 발견된 다이버전스: {len(signals)}개")
    
    if signals:
        bearish_count = sum(1 for s in signals if s['type'] == 'bearish')
        bullish_count = sum(1 for s in signals if s['type'] == 'bullish')
        print(f"   🔴 Bearish: {bearish_count}개")
        print(f"   🟢 Bullish: {bullish_count}개")
        
        process_results(df, signals)
        
        rois = [s['return_pct'] for s in signals if s['return_pct'] is not None]
        if rois:
            print(f"\n📈 성과 요약:")
            print(f"   총 거래: {len(rois)}개")
            print(f"   평균 수익률: {np.mean(rois):+.2f}%")
            print(f"   승률: {(np.array(rois) > 0).mean()*100:.1f}%")
            print(f"   최고 수익: {max(rois):+.2f}%")
            print(f"   최대 손실: {min(rois):+.2f}%")
            print(f"   총 수익: {sum(rois):+.2f}%")
    else:
        print("📭 신호가 발견되지 않았습니다.")

if __name__ == "__main__":
    main()