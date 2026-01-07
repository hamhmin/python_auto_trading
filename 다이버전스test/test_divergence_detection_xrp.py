import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 🔧 GUI 백엔드 사용 안함 (화면 표시 X)
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv
from binance.client import Client

# .env 로드
load_dotenv()
client = Client(os.getenv('API_KEY'), os.getenv('SECRET_KEY'))

# 설정값 (봇과 동일하게)
SYMBOL = "XRPUSDT"
TIMEFRAME = "15m"
RSI_PERIOD = 14
LOOKBACK_LEFT = 5
LOOKBACK_RIGHT = 1
RANGE_LOWER = 5
RANGE_UPPER = 60

# ============================================================================
# RSI 계산 (봇과 동일)
# ============================================================================

def calculate_rsi(data, period=14):
    """RSI 계산"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

# ============================================================================
# 피벗 감지 (봇과 동일)
# ============================================================================

def find_pivot_high(series, left, right, idx):
    """피벗 고점 찾기"""
    if idx - left < 0 or idx + right >= len(series):
        return False
    
    center_value = series.iloc[idx]
    
    # 왼쪽 체크
    left_lower = all(series.iloc[idx-left:idx] < center_value)
    
    # 오른쪽 체크
    right_lower = all(series.iloc[idx+1:idx+right+1] < center_value)
    
    return left_lower and right_lower

def find_pivot_low(series, left, right, idx):
    """피벗 저점 찾기"""
    if idx - left < 0 or idx + right >= len(series):
        return False
    
    center_value = series.iloc[idx]
    
    # 왼쪽 체크
    left_higher = all(series.iloc[idx-left:idx] > center_value)
    
    # 오른쪽 체크
    right_higher = all(series.iloc[idx+1:idx+right+1] > center_value)
    
    return left_higher and right_higher

# ============================================================================
# 다이버전스 감지 - 모든 다이버전스 찾기
# ============================================================================

def detect_all_divergences(df):
    """모든 다이버전스 감지 (전체 기간)"""
    all_signals = []
    
    rsi = df['rsi']
    high = df['high']
    low = df['low']
    
    # 🔧 모든 가능한 인덱스 체크 (전체 스캔)
    for check_idx in range(LOOKBACK_LEFT, len(df) - LOOKBACK_RIGHT):
        
        # Bearish Divergence
        if find_pivot_high(rsi, LOOKBACK_LEFT, LOOKBACK_RIGHT, check_idx):
            current_rsi = rsi.iloc[check_idx]
            current_high = high.iloc[check_idx]
            
            # 과거 피벗 찾기
            for j in range(check_idx - RANGE_LOWER, max(check_idx - RANGE_UPPER, LOOKBACK_LEFT), -1):
                if find_pivot_high(rsi, LOOKBACK_LEFT, LOOKBACK_RIGHT, j):
                    past_rsi = rsi.iloc[j]
                    past_high = high.iloc[j]
                    
                    # 가격은 상승했지만 RSI는 하락
                    if current_high > past_high and current_rsi < past_rsi:
                        all_signals.append({
                            'type': 'bearish',
                            'index': check_idx,
                            'current_idx': check_idx,
                            'past_idx': j,
                            'current_rsi': current_rsi,
                            'past_rsi': past_rsi,
                            'current_price': current_high,
                            'past_price': past_high
                        })
                        break
        
        # Bullish Divergence
        if find_pivot_low(rsi, LOOKBACK_LEFT, LOOKBACK_RIGHT, check_idx):
            current_rsi = rsi.iloc[check_idx]
            current_low = low.iloc[check_idx]
            
            # 과거 피벗 찾기
            for j in range(check_idx - RANGE_LOWER, max(check_idx - RANGE_UPPER, LOOKBACK_LEFT), -1):
                if find_pivot_low(rsi, LOOKBACK_LEFT, LOOKBACK_RIGHT, j):
                    past_rsi = rsi.iloc[j]
                    past_low = low.iloc[j]
                    
                    # 가격은 하락했지만 RSI는 상승
                    if current_low < past_low and current_rsi > past_rsi:
                        all_signals.append({
                            'type': 'bullish',
                            'index': check_idx,
                            'current_idx': check_idx,
                            'past_idx': j,
                            'current_rsi': current_rsi,
                            'past_rsi': past_rsi,
                            'current_price': current_low,
                            'past_price': past_low
                        })
                        break
    
    return all_signals

# ============================================================================
# 데이터 가져오기
# ============================================================================

def get_historical_data(symbol, interval, limit=500):
    """과거 데이터 가져오기"""
    try:
        klines = client.futures_klines(
            symbol=symbol,
            interval=interval,
            limit=limit
        )
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        
        return df
        
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return None

# ============================================================================
# 시각화 - 모든 다이버전스 표시
# ============================================================================

def plot_all_divergences(df, signals):
    """모든 다이버전스 시각화 (이미지만 저장)"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), sharex=True)
    
    # 가격 차트
    ax1.plot(df.index, df['close'], label='Close Price', color='black', linewidth=1.5, alpha=0.7)
    
    bearish_count = 0
    bullish_count = 0
    
    if signals:
        for signal in signals:
            current_idx = signal['current_idx']
            past_idx = signal['past_idx']
            
            if signal['type'] == 'bearish':
                bearish_count += 1
                # 🔴 Bearish Divergence
                ax1.plot([past_idx, current_idx], 
                        [signal['past_price'], signal['current_price']], 
                        'r--', linewidth=2, alpha=0.7, zorder=4)
                
                ax1.scatter([past_idx, current_idx], 
                           [signal['past_price'], signal['current_price']], 
                           color='red', s=100, zorder=5, alpha=0.8, edgecolors='darkred', linewidths=1)
                
            else:
                bullish_count += 1
                # 🟢 Bullish Divergence
                ax1.plot([past_idx, current_idx], 
                        [signal['past_price'], signal['current_price']], 
                        'g--', linewidth=2, alpha=0.7, zorder=4)
                
                ax1.scatter([past_idx, current_idx], 
                           [signal['past_price'], signal['current_price']], 
                           color='green', s=100, zorder=5, alpha=0.8, edgecolors='darkgreen', linewidths=1)
    
    # 제목
    total_signals = len(signals) if signals else 0
    ax1.set_title(f'{SYMBOL} {TIMEFRAME} - Price Chart\n'
                 f'Total Divergences: {total_signals} '
                 f'(🔴 Bearish: {bearish_count}, 🟢 Bullish: {bullish_count})', 
                 fontsize=14, fontweight='bold')
    
    ax1.set_ylabel('Price (USDT)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # RSI 차트
    ax2.plot(df.index, df['rsi'], label='RSI', color='blue', linewidth=1.5)
    ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5, linewidth=1, label='Overbought (70)')
    ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5, linewidth=1, label='Oversold (30)')
    ax2.axhline(y=50, color='gray', linestyle=':', alpha=0.3, linewidth=1)
    
    # RSI 영역 색칠
    ax2.fill_between(df.index, 70, 100, alpha=0.1, color='red')
    ax2.fill_between(df.index, 0, 30, alpha=0.1, color='green')
    
    if signals:
        for signal in signals:
            current_idx = signal['current_idx']
            past_idx = signal['past_idx']
            
            if signal['type'] == 'bearish':
                # 🔴 RSI 고점 연결
                ax2.plot([past_idx, current_idx], 
                        [signal['past_rsi'], signal['current_rsi']], 
                        'r--', linewidth=2, alpha=0.7, zorder=4)
                
                ax2.scatter([past_idx, current_idx], 
                           [signal['past_rsi'], signal['current_rsi']], 
                           color='red', s=100, zorder=5, alpha=0.8, edgecolors='darkred', linewidths=1)
                
            else:
                # 🟢 RSI 저점 연결
                ax2.plot([past_idx, current_idx], 
                        [signal['past_rsi'], signal['current_rsi']], 
                        'g--', linewidth=2, alpha=0.7, zorder=4)
                
                ax2.scatter([past_idx, current_idx], 
                           [signal['past_rsi'], signal['current_rsi']], 
                           color='green', s=100, zorder=5, alpha=0.8, edgecolors='darkgreen', linewidths=1)
    
    ax2.set_ylabel('RSI', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Candle Index', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.set_title(f'RSI Indicator - All Divergences Marked', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # 전체 레이아웃 조정
    plt.tight_layout()
    
    # 하단 설명
    if signals:
        fig.text(0.5, 0.02, 
                f'🔴 Bearish: Price ↗ RSI ↘ (하락 예상) | 🟢 Bullish: Price ↘ RSI ↗ (상승 예상)', 
                ha='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # 저장
    filename = f"divergence_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✅ 차트 저장: {filename}")
    
    # 🔧 화면에 표시 안함
    plt.close()  # 메모리 해제

# ============================================================================
# 메인 테스트
# ============================================================================

def main():
    print("="*80)
    print("🔍 모든 다이버전스 감지 테스트")
    print("="*80)
    print(f"심볼: {SYMBOL}")
    print(f"타임프레임: {TIMEFRAME}")
    print(f"RSI 기간: {RSI_PERIOD}")
    print(f"검색 범위: {RANGE_LOWER} ~ {RANGE_UPPER}")
    print("="*80)
    
    # 1. 데이터 로드
    print("\n📊 데이터 로드 중...")
    df = get_historical_data(SYMBOL, TIMEFRAME, limit=500)
    
    if df is None:
        print("❌ 데이터 로드 실패")
        return
    
    print(f"✅ {len(df)}개 캔들 로드")
    
    # 2. RSI 계산
    print("\n📈 RSI 계산 중...")
    df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
    df = df.dropna().reset_index(drop=True)
    
    print(f"✅ RSI 계산 완료: {len(df)}개 캔들")
    print(f"   최근 가격: ${df['close'].iloc[-1]:,.2f}")
    print(f"   최근 RSI: {df['rsi'].iloc[-1]:.2f}")
    
    # 3. 모든 다이버전스 감지
    print("\n🔍 모든 다이버전스 감지 중...")
    signals = detect_all_divergences(df)
    
    print(f"\n{'='*80}")
    print(f"📊 발견된 다이버전스: {len(signals)}개")
    print(f"{'='*80}")
    
    if signals:
        bearish_count = sum(1 for s in signals if s['type'] == 'bearish')
        bullish_count = sum(1 for s in signals if s['type'] == 'bullish')
        
        print(f"\n🔴 Bearish Divergence: {bearish_count}개")
        print(f"🟢 Bullish Divergence: {bullish_count}개")
        
        print(f"\n{'='*80}")
        print("📋 다이버전스 상세 정보 (최근 10개)")
        print(f"{'='*80}")
        
        # 최근 10개만 표시
        for i, signal in enumerate(signals[-10:], 1):
            emoji = "🔴" if signal['type'] == 'bearish' else "🟢"
            type_kr = "BEARISH" if signal['type'] == 'bearish' else "BULLISH"
            
            print(f"\n{i}. {emoji} {type_kr} DIVERGENCE")
            print(f"   위치: 인덱스 {signal['current_idx']} ({signal['current_idx'] - signal['past_idx']}봉 전과 비교)")
            print(f"   가격: ${signal['past_price']:,.0f} → ${signal['current_price']:,.0f} "
                  f"({((signal['current_price'] - signal['past_price']) / signal['past_price'] * 100):+.2f}%)")
            print(f"   RSI: {signal['past_rsi']:.1f} → {signal['current_rsi']:.1f} "
                  f"({signal['current_rsi'] - signal['past_rsi']:+.1f})")
        
        if len(signals) > 10:
            print(f"\n... 외 {len(signals) - 10}개 (차트에서 확인)")
        
        # 시각화
        print(f"\n📊 차트 생성 중...")
        plot_all_divergences(df, signals)
        
    else:
        print("\n📭 다이버전스 신호 없음")
        print("\n💡 팁:")
        print("  - 다른 타임프레임 시도: 1h, 4h, 1d")
        print("  - 다른 심볼 시도: ETHUSDT, BNBUSDT")
        print("  - 과거 데이터 더 가져오기: limit=1000")
        
        # 그래도 차트는 저장
        print(f"\n📊 현재 상태 차트 생성 중...")
        plot_all_divergences(df, None)
    
    print(f"\n{'='*80}")
    print("✅ 테스트 완료!")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()