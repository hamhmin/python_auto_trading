import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

# 한글 폰트 설정 (선택사항)
# plt.rcParams['font.family'] = 'AppleGothic'  # Mac
# plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 봇 로직 복사 (필요한 함수들)
# ============================================================================

RSI_PERIOD = 14
LOOKBACK_LEFT = 2
LOOKBACK_RIGHT = 5
RANGE_LOWER = 5
RANGE_UPPER = 60
HOLD_BARS = 38
PARTIAL_PROFIT_TARGET = 0.8
PARTIAL_PROFIT_RATIO = 0.5
STOP_LOSS_BEAR = 3
STOP_LOSS_BULL = 3
LEVERAGE = 30

def calculate_rsi(data, period=14):
    """RSI 계산"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def find_pivot_high(series, left, right, idx):
    """RSI 피봇 고점 찾기"""
    if idx < left or idx >= len(series) - right:
        return False
    center_value = series.iloc[idx]
    left_lower = all(series.iloc[idx-left:idx] < center_value)
    if right == 0:
        right_lower = True
    else:
        right_lower = all(series.iloc[idx+1:idx+right+1] < center_value)
    return left_lower and right_lower

def find_pivot_low(series, left, right, idx):
    """RSI 피봇 저점 찾기"""
    if idx < left or idx >= len(series) - right:
        return False
    center_value = series.iloc[idx]
    left_higher = all(series.iloc[idx-left:idx] > center_value)
    if right == 0:
        right_higher = True
    else:
        right_higher = all(series.iloc[idx+1:idx+right+1] > center_value)
    return left_higher and right_higher

def detect_divergences_backtest(df):
    """과거 데이터에서 모든 다이버전스 찾기"""
    divergences = []
    rsi = df['rsi']
    high = df['high']
    low = df['low']
    
    # 전체 데이터를 순회하면서 다이버전스 찾기
    for check_idx in range(LOOKBACK_LEFT + RANGE_UPPER, len(df) - LOOKBACK_RIGHT - 1):
        
        # Bearish Divergence
        if find_pivot_high(rsi, LOOKBACK_LEFT, LOOKBACK_RIGHT, check_idx):
            for j in range(check_idx - RANGE_LOWER, max(check_idx - RANGE_UPPER, LOOKBACK_LEFT), -1):
                if find_pivot_high(rsi, LOOKBACK_LEFT, LOOKBACK_RIGHT, j):
                    signal_idx = check_idx + LOOKBACK_RIGHT
                    
                    if signal_idx >= len(df):
                        break
                    
                    rsi_curr = rsi.iloc[check_idx]
                    rsi_prev = rsi.iloc[j]
                    price_curr = high.iloc[check_idx]
                    price_prev = high.iloc[j]
                    
                    if rsi_curr < rsi_prev and price_curr > price_prev:
                        divergences.append({
                            'type': 'bearish',
                            'pivot_idx': check_idx,
                            'prev_pivot_idx': j,
                            'entry_idx': signal_idx,
                            'entry_price': df['close'].iloc[signal_idx],
                            'entry_time': df['open_time'].iloc[signal_idx],
                            'rsi_curr': rsi_curr,
                            'rsi_prev': rsi_prev,
                            'price_curr': price_curr,
                            'price_prev': price_prev
                        })
                    break
        
        # Bullish Divergence
        if find_pivot_low(rsi, LOOKBACK_LEFT, LOOKBACK_RIGHT, check_idx):
            for j in range(check_idx - RANGE_LOWER, max(check_idx - RANGE_UPPER, LOOKBACK_LEFT), -1):
                if find_pivot_low(rsi, LOOKBACK_LEFT, LOOKBACK_RIGHT, j):
                    signal_idx = check_idx + LOOKBACK_RIGHT
                    
                    if signal_idx >= len(df):
                        break
                    
                    rsi_curr = rsi.iloc[check_idx]
                    rsi_prev = rsi.iloc[j]
                    price_curr = low.iloc[check_idx]
                    price_prev = low.iloc[j]
                    
                    if rsi_curr > rsi_prev and price_curr < price_prev:
                        divergences.append({
                            'type': 'bullish',
                            'pivot_idx': check_idx,
                            'prev_pivot_idx': j,
                            'entry_idx': signal_idx,
                            'entry_price': df['close'].iloc[signal_idx],
                            'entry_time': df['open_time'].iloc[signal_idx],
                            'rsi_curr': rsi_curr,
                            'rsi_prev': rsi_prev,
                            'price_curr': price_curr,
                            'price_prev': price_prev
                        })
                    break
    
    return divergences

def simulate_trade(df, divergence):
    """트레이드 시뮬레이션"""
    entry_idx = divergence['entry_idx']
    entry_price = divergence['entry_price']
    div_type = divergence['type']
    
    # 종료 인덱스
    exit_idx = min(entry_idx + HOLD_BARS, len(df) - 1)
    
    # 스탑로스 가격
    stop_loss_pct = STOP_LOSS_BEAR if div_type == 'bearish' else STOP_LOSS_BULL
    if div_type == 'bearish':
        stop_price = entry_price * (1 + stop_loss_pct / 100)
    else:
        stop_price = entry_price * (1 - stop_loss_pct / 100)
    
    # 부분익절 가격
    if div_type == 'bearish':
        partial_price = entry_price * (1 - PARTIAL_PROFIT_TARGET / 100)
    else:
        partial_price = entry_price * (1 + PARTIAL_PROFIT_TARGET / 100)
    
    result = {
        'stop_loss_hit': False,
        'partial_profit_hit': False,
        'partial_profit_idx': None,
        'exit_idx': exit_idx,
        'exit_price': None,
        'final_profit_pct': 0,
        'max_profit_pct': 0,
        'max_loss_pct': 0
    }
    
    max_profit = 0
    max_loss = 0
    
    # 캔들 하나씩 체크
    for i in range(entry_idx, exit_idx + 1):
        candle_high = df['high'].iloc[i]
        candle_low = df['low'].iloc[i]
        candle_close = df['close'].iloc[i]
        
        # 현재 수익률 계산
        if div_type == 'bearish':
            current_profit = ((entry_price - candle_close) / entry_price) * 100
            high_profit = ((entry_price - candle_low) / entry_price) * 100
            low_profit = ((entry_price - candle_high) / entry_price) * 100
        else:
            current_profit = ((candle_close - entry_price) / entry_price) * 100
            high_profit = ((candle_high - entry_price) / entry_price) * 100
            low_profit = ((candle_low - entry_price) / entry_price) * 100
        
        max_profit = max(max_profit, high_profit)
        max_loss = min(max_loss, low_profit)
        
        # 스탑로스 체크
        if div_type == 'bearish':
            if candle_high >= stop_price and not result['stop_loss_hit']:
                result['stop_loss_hit'] = True
                result['exit_idx'] = i
                result['exit_price'] = stop_price
                result['final_profit_pct'] = -stop_loss_pct
                break
        else:
            if candle_low <= stop_price and not result['stop_loss_hit']:
                result['stop_loss_hit'] = True
                result['exit_idx'] = i
                result['exit_price'] = stop_price
                result['final_profit_pct'] = -stop_loss_pct
                break
        
        # 부분익절 체크
        if not result['partial_profit_hit']:
            if div_type == 'bearish':
                if candle_low <= partial_price:
                    result['partial_profit_hit'] = True
                    result['partial_profit_idx'] = i
            else:
                if candle_high >= partial_price:
                    result['partial_profit_hit'] = True
                    result['partial_profit_idx'] = i
    
    # 스탑로스 안 걸렸으면 보유기간 종료
    if not result['stop_loss_hit']:
        final_price = df['close'].iloc[exit_idx]
        result['exit_price'] = final_price
        
        if div_type == 'bearish':
            result['final_profit_pct'] = ((entry_price - final_price) / entry_price) * 100
        else:
            result['final_profit_pct'] = ((final_price - entry_price) / entry_price) * 100
    
    result['max_profit_pct'] = max_profit
    result['max_loss_pct'] = max_loss
    
    return result

# ============================================================================
# 메인 분석 함수
# ============================================================================

def analyze_chart_data(json_file):
    """JSON 파일에서 차트 데이터 분석"""
    
    # 1. JSON 로드
    print("📂 데이터 로딩 중...")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 2. DataFrame 변환
    df = pd.DataFrame(data)
    
    print(f"📋 데이터 컬럼: {df.columns.tolist()}")
    
    # 🔧 시간 컬럼 찾기 및 변환
    time_column = None
    for col in ['open_time', 'timestamp', 'time', 'date', 'datetime']:
        if col in df.columns:
            time_column = col
            break
    
    if time_column is None:
        print("⚠️ 시간 컬럼을 찾을 수 없습니다. 사용 가능한 컬럼:", df.columns.tolist())
        return None, None, None
    
    print(f"⏰ 시간 컬럼 발견: {time_column}")
    
    # 시간 데이터 변환
    try:
        df['open_time'] = pd.to_datetime(df[time_column])
        print(f"✅ 시간 변환 성공")
    except Exception as e:
        print(f"❌ 시간 변환 실패: {e}")
        return None, None, None
    
    # 가격 데이터 변환
    try:
        df['open'] = pd.to_numeric(df['open'], errors='coerce')
        df['high'] = pd.to_numeric(df['high'], errors='coerce')
        df['low'] = pd.to_numeric(df['low'], errors='coerce')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        print(f"✅ 가격 데이터 변환 성공")
    except Exception as e:
        print(f"❌ 가격 변환 실패: {e}")
        return None, None, None
    
    # NaN 제거
    df = df.dropna(subset=['open', 'high', 'low', 'close']).reset_index(drop=True)
    
    print(f"✅ 데이터 로드 완료: {len(df)}개 캔들")
    print(f"   기간: {df['open_time'].iloc[0]} ~ {df['open_time'].iloc[-1]}")
    
    # 3. RSI 계산
    print("\n📊 RSI 계산 중...")
    df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
    df = df.dropna().reset_index(drop=True)
    print(f"✅ RSI 계산 완료: {len(df)}개 캔들")
    
    # 4. 다이버전스 감지
    print("\n🔍 다이버전스 감지 중...")
    divergences = detect_divergences_backtest(df)
    print(f"✅ 감지된 다이버전스: {len(divergences)}개")
    
    if len(divergences) == 0:
        print("⚠️ 다이버전스가 감지되지 않았습니다.")
        return df, [], pd.DataFrame()
    
    # 5. 트레이드 시뮬레이션
    print("\n💰 트레이드 시뮬레이션 중...")
    results = []
    
    for idx, div in enumerate(divergences, 1):
        trade_result = simulate_trade(df, div)
        
        results.append({
            'No': idx,
            '타입': '하락(SHORT)' if div['type'] == 'bearish' else '상승(LONG)',
            '진입시간': div['entry_time'],
            '진입가': div['entry_price'],
            '청산시간': df['open_time'].iloc[trade_result['exit_idx']],
            '청산가': trade_result['exit_price'],
            '보유봉수': trade_result['exit_idx'] - div['entry_idx'],
            'RSI_이전': f"{div['rsi_prev']:.1f}",
            'RSI_현재': f"{div['rsi_curr']:.1f}",
            '가격_이전': f"{div['price_prev']:.2f}",
            '가격_현재': f"{div['price_curr']:.2f}",
            '부분익절': '✅' if trade_result['partial_profit_hit'] else '❌',
            '스탑로스': '🚨' if trade_result['stop_loss_hit'] else '❌',
            '최종수익률(%)': f"{trade_result['final_profit_pct']:.2f}",
            '최대수익률(%)': f"{trade_result['max_profit_pct']:.2f}",
            '최대손실률(%)': f"{trade_result['max_loss_pct']:.2f}",
            '레버리지수익률(%)': f"{trade_result['final_profit_pct'] * LEVERAGE:.2f}"
        })
        
        # 진행 상황 출력
        if idx % 10 == 0:
            print(f"  진행: {idx}/{len(divergences)}")
    
    results_df = pd.DataFrame(results)
    
    # 6. CSV 저장
    csv_filename = 'divergence_analysis.csv'
    results_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"\n✅ CSV 저장 완료: {csv_filename}")
    
    # 7. 통계 출력
    print("\n" + "="*60)
    print("📈 분석 결과 요약")
    print("="*60)
    
    total = len(results_df)
    bearish_count = len(results_df[results_df['타입'] == '하락(SHORT)'])
    bullish_count = len(results_df[results_df['타입'] == '상승(LONG)'])
    
    print(f"총 다이버전스: {total}개")
    print(f"  - 하락(SHORT): {bearish_count}개")
    print(f"  - 상승(LONG): {bullish_count}개")
    print()
    
    partial_count = results_df['부분익절'].str.contains('✅').sum()
    stop_count = results_df['스탑로스'].str.contains('🚨').sum()
    print(f"부분익절 발생: {partial_count}개 ({partial_count/total*100:.1f}%)")
    print(f"스탑로스 발생: {stop_count}개 ({stop_count/total*100:.1f}%)")
    print()
    
    # 수익률 통계
    profits = [float(x) for x in results_df['최종수익률(%)']]
    win_trades = [p for p in profits if p > 0]
    loss_trades = [p for p in profits if p <= 0]
    
    print(f"승률: {len(win_trades)}/{total} ({len(win_trades)/total*100:.1f}%)")
    print(f"평균 수익률: {np.mean(profits):.2f}%")
    print(f"평균 레버리지 수익률: {np.mean(profits) * LEVERAGE:.2f}%")
    print(f"최대 수익: {max(profits):.2f}%")
    print(f"최대 손실: {min(profits):.2f}%")
    
    if win_trades:
        print(f"평균 수익(승리): {np.mean(win_trades):.2f}%")
    if loss_trades:
        print(f"평균 손실(패배): {np.mean(loss_trades):.2f}%")
    
    print("="*60)
    
    # 8. 차트 생성
    print("\n🎨 차트 생성 중...")
    create_divergence_charts(df, divergences, results_df)
    
    return df, divergences, results_df
    
    # 3. RSI 계산
    print("\n📊 RSI 계산 중...")
    df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
    df = df.dropna().reset_index(drop=True)
    
    # 4. 다이버전스 감지
    print("\n🔍 다이버전스 감지 중...")
    divergences = detect_divergences_backtest(df)
    print(f"✅ 감지된 다이버전스: {len(divergences)}개")
    
    if len(divergences) == 0:
        print("⚠️ 다이버전스가 감지되지 않았습니다.")
        return df, [], pd.DataFrame()
    
    # 5. 트레이드 시뮬레이션
    print("\n💰 트레이드 시뮬레이션 중...")
    results = []
    
    for idx, div in enumerate(divergences, 1):
        trade_result = simulate_trade(df, div)
        
        results.append({
            'No': idx,
            '타입': '하락(SHORT)' if div['type'] == 'bearish' else '상승(LONG)',
            '진입시간': div['entry_time'],
            '진입가': div['entry_price'],
            '청산시간': df['open_time'].iloc[trade_result['exit_idx']],
            '청산가': trade_result['exit_price'],
            '보유봉수': trade_result['exit_idx'] - div['entry_idx'],
            'RSI_이전': f"{div['rsi_prev']:.1f}",
            'RSI_현재': f"{div['rsi_curr']:.1f}",
            '가격_이전': f"{div['price_prev']:.2f}",
            '가격_현재': f"{div['price_curr']:.2f}",
            '부분익절': '✅' if trade_result['partial_profit_hit'] else '❌',
            '스탑로스': '🚨' if trade_result['stop_loss_hit'] else '❌',
            '최종수익률(%)': f"{trade_result['final_profit_pct']:.2f}",
            '최대수익률(%)': f"{trade_result['max_profit_pct']:.2f}",
            '최대손실률(%)': f"{trade_result['max_loss_pct']:.2f}",
            '레버리지수익률(%)': f"{trade_result['final_profit_pct'] * LEVERAGE:.2f}"
        })
        
        # 진행 상황 출력
        if idx % 10 == 0:
            print(f"  진행: {idx}/{len(divergences)}")
    
    results_df = pd.DataFrame(results)
    
    # 6. CSV 저장
    csv_filename = 'divergence_analysis.csv'
    results_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"\n✅ CSV 저장 완료: {csv_filename}")
    
    # 7. 통계 출력
    print("\n" + "="*60)
    print("📈 분석 결과 요약")
    print("="*60)
    
    total = len(results_df)
    bearish_count = len(results_df[results_df['타입'] == '하락(SHORT)'])
    bullish_count = len(results_df[results_df['타입'] == '상승(LONG)'])
    
    print(f"총 다이버전스: {total}개")
    print(f"  - 하락(SHORT): {bearish_count}개")
    print(f"  - 상승(LONG): {bullish_count}개")
    print()
    
    partial_count = results_df['부분익절'].str.contains('✅').sum()
    stop_count = results_df['스탑로스'].str.contains('🚨').sum()
    print(f"부분익절 발생: {partial_count}개 ({partial_count/total*100:.1f}%)")
    print(f"스탑로스 발생: {stop_count}개 ({stop_count/total*100:.1f}%)")
    print()
    
    # 수익률 통계
    profits = [float(x) for x in results_df['최종수익률(%)']]
    win_trades = [p for p in profits if p > 0]
    loss_trades = [p for p in profits if p <= 0]
    
    print(f"승률: {len(win_trades)}/{total} ({len(win_trades)/total*100:.1f}%)")
    print(f"평균 수익률: {np.mean(profits):.2f}%")
    print(f"평균 레버리지 수익률: {np.mean(profits) * LEVERAGE:.2f}%")
    print(f"최대 수익: {max(profits):.2f}%")
    print(f"최대 손실: {min(profits):.2f}%")
    
    if win_trades:
        print(f"평균 수익(승리): {np.mean(win_trades):.2f}%")
    if loss_trades:
        print(f"평균 손실(패배): {np.mean(loss_trades):.2f}%")
    
    print("="*60)
    
    # 8. 차트 생성
    print("\n🎨 차트 생성 중...")
    create_divergence_charts(df, divergences, results_df)
    
    return df, divergences, results_df

def create_divergence_charts(df, divergences, results_df):
    """다이버전스 차트 생성"""
    
    num_charts = min(20, len(divergences))
    
    # 개별 다이버전스마다 차트 생성
    for idx in range(num_charts):
        div = divergences[idx]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
        
        # 진입/청산 인덱스
        entry_idx = div['entry_idx']
        result = results_df.iloc[idx]
        
        # 표시할 범위 (진입 전후 100개 캔들)
        start_idx = max(0, entry_idx - 100)
        end_idx = min(len(df) - 1, entry_idx + HOLD_BARS + 20)
        
        df_slice = df.iloc[start_idx:end_idx]
        
        # 1. 가격 차트
        ax1.plot(df_slice['open_time'], df_slice['close'], 'k-', linewidth=0.8, label='Close')
        
        # 피봇 포인트 표시
        pivot_time = df['open_time'].iloc[div['pivot_idx']]
        prev_pivot_time = df['open_time'].iloc[div['prev_pivot_idx']]
        
        if div['type'] == 'bearish':
            ax1.scatter([prev_pivot_time, pivot_time], 
                       [div['price_prev'], div['price_curr']], 
                       color='red', s=100, zorder=5, label='Pivot High')
            ax1.plot([prev_pivot_time, pivot_time], 
                    [div['price_prev'], div['price_curr']], 
                    'r--', linewidth=2, alpha=0.5)
        else:
            ax1.scatter([prev_pivot_time, pivot_time], 
                       [div['price_prev'], div['price_curr']], 
                       color='green', s=100, zorder=5, label='Pivot Low')
            ax1.plot([prev_pivot_time, pivot_time], 
                    [div['price_prev'], div['price_curr']], 
                    'g--', linewidth=2, alpha=0.5)
        
        # 진입가 표시
        entry_time = df['open_time'].iloc[entry_idx]
        ax1.axvline(entry_time, color='blue', linestyle='--', linewidth=2, alpha=0.7, label='Entry')
        ax1.axhline(div['entry_price'], color='blue', linestyle=':', linewidth=1, alpha=0.5)
        
        # 청산가 표시
        exit_idx_val = int(result['보유봉수']) + entry_idx
        if exit_idx_val < len(df):
            exit_time = df['open_time'].iloc[exit_idx_val]
            ax1.axvline(exit_time, color='purple', linestyle='--', linewidth=2, alpha=0.7, label='Exit')
        
        # 스탑로스/부분익절 가격선
        stop_loss_pct = STOP_LOSS_BEAR if div['type'] == 'bearish' else STOP_LOSS_BULL
        if div['type'] == 'bearish':
            stop_price = div['entry_price'] * (1 + stop_loss_pct / 100)
            partial_price = div['entry_price'] * (1 - PARTIAL_PROFIT_TARGET / 100)
        else:
            stop_price = div['entry_price'] * (1 - stop_loss_pct / 100)
            partial_price = div['entry_price'] * (1 + PARTIAL_PROFIT_TARGET / 100)
        
        ax1.axhline(stop_price, color='red', linestyle=':', linewidth=1, alpha=0.5, label=f'Stop Loss {stop_loss_pct}%')
        ax1.axhline(partial_price, color='orange', linestyle=':', linewidth=1, alpha=0.5, label=f'Partial TP {PARTIAL_PROFIT_TARGET}%')
        
        ax1.set_ylabel('Price (USDT)', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 제목
        type_kr = 'RED SHORT' if div['type'] == 'bearish' else 'GREEN LONG'
        title = f"{idx+1}. {type_kr} Divergence | Entry: ${div['entry_price']:.0f} | "
        title += f"PnL: {result['최종수익률(%)']}% | "
        title += f"Partial: {result['부분익절']} | SL: {result['스탑로스']}"
        ax1.set_title(title, fontsize=14, fontweight='bold')
        
        # 2. RSI 차트
        ax2.plot(df_slice['open_time'], df_slice['rsi'], 'b-', linewidth=1, label='RSI')
        ax2.axhline(70, color='r', linestyle='--', linewidth=0.5, alpha=0.5)
        ax2.axhline(30, color='g', linestyle='--', linewidth=0.5, alpha=0.5)
        
        # RSI 피봇 표시
        if div['type'] == 'bearish':
            ax2.scatter([prev_pivot_time, pivot_time], 
                       [div['rsi_prev'], div['rsi_curr']], 
                       color='red', s=100, zorder=5)
            ax2.plot([prev_pivot_time, pivot_time], 
                    [div['rsi_prev'], div['rsi_curr']], 
                    'r--', linewidth=2, alpha=0.5, label='RSI Down')
        else:
            ax2.scatter([prev_pivot_time, pivot_time], 
                       [div['rsi_prev'], div['rsi_curr']], 
                       color='green', s=100, zorder=5)
            ax2.plot([prev_pivot_time, pivot_time], 
                    [div['rsi_prev'], div['rsi_curr']], 
                    'g--', linewidth=2, alpha=0.5, label='RSI Up')
        
        ax2.axvline(entry_time, color='blue', linestyle='--', linewidth=2, alpha=0.7)
        
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_ylabel('RSI', fontsize=12)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 100])
        
        # X축 날짜 포맷
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'divergence_{idx+1:03d}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        if (idx + 1) % 5 == 0:
            print(f"  Chart created: {idx+1}/{num_charts}")
    
    print(f"✅ Charts saved: divergence_001.png ~ divergence_{num_charts:03d}.png")
    
    # 전체 요약 차트
    create_summary_chart(df, divergences, results_df)

def create_summary_chart(df, divergences, results_df):
    """전체 요약 차트"""
    fig, axes = plt.subplots(3, 1, figsize=(20, 14))
    
    # 1. 가격 차트 + 모든 다이버전스
    ax1 = axes[0]
    ax1.plot(df['open_time'], df['close'], 'k-', linewidth=0.5, alpha=0.7)
    
    for div in divergences:
        entry_time = df['open_time'].iloc[div['entry_idx']]
        entry_price = div['entry_price']
        
        if div['type'] == 'bearish':
            ax1.scatter(entry_time, entry_price, color='red', s=30, alpha=0.6, marker='v')
        else:
            ax1.scatter(entry_time, entry_price, color='green', s=30, alpha=0.6, marker='^')
    
    ax1.set_ylabel('Price (USDT)', fontsize=12)
    ax1.set_title(f'All Divergence Signals (Total: {len(divergences)})', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(['Price', 'Bearish Div', 'Bullish Div'])
    
    # 2. RSI
    ax2 = axes[1]
    ax2.plot(df['open_time'], df['rsi'], 'b-', linewidth=0.5, alpha=0.7)
    ax2.axhline(70, color='r', linestyle='--', linewidth=0.5, alpha=0.3)
    ax2.axhline(30, color='g', linestyle='--', linewidth=0.5, alpha=0.3)
    ax2.set_ylabel('RSI', fontsize=12)
    ax2.set_title('RSI Indicator', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])
    
    # 3. 누적 수익률
    ax3 = axes[2]
    profits = [float(x) * LEVERAGE for x in results_df['최종수익률(%)']]
    cumulative = np.cumsum(profits)
    entry_times = [div['entry_time'] for div in divergences]
    
    ax3.plot(entry_times, cumulative, 'b-', linewidth=2, label='Cumulative PnL')
    ax3.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax3.fill_between(entry_times, 0, cumulative, alpha=0.3)
    ax3.set_xlabel('Time', fontsize=12)
    ax3.set_ylabel('Cumulative PnL (%)', fontsize=12)
    ax3.set_title(f'Cumulative PnL (Leverage {LEVERAGE}x)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # X축 날짜 포맷
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig('divergence_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Summary chart saved: divergence_summary.png")

# ============================================================================
# 실행
# ============================================================================

if __name__ == "__main__":
    # JSON 파일 경로 입력
    json_file = input("JSON 파일 경로를 입력하세요 (기본값: btc_15m_data.json): ").strip()
    
    if not json_file:
        json_file = "btc_15m_data.json"  # 기본값
    
    try:
        df, divergences, results_df = analyze_chart_data(json_file)
        
        if df is not None:
            print("\n✅ 모든 분석 완료!")
        
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {json_file}")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()