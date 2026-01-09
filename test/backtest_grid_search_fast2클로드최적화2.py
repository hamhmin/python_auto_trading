"""
RSI Divergence 그리드 서치 (완전한 Numba 배치 처리)
- 모든 계산을 Numba 내부에서 처리
- Python 루프 최소화
- 제미나이 방식 완벽 구현
"""
import pandas as pd
import numpy as np
import json
import itertools
from datetime import datetime
import sys
from multiprocessing import Pool, cpu_count
from functools import partial
from numba import njit

# ============================================================================
# Numba 최적화 함수들
# ============================================================================

@njit
def is_pivot_high_nb(rsi, idx, left, right):
    """피벗 고점 확인"""
    if idx - left < 0 or idx + right >= len(rsi):
        return False
    center = rsi[idx]
    for i in range(idx - left, idx):
        if rsi[i] >= center:
            return False
    for i in range(idx + 1, idx + right + 1):
        if rsi[i] >= center:
            return False
    return True

@njit
def is_pivot_low_nb(rsi, idx, left, right):
    """피벗 저점 확인"""
    if idx - left < 0 or idx + right >= len(rsi):
        return False
    center = rsi[idx]
    for i in range(idx - left, idx):
        if rsi[i] <= center:
            return False
    for i in range(idx + 1, idx + right + 1):
        if rsi[i] <= center:
            return False
    return True

@njit
def get_signals_nb(rsi, high, low, ll, lr):
    """다이버전스 감지"""
    n = len(rsi)
    bear_sigs = []
    bull_sigs = []
    
    for i in range(ll, n - lr):
        # Bearish
        if is_pivot_high_nb(rsi, i, ll, lr):
            for j in range(i - 5, max(i - 60, ll), -1):
                if j < ll:
                    break
                if is_pivot_high_nb(rsi, j, ll, lr):
                    signal_idx = i + lr
                    if signal_idx < n and rsi[i] < rsi[j] and high[i] > high[j]:
                        bear_sigs.append(signal_idx)
                    break
        
        # Bullish
        if is_pivot_low_nb(rsi, i, ll, lr):
            for j in range(i - 5, max(i - 60, ll), -1):
                if j < ll:
                    break
                if is_pivot_low_nb(rsi, j, ll, lr):
                    signal_idx = i + lr
                    if signal_idx < n and rsi[i] > rsi[j] and low[i] < low[j]:
                        bull_sigs.append(signal_idx)
                    break
    
    return np.array(bear_sigs, dtype=np.int32), np.array(bull_sigs, dtype=np.int32)

@njit
def execute_trade_nb(close, high, low, sig_idx, is_bear, pp, hb, sl):
    """거래 실행 - 결과를 배열로 반환"""
    n = len(close)
    if sig_idx + hb >= n:
        return None
    
    entry_price = close[sig_idx]
    max_profit = 0.0
    max_loss = 0.0
    partial_pnl = 0.0
    partial_closed = False
    actual_exit_bar = hb  # 실제 청산 시점
    
    # 🔥 1단계: 전체 보유기간의 max_loss/max_profit 먼저 계산 (스탑로스 무시)
    for i in range(sig_idx, sig_idx + hb + 1):
        if is_bear:
            current_profit = ((entry_price - low[i]) / entry_price) * 100
            current_loss = ((entry_price - high[i]) / entry_price) * 100
        else:
            current_profit = ((high[i] - entry_price) / entry_price) * 100
            current_loss = ((low[i] - entry_price) / entry_price) * 100
        
        if current_profit > max_profit:
            max_profit = current_profit
        if current_loss < max_loss:
            max_loss = current_loss
    
    # 🔥 2단계: 스탑로스 체크 (실제 청산 시점 결정)
    for i in range(sig_idx, sig_idx + hb + 1):
        if is_bear:
            current_profit = ((entry_price - low[i]) / entry_price) * 100
            current_loss = ((entry_price - high[i]) / entry_price) * 100
        else:
            current_profit = ((high[i] - entry_price) / entry_price) * 100
            current_loss = ((low[i] - entry_price) / entry_price) * 100
        
        # 스탑로스
        if sl > 0 and current_loss <= -sl:
            if is_bear:
                total_pnl = ((entry_price - high[i]) / entry_price) * 100
            else:
                total_pnl = ((low[i] - entry_price) / entry_price) * 100
            
            result = np.zeros(4)
            result[0] = total_pnl
            result[1] = max_loss  # 🔥 전체 보유기간의 max_loss
            result[2] = max_profit  # 🔥 전체 보유기간의 max_profit
            result[3] = 1.0
            return result
        
        # 부분익절
        if not partial_closed and current_profit >= pp:
            partial_pnl = current_profit * 0.5
            partial_closed = True
    
    # 정상 청산
    exit_price = close[sig_idx + hb]
    if is_bear:
        remaining_pnl = ((entry_price - exit_price) / entry_price) * 100 * 0.5
    else:
        remaining_pnl = ((exit_price - entry_price) / entry_price) * 100 * 0.5
    
    total_pnl = partial_pnl + remaining_pnl
    
    result = np.zeros(4)
    result[0] = total_pnl
    result[1] = max_loss
    result[2] = max_profit
    result[3] = 0.0
    return result

@njit
def process_all_combos_nb(close, high, low, bear_sigs, bull_sigs, pp_arr, hb_arr, sl_arr):
    """🔥 모든 조합을 Numba 내부에서 처리!"""
    n_pp = len(pp_arr)
    n_hb = len(hb_arr)
    n_sl = len(sl_arr)
    n_bars = len(close)
    
    # 결과 저장: [pp_idx, hb_idx, sl_idx, n_trades, sum_pnl, sum_win, ..., max_concurrent]
    results = []
    
    for pp_idx in range(n_pp):
        pp = pp_arr[pp_idx]
        for hb_idx in range(n_hb):
            hb = hb_arr[hb_idx]
            for sl_idx in range(n_sl):
                sl = sl_arr[sl_idx]
                
                # 거래 결과 수집
                pnls = []
                max_losses = []
                max_profits = []
                sl_count = 0
                entry_bars = []  # 🔥 진입 인덱스 저장
                exit_bars = []   # 🔥 청산 인덱스 저장
                
                # Bear 거래
                for sig_idx in bear_sigs:
                    r = execute_trade_nb(close, high, low, sig_idx, True, pp, hb, sl)
                    if r is not None:
                        pnls.append(r[0])
                        max_losses.append(r[1])
                        max_profits.append(r[2])
                        sl_count += int(r[3])
                        
                        # 🔥 진입/청산 시점 저장
                        entry_bars.append(sig_idx)
                        # 스탑로스 발동 시 조기 청산 가능 (간단히 hb로 계산)
                        exit_bars.append(sig_idx + hb)
                
                # Bull 거래
                for sig_idx in bull_sigs:
                    r = execute_trade_nb(close, high, low, sig_idx, False, pp, hb, sl)
                    if r is not None:
                        pnls.append(r[0])
                        max_losses.append(r[1])
                        max_profits.append(r[2])
                        sl_count += int(r[3])
                        
                        entry_bars.append(sig_idx)
                        exit_bars.append(sig_idx + hb)
                
                if len(pnls) == 0:
                    continue
                
                # 🔥 최대 동시 포지션 계산
                max_concurrent = 0
                for bar in range(0, n_bars, 10):  # 샘플링 (10봉마다)
                    concurrent = 0
                    for i in range(len(entry_bars)):
                        if entry_bars[i] <= bar <= exit_bars[i]:
                            concurrent += 1
                    if concurrent > max_concurrent:
                        max_concurrent = concurrent
                
                # 통계 계산
                n_trades = len(pnls)
                sum_pnl = 0.0
                n_wins = 0
                sum_wins = 0.0
                sum_losses = 0.0
                min_pnl = pnls[0]
                max_profit_val = max_profits[0]
                max_loss_val = max_losses[0]  # 🔥 추가!
                sum_max_loss = 0.0
                sum_max_profit = 0.0  # 🔥 추가!
                
                for i in range(n_trades):
                    pnl = pnls[i]
                    sum_pnl += pnl
                    sum_max_loss += max_losses[i]
                    sum_max_profit += max_profits[i]  # 🔥 추가!
                    
                    if pnl > 0:
                        n_wins += 1
                        sum_wins += pnl
                    else:
                        sum_losses += pnl
                    
                    if pnl < min_pnl:
                        min_pnl = pnl
                    
                    if max_profits[i] > max_profit_val:
                        max_profit_val = max_profits[i]
                    
                    # 🔥 max_loss 최소값(최악) 찾기
                    if max_losses[i] < max_loss_val:
                        max_loss_val = max_losses[i]
                
                n_losses = n_trades - n_wins
                
                # 결과 저장 (16개: sum_max_profit 추가)
                result = np.zeros(16)
                result[0] = pp_idx
                result[1] = hb_idx
                result[2] = sl_idx
                result[3] = n_trades
                result[4] = sum_pnl
                result[5] = n_wins
                result[6] = sum_wins
                result[7] = n_losses
                result[8] = sum_losses
                result[9] = min_pnl
                result[10] = sum_max_loss
                result[11] = max_profit_val
                result[12] = sl_count
                result[13] = max_concurrent
                result[14] = max_loss_val  # 🔥 실제 최악의 max_loss
                result[15] = sum_max_profit  # 🔥 추가!
                
                results.append(result)
    
    return results

# ============================================================================
# 배치 워커
# ============================================================================

def process_batch_ultra(batch_data, close, high, low, rsi, pp_arr, hb_arr, sl_arr, fee_rate):
    """배치 처리: (ll, lr)마다 다이버전스 1번 계산 후 Numba에서 전체 처리"""
    ll, lr = batch_data['ll_lr']
    
    # 다이버전스 계산 (1번만!)
    bear_sigs, bull_sigs = get_signals_nb(rsi, high, low, ll, lr)
    
    if len(bear_sigs) == 0 and len(bull_sigs) == 0:
        return []
    
    # 🔥 모든 조합을 Numba 내부에서 처리!
    numba_results = process_all_combos_nb(close, high, low, bear_sigs, bull_sigs, pp_arr, hb_arr, sl_arr)
    
    # Python 딕셔너리로 변환
    batch_results = []
    for r in numba_results:
        pp_idx = int(r[0])
        hb_idx = int(r[1])
        sl_idx = int(r[2])
        n_trades = int(r[3])
        sum_pnl = r[4]
        n_wins = int(r[5])
        sum_wins = r[6]
        n_losses = int(r[7])
        sum_losses = r[8]
        min_pnl = r[9]
        sum_max_loss = r[10]
        max_profit_val = r[11]
        sl_count = int(r[12])
        max_concurrent = int(r[13])
        max_loss_val = r[14]  # 🔥 실제 최악의 max_loss
        sum_max_profit = r[15]  # 🔥 추가!
        
        pp = pp_arr[pp_idx]
        hb = hb_arr[hb_idx]
        sl = sl_arr[sl_idx]
        
        total_pnl_before_fee = sum_pnl
        total_fee = n_trades * 2 * fee_rate
        total_pnl = total_pnl_before_fee - total_fee
        
        win_rate = (n_wins / n_trades) * 100
        avg_pnl = total_pnl / n_trades
        avg_win = sum_wins / n_wins if n_wins > 0 else 0
        avg_loss = sum_losses / n_losses if n_losses > 0 else 0
        avg_max_loss = sum_max_loss / n_trades
        avg_max_profit = sum_max_profit / n_trades  # 🔥 추가!
        stop_loss_rate = (sl_count / n_trades) * 100
        
        # 🔥 추가 지표 계산
        # 1. 켈리 지수 (Kelly Criterion Score)
        # = (승률 * (평균익절/abs(평균손실)) - (1-승률)) / (평균익절/abs(평균손실))
        win_rate_decimal = win_rate / 100.0
        if avg_loss != 0 and avg_win != 0:
            win_loss_ratio = avg_win / abs(avg_loss)
            kelly_criterion = ((win_rate_decimal * win_loss_ratio) - (1 - win_rate_decimal)) / win_loss_ratio
        else:
            kelly_criterion = 0.0
        
        # 2. 수익 안정성 점수 (Expectancy Score)
        # = 총수익 * 승률
        expectancy_score = total_pnl * win_rate_decimal
        
        # 3. 통합 최강 전략 점수 (Ultimate Rank)
        # = ((승률 * 평균익절) + ((1-승률) * 평균손실)) / abs(최대손실)
        if max_loss_val != 0:
            expected_return = (win_rate_decimal * avg_win) + ((1 - win_rate_decimal) * avg_loss)
            ultimate_rank = expected_return / abs(max_loss_val)  # 🔥 수정! max_loss_val 사용
        else:
            ultimate_rank = 0.0
        
        # 4. 최대 레버리지 + 분할 = PNL 지수
        # = total_pnl / max_concurrent * 100 / abs(max_loss_val)
        if max_concurrent > 0 and max_loss_val != 0:
            leverage_calc = total_pnl / max_concurrent * 100 / abs(max_loss_val)
        else:
            leverage_calc = 0.0
        #손익비
        loss_profit_per = (avg_win + avg_loss) * win_rate / 100

        batch_results.append({
            'lookback_left': ll,
            'lookback_right': lr,
            'partial_profit': pp,
            'hold_bars': hb,
            'stop_loss': sl,
            'fee_rate': fee_rate,
            'total_trades': n_trades,
            'win_rate': round(win_rate, 2),
            'total_pnl_before_fee': round(total_pnl_before_fee, 4),
            'total_fee': round(total_fee, 4),
            'total_pnl': round(total_pnl, 4),
            'avg_pnl': round(avg_pnl, 4),
            'avg_win': round(avg_win, 4),
            'avg_loss': round(avg_loss, 4),
            'max_loss': round(max_loss_val, 4),  # 🔥 수정! 실제 최악의 손실 폭
            'worst_trade_pnl': round(min_pnl, 4),  # 🔥 추가! 최악의 거래 총수익
            'avg_max_loss': round(avg_max_loss, 4),
            'avg_max_profit': round(avg_max_profit, 4),  # 🔥 추가!
            'max_profit': round(max_profit_val, 4),
            'stop_loss_count': sl_count,
            'stop_loss_rate': round(stop_loss_rate, 2),
            'max_concurrent_positions': max_concurrent,
            'kelly_criterion': round(kelly_criterion, 4),
            'expectancy_score': round(expectancy_score, 4),
            'ultimate_rank': round(ultimate_rank, 4),
            'max_lvg+분할=pnl': round(leverage_calc, 4),  # 🔥 추가!
            '손익비': round(loss_profit_per, 4)  # 🔥 추가!
        })
    
    return batch_results

# ============================================================================
# 메인 실행
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print("사용법: python script.py <data.json>")
        return
    
    json_file = sys.argv[1]
    
    print("\n📊 데이터 로딩...")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    df.columns = [c.lower() for c in df.columns]
    
    # Numpy 배열로 변환
    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    
    # RSI 계산
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rsi = (100 - (100 / (1 + (gain / loss.replace(0, 1e-10))))).values
    
    print(f"✅ 데이터: {len(df)}개 캔들\n")
    
    # 파라미터 입력
    print("="*60)
    print("🎯 파라미터 범위 입력 (시작-끝)")
    print("="*60)
    
    def get_input(msg, def_s, def_e, is_float=False):
        raw = input(f"{msg} (기본 {def_s}-{def_e}): ").strip()
        if not raw:
            return (def_s, def_e)
        s, e = raw.split('-')
        return (float(s), float(e)) if is_float else (int(s), int(e))
    
    ll_s, ll_e = get_input("lookback_left", 5, 5)
    lr_s, lr_e = get_input("lookback_right", 1, 10)
    pp_s, pp_e = get_input("부분익절 %", 0.4, 2.0, True)
    hb_s, hb_e = get_input("보유기간(봉)", 15, 35)
    sl_s, sl_e = get_input("스탑로스 %", 2.0, 4.0, True)
    
    fee_input = input("수수료율 % (기본 0.05): ").strip()
    fee_rate = float(fee_input) if fee_input else 0.05
    
    # 범위 생성 (Numpy 배열)
    ll_range = np.arange(ll_s, ll_e + 1, dtype=np.int32)
    lr_range = np.arange(lr_s, lr_e + 1, dtype=np.int32)
    pp_arr = np.round(np.arange(pp_s, pp_e + 0.01, 0.1), 1)
    hb_arr = np.arange(hb_s, hb_e + 1, dtype=np.int32)
    sl_arr = np.round(np.arange(sl_s, sl_e + 0.01, 0.1), 1)
    
    # 배치 생성
    tasks = [{'ll_lr': (int(ll), int(lr))} for ll, lr in itertools.product(ll_range, lr_range)]
    
    total_combos = len(tasks) * len(pp_arr) * len(hb_arr) * len(sl_arr)
    
    print(f"\n🚀 초고속 Numba 배치 처리 시작")
    print(f"   lookback 조합: {len(tasks)}개")
    print(f"   각 조합당: {len(pp_arr) * len(hb_arr) * len(sl_arr)}개")
    print(f"   총 조합: {total_combos:,}개")
    print(f"   CPU: {cpu_count()-2} 코어\n")
    
    start_time = datetime.now()
    
    # 워커 함수
    worker = partial(
        process_batch_ultra,
        close=close, high=high, low=low, rsi=rsi,
        pp_arr=pp_arr, hb_arr=hb_arr, sl_arr=sl_arr,
        fee_rate=fee_rate
    )
    
    # 병렬 처리
    all_results = []
    with Pool(processes=cpu_count() - 2) as pool:
        for i, batch_results in enumerate(pool.imap_unordered(worker, tasks), 1):
            all_results.extend(batch_results)
            
            if i % 2 == 0 or i == len(tasks):
                elapsed = (datetime.now() - start_time).total_seconds()
                done = i * len(pp_arr) * len(hb_arr) * len(sl_arr)
                speed = done / elapsed if elapsed > 0 else 0
                
                print(f"\r진행: {i}/{len(tasks)} 배치 | "
                      f"속도: {speed:.0f}조합/초 | "
                      f"{elapsed:.1f}초", end='', flush=True)
    
    print()
    
    elapsed_total = (datetime.now() - start_time).total_seconds()
    
    # 결과 저장
    if all_results:
        result_df = pd.DataFrame(all_results)
        
        # 🔧 마이너스 수익률 제외
        positive_results = result_df[result_df['total_pnl'] > 0].copy()
        
        if len(positive_results) == 0:
            print("\n❌ 수익이 나는 조합이 없습니다!")
            return
        
        positive_results = positive_results.sort_values('total_pnl', ascending=False)
        filename = f"backtest_ultra_{datetime.now().strftime('%m%d_%H%M%S')}.csv"
        positive_results.to_csv(filename, index=False, encoding='utf-8-sig')
        
        # 필터링 정보 출력
        filtered_count = len(result_df) - len(positive_results)
        print(f"\n📊 전체 결과: {len(result_df):,}개 | 수익 조합: {len(positive_results):,}개 | 제외: {filtered_count:,}개")
        
        print(f"\n✅ 완료! {elapsed_total:.1f}초 | {total_combos/elapsed_total:.0f}조합/초")
        print(f"💾 {filename}\n")
        
        # TOP 10
        print("="*100)
        print("🏆 TOP 1 (수익 조합만)")
        print("="*100)
        print(positive_results.head(1).to_string(index=False))
        
        # 최고 결과
        best = positive_results.iloc[0]
        print(f"\n🥇 최고 수익: {best['total_pnl']:+.2f}%")
        print(f"   ll={best['lookback_left']}, lr={best['lookback_right']}, "
              f"pp={best['partial_profit']}, hb={best['hold_bars']}, sl={best['stop_loss']}")
        print(f"\n📊 추가 지표:")
        print(f"   켈리 지수: {best['kelly_criterion']:.4f}")
        print(f"   수익 안정성: {best['expectancy_score']:.4f}")
        print(f"   통합 점수: {best['ultimate_rank']:.4f}")
        
        # 각 지표별 최고 전략
        print(f"\n🎯 지표별 최고 전략:")
        
        best_kelly = positive_results.nlargest(1, 'kelly_criterion').iloc[0]
        print(f"\n   켈리 지수 최고: {best_kelly['kelly_criterion']:.4f}")
        print(f"   → ll={best_kelly['lookback_left']}, lr={best_kelly['lookback_right']}, "
              f"pp={best_kelly['partial_profit']}, hb={best_kelly['hold_bars']}, sl={best_kelly['stop_loss']}")
        print(f"   → 총수익: {best_kelly['total_pnl']:+.2f}%, 승률: {best_kelly['win_rate']:.1f}%")
        
        best_expectancy = positive_results.nlargest(1, 'expectancy_score').iloc[0]
        print(f"\n   수익 안정성 최고: {best_expectancy['expectancy_score']:.4f}")
        print(f"   → ll={best_expectancy['lookback_left']}, lr={best_expectancy['lookback_right']}, "
              f"pp={best_expectancy['partial_profit']}, hb={best_expectancy['hold_bars']}, sl={best_expectancy['stop_loss']}")
        print(f"   → 총수익: {best_expectancy['total_pnl']:+.2f}%, 승률: {best_expectancy['win_rate']:.1f}%")
        
        best_ultimate = positive_results.nlargest(1, 'ultimate_rank').iloc[0]
        print(f"\n   통합 점수 최고: {best_ultimate['ultimate_rank']:.4f}")
        print(f"   → ll={best_ultimate['lookback_left']}, lr={best_ultimate['lookback_right']}, "
              f"pp={best_ultimate['partial_profit']}, hb={best_ultimate['hold_bars']}, sl={best_ultimate['stop_loss']}")
        print(f"   → 총수익: {best_ultimate['total_pnl']:+.2f}%, 승률: {best_ultimate['win_rate']:.1f}%")

        best_lvg_pnl = positive_results.nlargest(1, 'max_lvg+분할=pnl').iloc[0]
        print(f"\n   최대 레버리지+분할 적용 pnl: {best_lvg_pnl['ultimate_rank']:.4f}")
        print(f"   → ll={best_lvg_pnl['lookback_left']}, lr={best_lvg_pnl['lookback_right']}, "
              f"pp={best_lvg_pnl['partial_profit']}, hb={best_lvg_pnl['hold_bars']}, sl={best_lvg_pnl['stop_loss']}")
        print(f"   → 총수익: {best_lvg_pnl['total_pnl']:+.2f}%, 승률: {best_lvg_pnl['win_rate']:.1f}%")

        best_pnl_ratio = positive_results.nlargest(1, '손익비').iloc[0]
        print(f"\n   손익비: {best_pnl_ratio['ultimate_rank']:.4f}")
        print(f"   → ll={best_pnl_ratio['lookback_left']}, lr={best_pnl_ratio['lookback_right']}, "
              f"pp={best_pnl_ratio['partial_profit']}, hb={best_pnl_ratio['hold_bars']}, sl={best_pnl_ratio['stop_loss']}")
        print(f"   → 총수익: {best_pnl_ratio['total_pnl']:+.2f}%, 승률: {best_pnl_ratio['win_rate']:.1f}%")

    else:
        print("\n❌ 결과 없음")

if __name__ == "__main__":
    main()