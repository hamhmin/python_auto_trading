import pandas as pd
import numpy as np
import json
import itertools
from datetime import datetime
import sys
import os
from multiprocessing import Pool, cpu_count
from functools import partial
from numba import njit

# ============================================================================
# 1. Numba 가속 엔진 (핵심 연산)
# ============================================================================

@njit
def is_pivot_high_nb(rsi, idx, left, right):
    if idx - left < 0 or idx + right >= len(rsi): return False
    center = rsi[idx]
    for i in range(idx - left, idx):
        if rsi[i] >= center: return False
    for i in range(idx + 1, idx + right + 1):
        if rsi[i] >= center: return False
    return True

@njit
def is_pivot_low_nb(rsi, idx, left, right):
    if idx - left < 0 or idx + right >= len(rsi): return False
    center = rsi[idx]
    for i in range(idx - left, idx):
        if rsi[i] <= center: return False
    for i in range(idx + 1, idx + right + 1):
        if rsi[i] <= center: return False
    return True

@njit
def get_signals_nb(rsi, high, low, ll, lr):
    n = len(rsi)
    bear_sigs, bull_sigs = [], []
    for i in range(ll, n - lr):
        if is_pivot_high_nb(rsi, i, ll, lr):
            for j in range(i - 5, max(i - 60, ll), -1):
                if is_pivot_high_nb(rsi, j, ll, lr):
                    if rsi[i] < rsi[j] and high[i] > high[j]:
                        bear_sigs.append(i + lr)
                    break
        if is_pivot_low_nb(rsi, i, ll, lr):
            for j in range(i - 5, max(i - 60, ll), -1):
                if is_pivot_low_nb(rsi, j, ll, lr):
                    if rsi[i] > rsi[j] and low[i] < low[j]:
                        bull_sigs.append(i + lr)
                    break
    return np.array(bear_sigs), np.array(bull_sigs)

@njit
def execute_trade_nb(close, high, low, sig_idx, sig_type, pp, hb, sl):
    n = len(close)
    if sig_idx + hb >= n: return None
    entry_p = close[sig_idx]
    max_p, max_l = 0.0, 0.0
    res = np.zeros(4) # [pnl, max_l, max_p, sl_hit]
    
    for i in range(sig_idx, sig_idx + hb + 1):
        if sig_type == 1: # bear
            c_p, c_l = (entry_p - low[i])/entry_p*100, (entry_p - high[i])/entry_p*100
        else: # bull
            c_p, c_l = (high[i] - entry_p)/entry_p*100, (low[i] - entry_p)/entry_p*100
        if c_p > max_p: max_p = c_p
        if c_l < max_l: max_l = c_l
        if sl > 0 and c_l <= -sl:
            res[0], res[1], res[2], res[3] = -sl, max_l, max_p, 1.0
            return res
            
    exit_p = close[sig_idx + hb]
    raw = (entry_p - exit_p)/entry_p*100 if sig_type == 1 else (exit_p - entry_p)/entry_p*100
    res[0] = (pp*0.5 + raw*0.5) if max_p >= pp else raw
    res[1], res[2], res[3] = max_l, max_p, 0.0
    return res

# ============================================================================
# 2. 병렬 워커 (모든 지표 계산)
# ============================================================================

def process_batch(batch_data, close, high, low, rsi, fee_rate):
    ll, lr = batch_data['ll_lr']
    combos = batch_data['combos']
    bear_sigs, bull_sigs = get_signals_nb(rsi, high, low, ll, lr)
    
    batch_results = []
    for pp, hb, sl in combos:
        pnls, max_ls, max_ps = [], [], []
        sl_count = 0
        
        for sigs, s_type in [(bear_sigs, 1), (bull_sigs, 2)]:
            for s in sigs:
                r = execute_trade_nb(close, high, low, s, s_type, pp, hb, sl)
                if r is not None:
                    pnls.append(r[0]); max_ls.append(r[1]); max_ps.append(r[2])
                    sl_count += int(r[3])
        
        if not pnls: continue
        
        n = len(pnls)
        wins = [x for x in pnls if x > 0]
        losses = [x for x in pnls if x <= 0]
        total_pnl_before = sum(pnls)
        total_fee = n * 2 * fee_rate
        total_pnl = total_pnl_before - total_fee
        
        batch_results.append({
            'lookback_left': ll, 'lookback_right': lr, 'partial_profit': pp, 'hold_bars': hb, 'stop_loss': sl,
            'fee_rate': fee_rate,
            'total_trades': n,
            'win_rate': round(len(wins)/n*100, 2),
            'total_pnl_before_fee': round(total_pnl_before, 4),
            'total_fee': round(total_fee, 4),
            'total_pnl': round(total_pnl, 4),
            'avg_pnl': round(total_pnl/n, 4),
            'avg_win': round(sum(wins)/len(wins), 4) if wins else 0,
            'avg_loss': round(sum(losses)/len(losses), 4) if losses else 0,
            'max_loss': round(min(pnls), 4),
            'avg_max_loss': round(sum(max_ls)/n, 4),
            'max_profit': round(max(max_ps), 4),
            'stop_loss_count': sl_count,
            'stop_loss_rate': round(sl_count/n*100, 2),
            'max_concurrent_positions': 1
        })
    return batch_results

# ============================================================================
# 3. 메인 실행 및 입력 인터페이스
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print("사용법: python script.py <data.json>")
        return

    print("📊 데이터 로딩 중...")
    with open(sys.argv[1], 'r', encoding='utf-8') as f:
        df = pd.DataFrame(json.load(f))
    
    df.columns = [c.lower() for c in df.columns]
    close, high, low = df['close'].values.astype(np.float64), df['high'].values.astype(np.float64), df['low'].values.astype(np.float64)
    
    # RSI 계산
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rsi = (100 - (100 / (1 + (gain / loss.replace(0, 1e-10))))).values

    print("\n" + "="*50)
    print("🎯 파라미터 테스트 범위 입력 (시작-끝)")
    print("="*50)

    def get_input(msg, def_s, def_e, is_f=False):
        raw = input(f"{msg} (기본 {def_s}-{def_e}): ").strip()
        if not raw: return (def_s, def_e)
        s, e = map(float if is_f else int, raw.split('-'))
        return (s, e)

    ll_s, ll_e = get_input("lookback_left", 5, 15)
    lr_s, lr_e = get_input("lookback_right", 1, 10)
    pp_s, pp_e = get_input("partial_profit %", 0.4, 2.0, True)
    hb_s, hb_e = get_input("hold_bars", 25, 50)
    sl_s, sl_e = get_input("stop_loss %", 2.0, 4.0, True)
    fee = float(input("수수료율 % (기본 0.05): ") or 0.05)

    # 범위 생성
    ll_range = range(int(ll_s), int(ll_e) + 1)
    lr_range = range(int(lr_s), int(lr_e) + 1)
    pp_range = np.round(np.arange(pp_s, pp_e + 0.01, 0.1), 1)
    hb_range = range(int(hb_s), int(hb_e) + 1)
    sl_range = np.round(np.arange(sl_s, sl_e + 0.01, 0.1), 1)

    tasks = [{'ll_lr': (ll, lr), 'combos': list(itertools.product(pp_range, hb_range, sl_range))} 
             for ll, lr in itertools.product(ll_range, lr_range)]

    total_c = len(tasks) * len(tasks[0]['combos'])
    print(f"\n🚀 총 {total_c:,}개 조합 테스트 시작... (CPU: {cpu_count()-2} 스레드 사용)")

    start_t = datetime.now()
    worker = partial(process_batch, close=close, high=high, low=low, rsi=rsi, fee_rate=fee)
    
    all_res = []
    with Pool(processes=cpu_count() - 2) as pool:
        for i, res in enumerate(pool.imap_unordered(worker, tasks), 1):
            all_res.extend(res)
            if i % 5 == 0 or i == len(tasks):
                elapsed = (datetime.now() - start_t).total_seconds()
                done = i * len(tasks[0]['combos'])
                print(f"\r진행: {i}/{len(tasks)} 그룹 | 속도: {done/elapsed:.0f}조합/초 | 소요시간: {elapsed:.1f}초", end='')

    # 결과 저장
    if all_res:
        result_df = pd.DataFrame(all_res).sort_values('total_pnl', ascending=False)
        filename = f"backtest_report_{datetime.now().strftime('%m%d_%H%M%S')}.csv"
        result_df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\n\n✅ 완료! 결과 파일: {filename}")
        print(f"🥇 최고 수익률: {result_df.iloc[0]['total_pnl']}%")
    else:
        print("\n❌ 조건에 맞는 거래 결과가 없습니다.")

if __name__ == "__main__":
    main()