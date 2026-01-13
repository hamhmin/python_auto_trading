"""
RSI Divergence 파라미터 그리드 서치 (캐싱 최적화 버전 - Numba 불필요)
- 다이버전스 캐싱 (99% 중복 제거)
- Numpy 배열 최적화
- CPU 코어 최대 활용
"""
import pandas as pd
import numpy as np
import json
import itertools
from datetime import datetime
import sys
from multiprocessing import Pool, cpu_count
from functools import partial

class RSIDivergenceGridSearchCached:
    def __init__(self, json_file_path):
        self.json_file_path = json_file_path
        self.df = None
        self.all_results = []
        self.divergence_cache = {}
        
    def load_data(self):
        """JSON 파일에서 캔들 데이터 로드"""
        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.df = pd.DataFrame(data)
        
        # 컬럼명 자동 감지
        col_mapping = {}
        for col in self.df.columns:
            col_lower = col.lower()
            if col_lower in ['open', 'o']:
                col_mapping['open'] = col
            elif col_lower in ['high', 'h']:
                col_mapping['high'] = col
            elif col_lower in ['low', 'l']:
                col_mapping['low'] = col
            elif col_lower in ['close', 'c']:
                col_mapping['close'] = col
        
        if len(col_mapping) < 4:
            raise ValueError(f"필수 컬럼 없음. 발견된 컬럼: {list(self.df.columns)}")
        
        self.df = self.df.rename(columns={v: k for k, v in col_mapping.items()})
        
        # RSI 계산
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df['rsi'] = 100 - (100 / (1 + rs))
        self.df = self.df.dropna().reset_index(drop=True)
        
        print(f"✅ 데이터 로드 완료: {len(self.df)}개 캔들")
        return self
    
    def precompute_divergences(self, lookback_left_range, lookback_right_range):
        """다이버전스를 미리 계산하여 캐싱"""
        print(f"\n🔥 다이버전스 사전 계산 중...")
        
        # Numpy 배열로 변환
        rsi = np.array(self.df['rsi'].tolist())
        high = np.array(self.df['high'].tolist())
        low = np.array(self.df['low'].tolist())
        
        total = len(lookback_left_range) * len(lookback_right_range)
        count = 0
        
        for ll in lookback_left_range:
            for lr in lookback_right_range:
                bear_signals, bull_signals = detect_divergences_fast(rsi, high, low, ll, lr)
                self.divergence_cache[(ll, lr)] = (bear_signals, bull_signals)
                count += 1
                print(f"\r진행: {count}/{total} ({count*100/total:.1f}%)", end='', flush=True)
        
        print(f"\n✅ 다이버전스 캐싱 완료: {len(self.divergence_cache)}개 조합")
    
    def grid_search(self, 
                   lookback_left_range,
                   lookback_right_range, 
                   partial_profit_range, 
                   hold_bars_range,
                   stop_loss_range,
                   fee_rate=0.0,
                   n_jobs=None):
        """그리드 서치 실행 (캐싱 최적화)"""
        if self.df is None:
            self.load_data()
        
        # 1. 다이버전스 사전 계산 (캐싱)
        self.precompute_divergences(lookback_left_range, lookback_right_range)
        
        # 2. 모든 조합 생성
        combinations = list(itertools.product(
            lookback_left_range,
            lookback_right_range, 
            partial_profit_range, 
            hold_bars_range,
            stop_loss_range
        ))
        
        total_combinations = len(combinations)
        
        if n_jobs is None:
            n_jobs = cpu_count()
        
        print(f"\n🚀 캐싱 최적화 그리드 서치 시작")
        print(f"   lookback_left: {list(lookback_left_range)}")
        print(f"   lookback_right: {list(lookback_right_range)}")
        print(f"   partial_profit: {list(partial_profit_range)[:5]}{'...' if len(partial_profit_range) > 5 else ''}")
        print(f"   hold_bars: {list(hold_bars_range)[:5]}{'...' if len(hold_bars_range) > 5 else ''}")
        print(f"   stop_loss: {list(stop_loss_range)[:5]}{'...' if len(stop_loss_range) > 5 else ''}")
        print(f"   수수료율: {fee_rate}%")
        print(f"   총 테스트 조합: {total_combinations:,}개")
        print(f"   병렬 작업 수: {n_jobs}개 CPU 코어")
        print(f"   최적화: 다이버전스 캐싱 (중복 제거)\n")
        
        start_time = datetime.now()
        
        # 3. Numpy 배열로 변환
        df_arrays = {
            'close': np.array(self.df['close'].tolist()),
            'high': np.array(self.df['high'].tolist()),
            'low': np.array(self.df['low'].tolist())
        }
        
        # 4. 워커 함수 준비
        worker_func = partial(
            process_single_combination_cached, 
            df_arrays=df_arrays, 
            divergence_cache=self.divergence_cache,
            fee_rate=fee_rate
        )
        
        # 5. 병렬 처리
        with Pool(processes=n_jobs) as pool:
            results = []
            
            for i, result in enumerate(pool.imap(worker_func, combinations), 1):
                if result:
                    results.append(result)
                
                # 진행률 표시
                if i % 50 == 0 or i == total_combinations:
                    progress = (i / total_combinations) * 100
                    elapsed = (datetime.now() - start_time).total_seconds()
                    rate = i / elapsed if elapsed > 0 else 0
                    remaining = (total_combinations - i) / rate if rate > 0 else 0
                    
                    best_pnl = max([r['total_pnl'] for r in results]) if results else 0
                    
                    print(f"\r진행: {progress:5.1f}% ({i:,}/{total_combinations:,}) | "
                          f"속도: {rate:.0f}개/초 | 남은시간: {remaining:.0f}초 | "
                          f"현재 최고: {best_pnl:+.2f}%", end='', flush=True)
        
        print()
        
        elapsed_total = (datetime.now() - start_time).total_seconds()
        print(f"\n✅ 완료! 소요시간: {elapsed_total:.1f}초 | 유효한 결과: {len(results):,}개")
        print(f"   속도: {total_combinations/elapsed_total:.0f}개 조합/초\n")
        
        self.all_results = results
        return pd.DataFrame(results)
    
    def get_top_results(self, n=10, sort_by='total_pnl'):
        """상위 결과 조회"""
        if not self.all_results:
            print("먼저 grid_search()를 실행하세요")
            return None
        
        df = pd.DataFrame(self.all_results)
        df = df.sort_values(sort_by, ascending=False).head(n)
        return df
    
    def save_results(self, filename='grid_search_results_cached.csv'):
        """결과를 CSV 파일로 저장"""
        if not self.all_results:
            print("먼저 grid_search()를 실행하세요")
            return
        
        df = pd.DataFrame(self.all_results)
        df = df.sort_values('total_pnl', ascending=False)
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        
        print(f"💾 결과 저장: {filename}")


# ============================================================================
# 최적화 함수들 (Numpy 배열 사용)
# ============================================================================

def is_pivot_high_fast(series, idx, left, right):
    """피벗 고점 확인 (Numpy 최적화)"""
    n = len(series)
    if idx >= n:
        return False
    
    center = series[idx]
    
    # 왼쪽 체크
    start = max(0, idx - left)
    if np.any(series[start:idx] >= center):
        return False
    
    # 오른쪽 체크
    if right == 0:
        return True
    
    end = min(n, idx + right + 1)
    if np.any(series[idx+1:end] >= center):
        return False
    
    return True


def is_pivot_low_fast(series, idx, left, right):
    """피벗 저점 확인 (Numpy 최적화)"""
    n = len(series)
    if idx >= n:
        return False
    
    center = series[idx]
    
    # 왼쪽 체크
    start = max(0, idx - left)
    if np.any(series[start:idx] <= center):
        return False
    
    # 오른쪽 체크
    if right == 0:
        return True
    
    end = min(n, idx + right + 1)
    if np.any(series[idx+1:end] <= center):
        return False
    
    return True


def detect_divergences_fast(rsi, high, low, lookback_left, lookback_right):
    """다이버전스 감지 (Numpy 최적화)"""
    n = len(rsi)
    range_lower = 5
    range_upper = 60
    
    bear_signals = []
    bull_signals = []
    
    for i in range(lookback_left, n - lookback_right):
        # Bearish
        if is_pivot_high_fast(rsi, i, lookback_left, lookback_right):
            for j in range(i - range_lower, max(i - range_upper, lookback_left) - 1, -1):
                if j < lookback_left:
                    break
                if is_pivot_high_fast(rsi, j, lookback_left, lookback_right):
                    signal_idx = i + lookback_right
                    if signal_idx < n and rsi[i] < rsi[j] and high[i] > high[j]:
                        bear_signals.append(signal_idx)
                    break
        
        # Bullish
        if is_pivot_low_fast(rsi, i, lookback_left, lookback_right):
            for j in range(i - range_lower, max(i - range_upper, lookback_left) - 1, -1):
                if j < lookback_left:
                    break
                if is_pivot_low_fast(rsi, j, lookback_left, lookback_right):
                    signal_idx = i + lookback_right
                    if signal_idx < n and rsi[i] > rsi[j] and low[i] < low[j]:
                        bull_signals.append(signal_idx)
                    break
    
    return bear_signals, bull_signals


def execute_trade_fast(close, high, low, signal_idx, signal_type, 
                       partial_profit_target, hold_bars, stop_loss):
    """거래 실행 (Numpy 최적화)"""
    n = len(close)
    
    if signal_idx + hold_bars >= n:
        return None
    
    entry_price = close[signal_idx]
    
    partial_closed = False
    partial_pnl = 0.0
    max_profit = 0.0
    max_loss = 0.0
    stop_loss_hit = False
    exit_bar = hold_bars
    
    for i in range(signal_idx, signal_idx + hold_bars + 1):
        if signal_type == 'bear':
            current_profit = ((entry_price - low[i]) / entry_price) * 100
            current_loss = ((entry_price - high[i]) / entry_price) * 100
        else:
            current_profit = ((high[i] - entry_price) / entry_price) * 100
            current_loss = ((low[i] - entry_price) / entry_price) * 100
        
        # 최고 수익/손실 추적
        if current_profit > max_profit:
            max_profit = current_profit
        if current_loss < max_loss:
            max_loss = current_loss
        
        # 스탑로스 체크
        if stop_loss > 0 and current_loss <= -stop_loss:
            stop_loss_hit = True
            exit_bar = i - signal_idx
            if signal_type == 'bear':
                total_pnl = ((entry_price - high[i]) / entry_price) * 100
            else:
                total_pnl = ((low[i] - entry_price) / entry_price) * 100
            return {
                'pnl': total_pnl,
                'max_loss': max_loss,
                'max_profit': max_profit,
                'stop_loss_hit': True,
                'exit_bar': exit_bar
            }
        
        # 부분익절 체크
        if not partial_closed and current_profit >= partial_profit_target:
            partial_pnl = current_profit * 0.5
            partial_closed = True
    
    # 정상 청산
    exit_price = close[signal_idx + hold_bars]
    if signal_type == 'bear':
        remaining_pnl = ((entry_price - exit_price) / entry_price) * 100 * 0.5
    else:
        remaining_pnl = ((exit_price - entry_price) / entry_price) * 100 * 0.5
    
    total_pnl = partial_pnl + remaining_pnl
    
    return {
        'pnl': total_pnl,
        'max_loss': max_loss,
        'max_profit': max_profit,
        'stop_loss_hit': False,
        'exit_bar': exit_bar
    }


def process_single_combination_cached(params, df_arrays, divergence_cache, fee_rate):
    """단일 조합 처리 (캐싱 사용)"""
    ll, lr, pp, hb, sl = params
    
    # 캐시에서 다이버전스 가져오기
    bear_signals, bull_signals = divergence_cache[(ll, lr)]
    
    if len(bear_signals) == 0 and len(bull_signals) == 0:
        return None
    
    close = df_arrays['close']
    high = df_arrays['high']
    low = df_arrays['low']
    
    all_trades = []
    
    # Bear 거래
    for signal_idx in bear_signals:
        result = execute_trade_fast(close, high, low, signal_idx, 'bear', pp, hb, sl)
        if result:
            result['entry_bar'] = signal_idx
            result['signal_type'] = 'bear'
            all_trades.append(result)
    
    # Bull 거래
    for signal_idx in bull_signals:
        result = execute_trade_fast(close, high, low, signal_idx, 'bull', pp, hb, sl)
        if result:
            result['entry_bar'] = signal_idx
            result['signal_type'] = 'bull'
            all_trades.append(result)
    
    if not all_trades:
        return None
    
    # 시간순 정렬
    all_trades.sort(key=lambda x: x['entry_bar'])
    
    # 통계 계산
    total_trades = len(all_trades)
    total_pnl_before_fee = sum([t['pnl'] for t in all_trades])
    total_fee = total_trades * 2 * fee_rate
    total_pnl = total_pnl_before_fee - total_fee
    wins = sum(1 for t in all_trades if t['pnl'] > 0)
    win_rate = (wins / total_trades) * 100
    
    max_loss = min([t['max_loss'] for t in all_trades])
    max_profit = max([t['max_profit'] for t in all_trades])
    avg_max_loss = sum([t['max_loss'] for t in all_trades]) / total_trades
    
    winning_trades = [t['pnl'] for t in all_trades if t['pnl'] > 0]
    losing_trades = [t['pnl'] for t in all_trades if t['pnl'] <= 0]
    
    avg_win = sum(winning_trades) / len(winning_trades) if winning_trades else 0
    avg_loss = sum(losing_trades) / len(losing_trades) if losing_trades else 0
    avg_pnl = total_pnl / total_trades
    
    stop_loss_count = sum(1 for t in all_trades if t['stop_loss_hit'])
    stop_loss_rate = (stop_loss_count / total_trades) * 100
    
    # 동시 포지션 계산
    max_concurrent = 0
    n_bars = len(close)
    
    for bar in range(0, n_bars, 100):  # 샘플링으로 속도 향상
        concurrent_count = 0
        for trade in all_trades:
            entry = trade['entry_bar']
            exit_bar = entry + trade['exit_bar']
            if entry <= bar <= exit_bar:
                concurrent_count += 1
        if concurrent_count > max_concurrent:
            max_concurrent = concurrent_count
    
    return {
        'lookback_left': ll,
        'lookback_right': lr,
        'partial_profit': pp,
        'hold_bars': hb,
        'stop_loss': sl,
        'fee_rate': fee_rate,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_pnl_before_fee': total_pnl_before_fee,
        'total_fee': total_fee,
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'max_loss': max_loss,
        'avg_max_loss': avg_max_loss,
        'max_profit': max_profit,
        'stop_loss_count': stop_loss_count,
        'stop_loss_rate': stop_loss_rate,
        'max_concurrent_positions': max_concurrent
    }


def main():
    """메인 함수"""
    if len(sys.argv) < 2:
        print("""
사용법:
    python backtest_grid_search_cached.py <json_파일>

예시:
    python backtest_grid_search_cached.py btc_15m_data.json
        """)
        return
    
    json_file = sys.argv[1]
    
    searcher = RSIDivergenceGridSearchCached(json_file)
    
    print("\n" + "="*80)
    print("파라미터 범위를 입력하세요 (Enter = 기본값)")
    print("="*80)
    
    ll_input = input("lookback_left 범위 (예: 3-7) [기본: 5-5]: ").strip()
    if ll_input and '-' in ll_input:
        ll_start, ll_end = map(int, ll_input.split('-'))
        lookback_left_range = range(ll_start, ll_end + 1)
    else:
        lookback_left_range = range(5, 6)
    
    lr_input = input("lookback_right 범위 (예: 1-10) [기본: 1-5]: ").strip()
    if lr_input and '-' in lr_input:
        lr_start, lr_end = map(int, lr_input.split('-'))
        lookback_right_range = range(lr_start, lr_end + 1)
    else:
        lookback_right_range = range(1, 6)
    
    pp_input = input("부분익절% 범위 (예: 0.1-2.0-0.1) [기본: 0.4-2.0-0.1]: ").strip()
    if pp_input and '-' in pp_input:
        parts = pp_input.split('-')
        pp_start, pp_end, pp_step = float(parts[0]), float(parts[1]), float(parts[2])
        partial_profit_range = np.arange(pp_start, pp_end + pp_step/2, pp_step)
        partial_profit_range = np.round(partial_profit_range, 2)
    else:
        partial_profit_range = np.arange(0.4, 2.0, 0.1)
        partial_profit_range = np.round(partial_profit_range, 2)
    
    hb_input = input("보유기간(봉) 범위 (예: 5-30) [기본: 15-35]: ").strip()
    if hb_input and '-' in hb_input:
        hb_start, hb_end = map(int, hb_input.split('-'))
        hold_bars_range = range(hb_start, hb_end + 1)
    else:
        hold_bars_range = range(15, 36)
    
    sl_input = input("스탑로스% 범위 (예: 0.5-2.0-0.5 또는 0은 없음) [기본: 2.0-4.0-0.1]: ").strip()
    if sl_input and '-' in sl_input:
        parts = sl_input.split('-')
        sl_start, sl_end, sl_step = float(parts[0]), float(parts[1]), float(parts[2])
        stop_loss_range = np.arange(sl_start, sl_end + sl_step/2, sl_step)
        stop_loss_range = np.round(stop_loss_range, 2)
    else:
        stop_loss_range = np.arange(2, 4.0, 0.1)
        stop_loss_range = np.round(stop_loss_range, 2)
    
    fee_input = input("수수료율% (예: 0.05) [기본: 0.05]: ").strip()
    fee_rate = float(fee_input) if fee_input else 0.05
    
    # 그리드 서치 실행
    df_results = searcher.grid_search(
        lookback_left_range=lookback_left_range,
        lookback_right_range=lookback_right_range,
        partial_profit_range=partial_profit_range,
        hold_bars_range=hold_bars_range,
        stop_loss_range=stop_loss_range,
        fee_rate=fee_rate
    )
    
    print("\n" + "="*100)
    print("🏆 TOP 20 결과 (총수익 기준)")
    print("="*100)
    
    top_20 = searcher.get_top_results(n=20)
    print(top_20.to_string(index=False))
    
    searcher.save_results('grid_search_results_cached.csv')
    
    best = top_20.iloc[0]
    print("\n" + "="*100)
    print("🥇 최고 성과 파라미터")
    print("="*100)
    print(f"lookback_left: {best['lookback_left']}")
    print(f"lookback_right: {best['lookback_right']}")
    print(f"부분익절: {best['partial_profit']}%")
    print(f"보유기간: {best['hold_bars']}봉")
    print(f"스탑로스: {best['stop_loss']}%")
    print(f"총 거래: {best['total_trades']}개")
    print(f"승률: {best['win_rate']:.1f}%")
    print(f"총 수익: {best['total_pnl']:+.2f}%")
    print(f"평균 수익: {best['avg_pnl']:+.3f}%")
    print(f"평균 수익(승): {best['avg_win']:+.3f}%")
    print(f"평균 손실(패): {best['avg_loss']:+.3f}%")
    print(f"최고 손실: {best['max_loss']:.2f}%")
    print(f"평균 최고 손실: {best['avg_max_loss']:.2f}%")
    print(f"스탑로스 발동: {best['stop_loss_count']}회 ({best['stop_loss_rate']:.1f}%)")
    print(f"최대 동시 포지션: {best['max_concurrent_positions']}개")


if __name__ == "__main__":
    main()