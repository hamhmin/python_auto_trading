"""
RSI Divergence 파라미터 그리드 서치
모든 경우의 수를 테스트하여 최적 파라미터 찾기
"""
import pandas as pd
import numpy as np
import json
import itertools
from datetime import datetime
import sys

class RSIDivergenceGridSearch:
    def __init__(self, json_file_path):
        """
        Parameters:
        -----------
        json_file_path : str
            캔들 데이터 JSON 파일 경로
        """
        self.json_file_path = json_file_path
        self.df = None
        self.all_results = []
        
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
        
        # 컬럼명 표준화
        self.df = self.df.rename(columns={v: k for k, v in col_mapping.items()})
        
        # RSI 계산
        self.df['rsi'] = self._calculate_rsi(self.df['close'], 14)
        self.df = self.df.dropna().reset_index(drop=True)
        
        print(f"✅ 데이터 로드 완료: {len(self.df)}개 캔들")
        return self
    
    def _calculate_rsi(self, data, period=14):
        """RSI 계산"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _find_pivot_high(self, series, left, right, idx):
        """피벗 고점 찾기"""
        if idx < left or idx >= len(series) - right:
            return False
        center_value = series.iloc[idx]
        left_lower = all(series.iloc[idx-left:idx] < center_value)
        if right == 0:
            right_lower = True
        else:
            right_lower = all(series.iloc[idx+1:idx+right+1] < center_value)
        return left_lower and right_lower
    
    def _find_pivot_low(self, series, left, right, idx):
        """피벗 저점 찾기"""
        if idx < left or idx >= len(series) - right:
            return False
        center_value = series.iloc[idx]
        left_higher = all(series.iloc[idx-left:idx] > center_value)
        if right == 0:
            right_higher = True
        else:
            right_higher = all(series.iloc[idx+1:idx+right+1] > center_value)
        return left_higher and right_higher
    
    def detect_divergences(self, lookback_right):
        """Regular Divergence 감지"""
        lookback_left = 5
        range_lower = 5
        range_upper = 60
        
        regular_bear = []
        regular_bull = []
        
        for i in range(len(self.df)):
            # Bearish Divergence
            if self._find_pivot_high(self.df['rsi'], lookback_left, lookback_right, i):
                prev_pivot_idx = None
                for j in range(i - range_lower, max(i - range_upper, 0), -1):
                    if self._find_pivot_high(self.df['rsi'], lookback_left, lookback_right, j):
                        prev_pivot_idx = j
                        break
                
                if prev_pivot_idx is not None:
                    signal_idx = i + lookback_right
                    if signal_idx < len(self.df):
                        rsi_curr = self.df['rsi'].iloc[i]
                        rsi_prev = self.df['rsi'].iloc[prev_pivot_idx]
                        price_curr = self.df['high'].iloc[i]
                        price_prev = self.df['high'].iloc[prev_pivot_idx]
                        
                        if rsi_curr < rsi_prev and price_curr > price_prev:
                            regular_bear.append({'signal_index': signal_idx})
            
            # Bullish Divergence
            if self._find_pivot_low(self.df['rsi'], lookback_left, lookback_right, i):
                prev_pivot_idx = None
                for j in range(i - range_lower, max(i - range_upper, 0), -1):
                    if self._find_pivot_low(self.df['rsi'], lookback_left, lookback_right, j):
                        prev_pivot_idx = j
                        break
                
                if prev_pivot_idx is not None:
                    signal_idx = i + lookback_right
                    if signal_idx < len(self.df):
                        rsi_curr = self.df['rsi'].iloc[i]
                        rsi_prev = self.df['rsi'].iloc[prev_pivot_idx]
                        price_curr = self.df['low'].iloc[i]
                        price_prev = self.df['low'].iloc[prev_pivot_idx]
                        
                        if rsi_curr > rsi_prev and price_curr < price_prev:
                            regular_bull.append({'signal_index': signal_idx})
        
        return regular_bear, regular_bull
    
    def _execute_trade(self, signal_idx, signal_type, partial_profit_target, hold_bars):
        """거래 실행 및 결과 계산"""
        if signal_idx + hold_bars >= len(self.df):
            return None
        
        entry_price = self.df['close'].iloc[signal_idx]
        
        partial_closed = False
        partial_pnl = 0
        
        # 보유 기간 동안 부분 익절 체크
        for i in range(signal_idx, signal_idx + hold_bars + 1):
            current_high = self.df['high'].iloc[i]
            current_low = self.df['low'].iloc[i]
            
            if signal_type == 'bear':
                current_profit = ((entry_price - current_low) / entry_price) * 100
            else:
                current_profit = ((current_high - entry_price) / entry_price) * 100
            
            # 목표가 도달 시 부분 익절
            if not partial_closed and current_profit >= partial_profit_target:
                partial_pnl = current_profit * 0.5  # 50% 포지션
                partial_closed = True
        
        # 나머지 포지션 청산
        exit_price = self.df['close'].iloc[signal_idx + hold_bars]
        if signal_type == 'bear':
            remaining_pnl = ((entry_price - exit_price) / entry_price) * 100 * 0.5
        else:
            remaining_pnl = ((exit_price - entry_price) / entry_price) * 100 * 0.5
        
        total_pnl = partial_pnl + remaining_pnl
        
        return {
            'pnl': total_pnl,
            'partial_closed': partial_closed
        }
    
    def run_single_test(self, lookback_right, partial_profit, hold_bars, fee_rate):
        """단일 파라미터 조합 테스트"""
        # 다이버전스 신호 감지
        bear_signals, bull_signals = self.detect_divergences(lookback_right)
        
        # 거래 실행
        bear_trades = []
        for signal in bear_signals:
            result = self._execute_trade(signal['signal_index'], 'bear', 
                                        partial_profit, hold_bars)
            if result:
                bear_trades.append(result)
        
        bull_trades = []
        for signal in bull_signals:
            result = self._execute_trade(signal['signal_index'], 'bull', 
                                        partial_profit, hold_bars)
            if result:
                bull_trades.append(result)
        
        # 통계 계산
        all_trades = bear_trades + bull_trades
        total_trades = len(all_trades)
        
        if total_trades == 0:
            return None
        
        # 수익 계산 (수수료 전)
        total_pnl_before_fee = sum([t['pnl'] for t in all_trades])
        
        # 수수료 계산: 거래 횟수 * 2 * 수수료율
        total_fee = total_trades * 2 * fee_rate
        
        # 최종 수익 (수수료 후)
        total_pnl = total_pnl_before_fee - total_fee
        
        # 승률
        wins = sum(1 for t in all_trades if t['pnl'] > 0)
        win_rate = (wins / total_trades) * 100
        
        return {
            'lookback_right': lookback_right,
            'partial_profit': partial_profit,
            'hold_bars': hold_bars,
            'fee_rate': fee_rate,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl_before_fee': total_pnl_before_fee,
            'total_fee': total_fee,
            'total_pnl': total_pnl,
            'bear_signals': len(bear_signals),
            'bull_signals': len(bull_signals)
        }
    
    def grid_search(self, 
                   lookback_right_range, 
                   partial_profit_range, 
                   hold_bars_range,
                   fee_rate=0.0):
        """
        그리드 서치 실행
        
        Parameters:
        -----------
        lookback_right_range : list or range
            예: [1, 2, 3] 또는 range(1, 11)
        partial_profit_range : list or range  
            예: [0.1, 0.2, 0.3] 또는 np.arange(0.1, 2.1, 0.1)
        hold_bars_range : list or range
            예: [5, 10, 15] 또는 range(1, 41)
        fee_rate : float
            수수료율 (기본값: 0.0)
        """
        if self.df is None:
            self.load_data()
        
        # 모든 조합 생성
        total_combinations = len(list(lookback_right_range)) * \
                            len(list(partial_profit_range)) * \
                            len(list(hold_bars_range))
        
        print(f"\n🔍 그리드 서치 시작")
        print(f"   lookback_right: {list(lookback_right_range)}")
        print(f"   partial_profit: {list(partial_profit_range)}")
        print(f"   hold_bars: {list(hold_bars_range)}")
        print(f"   수수료율: {fee_rate}%")
        print(f"   총 테스트 조합: {total_combinations:,}개\n")
        
        results = []
        count = 0
        
        for lr, pp, hb in itertools.product(lookback_right_range, 
                                            partial_profit_range, 
                                            hold_bars_range):
            count += 1
            
            # 진행률 표시 (10% 단위)
            if count % max(1, total_combinations // 10) == 0:
                progress = (count / total_combinations) * 100
                print(f"진행중... {progress:.0f}% ({count:,}/{total_combinations:,})")
            
            result = self.run_single_test(lr, pp, hb, fee_rate)
            
            if result:
                results.append(result)
        
        self.all_results = results
        
        print(f"\n✅ 완료! 유효한 결과: {len(results):,}개\n")
        
        return pd.DataFrame(results)
    
    def get_top_results(self, n=10, sort_by='total_pnl'):
        """
        상위 결과 조회
        
        Parameters:
        -----------
        n : int
            조회할 상위 개수
        sort_by : str
            정렬 기준 ('total_pnl', 'win_rate', 'total_trades')
        """
        if not self.all_results:
            print("먼저 grid_search()를 실행하세요")
            return None
        
        df = pd.DataFrame(self.all_results)
        df = df.sort_values(sort_by, ascending=False).head(n)
        
        return df
    
    def save_results(self, filename='grid_search_results.csv'):
        """결과를 CSV 파일로 저장"""
        if not self.all_results:
            print("먼저 grid_search()를 실행하세요")
            return
        
        df = pd.DataFrame(self.all_results)
        df = df.sort_values('total_pnl', ascending=False)
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        
        print(f"💾 결과 저장: {filename}")
        print(f"   총 {len(df)}개 조합 저장됨")


def main():
    """메인 함수"""
    if len(sys.argv) < 2:
        print("""
사용법:
    python backtest_grid_search.py <json_파일>

예시:
    python backtest_grid_search.py btc_15m_data.json
        """)
        return
    
    json_file = sys.argv[1]
    
    # 그리드 서치 실행
    searcher = RSIDivergenceGridSearch(json_file)
    
    # 파라미터 범위 설정
    print("\n" + "="*80)
    print("파라미터 범위를 입력하세요 (Enter = 기본값)")
    print("="*80)
    
    # lookback_right
    lr_input = input("lookback_right 범위 (예: 1-10) [기본: 1-5]: ").strip()
    if lr_input and '-' in lr_input:
        lr_start, lr_end = map(int, lr_input.split('-'))
        lookback_right_range = range(lr_start, lr_end + 1)
    else:
        lookback_right_range = range(1, 6)
    
    # partial_profit
    pp_input = input("부분익절% 범위 (예: 0.1-2.0-0.1) [기본: 0.3-1.0-0.1]: ").strip()
    if pp_input and '-' in pp_input:
        parts = pp_input.split('-')
        pp_start, pp_end, pp_step = float(parts[0]), float(parts[1]), float(parts[2])
        partial_profit_range = np.arange(pp_start, pp_end + pp_step/2, pp_step)
        partial_profit_range = np.round(partial_profit_range, 2)
    else:
        partial_profit_range = np.arange(0.3, 1.1, 0.1)
        partial_profit_range = np.round(partial_profit_range, 2)
    
    # hold_bars
    hb_input = input("보유기간(봉) 범위 (예: 5-30) [기본: 10-25]: ").strip()
    if hb_input and '-' in hb_input:
        hb_start, hb_end = map(int, hb_input.split('-'))
        hold_bars_range = range(hb_start, hb_end + 1)
    else:
        hold_bars_range = range(10, 26)
    
    # 수수료
    fee_input = input("수수료율% (예: 0.02) [기본: 0]: ").strip()
    fee_rate = float(fee_input) if fee_input else 0.0
    
    # 그리드 서치 실행
    df_results = searcher.grid_search(
        lookback_right_range=lookback_right_range,
        partial_profit_range=partial_profit_range,
        hold_bars_range=hold_bars_range,
        fee_rate=fee_rate
    )
    
    # 상위 결과 출력
    print("\n" + "="*100)
    print("🏆 TOP 20 결과 (총수익 기준)")
    print("="*100)
    
    top_20 = searcher.get_top_results(n=20)
    
    # 포맷팅하여 출력
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_columns', None)
    
    print(top_20.to_string(index=False))
    
    # 결과 저장
    searcher.save_results('grid_search_results.csv')
    
    # 최고 결과
    best = top_20.iloc[0]
    print("\n" + "="*100)
    print("🥇 최고 성과 파라미터")
    print("="*100)
    print(f"lookback_right: {best['lookback_right']}")
    print(f"부분익절: {best['partial_profit']}%")
    print(f"보유기간: {best['hold_bars']}봉")
    print(f"총 거래: {best['total_trades']}개")
    print(f"승률: {best['win_rate']:.1f}%")
    print(f"총 수익: {best['total_pnl']:+.2f}%")
    print(f"수수료: {best['total_fee']:.2f}%")


if __name__ == "__main__":
    main()