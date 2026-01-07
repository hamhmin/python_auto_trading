import pandas as pd
import numpy as np
import json
import sys
from datetime import datetime

class RSIDivergenceBacktester:
    def __init__(self, 
                 lookback_left=5,
                 lookback_right=1,
                 range_lower=5,
                 range_upper=60,
                 rsi_period=14,
                 partial_profit_target=0.4,
                 partial_profit_ratio=0.5,
                 hold_bars=15,
                 trading_fee=0.02):
        """
        RSI Divergence 백테스터
        
        Parameters:
        -----------
        lookback_left : int (기본값 5)
            피벗 왼쪽 확인 봉 수
        lookback_right : int (기본값 1) ⭐ 설정 가능
            피벗 오른쪽 확인 봉 수 (신호 지연)
        range_lower : int (기본값 5)
            이전 피벗 최소 간격
        range_upper : int (기본값 60)
            이전 피벗 최대 간격
        rsi_period : int (기본값 14)
            RSI 계산 기간
        partial_profit_target : float (기본값 0.4) ⭐ 설정 가능
            부분 익절 목표 (%)
        partial_profit_ratio : float (기본값 0.5)
            익절 비율 (50% = 0.5)
        hold_bars : int (기본값 15) ⭐ 설정 가능
            포지션 보유 기간 (봉 수)
        trading_fee : float (기본값 0.02)
            거래 수수료 (%)
        """
        self.lookback_left = lookback_left
        self.lookback_right = lookback_right
        self.range_lower = range_lower
        self.range_upper = range_upper
        self.rsi_period = rsi_period
        self.partial_profit_target = partial_profit_target
        self.partial_profit_ratio = partial_profit_ratio
        self.hold_bars = hold_bars
        self.trading_fee = trading_fee
        
        self.df = None
        self.results = None
    
    def load_data(self, json_file_path):
        """JSON 파일에서 캔들 데이터 로드"""
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        self.df = pd.DataFrame(data)
        
        # 필수 컬럼 확인
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in self.df.columns for col in required_cols):
            raise ValueError(f"데이터에 필수 컬럼이 없습니다: {required_cols}")
        
        # RSI 계산
        self.df['rsi'] = self._calculate_rsi(self.df['close'], self.rsi_period)
        self.df = self.df.dropna().reset_index(drop=True)
        
        print(f"✅ 데이터 로드 완료: {len(self.df)}개 캔들")
        if 'timestamp' in self.df.columns:
            print(f"   기간: {self.df['timestamp'].iloc[0]} ~ {self.df['timestamp'].iloc[-1]}")
        
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
    
    def detect_divergences(self):
        """Regular Divergence 감지"""
        regular_bear = []
        regular_bull = []
        
        for i in range(len(self.df)):
            # Bearish Divergence
            if self._find_pivot_high(self.df['rsi'], self.lookback_left, self.lookback_right, i):
                prev_pivot_idx = None
                for j in range(i - self.range_lower, max(i - self.range_upper, 0), -1):
                    if self._find_pivot_high(self.df['rsi'], self.lookback_left, self.lookback_right, j):
                        prev_pivot_idx = j
                        break
                
                if prev_pivot_idx is not None:
                    signal_idx = i + self.lookback_right
                    if signal_idx < len(self.df):
                        rsi_curr = self.df['rsi'].iloc[i]
                        rsi_prev = self.df['rsi'].iloc[prev_pivot_idx]
                        price_curr = self.df['high'].iloc[i]
                        price_prev = self.df['high'].iloc[prev_pivot_idx]
                        
                        if rsi_curr < rsi_prev and price_curr > price_prev:
                            regular_bear.append({
                                'signal_index': signal_idx,
                                'pivot_index': i,
                                'prev_pivot_index': prev_pivot_idx
                            })
            
            # Bullish Divergence
            if self._find_pivot_low(self.df['rsi'], self.lookback_left, self.lookback_right, i):
                prev_pivot_idx = None
                for j in range(i - self.range_lower, max(i - self.range_upper, 0), -1):
                    if self._find_pivot_low(self.df['rsi'], self.lookback_left, self.lookback_right, j):
                        prev_pivot_idx = j
                        break
                
                if prev_pivot_idx is not None:
                    signal_idx = i + self.lookback_right
                    if signal_idx < len(self.df):
                        rsi_curr = self.df['rsi'].iloc[i]
                        rsi_prev = self.df['rsi'].iloc[prev_pivot_idx]
                        price_curr = self.df['low'].iloc[i]
                        price_prev = self.df['low'].iloc[prev_pivot_idx]
                        
                        if rsi_curr > rsi_prev and price_curr < price_prev:
                            regular_bull.append({
                                'signal_index': signal_idx,
                                'pivot_index': i,
                                'prev_pivot_index': prev_pivot_idx
                            })
        
        print(f"\n📊 다이버전스 신호 감지:")
        print(f"   Bearish: {len(regular_bear)}개")
        print(f"   Bullish: {len(regular_bull)}개")
        print(f"   총: {len(regular_bear) + len(regular_bull)}개")
        
        return regular_bear, regular_bull
    
    def _execute_trade(self, signal_idx, signal_type):
        """거래 실행 및 결과 계산"""
        if signal_idx + self.hold_bars >= len(self.df):
            return None
        
        entry_price = self.df['close'].iloc[signal_idx]
        entry_fee = self.trading_fee
        
        partial_closed = False
        partial_pnl = 0
        partial_fee = 0
        partial_close_bar = None
        
        # 보유 기간 동안 부분 익절 체크
        for i in range(signal_idx, signal_idx + self.hold_bars + 1):
            current_high = self.df['high'].iloc[i]
            current_low = self.df['low'].iloc[i]
            
            if signal_type == 'bear':
                current_profit = ((entry_price - current_low) / entry_price) * 100
            else:
                current_profit = ((current_high - entry_price) / entry_price) * 100
            
            # 목표가 도달 시 부분 익절
            if not partial_closed and current_profit >= self.partial_profit_target:
                partial_pnl = current_profit * self.partial_profit_ratio
                partial_fee = self.trading_fee * self.partial_profit_ratio
                partial_closed = True
                partial_close_bar = i - signal_idx
        
        # 나머지 포지션 청산
        exit_price = self.df['close'].iloc[signal_idx + self.hold_bars]
        if signal_type == 'bear':
            remaining_pnl = ((entry_price - exit_price) / entry_price) * 100 * self.partial_profit_ratio
        else:
            remaining_pnl = ((exit_price - entry_price) / entry_price) * 100 * self.partial_profit_ratio
        
        remaining_fee = self.trading_fee * self.partial_profit_ratio
        
        # 총 수익 계산
        total_pnl_before_fee = partial_pnl + remaining_pnl
        total_fees = entry_fee + partial_fee + remaining_fee
        total_pnl = total_pnl_before_fee - total_fees
        
        return {
            'entry_index': signal_idx,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'signal_type': signal_type,
            'pnl_before_fee': total_pnl_before_fee,
            'total_fees': total_fees,
            'pnl': total_pnl,
            'partial_closed': partial_closed,
            'partial_close_bar': partial_close_bar,
            'exit_bar': self.hold_bars
        }
    
    def run_backtest(self):
        """백테스팅 실행"""
        if self.df is None:
            raise ValueError("먼저 load_data()를 호출하세요")
        
        # 다이버전스 신호 감지
        bear_signals, bull_signals = self.detect_divergences()
        
        # 거래 실행
        bear_trades = []
        for signal in bear_signals:
            result = self._execute_trade(signal['signal_index'], 'bear')
            if result:
                bear_trades.append(result)
        
        bull_trades = []
        for signal in bull_signals:
            result = self._execute_trade(signal['signal_index'], 'bull')
            if result:
                bull_trades.append(result)
        
        # 결과 저장
        self.results = {
            'bear_trades': bear_trades,
            'bull_trades': bull_trades,
            'parameters': {
                'lookback_right': self.lookback_right,
                'partial_profit_target': self.partial_profit_target,
                'hold_bars': self.hold_bars,
                'trading_fee': self.trading_fee
            }
        }
        
        return self
    
    def print_results(self):
        """결과 출력"""
        if self.results is None:
            raise ValueError("먼저 run_backtest()를 호출하세요")
        
        bear_trades = self.results['bear_trades']
        bull_trades = self.results['bull_trades']
        params = self.results['parameters']
        
        print("\n" + "="*100)
        print("📈 백테스팅 결과")
        print("="*100)
        
        # 파라미터 출력
        print("\n⚙️  설정 파라미터:")
        print(f"   lookback_right: {params['lookback_right']}봉")
        print(f"   부분익절 목표: {params['partial_profit_target']}%")
        print(f"   포지션 보유: {params['hold_bars']}봉")
        print(f"   거래 수수료: {params['trading_fee']}%")
        
        # 통계 계산
        total_trades = len(bear_trades) + len(bull_trades)
        
        if total_trades == 0:
            print("\n⚠️  거래가 없습니다")
            return
        
        # Bearish 통계
        if bear_trades:
            bear_wins = sum(1 for t in bear_trades if t['pnl'] > 0)
            bear_win_rate = (bear_wins / len(bear_trades)) * 100
            bear_total_pnl = sum([t['pnl'] for t in bear_trades])
            bear_avg_pnl = np.mean([t['pnl'] for t in bear_trades])
            bear_total_fees = sum([t['total_fees'] for t in bear_trades])
            bear_partial_count = sum(1 for t in bear_trades if t['partial_closed'])
        else:
            bear_wins = bear_win_rate = bear_total_pnl = bear_avg_pnl = bear_total_fees = bear_partial_count = 0
        
        # Bullish 통계
        if bull_trades:
            bull_wins = sum(1 for t in bull_trades if t['pnl'] > 0)
            bull_win_rate = (bull_wins / len(bull_trades)) * 100
            bull_total_pnl = sum([t['pnl'] for t in bull_trades])
            bull_avg_pnl = np.mean([t['pnl'] for t in bull_trades])
            bull_total_fees = sum([t['total_fees'] for t in bull_trades])
            bull_partial_count = sum(1 for t in bull_trades if t['partial_closed'])
        else:
            bull_wins = bull_win_rate = bull_total_pnl = bull_avg_pnl = bull_total_fees = bull_partial_count = 0
        
        # 전체 통계
        total_wins = bear_wins + bull_wins
        total_win_rate = (total_wins / total_trades) * 100
        total_pnl = bear_total_pnl + bull_total_pnl
        total_fees = bear_total_fees + bull_total_fees
        total_partial_count = bear_partial_count + bull_partial_count
        
        # 결과 출력
        print(f"\n📊 전체 성과:")
        print(f"   총 거래: {total_trades}개")
        print(f"   승률: {total_win_rate:.1f}% ({total_wins}/{total_trades})")
        print(f"   총 수익: {total_pnl:+.2f}%")
        print(f"   총 수수료: {total_fees:.2f}%")
        print(f"   부분익절 발생: {total_partial_count}/{total_trades} ({total_partial_count/total_trades*100:.1f}%)")
        
        print(f"\n📉 Bearish (Short):")
        print(f"   거래 수: {len(bear_trades)}개")
        if bear_trades:
            print(f"   승률: {bear_win_rate:.1f}% ({bear_wins}/{len(bear_trades)})")
            print(f"   총 수익: {bear_total_pnl:+.2f}%")
            print(f"   평균 수익: {bear_avg_pnl:+.3f}%")
            print(f"   수수료: {bear_total_fees:.2f}%")
            print(f"   부분익절: {bear_partial_count}/{len(bear_trades)} ({bear_partial_count/len(bear_trades)*100:.1f}%)")
        
        print(f"\n📈 Bullish (Long):")
        print(f"   거래 수: {len(bull_trades)}개")
        if bull_trades:
            print(f"   승률: {bull_win_rate:.1f}% ({bull_wins}/{len(bull_trades)})")
            print(f"   총 수익: {bull_total_pnl:+.2f}%")
            print(f"   평균 수익: {bull_avg_pnl:+.3f}%")
            print(f"   수수료: {bull_total_fees:.2f}%")
            print(f"   부분익절: {bull_partial_count}/{len(bull_trades)} ({bull_partial_count/len(bull_trades)*100:.1f}%)")
        
        # 상위/하위 거래
        all_trades = bear_trades + bull_trades
        all_trades.sort(key=lambda x: x['pnl'], reverse=True)
        
        print(f"\n🏆 최고 수익 거래 TOP 3:")
        for i, trade in enumerate(all_trades[:3], 1):
            print(f"   {i}. {trade['signal_type'].upper()}: {trade['pnl']:+.2f}% (진입: {trade['entry_price']:.2f})")
        
        print(f"\n💀 최악 손실 거래 TOP 3:")
        for i, trade in enumerate(all_trades[-3:][::-1], 1):
            print(f"   {i}. {trade['signal_type'].upper()}: {trade['pnl']:+.2f}% (진입: {trade['entry_price']:.2f})")
        
        print("\n" + "="*100)
    
    def get_trade_history(self):
        """거래 내역을 DataFrame으로 반환"""
        if self.results is None:
            raise ValueError("먼저 run_backtest()를 호출하세요")
        
        all_trades = self.results['bear_trades'] + self.results['bull_trades']
        
        if not all_trades:
            return pd.DataFrame()
        
        df_trades = pd.DataFrame(all_trades)
        df_trades = df_trades.sort_values('entry_index').reset_index(drop=True)
        
        return df_trades


def main():
    """메인 함수 - 사용 예시"""
    
    # 사용법 출력
    if len(sys.argv) < 2:
        print("""
사용법:
    python backtest_divergence.py <json_파일_경로> [옵션]

옵션:
    --lookback_right <숫자>          피벗 오른쪽 확인 봉 수 (기본값: 1)
    --partial_profit <숫자>          부분익절 목표 % (기본값: 0.4)
    --hold_bars <숫자>               포지션 보유 봉 수 (기본값: 15)
    --fee <숫자>                     거래 수수료 % (기본값: 0.02)

예시:
    python backtest_divergence.py btc_15m_data.json
    python backtest_divergence.py btc_15m_data.json --lookback_right 2 --partial_profit 0.5
    python backtest_divergence.py btc_15m_data.json --hold_bars 20 --fee 0.05
        """)
        return
    
    # 파일 경로
    json_file = sys.argv[1]
    
    # 파라미터 파싱
    lookback_right = 1
    partial_profit = 0.4
    hold_bars = 15
    fee = 0.02
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--lookback_right':
            lookback_right = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--partial_profit':
            partial_profit = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--hold_bars':
            hold_bars = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--fee':
            fee = float(sys.argv[i + 1])
            i += 2
        else:
            i += 1
    
    # 백테스터 생성 및 실행
    print("🚀 RSI Divergence 백테스터 시작")
    print(f"📁 파일: {json_file}")
    
    backtester = RSIDivergenceBacktester(
        lookback_right=lookback_right,
        partial_profit_target=partial_profit,
        hold_bars=hold_bars,
        trading_fee=fee
    )
    
    # 데이터 로드 및 백테스팅 실행
    backtester.load_data(json_file)
    backtester.run_backtest()
    backtester.print_results()
    
    # 거래 내역 저장 (옵션)
    trade_history = backtester.get_trade_history()
    if not trade_history.empty:
        output_file = 'backtest_trades.csv'
        trade_history.to_csv(output_file, index=False)
        print(f"\n💾 거래 내역 저장: {output_file}")


if __name__ == "__main__":
    main()