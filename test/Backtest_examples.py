"""
RSI Divergence 백테스터 사용 예시
"""
from backtest_divergence import RSIDivergenceBacktester
import pandas as pd

# 예시 1: 기본 설정으로 백테스팅
print("="*100)
print("예시 1: 기본 설정 (lookback_right=4, 부분익절=0.4%, 보유=15봉)")
print("="*100)

backtester1 = RSIDivergenceBacktester(
    lookback_right=4,
    partial_profit_target=0.8,
    hold_bars=5,
    trading_fee=0
)

backtester1.load_data('btc_15m_data.json')
backtester1.run_backtest()
backtester1.print_results()

# 예시 2: lookback_right를 2로 변경
print("\n\n" + "="*100)
print("예시 2: lookback_right=4로 변경 (신호 지연 증가)")
print("="*100)

backtester2 = RSIDivergenceBacktester(
    lookback_right=4,  # 변경!
    partial_profit_target=0.8,
    hold_bars=10,
    trading_fee=0
)

backtester2.load_data('btc_15m_data.json')
backtester2.run_backtest()
backtester2.print_results()

# 예시 3: 부분익절 목표를 0.6%로 상향
print("\n\n" + "="*100)
print("예시 3: 부분익절 목표를 0.6%로 상향")
print("="*100)

backtester3 = RSIDivergenceBacktester(
    lookback_right=4,
    partial_profit_target=0.8,  # 변경!
    hold_bars=15,
    trading_fee=0
)

backtester3.load_data('btc_15m_data.json')
backtester3.run_backtest()
backtester3.print_results()

# 예시 4: 포지션 보유 기간을 20봉으로 연장
print("\n\n" + "="*100)
print("예시 4: 포지션 보유 기간을 20봉으로 연장")
print("="*100)

backtester4 = RSIDivergenceBacktester(
    lookback_right=4,
    partial_profit_target=0.8,
    hold_bars=20,  # 변경!
    trading_fee=0
)

backtester4.load_data('btc_15m_data.json')
backtester4.run_backtest()
backtester4.print_results()

# 예시 5: 거래 수수료를 0.05%로 상향 (바이낸스 일반 수수료)
print("\n\n" + "="*100)
print("예시 5: 거래 수수료 0.05%로 상향")
print("="*100)

backtester5 = RSIDivergenceBacktester(
    lookback_right=4,
    partial_profit_target=0.8,
    hold_bars=25,
    trading_fee=0  # 변경!
)

backtester5.load_data('btc_15m_data.json')
backtester5.run_backtest()
backtester5.print_results()

# 결과 비교
print("\n\n" + "="*100)
print("📊 전체 결과 비교")
print("="*100)

results_comparison = []

for idx, bt in enumerate([backtester1, backtester2, backtester3, backtester4, backtester5], 1):
    bear_trades = bt.results['bear_trades']
    bull_trades = bt.results['bull_trades']
    total_trades = len(bear_trades) + len(bull_trades)
    
    if total_trades > 0:
        total_wins = sum(1 for t in bear_trades + bull_trades if t['pnl'] > 0)
        total_win_rate = (total_wins / total_trades) * 100
        total_pnl = sum([t['pnl'] for t in bear_trades + bull_trades])
    else:
        total_win_rate = 0
        total_pnl = 0
    
    params = bt.results['parameters']
    
    results_comparison.append({
        '예시': f"예시 {idx}",
        'lookback_right': params['lookback_right'],
        '부분익절(%)': params['partial_profit_target'],
        '보유기간(봉)': params['hold_bars'],
        '수수료(%)': params['trading_fee'],
        '총거래': total_trades,
        '승률(%)': f"{total_win_rate:.1f}",
        '총수익(%)': f"{total_pnl:+.2f}"
    })

df_comparison = pd.DataFrame(results_comparison)
print("\n")
print(df_comparison.to_string(index=False))

# 최고 성과 찾기
best_idx = df_comparison['총수익(%)'].apply(lambda x: float(x)).argmax()
print(f"\n🏆 최고 성과: {df_comparison.iloc[best_idx]['예시']}")
print(f"   총 수익: {df_comparison.iloc[best_idx]['총수익(%)']}%")
print(f"   승률: {df_comparison.iloc[best_idx]['승률(%)']}%")