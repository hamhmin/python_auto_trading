import os
import time
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException

load_dotenv()
client = Client(os.getenv('API_KEY'), os.getenv('SECRET_KEY'))

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

POSITION_COUNTER = 0

def get_next_position_id():
    global POSITION_COUNTER
    POSITION_COUNTER += 1
    return POSITION_COUNTER

# ============================================================================
# 설정값
# ============================================================================

SYMBOL = "XRPUSDT"
TIMEFRAME = "15m"
LEVERAGE = 25
POSITION_SIZE = 25  # XRP 수량

# 전략 파라미터
RSI_PERIOD = 14
LOOKBACK_LEFT = 5
LOOKBACK_RIGHT = 1
RANGE_LOWER = 5
RANGE_UPPER = 60

# 청산 설정
HOLD_BARS = 18  # 15봉 = 225분 = 3.75시간
PARTIAL_PROFIT_TARGET = 0.8  # 0.8% 도달 시
PARTIAL_PROFIT_RATIO = 0.5  # 50% 청산

# 포지션 관리
MAX_POSITIONS = 3  # 최대 동시 포지션 수

# 리스크 관리
STOP_LOSS_BEAR = 3.6  # Bearish 스탑로스 (%)
STOP_LOSS_BULL = 3.6  # Bullish 스탑로스 (%)

# 데이터 설정
CANDLES_TO_LOAD = 300  # RSI 계산 후 dropna를 고려하여 여유있게 설정

# ============================================================================
# 텔레그램 (간소화)
# ============================================================================

def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
        response = requests.post(url, data=data, timeout=10)
        return response.status_code == 200
    except:
        return False

def send_entry_alert(position):
    emoji = "🔴" if position['type'] == "bearish" else "🟢"
    type_kr = "숏(SHORT)" if position['type'] == "bearish" else "롱(LONG)"
    expected_close = position['entry_time'] + timedelta(minutes=HOLD_BARS*15)
    message = f"""
{emoji} <b>포지션 진입!</b>

📊 {type_kr}
💰 ${position['entry_price']:,.2f}
⏰ {position['entry_time'].strftime('%H:%M:%S')}
🎯 예상 청산: {expected_close.strftime('%H:%M:%S')} ({HOLD_BARS}봉)
"""
    send_telegram_message(message)

def send_exit_alert(position, reason, final_profit):
    emoji = "🎉" if final_profit > 0 else "😢"
    time_held = datetime.now() - position['entry_time']
    hours = time_held.total_seconds() / 3600
    message = f"""
{emoji} <b>{reason}</b>

📊 {'숏' if position['type'] == 'bearish' else '롱'}
💰 ${position['entry_price']:,.2f}
📈 수익: {final_profit:+.2f}%
⏱️ 보유: {hours:.1f}시간
⏰ {datetime.now().strftime('%H:%M:%S')}
"""
    send_telegram_message(message)

# ============================================================================
# 유틸리티
# ============================================================================

def log(message, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    if level == "INFO":
        print(f"[{timestamp}] {message}")
    elif level == "EVENT":
        print(f"\n{'='*60}")
        print(f"[{timestamp}] {message}")
        print(f"{'='*60}")
    elif level == "ERROR":
        print(f"\n❌ [{timestamp}] {message}")
    elif level == "DEBUG":
        print(f"🔍 [{timestamp}] {message}")

def get_historical_data(symbol, interval, limit=100):
    try:
        klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close'] = df['close'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['open'] = df['open'].astype(float)
        return df
    except Exception as e:
        log(f"데이터 로드 실패: {e}", "ERROR")
        return None

def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def find_pivot_high(series, left, right, idx):
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
    if idx < left or idx >= len(series) - right:
        return False
    center_value = series.iloc[idx]
    left_higher = all(series.iloc[idx-left:idx] > center_value)
    if right == 0:
        right_higher = True
    else:
        right_higher = all(series.iloc[idx+1:idx+right+1] > center_value)
    return left_higher and right_higher

def detect_regular_divergence(df):
    signals = []
    rsi = df['rsi']
    high = df['high']
    low = df['low']
    check_idx = len(df) - LOOKBACK_RIGHT - 1
    
    if check_idx < LOOKBACK_LEFT:
        return signals
    
    # Bearish
    if find_pivot_high(rsi, LOOKBACK_LEFT, LOOKBACK_RIGHT, check_idx):
        for j in range(check_idx - RANGE_LOWER, max(check_idx - RANGE_UPPER, LOOKBACK_LEFT), -1):
            if find_pivot_high(rsi, LOOKBACK_LEFT, LOOKBACK_RIGHT, j):
                signal_idx = check_idx + LOOKBACK_RIGHT
                if signal_idx < len(df):
                    rsi_curr = rsi.iloc[check_idx]
                    rsi_prev = rsi.iloc[j]
                    price_curr = high.iloc[check_idx]
                    price_prev = high.iloc[j]
                    
                    if rsi_curr < rsi_prev and price_curr > price_prev:
                        signals.append({
                            'type': 'bearish',
                            'index': signal_idx,
                            'entry_price': df['close'].iloc[signal_idx],
                            'time': df['open_time'].iloc[signal_idx]
                        })
                        log(f"🔴 Bearish Divergence! RSI: {rsi_prev:.1f}→{rsi_curr:.1f}", "EVENT")
                break
    
    # Bullish
    if find_pivot_low(rsi, LOOKBACK_LEFT, LOOKBACK_RIGHT, check_idx):
        for j in range(check_idx - RANGE_LOWER, max(check_idx - RANGE_UPPER, LOOKBACK_LEFT), -1):
            if find_pivot_low(rsi, LOOKBACK_LEFT, LOOKBACK_RIGHT, j):
                signal_idx = check_idx + LOOKBACK_RIGHT
                if signal_idx < len(df):
                    rsi_curr = rsi.iloc[check_idx]
                    rsi_prev = rsi.iloc[j]
                    price_curr = low.iloc[check_idx]
                    price_prev = low.iloc[j]
                    
                    if rsi_curr > rsi_prev and price_curr < price_prev:
                        signals.append({
                            'type': 'bullish',
                            'index': signal_idx,
                            'entry_price': df['close'].iloc[signal_idx],
                            'time': df['open_time'].iloc[signal_idx]
                        })
                        log(f"🟢 Bullish Divergence! RSI: {rsi_prev:.1f}→{rsi_curr:.1f}", "EVENT")
                break
    
    return signals

# ============================================================================
# 주문 실행
# ============================================================================

def execute_entry(signal_type, amount=POSITION_SIZE):
    try:
        try:
            client.futures_change_margin_type(symbol=SYMBOL, marginType='ISOLATED')
        except:
            pass
        
        client.futures_change_leverage(symbol=SYMBOL, leverage=LEVERAGE)
        side = SIDE_SELL if signal_type == 'bearish' else SIDE_BUY
        
        order = client.futures_create_order(
            symbol=SYMBOL,
            side=side,
            type=ORDER_TYPE_MARKET,
            quantity=amount
        )
        
        entry_price = 0.0
        if 'avgPrice' in order and order['avgPrice']:
            entry_price = float(order['avgPrice'])
        elif 'fills' in order and order['fills']:
            total_qty = 0
            total_cost = 0
            for fill in order['fills']:
                qty = float(fill['qty'])
                price = float(fill['price'])
                total_qty += qty
                total_cost += qty * price
            if total_qty > 0:
                entry_price = total_cost / total_qty
        
        if entry_price == 0.0:
            ticker = client.futures_symbol_ticker(symbol=SYMBOL)
            entry_price = float(ticker['price'])
        
        if entry_price <= 0:
            log(f"진입 가격 유효하지 않음: {entry_price}", "ERROR")
            return None
        
        log(f"✅ {'숏' if signal_type == 'bearish' else '롱'} 진입 ${entry_price:,.2f}", "EVENT")
        
        position = {
            'order_id': order['orderId'],
            'type': signal_type,
            'side': side,
            'entry_price': entry_price,
            'amount': amount,
            'entry_time': datetime.now()
        }
        
        send_entry_alert(position)
        return position
    
    except BinanceAPIException as e:
        if e.code == -2019:
            log(f"잔고 부족! 필요: {amount} BTC", "ERROR")
        elif e.code == -4131:
            log(f"Reduce-only 거부", "ERROR")
        else:
            log(f"바이낸스 API 에러 [{e.code}]: {e.message}", "ERROR")
        return None
    except Exception as e:
        log(f"진입 주문 실패: {e}", "ERROR")
        return None

def execute_partial_close(position, ratio=0.5):
    try:
        close_amount = round(position['amount'] * ratio, 3)
        side = SIDE_BUY if position['side'] == SIDE_SELL else SIDE_SELL
        
        order = client.futures_create_order(
            symbol=SYMBOL,
            side=side,
            type=ORDER_TYPE_MARKET,
            quantity=close_amount,
            reduceOnly=True
        )
        
        log(f"✅ 부분 익절 {close_amount:.4f} BTC", "EVENT")
        return order
    except Exception as e:
        log(f"부분 청산 실패: {e}", "ERROR")
        return None

def execute_full_close(position):
    try:
        close_amount = round(position['amount'], 3)
        side = SIDE_BUY if position['side'] == SIDE_SELL else SIDE_SELL
        
        order = client.futures_create_order(
            symbol=SYMBOL,
            side=side,
            type=ORDER_TYPE_MARKET,
            quantity=close_amount,
            reduceOnly=True
        )
        
        log(f"✅ 전체 청산 {close_amount:.4f} BTC", "EVENT")
        return order
    except Exception as e:
        log(f"전체 청산 실패: {e}", "ERROR")
        return None

def get_current_price():
    try:
        ticker = client.futures_symbol_ticker(symbol=SYMBOL)
        return float(ticker['price'])
    except:
        return None

def calculate_profit(position, current_price):
    entry_price = position['entry_price']
    signal_type = position['type']
    
    if entry_price <= 0:
        return 0.0
    
    if signal_type == 'bearish':
        profit = ((entry_price - current_price) / entry_price) * 100
    else:
        profit = ((current_price - entry_price) / entry_price) * 100
    
    return profit

def get_current_candle():
    try:
        klines = client.futures_klines(symbol=SYMBOL, interval=TIMEFRAME, limit=1)
        if klines:
            return {
                'high': float(klines[0][2]),
                'low': float(klines[0][3]),
                'close': float(klines[0][4])
            }
        return None
    except:
        return None

def calculate_max_profit_in_candle(position, candle):
    if candle is None:
        return 0
    
    entry_price = position['entry_price']
    signal_type = position['type']
    
    if entry_price <= 0:
        return 0.0
    
    if signal_type == 'bearish':
        max_profit = ((entry_price - candle['low']) / entry_price) * 100
    else:
        max_profit = ((candle['high'] - entry_price) / entry_price) * 100
    
    return max_profit

# ============================================================================
# 메인 봇 로직 - 🔧 보유기간 체크 완전 수정
# ============================================================================

def main():
    log("="*80, "EVENT")
    log("🤖 RSI Divergence 자동매매 봇 시작", "EVENT")
    log(f"심볼: {SYMBOL} | 타임프레임: {TIMEFRAME} | 레버리지: {LEVERAGE}배")
    log(f"포지션 크기: {POSITION_SIZE} BTC | 최대: {MAX_POSITIONS}개")
    log(f"부분 익절: {PARTIAL_PROFIT_TARGET}% | 보유: {HOLD_BARS}봉 (약 {HOLD_BARS*15/60:.1f}시간)")
    log(f"스탑로스: Bear {STOP_LOSS_BEAR}% / Bull {STOP_LOSS_BULL}%")
    log("="*80, "EVENT")
    
    active_positions = {}
    entered_signals = set()
    last_signal_check_time = datetime.now()
    
    while True:
        try:
            current_time = datetime.now()
            
            # 신호 체크
            minutes_since_last_check = (current_time - last_signal_check_time).total_seconds() / 60
            should_check_signals = minutes_since_last_check >= 15
            
            if should_check_signals:
                df = get_historical_data(SYMBOL, TIMEFRAME, limit=CANDLES_TO_LOAD)
                
                if df is None:
                    log("데이터 로드 실패", "ERROR")
                    time.sleep(60)
                    continue
                
                df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
                df = df.dropna().reset_index(drop=True)
                
                required_candles = RSI_PERIOD + LOOKBACK_LEFT + RANGE_UPPER
                
                if len(df) < required_candles:
                    log(f"데이터 부족: {len(df)}/{required_candles}", "ERROR")
                    time.sleep(60)
                    continue
                
                # 신호 감지
                if len(active_positions) < MAX_POSITIONS:
                    signals = detect_regular_divergence(df)
                    
                    if signals:
                        for signal in signals:
                            signal_index = signal['index']
                            
                            if signal_index in entered_signals:
                                continue
                            
                            position = execute_entry(signal['type'], POSITION_SIZE)
                            
                            if position:
                                position_id = get_next_position_id()
                                
                                position['position_id'] = position_id
                                position['stop_order_id'] = None
                                position['partial_closed'] = False
                                position['signal_index'] = signal_index
                                position['initial_amount'] = POSITION_SIZE
                                
                                active_positions[position_id] = position
                                entered_signals.add(signal_index)
                                
                                log(f"ID={position_id} 진입시간: {position['entry_time'].strftime('%H:%M:%S')}", "DEBUG")
                                log(f"ID={position_id} 청산예정: {(position['entry_time'] + timedelta(minutes=HOLD_BARS*15)).strftime('%H:%M:%S')}", "DEBUG")
                                
                                if len(active_positions) >= MAX_POSITIONS:
                                    break
                            else:
                                entered_signals.add(signal_index)
                
                last_signal_check_time = current_time
            
            # 🔧 포지션 관리
            for pos_id in list(active_positions.keys()):
                position = active_positions[pos_id]
                
                current_candle = get_current_candle()
                if current_candle is None:
                    continue
                
                current_price = current_candle['close']
                profit = calculate_profit(position, current_price)
                
                # 🔧 보유 시간 계산 (분 단위)
                time_held = datetime.now() - position['entry_time']
                minutes_held = time_held.total_seconds() / 60
                bars_held = minutes_held / 15  # float 유지
                
                # 1️⃣ 스탑로스 체크
                stop_loss_pct = STOP_LOSS_BEAR if position['type'] == 'bearish' else STOP_LOSS_BULL
                
                if profit <= -stop_loss_pct:
                    log(f"🚨 ID={pos_id} 스탑로스! {profit:.2f}%", "EVENT")
                    
                    result = execute_full_close(position)
                    
                    if result:
                        final_price = get_current_price()
                        final_profit = calculate_profit(position, final_price)
                        send_exit_alert(position, "스탑로스", final_profit)
                        del active_positions[pos_id]
                        if 'signal_index' in position:
                            entered_signals.discard(position['signal_index'])
                    
                    continue
                
                max_profit_in_candle = calculate_max_profit_in_candle(position, current_candle)
                
                # 2️⃣ 부분 익절 체크
                if not position['partial_closed'] and max_profit_in_candle >= PARTIAL_PROFIT_TARGET:
                    log(f"🎯 ID={pos_id} 부분 익절 {max_profit_in_candle:.2f}%", "EVENT")
                    
                    result = execute_partial_close(position, PARTIAL_PROFIT_RATIO)
                    
                    if result:
                        closed_amount = position['amount'] * PARTIAL_PROFIT_RATIO
                        position['amount'] = position['amount'] - closed_amount
                        position['partial_closed'] = True
                        send_exit_alert(position, "부분 익절", max_profit_in_candle)
                
                # 3️⃣ 보유기간 도달 체크 (🔧 분 단위로 체크)
                target_minutes = HOLD_BARS * 15
                
                if minutes_held >= target_minutes:
                    log(f"⏰ ID={pos_id} {HOLD_BARS}봉({target_minutes}분) 도달 (실제: {minutes_held:.1f}분)", "EVENT")
                    
                    result = execute_full_close(position)
                    
                    if result:
                        final_price = get_current_price()
                        final_profit = calculate_profit(position, final_price)
                        send_exit_alert(position, "보유기간 종료", final_profit)
                        del active_positions[pos_id]
                        if 'signal_index' in position:
                            entered_signals.discard(position['signal_index'])
            
            # 간소화된 로그
            next_signal_check = 15 - int(minutes_since_last_check)
            pos_summary = ""
            if active_positions:
                for pos_id, pos in active_positions.items():
                    candle = get_current_candle()
                    if candle:
                        p = calculate_profit(pos, candle['close'])
                        mins = (datetime.now() - pos['entry_time']).total_seconds() / 60
                        pos_summary += f" | P{pos_id}: {p:+.2f}% ({mins:.0f}/{HOLD_BARS*15}분)"
            
            current_price = get_current_price() or 0
            df_temp = get_historical_data(SYMBOL, TIMEFRAME, limit=50)
            if df_temp is not None:
                df_temp['rsi'] = calculate_rsi(df_temp['close'], RSI_PERIOD)
                current_rsi = df_temp['rsi'].iloc[-1] if not df_temp.empty else 0
            else:
                current_rsi = 0
            
            log(f"✓ ${current_price:,.0f} | RSI {current_rsi:.1f} | "
                f"{len(active_positions)}개{pos_summary} | 신호 {next_signal_check}분")
            
            time.sleep(60)
            
        except KeyboardInterrupt:
            log("\n🛑 봇 종료", "EVENT")
            break
        except Exception as e:
            import traceback
            log(f"오류: {e}", "ERROR")
            log(f"{traceback.format_exc()}")
            time.sleep(60)

if __name__ == "__main__":
    main()