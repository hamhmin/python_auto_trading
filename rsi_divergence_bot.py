import os
import time
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from dotenv import load_dotenv
from binance.client import Client
from binance.enums import *

# .env 로드 및 클라이언트 생성
load_dotenv()
client = Client(os.getenv('API_KEY'), os.getenv('SECRET_KEY'))

# 텔레그램 설정
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# ============================================================================
# 설정값
# ============================================================================

SYMBOL = "BTCUSDT"
TIMEFRAME = "15m"
LEVERAGE = 20
POSITION_SIZE = 0.002  # BTC 수량

# 전략 파라미터
RSI_PERIOD = 14
LOOKBACK_LEFT = 5
LOOKBACK_RIGHT = 1
RANGE_LOWER = 5
RANGE_UPPER = 60

# 청산 설정
HOLD_BARS = 15  # 15봉 = 225분 = 3.75시간
PARTIAL_PROFIT_TARGET = 0.4  # 0.4% 도달 시
PARTIAL_PROFIT_RATIO = 0.5  # 50% 청산

# 포지션 관리
MAX_POSITIONS = 3  # 최대 동시 포지션 수

# 리스크 관리
STOP_LOSS_BEAR = 2.5  # Bearish 스탑로스 (%)
STOP_LOSS_BULL = 1.0  # Bullish 스탑로스 (%)

# 데이터 설정
CANDLES_TO_LOAD = 100  # 최소 60개 이상 필요

# ============================================================================
# 텔레그램 알림 함수
# ============================================================================

def send_telegram_message(message):
    """텔레그램 메시지 전송"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log("⚠️ 텔레그램 설정이 없습니다. 메시지 전송 건너뜀")
        return False
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }
        response = requests.post(url, data=data, timeout=10)
        
        if response.status_code == 200:
            return True
        else:
            log(f"⚠️ 텔레그램 전송 실패: {response.status_code}")
            return False
            
    except Exception as e:
        log(f"⚠️ 텔레그램 오류: {e}")
        return False

def send_divergence_alert(signal_type, current_price, current_rsi):
    """다이버전스 감지 알림"""
    emoji = "🔴" if signal_type == "bearish" else "🟢"
    type_kr = "Bearish (숏)" if signal_type == "bearish" else "Bullish (롱)"
    
    message = f"""
{emoji} <b>다이버전스 신호 감지!</b>

📊 타입: {type_kr}
💰 현재가: ${current_price:,.2f}
📈 RSI: {current_rsi:.2f}
⏰ 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🎯 진입 준비 중...
"""
    send_telegram_message(message)

def send_entry_alert(position):
    """진입 체결 알림"""
    from datetime import timedelta
    
    emoji = "🔴" if position['type'] == "bearish" else "🟢"
    type_kr = "숏(SHORT)" if position['type'] == "bearish" else "롱(LONG)"
    stop_loss = STOP_LOSS_BEAR if position['type'] == "bearish" else STOP_LOSS_BULL
    
    # 예상 종료 시간 계산 (15봉 = 225분)
    expected_close_time = position['entry_time'] + timedelta(minutes=HOLD_BARS * 15)
    
    message = f"""
{emoji} <b>포지션 진입 완료!</b>

📊 방향: {type_kr}
💰 진입가: ${position['entry_price']:,.2f}
📦 수량: {position['amount']} BTC
🔢 레버리지: {LEVERAGE}배
🛡️ 스탑로스: {stop_loss}%

⏰ 진입: {position['entry_time'].strftime('%Y-%m-%d %H:%M:%S')}
⏰ 예상 종료: {expected_close_time.strftime('%Y-%m-%d %H:%M:%S')} ({HOLD_BARS}봉 후)

📌 목표:
  • 부분 익절: 0.4% 도달 시 50%
  • 전체 청산: 15봉 후 (약 3.75시간)
"""
    send_telegram_message(message)

def send_partial_close_alert(position, profit):
    """부분 익절 알림"""
    message = f"""
💰 <b>부분 익절 체결!</b>

📊 포지션: {'숏' if position['type'] == 'bearish' else '롱'}
✅ 익절 비율: 50%
📈 현재 수익률: {profit:+.2f}%
💵 진입가: ${position['entry_price']:,.2f}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🔄 남은 50%는 15봉까지 보유 예정
"""
    send_telegram_message(message)

def send_final_close_alert(position, final_profit, final_price):
    """최종 청산 알림"""
    emoji = "🎉" if final_profit > 0 else "😢"
    result = "수익" if final_profit > 0 else "손실"
    
    # 보유 시간 계산
    hold_time = datetime.now() - position['entry_time']
    hours = hold_time.total_seconds() / 3600
    
    message = f"""
{emoji} <b>포지션 최종 청산!</b>

📊 포지션: {'숏' if position['type'] == 'bearish' else '롱'}
💰 진입가: ${position['entry_price']:,.2f}
💵 청산가: ${final_price:,.2f}

<b>📈 최종 수익률: {final_profit:+.2f}%</b>

⏱️ 보유 시간: {hours:.1f}시간
⏰ 청산 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'🎊 수익 달성!' if final_profit > 0 else '📉 손실 발생'}
"""
    send_telegram_message(message)

def send_stop_loss_alert(position):
    """스탑로스 체결 알림"""
    stop_loss = STOP_LOSS_BEAR if position['type'] == 'bearish' else STOP_LOSS_BULL
    
    message = f"""
🚨 <b>스탑로스 체결!</b>

📊 포지션: {'숏' if position['type'] == 'bearish' else '롱'}
💰 진입가: ${position['entry_price']:,.2f}
🛡️ 손실률: -{stop_loss}%

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

⚠️ 손실 제한으로 포지션 종료
"""
    send_telegram_message(message)

# ============================================================================
# 유틸리티 함수
# ============================================================================

def log(message):
    """로그 출력"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def get_historical_data(symbol, interval, limit=100):
    """과거 데이터 가져오기"""
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
        log(f"❌ 데이터 로드 실패: {e}")
        return None

def calculate_rsi(data, period=14):
    """RSI 계산 (division by zero 방지)"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # division by zero 방지
    rs = gain / loss.replace(0, 1e-10)  # loss가 0이면 아주 작은 값으로 대체
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def find_pivot_high(series, left, right, idx):
    """RSI 피벗 고점 감지"""
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
    """RSI 피벗 저점 감지"""
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
    """Regular Divergence 감지 (현재 봉에서 확정된 신호만)"""
    signals = []
    current_idx = len(df) - 1
    
    # Bearish Divergence 체크
    for i in range(len(df) - LOOKBACK_RIGHT - 1, len(df) - LOOKBACK_RIGHT):
        if find_pivot_high(df['rsi'], LOOKBACK_LEFT, LOOKBACK_RIGHT, i):
            # 이전 피벗 찾기
            for j in range(i - RANGE_LOWER, max(i - RANGE_UPPER, 0), -1):
                if find_pivot_high(df['rsi'], LOOKBACK_LEFT, LOOKBACK_RIGHT, j):
                    signal_idx = i + LOOKBACK_RIGHT
                    
                    # 신호가 현재 봉인지 확인
                    if signal_idx == current_idx:
                        rsi_curr = df['rsi'].iloc[i]
                        rsi_prev = df['rsi'].iloc[j]
                        price_curr = df['high'].iloc[i]
                        price_prev = df['high'].iloc[j]
                        
                        # Regular Bearish: RSI LH + Price HH
                        if rsi_curr < rsi_prev and price_curr > price_prev:
                            signals.append({
                                'type': 'bearish',
                                'index': signal_idx,
                                'entry_price': df['close'].iloc[signal_idx],
                                'time': df['open_time'].iloc[signal_idx]
                            })
                            log(f"🔴 Bearish Divergence 감지! RSI: {rsi_prev:.1f}→{rsi_curr:.1f}, Price: {price_prev:.2f}→{price_curr:.2f}")
                            
                            # 텔레그램 알림
                            send_divergence_alert('bearish', df['close'].iloc[signal_idx], rsi_curr)
                    break
    
    # Bullish Divergence 체크
    for i in range(len(df) - LOOKBACK_RIGHT - 1, len(df) - LOOKBACK_RIGHT):
        if find_pivot_low(df['rsi'], LOOKBACK_LEFT, LOOKBACK_RIGHT, i):
            # 이전 피벗 찾기
            for j in range(i - RANGE_LOWER, max(i - RANGE_UPPER, 0), -1):
                if find_pivot_low(df['rsi'], LOOKBACK_LEFT, LOOKBACK_RIGHT, j):
                    signal_idx = i + LOOKBACK_RIGHT
                    
                    # 신호가 현재 봉인지 확인
                    if signal_idx == current_idx:
                        rsi_curr = df['rsi'].iloc[i]
                        rsi_prev = df['rsi'].iloc[j]
                        price_curr = df['low'].iloc[i]
                        price_prev = df['low'].iloc[j]
                        
                        # Regular Bullish: RSI HL + Price LL
                        if rsi_curr > rsi_prev and price_curr < price_prev:
                            signals.append({
                                'type': 'bullish',
                                'index': signal_idx,
                                'entry_price': df['close'].iloc[signal_idx],
                                'time': df['open_time'].iloc[signal_idx]
                            })
                            log(f"🟢 Bullish Divergence 감지! RSI: {rsi_prev:.1f}→{rsi_curr:.1f}, Price: {price_prev:.2f}→{price_curr:.2f}")
                            
                            # 텔레그램 알림
                            send_divergence_alert('bullish', df['close'].iloc[signal_idx], rsi_curr)
                    break
    
    return signals

# ============================================================================
# 주문 실행 함수
# ============================================================================

def execute_entry(signal_type, amount=POSITION_SIZE):
    """진입 주문 실행"""
    try:
        # 1. 마진 모드 설정 (ISOLATED)
        try:
            client.futures_change_margin_type(symbol=SYMBOL, marginType='ISOLATED')
            log(f"격리 모드 설정 완료")
        except:
            pass
        
        # 2. 레버리지 설정
        client.futures_change_leverage(symbol=SYMBOL, leverage=LEVERAGE)
        log(f"레버리지 {LEVERAGE}배 설정 완료")
        
        # 3. 포지션 방향 결정
        side = SIDE_SELL if signal_type == 'bearish' else SIDE_BUY
        
        # 4. 시장가 주문
        order = client.futures_create_order(
            symbol=SYMBOL,
            side=side,
            type=ORDER_TYPE_MARKET,
            quantity=amount
        )
        
        entry_price = float(order['avgPrice'])
        log(f"✅ {'숏' if signal_type == 'bearish' else '롱'} 진입 성공! 가격: {entry_price}, 수량: {amount}")
        
        position = {
            'order_id': order['orderId'],
            'type': signal_type,
            'side': side,
            'entry_price': entry_price,
            'amount': amount,
            'entry_time': datetime.now()
        }
        
        # 텔레그램 알림
        send_entry_alert(position)
        
        return position
        
    except Exception as e:
        log(f"❌ 진입 주문 실패: {e}")
        return None

def set_stop_loss(position):
    """스탑로스 설정"""
    try:
        entry_price = position['entry_price']
        signal_type = position['type']
        
        # 스탑로스 가격 계산
        if signal_type == 'bearish':
            # 숏: 진입가보다 위
            stop_price = entry_price * (1 + STOP_LOSS_BEAR / 100)
            side = SIDE_BUY  # 숏 청산 = 매수
        else:
            # 롱: 진입가보다 아래
            stop_price = entry_price * (1 - STOP_LOSS_BULL / 100)
            side = SIDE_SELL  # 롱 청산 = 매도
        
        # 스탑로스 주문
        stop_order = client.futures_create_order(
            symbol=SYMBOL,
            side=side,
            type=FUTURE_ORDER_TYPE_STOP_MARKET,
            stopPrice=round(stop_price, 2),
            quantity=position['amount'],
            closePosition=True
        )
        
        log(f"🛡️ 스탑로스 설정: {stop_price:.2f} ({STOP_LOSS_BEAR if signal_type == 'bearish' else STOP_LOSS_BULL}%)")
        return stop_order['orderId']
        
    except Exception as e:
        log(f"❌ 스탑로스 설정 실패: {e}")
        return None

def execute_partial_close(position, ratio=0.5):
    """부분 청산"""
    try:
        amount = position['amount'] * ratio
        side = SIDE_BUY if position['side'] == SIDE_SELL else SIDE_SELL
        
        order = client.futures_create_order(
            symbol=SYMBOL,
            side=side,
            type=ORDER_TYPE_MARKET,
            quantity=amount
        )
        
        log(f"💰 부분 익절 ({ratio*100}%) 성공! 수량: {amount}")
        return order
        
    except Exception as e:
        log(f"❌ 부분 청산 실패: {e}")
        return None

def execute_full_close(position):
    """전체 청산"""
    try:
        side = SIDE_BUY if position['side'] == SIDE_SELL else SIDE_SELL
        
        order = client.futures_create_order(
            symbol=SYMBOL,
            side=side,
            type=ORDER_TYPE_MARKET,
            quantity=position['amount'],
            reduceOnly=True
        )
        
        log(f"🏁 전체 청산 성공! 수량: {position['amount']}")
        return order
        
    except Exception as e:
        log(f"❌ 전체 청산 실패: {e}")
        return None

def cancel_stop_loss(stop_order_id):
    """스탑로스 취소"""
    try:
        client.futures_cancel_order(symbol=SYMBOL, orderId=stop_order_id)
        log(f"🗑️ 스탑로스 주문 취소: {stop_order_id}")
    except Exception as e:
        log(f"⚠️ 스탑로스 취소 실패: {e}")

def get_current_price():
    """현재 가격 조회"""
    try:
        ticker = client.futures_symbol_ticker(symbol=SYMBOL)
        return float(ticker['price'])
    except:
        return None

def calculate_profit(position, current_price):
    """현재 수익률 계산 (종가 기준)"""
    entry_price = position['entry_price']
    signal_type = position['type']
    
    if signal_type == 'bearish':
        # 숏: 가격 하락이 이익
        profit = ((entry_price - current_price) / entry_price) * 100
    else:
        # 롱: 가격 상승이 이익
        profit = ((current_price - entry_price) / entry_price) * 100
    
    return profit

def get_current_candle():
    """현재 봉 정보 가져오기 (고가/저가 포함)"""
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
    """현재 봉에서 도달 가능한 최대 수익률 계산 (고가/저가 기준)"""
    if candle is None:
        return 0
    
    entry_price = position['entry_price']
    signal_type = position['type']
    
    if signal_type == 'bearish':
        # 숏: 저가에서 최대 이익
        max_profit = ((entry_price - candle['low']) / entry_price) * 100
    else:
        # 롱: 고가에서 최대 이익
        max_profit = ((candle['high'] - entry_price) / entry_price) * 100
    
    return max_profit

# ============================================================================
# 메인 봇 로직
# ============================================================================

def main():
    log("="*80)
    log("🤖 RSI Divergence 자동매매 봇 시작")
    log("="*80)
    log(f"심볼: {SYMBOL}")
    log(f"타임프레임: {TIMEFRAME}")
    log(f"레버리지: {LEVERAGE}배")
    log(f"포지션 크기: {POSITION_SIZE} BTC")
    log(f"부분 익절: {PARTIAL_PROFIT_TARGET}% 도달 시 {PARTIAL_PROFIT_RATIO*100}%")
    log(f"보유 기간: {HOLD_BARS}봉 (225분)")
    log(f"스탑로스: Bear {STOP_LOSS_BEAR}%, Bull {STOP_LOSS_BULL}%")
    log("="*80)
    
    # 포지션 추적
    active_positions = {}
    
    while True:
        try:
            log(f"\n{'='*60}")
            log(f"📊 데이터 업데이트 중... ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
            
            # 1. 최신 데이터 가져오기
            df = get_historical_data(SYMBOL, TIMEFRAME, limit=CANDLES_TO_LOAD)
            
            if df is None or len(df) < RSI_PERIOD + LOOKBACK_LEFT + RANGE_UPPER:
                log("⚠️ 데이터 부족, 다음 주기 대기...")
                time.sleep(60)
                continue
            
            # 2. RSI 계산
            df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
            df = df.dropna().reset_index(drop=True)
            
            current_price = df['close'].iloc[-1]
            current_rsi = df['rsi'].iloc[-1]
            log(f"현재 가격: ${current_price:,.2f}, RSI: {current_rsi:.2f}")
            
            # 3. 다이버전스 신호 감지 (현재 봉에서 확정된 것만)
            # 다중 포지션 허용 (최대 3개)
            MAX_POSITIONS = 3
            
            if len(active_positions) < MAX_POSITIONS:
                signals = detect_regular_divergence(df)
                
                if signals:
                    signal = signals[0]  # 첫 번째 신호만 사용
                    
                    # 진입
                    position = execute_entry(signal['type'], POSITION_SIZE)
                    
                    if position:
                        # 스탑로스 설정
                        stop_order_id = set_stop_loss(position)
                        
                        # 포지션 기록
                        position['stop_order_id'] = stop_order_id
                        position['partial_closed'] = False
                        
                        active_positions[position['order_id']] = position
                        
                        log(f"✅ 포지션 오픈 완료: {signal['type'].upper()} (총 {len(active_positions)}개)")
                else:
                    if len(active_positions) == 0:
                        log("📭 신호 없음")
            else:
                log(f"⚠️ 최대 포지션 수 도달 ({MAX_POSITIONS}개), 신호 무시")
            
            # 4. 기존 포지션 관리
            for pos_id in list(active_positions.keys()):
                position = active_positions[pos_id]
                
                # 현재 봉 데이터 가져오기
                current_candle = get_current_candle()
                if current_candle is None:
                    log("⚠️ 현재 봉 데이터 가져오기 실패")
                    continue
                
                current_price = current_candle['close']
                
                # 현재 수익률 계산 (종가 기준)
                profit = calculate_profit(position, current_price)
                
                # 현재 봉에서 도달 가능한 최대 수익률 (고가/저가 기준)
                max_profit_in_candle = calculate_max_profit_in_candle(position, current_candle)
                
                # 보유 시간 계산 (실제 시간 기준)
                time_held = datetime.now() - position['entry_time']
                minutes_held = time_held.total_seconds() / 60
                bars_held = int(minutes_held / 15)  # 15분 = 1봉
                
                log(f"📍 포지션 #{pos_id}: {position['type'].upper()}, "
                    f"진입가: ${position['entry_price']:,.2f}, "
                    f"현재: ${current_price:,.2f}, "
                    f"수익(종가): {profit:+.2f}%, "
                    f"최대수익(봉내): {max_profit_in_candle:+.2f}%, "
                    f"보유: {bars_held}봉 ({minutes_held:.0f}분)")
                
                # 부분 익절 체크 (고가/저가 기준으로 0.4% 도달 확인)
                if not position['partial_closed'] and max_profit_in_candle >= PARTIAL_PROFIT_TARGET:
                    log(f"🎯 부분 익절 조건 달성! (최대 {max_profit_in_candle:.2f}% >= {PARTIAL_PROFIT_TARGET}%)")
                    
                    result = execute_partial_close(position, PARTIAL_PROFIT_RATIO)
                    
                    if result:
                        position['partial_closed'] = True
                        position['amount'] *= (1 - PARTIAL_PROFIT_RATIO)  # 남은 수량 업데이트
                        log(f"✅ 부분 익절 완료, 남은 수량: {position['amount']}")
                        
                        # 텔레그램 알림
                        send_partial_close_alert(position, max_profit_in_candle)
                
                # 15봉 도달 체크 (실제 시간 기준)
                if bars_held >= HOLD_BARS:
                    log(f"⏰ {HOLD_BARS}봉 도달! ({minutes_held:.0f}분 경과) 전체 청산 실행")
                    
                    # 전체 청산
                    result = execute_full_close(position)
                    
                    if result:
                        # 스탑로스 취소
                        if position.get('stop_order_id'):
                            cancel_stop_loss(position['stop_order_id'])
                        
                        # 최종 수익 계산
                        final_price = get_current_price()
                        final_profit = calculate_profit(position, final_price)
                        
                        log(f"🏁 포지션 종료: 최종 수익률 {final_profit:+.2f}%")
                        
                        # 텔레그램 알림
                        send_final_close_alert(position, final_profit, final_price)
                        
                        # 포지션 제거
                        del active_positions[pos_id]
            
            # 5. 다음 봉까지 대기
            current_time = datetime.now()
            log(f"\n⏳ 다음 봉까지 대기 중... (15분) - 현재: {current_time.strftime('%H:%M:%S')}")
            time.sleep(900)  # 15분 = 900초
            
        except KeyboardInterrupt:
            log("\n🛑 봇 종료 (사용자 중단)")
            break
            
        except Exception as e:
            log(f"❌ 오류 발생: {e}")
            log("⏳ 60초 후 재시도...")
            time.sleep(60)

# ============================================================================
# 실행
# ============================================================================

if __name__ == "__main__":
    main()