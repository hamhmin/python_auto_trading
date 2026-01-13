import os
import json
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


# ============================================================================
# 로그 시스템
# ============================================================================

# 로그 디렉토리
LOG_DIR = "bot_logs"

# 봇 시작 시간
BOT_START_TIME = datetime.now()
LOG_FILENAME = BOT_START_TIME.strftime("bot_log_%Y%m%d_%H%M%S.json")
LOG_FILEPATH = os.path.join(LOG_DIR, LOG_FILENAME)

# 로그 데이터
LOG_DATA = {
    "bot_start_time": BOT_START_TIME.strftime("%Y-%m-%d %H:%M:%S"),
    "logs": []
}

def init_log_system():
    """로그 시스템 초기화"""
    global LOG_DATA
    
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
        print(f"📁 로그 디렉토리 생성: {LOG_DIR}")
    
    LOG_DATA["symbol"] = SYMBOL
    LOG_DATA["leverage"] = LEVERAGE
    LOG_DATA["stop_loss_bear"] = STOP_LOSS_BEAR
    LOG_DATA["stop_loss_bull"] = STOP_LOSS_BULL
    
    print(f"📝 로그 파일: {LOG_FILEPATH}")

def save_log_to_file():
    """로그를 파일에 저장"""
    try:
        with open(LOG_FILEPATH, 'w', encoding='utf-8') as f:
            json.dump(LOG_DATA, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ 로그 저장 실패: {e}")

def add_log_entry(message, level="INFO", log_type="TERMINAL"):
    """로그 항목 추가"""
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "level": level,
        "type": log_type,
        "message": message
    }
    
    LOG_DATA["logs"].append(entry)
    
    # 100개마다 저장
    if len(LOG_DATA["logs"]) % 100 == 0:
        save_log_to_file()

def cleanup_old_logs(days=30):
    """오래된 로그 삭제"""
    import glob
    from datetime import timedelta
    
    try:
        cutoff = datetime.now() - timedelta(days=days)
        
        for log_file in glob.glob(f"{LOG_DIR}/bot_log_*.json"):
            try:
                filename = os.path.basename(log_file)
                date_str = filename.replace('bot_log_', '').replace('.json', '')
                file_date = datetime.strptime(date_str, "%Y%m%d_%H%M%S")
                
                if file_date < cutoff:
                    os.remove(log_file)
                    print(f"🗑️ 오래된 로그 삭제: {filename}")
            except:
                pass
    except Exception as e:
        print(f"⚠️ 로그 정리 실패: {e}")

def finalize_log():
    """봇 종료 시 통계 추가"""
    global LOG_DATA
    
    total_logs = len(LOG_DATA["logs"])
    telegram_count = sum(1 for log in LOG_DATA["logs"] if log["type"] == "TELEGRAM")
    error_count = sum(1 for log in LOG_DATA["logs"] if log["level"] == "ERROR")
    event_count = sum(1 for log in LOG_DATA["logs"] if log["level"] == "EVENT")
    
    LOG_DATA["statistics"] = {
        "total_logs": total_logs,
        "telegram_messages": telegram_count,
        "errors": error_count,
        "events": event_count,
        "bot_end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    save_log_to_file()


POSITION_COUNTER = 0

def get_next_position_id():
    global POSITION_COUNTER
    POSITION_COUNTER += 1
    return POSITION_COUNTER

# ============================================================================
# 설정값
# ============================================================================

# 포지션 영속성
POSITIONS_FILE = "positions_data.json"

SYMBOL = "BTCUSDT"
TIMEFRAME = "15m"
LEVERAGE = 30
POSITION_SIZE = 0.004  # BTC 수량

# 전략 파라미터
RSI_PERIOD = 14
LOOKBACK_LEFT = 2
LOOKBACK_RIGHT = 5
RANGE_LOWER = 5
RANGE_UPPER = 60

# 청산 설정
HOLD_BARS = 38  # 15봉 = 225분 = 3.75시간
PARTIAL_PROFIT_TARGET = 0.8  # 0.4% 도달 시
PARTIAL_PROFIT_RATIO = 0.5  # 50% 청산

# 포지션 관리
MAX_POSITIONS = 10  # 최대 동시 포지션 수

# 리스크 관리
STOP_LOSS_BEAR = 3  # Bearish 스탑로스 (%)
STOP_LOSS_BULL = 3  # Bullish 스탑로스 (%)

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

def send_divergence_alert(div_type, pivot_idx, rsi_prev, rsi_curr, price_prev, price_curr):
    """다이버전스 발견 시 알림"""
    emoji = "🔴" if div_type == "bearish" else "🟢"
    type_kr = "하락(BEARISH)" if div_type == "bearish" else "상승(BULLISH)"
    
    message = f"""
{emoji} <b>다이버전스 발견!</b>

📊 {type_kr} 다이버전스
📈 RSI: {rsi_prev:.1f} → {rsi_curr:.1f}
💰 가격: ${price_prev:,.0f} → ${price_curr:,.0f}
📍 피봇 인덱스: {pivot_idx}
⏰ {datetime.now().strftime('%H:%M:%S')}

🎯 포지션 진입 예정...
"""
    send_telegram_message(message)


def send_error_alert(error_type, error_message, context=""):
    """에러 발생 시 텔레그램 알림"""
    msg = f"⚠️ 봇 에러!\n\n🚨 {error_type}\n💬 {error_message}"
    if context:
        msg += f"\n📍 {context}"
    msg += f"\n⏰ {datetime.now().strftime('%H:%M:%S')}"
    send_telegram_message(msg)


def send_entry_alert(position):
    """포지션 진입 알림 (상세)"""
    emoji = "🔴" if position['type'] == "bearish" else "🟢"
    type_kr = "숏(SHORT)" if position['type'] == "bearish" else "롱(LONG)"
    
    position_value = position['entry_price'] * position['amount']
    
    # 스탑로스 가격 계산
    stop_loss_pct = STOP_LOSS_BEAR if position['type'] == 'bearish' else STOP_LOSS_BULL
    if position['type'] == 'bearish':
        stop_price = position['entry_price'] * (1 + stop_loss_pct / 100)
    else:
        stop_price = position['entry_price'] * (1 - stop_loss_pct / 100)
    
    # 부분익절 가격 계산
    if position['type'] == 'bearish':
        partial_price = position['entry_price'] * (1 - PARTIAL_PROFIT_TARGET / 100)
    else:
        partial_price = position['entry_price'] * (1 + PARTIAL_PROFIT_TARGET / 100)
    
    expected_close = position['entry_time'] + timedelta(minutes=HOLD_BARS*15)
    hold_hours = HOLD_BARS * 15 / 60
    
    message = f"""
{emoji} <b>포지션 진입! #{position.get('position_id', '?')}</b>

📊 {SYMBOL} {type_kr}
💰 진입가: ${position['entry_price']:,.2f}
📦 수량: {position['amount']:.4f} BTC
💵 포지션 크기: ${position_value:,.2f}

🛡️ 스탑로스: ${stop_price:,.0f} (-{stop_loss_pct}%)
🎯 부분익절: ${partial_price:,.0f} (+{PARTIAL_PROFIT_TARGET}%)
⏰ 보유기간: {HOLD_BARS}봉 ({hold_hours:.1f}시간)

⏰ {position['entry_time'].strftime('%H:%M:%S')}
"""
    send_telegram_message(message)
    
    # 터미널 로그
    log("="*80, "EVENT")
    log(f"{emoji} 포지션 진입! ID={position.get('position_id', '?')}", "EVENT")
    log("="*80, "EVENT")
    log(f"📊 심볼: {SYMBOL} | 방향: {type_kr} | 레버리지: {LEVERAGE}배", "INFO")
    log(f"💰 진입가: ${position['entry_price']:,.2f} | 수량: {position['amount']:.4f} BTC | 포지션 크기: ${position_value:,.2f}", "INFO")
    log(f"🛡️ 스탑로스: ${stop_price:,.2f} (-{stop_loss_pct}%) | 🎯 부분익절: ${partial_price:,.2f} (+{PARTIAL_PROFIT_TARGET}%)", "INFO")
    log(f"⏰ 예상 청산: {expected_close.strftime('%Y-%m-%d %H:%M:%S')} ({HOLD_BARS}봉, {hold_hours:.1f}시간)", "INFO")
    if 'stop_order_id' in position:
        log(f"📋 스탑로스 주문 ID: {position['stop_order_id']}", "DEBUG")
    log("="*80, "EVENT")


def send_exit_alert(position, reason, final_profit):
    """포지션 청산 알림 (상세)"""
    emoji = "🎉" if final_profit > 0 else "😢"
    if "스탑로스" in reason:
        emoji = "🚨"
    elif "보유기간" in reason:
        emoji = "⏰"
    
    type_kr = "숏(SHORT)" if position['type'] == 'bearish' else "롱(LONG)"
    
    time_held = datetime.now() - position['entry_time']
    hours = time_held.total_seconds() / 3600
    
    # 현재가 추정
    current_price = position['entry_price'] * (1 + final_profit / 100) if position['type'] == 'bullish' else position['entry_price'] * (1 - final_profit / 100)
    
    closed_amount = position['amount']
    closed_value = current_price * closed_amount
    
    # 실현 손익
    if position['type'] == 'bearish':
        realized_pnl = (position['entry_price'] - current_price) * closed_amount
    else:
        realized_pnl = (current_price - position['entry_price']) * closed_amount
    
    message = f"""
{emoji} <b>{reason}! #{position.get('position_id', '?')}</b>

📊 {SYMBOL} {type_kr}
💰 진입가: ${position['entry_price']:,.2f}
📈 청산가: ${current_price:,.2f}
📦 청산 수량: {closed_amount:.4f} BTC
💵 청산 금액: ${closed_value:,.2f}

📊 수익률: {final_profit:+.2f}%
💵 실현 손익: ${realized_pnl:+.2f}
⏱️ 보유: {hours:.1f}시간

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
    
    # 스탑로스인 경우 추가 정보
    if "스탑로스" in reason:
        stop_loss_pct = STOP_LOSS_BEAR if position['type'] == 'bearish' else STOP_LOSS_BULL
        if position['type'] == 'bearish':
            expected_stop = position['entry_price'] * (1 + stop_loss_pct / 100)
            liquidation_price = position['entry_price'] * (1 + 100 / LEVERAGE / 100)
        else:
            expected_stop = position['entry_price'] * (1 - stop_loss_pct / 100)
            liquidation_price = position['entry_price'] * (1 - 100 / LEVERAGE / 100)
        
        message += f"""
⚠️ 청산 원인: {reason}
🛡️ 스탑로스가: ${expected_stop:,.0f}
💀 강제청산가: ${liquidation_price:,.0f}
"""
    
    send_telegram_message(message)
    
    # 터미널 로그
    log("="*80, "EVENT")
    log(f"{emoji} {reason}! ID={position.get('position_id', '?')}", "EVENT")
    log("="*80, "EVENT")
    log(f"📊 {SYMBOL} {type_kr} | 진입가: ${position['entry_price']:,.2f}", "INFO")
    log(f"📈 청산가: ${current_price:,.2f} | 수익: {final_profit:+.2f}%", "INFO")
    log(f"📦 청산: {closed_amount:.4f} BTC | 금액: ${closed_value:,.2f}", "INFO")
    log(f"💵 실현 손익: ${realized_pnl:+.2f} | ⏱️ 보유: {hours:.1f}시간", "INFO")
    log("="*80, "EVENT")


def send_bot_end_alert(reason=""):
    """봇 종료 알림"""
    message = f"""
🔄 <b>봇 종료</b>

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
    send_telegram_message(message)

# ============================================================================
# 바이낸스 스탑로스 주문
# ============================================================================

def place_stop_loss_order(position):
    """바이낸스에 스탑로스 주문 등록"""
    try:
        stop_loss_pct = STOP_LOSS_BEAR if position['type'] == 'bearish' else STOP_LOSS_BULL
        
        if position['type'] == 'bearish':
            # SHORT: 진입가보다 높은 가격에 스탑로스
            stop_price = round(position['entry_price'] * (1 + stop_loss_pct / 100), 1)
            side = SIDE_BUY
            position_side = 'SHORT'
        else:
            # LONG: 진입가보다 낮은 가격에 스탑로스
            stop_price = round(position['entry_price'] * (1 - stop_loss_pct / 100), 1)
            side = SIDE_SELL
            position_side = 'LONG'
        
        # 스탑 마켓 주문
        order = client.futures_create_order(
            symbol=SYMBOL,
            side=side,
            type='STOP_MARKET',
            stopPrice=stop_price,
            quantity=position['amount'],
            positionSide=position_side
        )
        
        log(f"✅ 스탑로스 주문 등록: ${stop_price:,.0f} (주문ID: {order['orderId']})", "INFO")
        return order['orderId']
        
    except BinanceAPIException as e:
        msg = f"[{e.code}] {e.message}"
        log(f"스탑로스 주문 실패: {msg}", "ERROR")
        send_error_alert("스탑로스 주문 실패", msg, "place_stop_loss_order")
        return None
    except Exception as e:
        log(f"스탑로스 주문 오류: {e}", "ERROR")
        return None

def cancel_stop_loss_order(stop_order_id):
    """스탑로스 주문 취소"""
    if not stop_order_id:
        return
    
    try:
        client.futures_cancel_order(
            symbol=SYMBOL,
            orderId=stop_order_id
        )
        log(f"스탑로스 주문 취소: {stop_order_id}", "DEBUG")
    except Exception as e:
        log(f"스탑로스 주문 취소 실패: {e}", "DEBUG")

def check_stop_loss_filled(position):
    """스탑로스 주문 체결 확인"""
    if not position.get('stop_order_id'):
        return None
    
    try:
        order = client.futures_get_order(
            symbol=SYMBOL,
            orderId=position['stop_order_id']
        )
        
        if order['status'] == 'FILLED':
            return {
                'filled': True,
                'avg_price': float(order['avgPrice']),
                'reason': '스탑로스 주문 체결'
            }
        elif order['status'] in ['NEW', 'PARTIALLY_FILLED']:
            return {'filled': False}
        else:
            # CANCELED, EXPIRED 등
            return {'filled': False, 'canceled': True}
            
    except Exception as e:
        log(f"주문 상태 조회 실패: {e}", "ERROR")
        return None


def send_bot_start_alert(reason=""):
    """봇 시작 알림"""
    message = f"""
🔄 <b>봇 시작</b>

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
    check_idx = len(df) - LOOKBACK_RIGHT - 2  # 진행중 캔들 제외
    
    # 🔥 디버깅 로그
    log(f"[신호체크] check_idx={check_idx}, len(df)={len(df)}, RSI={rsi.iloc[check_idx]:.1f}", "INFO")
    
    if check_idx < LOOKBACK_LEFT:
        log(f"[신호체크] check_idx < LOOKBACK_LEFT - 데이터 부족", "DEBUG")
        return signals
    
    # Bearish
    is_pivot_high = find_pivot_high(rsi, LOOKBACK_LEFT, LOOKBACK_RIGHT, check_idx)
    log(f"[신호체크] Bearish 피벗 체크: {is_pivot_high}", "DEBUG")
    
    if is_pivot_high:
        log(f"[신호체크] ✅ RSI 피벗 고점 발견! 이전 피벗 검색 중...", "DEBUG")
        
        for j in range(check_idx - RANGE_LOWER, max(check_idx - RANGE_UPPER, LOOKBACK_LEFT), -1):
            if find_pivot_high(rsi, LOOKBACK_LEFT, LOOKBACK_RIGHT, j):
                log(f"[신호체크] ✅ 이전 피벗 발견! idx={j}", "DEBUG")
                
                signal_idx = check_idx + LOOKBACK_RIGHT
                
                rsi_curr = rsi.iloc[check_idx]
                rsi_prev = rsi.iloc[j]
                price_curr = high.iloc[check_idx]
                price_prev = high.iloc[j]
                
                log(f"[신호체크] RSI: {rsi_prev:.1f}→{rsi_curr:.1f} (하락:{rsi_curr < rsi_prev})", "DEBUG")
                log(f"[신호체크] 가격: ${price_prev:.0f}→${price_curr:.0f} (상승:{price_curr > price_prev})", "DEBUG")
                
                if rsi_curr < rsi_prev and price_curr > price_prev:
                    if signal_idx < len(df):
                        signals.append({
                            'type': 'bearish',
                            'index': signal_idx,
                            'entry_price': df['close'].iloc[signal_idx],
                            'time': df['open_time'].iloc[signal_idx]
                        })
                        log(f"🔴 Bearish Divergence! RSI: {rsi_prev:.1f}→{rsi_curr:.1f}", "EVENT")
                        send_divergence_alert('bearish', check_idx, rsi_prev, rsi_curr, price_prev, price_curr)
                    else:
                        log(f"⚠️ Bearish Divergence 감지! RSI: {rsi_prev:.1f}→{rsi_curr:.1f}", "EVENT")
                        log(f"   진입 시점(idx={signal_idx})이 데이터 범위({len(df)}) 밖 - 다음 체크 시 진입", "DEBUG")
                break
    
    # Bullish
    is_pivot_low = find_pivot_low(rsi, LOOKBACK_LEFT, LOOKBACK_RIGHT, check_idx)
    log(f"[신호체크] Bullish 피벗 체크: {is_pivot_low}", "DEBUG")
    
    if is_pivot_low:
        log(f"[신호체크] ✅ RSI 피벗 저점 발견! 이전 피벗 검색 중...", "DEBUG")
        
        for j in range(check_idx - RANGE_LOWER, max(check_idx - RANGE_UPPER, LOOKBACK_LEFT), -1):
            if find_pivot_low(rsi, LOOKBACK_LEFT, LOOKBACK_RIGHT, j):
                log(f"[신호체크] ✅ 이전 피벗 발견! idx={j}", "DEBUG")
                
                signal_idx = check_idx + LOOKBACK_RIGHT
                
                rsi_curr = rsi.iloc[check_idx]
                rsi_prev = rsi.iloc[j]
                price_curr = low.iloc[check_idx]
                price_prev = low.iloc[j]
                
                log(f"[신호체크] RSI: {rsi_prev:.1f}→{rsi_curr:.1f} (상승:{rsi_curr > rsi_prev})", "DEBUG")
                log(f"[신호체크] 가격: ${price_prev:.0f}→${price_curr:.0f} (하락:{price_curr < price_prev})", "DEBUG")
                
                if rsi_curr > rsi_prev and price_curr < price_prev:
                    if signal_idx < len(df):
                        signals.append({
                            'type': 'bullish',
                            'index': signal_idx,
                            'entry_price': df['close'].iloc[signal_idx],
                            'time': df['open_time'].iloc[signal_idx]
                        })
                        log(f"🟢 Bullish Divergence! RSI: {rsi_prev:.1f}→{rsi_curr:.1f}", "EVENT")
                        send_divergence_alert('bullish', check_idx, rsi_prev, rsi_curr, price_prev, price_curr)
                    else:
                        log(f"⚠️ Bullish Divergence 감지! RSI: {rsi_prev:.1f}→{rsi_curr:.1f}", "EVENT")
                        log(f"   진입 시점(idx={signal_idx})이 데이터 범위({len(df)}) 밖 - 다음 체크 시 진입", "DEBUG")
                break
    
    log(f"[신호체크] ✅ 감지된 신호: {len(signals)}개", "INFO")
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
        
        # 양방향 포지션 모드 지원
        if signal_type == 'bearish':
            side = SIDE_SELL
            position_side = 'SHORT'
        else:
            side = SIDE_BUY
            position_side = 'LONG'
        
        order = client.futures_create_order(
            symbol=SYMBOL,
            side=side,
            type=ORDER_TYPE_MARKET,
            quantity=amount,
            positionSide=position_side
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
        
        # 스탑로스 주문 등록
        stop_order_id = place_stop_loss_order(position)
        position['stop_order_id'] = stop_order_id
        
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
    except BinanceAPIException as e:
        msg = f"[{e.code}] {e.message}"
        log(f"API 에러: {msg}", "ERROR")
        send_error_alert("API 에러", msg, "진입")
        return None
    except Exception as e:
        log(f"진입 실패: {e}", "ERROR")
        send_error_alert("진입 실패", str(e), "execute_entry")
        return None

def execute_partial_close(position, ratio=0.5):
    try:
        close_amount = round(position['amount'] * ratio, 3)
        
        # 양방향 모드: 포지션 타입에 따라 청산 방향 결정
        if position['type'] == 'bearish':
            side = SIDE_BUY  # SHORT 청산은 BUY
            position_side = 'SHORT'
        else:
            side = SIDE_SELL  # LONG 청산은 SELL
            position_side = 'LONG'
        
        order = client.futures_create_order(
            symbol=SYMBOL,
            side=side,
            type=ORDER_TYPE_MARKET,
            quantity=close_amount,
            positionSide=position_side
        )
        
        log(f"✅ 부분 익절 {close_amount:.4f} BTC", "EVENT")
        return order
    except BinanceAPIException as e:
        msg = f"[{e.code}] {e.message}"
        log(f"API 에러: {msg}", "ERROR")
        send_error_alert("API 에러", msg, "부분청산")
        return None
    except Exception as e:
        log(f"부분청산 실패: {e}", "ERROR")
        send_error_alert("부분청산 실패", str(e), "execute_partial_close")
        return None

def execute_full_close(position):
    try:
        close_amount = round(position['amount'], 3)
        
        # 양방향 모드: 포지션 타입에 따라 청산 방향 결정
        if position['type'] == 'bearish':
            side = SIDE_BUY  # SHORT 청산은 BUY
            position_side = 'SHORT'
        else:
            side = SIDE_SELL  # LONG 청산은 SELL
            position_side = 'LONG'
        
        order = client.futures_create_order(
            symbol=SYMBOL,
            side=side,
            type=ORDER_TYPE_MARKET,
            quantity=close_amount,
            positionSide=position_side
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


# ============================================================================
# 포지션 영속성
# ============================================================================

def save_positions(active_positions, entered_signals):
    """포지션 데이터를 JSON 파일로 저장"""
    try:
        positions_to_save = {}
        for pos_id, pos in active_positions.items():
            pos_copy = pos.copy()
            if isinstance(pos_copy.get('entry_time'), datetime):
                pos_copy['entry_time'] = pos_copy['entry_time'].isoformat()
            if 'side' in pos_copy:
                pos_copy['side'] = str(pos_copy['side'])
            positions_to_save[str(pos_id)] = pos_copy
        
        data = {
            "active_positions": positions_to_save,
            "entered_signals": list(entered_signals),
            "last_updated": datetime.now().isoformat(),
            "position_counter": POSITION_COUNTER
        }
        
        with open(POSITIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        log(f"💾 저장: {len(active_positions)}개 포지션", "DEBUG")
        return True
    except Exception as e:
        log(f"저장 실패: {e}", "ERROR")
        return False

def load_positions():
    """JSON에서 포지션 로드"""
    global POSITION_COUNTER
    
    try:
        if not os.path.exists(POSITIONS_FILE):
            log("💾 저장된 포지션 없음", "INFO")
            return {}, set()
        
        with open(POSITIONS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        active_positions = {}
        for pos_id, pos in data.get("active_positions", {}).items():
            pos_copy = pos.copy()
            if 'entry_time' in pos_copy:
                try:
                    pos_copy['entry_time'] = datetime.fromisoformat(pos_copy['entry_time'])
                except:
                    pos_copy['entry_time'] = datetime.now()
            if 'side' in pos_copy:
                if 'SELL' in str(pos_copy['side']):
                    pos_copy['side'] = SIDE_SELL
                else:
                    pos_copy['side'] = SIDE_BUY
            active_positions[int(pos_id)] = pos_copy
        
        entered_signals = set(data.get("entered_signals", []))
        
        if "position_counter" in data:
            POSITION_COUNTER = data["position_counter"]
        
        log(f"💾 로드: {len(active_positions)}개 포지션", "INFO")
        
        if active_positions:
            log("="*60, "INFO")
            for pos_id, pos in active_positions.items():
                elapsed = (datetime.now() - pos['entry_time']).total_seconds() / 60
                log(f"  #{pos_id}: {pos['type']} ${pos['entry_price']:.0f} "
                    f"{pos['amount']:.4f}BTC ({elapsed:.0f}분)", "INFO")
            log("="*60, "INFO")
        
        return active_positions, entered_signals
    except Exception as e:
        log(f"로드 실패: {e}", "ERROR")
        return {}, set()


def main():
    log("="*80, "EVENT")
    log("🤖 RSI Divergence 자동매매 봇 시작", "EVENT")
    log(f"심볼: {SYMBOL} | 타임프레임: {TIMEFRAME} | 레버리지: {LEVERAGE}배")
    log(f"포지션 크기: {POSITION_SIZE} BTC | 최대: {MAX_POSITIONS}개")
    log(f"부분 익절: {PARTIAL_PROFIT_TARGET}% | 보유: {HOLD_BARS}봉 (약 {HOLD_BARS*15/60:.1f}시간)")
    log(f"스탑로스: Bear {STOP_LOSS_BEAR}% / Bull {STOP_LOSS_BULL}%")
    log("="*80, "EVENT")
    send_bot_start_alert()
    
    # 포지션 로드
    active_positions, entered_signals = load_positions()
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
                                save_positions(active_positions, entered_signals)
                                
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
                
                # 1️⃣ 스탑로스 체결 확인
                stop_result = check_stop_loss_filled(position)
                
                if stop_result and stop_result.get('filled'):
                    log(f"🚨 ID={pos_id} 스탑로스 체결!", "EVENT")
                    
                    avg_price = stop_result['avg_price']
                    final_profit = calculate_profit(position, avg_price)
                    send_exit_alert(position, "스탑로스", final_profit)
                    
                    del active_positions[pos_id]
                    if 'signal_index' in position:
                        entered_signals.discard(position['signal_index'])
                    save_positions(active_positions, entered_signals)
                    
                    continue
                
                max_profit_in_candle = calculate_max_profit_in_candle(position, current_candle)
                
                # 2️⃣ 부분 익절 체크
                if not position['partial_closed'] and max_profit_in_candle >= PARTIAL_PROFIT_TARGET:
                    log(f"🎯 ID={pos_id} 부분 익절 {max_profit_in_candle:.2f}%", "EVENT")
                    
                    result = execute_partial_close(position, PARTIAL_PROFIT_RATIO)
                    
                    if result:
                        closed_amount = position['amount'] * PARTIAL_PROFIT_RATIO
                        # 기존 스탑로스 취소
                        cancel_stop_loss_order(position.get('stop_order_id'))
                        
                        # 남은 수량 업데이트
                        position['amount'] = position['amount'] - closed_amount
                        position['partial_closed'] = True
                        
                        # 새 스탑로스 주문
                        new_stop_order_id = place_stop_loss_order(position)
                        position['stop_order_id'] = new_stop_order_id
                        
                        save_positions(active_positions, entered_signals)
                        send_exit_alert(position, "부분 익절", max_profit_in_candle)
                
                # 3️⃣ 보유기간 도달 체크 (🔧 분 단위로 체크)
                target_minutes = HOLD_BARS * 15
                
                if minutes_held >= target_minutes:
                    log(f"⏰ ID={pos_id} {HOLD_BARS}봉({target_minutes}분) 도달 (실제: {minutes_held:.1f}분)", "EVENT")
                    
                    # 스탑로스 주문 취소
                    cancel_stop_loss_order(position.get('stop_order_id'))
                    
                    result = execute_full_close(position)
                    
                    if result:
                        final_price = get_current_price()
                        final_profit = calculate_profit(position, final_price)
                        send_exit_alert(position, "보유기간 종료", final_profit)
                        del active_positions[pos_id]
                        if 'signal_index' in position:
                            entered_signals.discard(position['signal_index'])
                        save_positions(active_positions, entered_signals)
            
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
            send_bot_end_alert()
            break
        except Exception as e:
            import traceback
            log(f"오류: {e}", "ERROR")
            log(f"{traceback.format_exc()}")
            time.sleep(60)

if __name__ == "__main__":
    main()