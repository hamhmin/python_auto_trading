
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

def send_positions_status(active_positions):
    """현재 포지션 상태 전송"""
    if not active_positions:
        message = """
📊 <b>현재 포지션 현황</b>

포지션이 없습니다.

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""".replace("{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        send_telegram_message(message)
        return
    
    message = f"""
📊 <b>현재 포지션 현황</b>

총 {len(active_positions)}개 포지션 보유

"""
    
    for pos_id, pos in active_positions.items():
        type_kr = "숏(SHORT)" if pos['type'] == 'bearish' else "롱(LONG)"
        partial_status = "✅ 완료" if pos['partial_closed'] else "❌ 미완료"
        
        # 보유 시간 계산
        time_held = datetime.now() - pos['entry_time']
        hours = time_held.total_seconds() / 3600
        bars_held = int(time_held.total_seconds() / 900)  # 15분 = 900초
        
        # 현재 수익률 계산
        current_price = get_current_price()
        if current_price:
            profit = calculate_profit(pos, current_price)
            profit_text = f"{profit:+.2f}%"
        else:
            profit_text = "계산 불가"
        
        message += f"""
━━━━━━━━━━━━━━━━━━━━
🔖 포지션 ID: {pos['position_id']}
📊 방향: {type_kr}
💰 진입가: ${pos['entry_price']:,.2f}
📦 현재 수량: {pos['amount']:.4f} BTC
💎 부분 익절: {partial_status}
📈 수익률: {profit_text}
⏱️ 보유: {bars_held}봉 ({hours:.1f}시간)

"""
    
    message += f"""
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
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
    """Regular Divergence 감지 - 최근 확정된 봉만 체크"""
    signals = []
    
    rsi = df['rsi']
    high = df['high']
    low = df['low']
    
    # 🔧 가장 최근 확정된 봉만 체크 (마지막 봉은 아직 확정 안됨)
    check_idx = len(df) - LOOKBACK_RIGHT - 1
    
    if check_idx < LOOKBACK_LEFT:
        return signals
    
    # Bearish Divergence - 최근 봉만
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
                        log(f"🔴 Bearish Divergence 감지! RSI: {rsi_prev:.1f}→{rsi_curr:.1f}, Price: {price_prev:.2f}→{price_curr:.2f}")
                        
                        # 텔레그램 알림
                        send_divergence_alert('bearish', df['close'].iloc[signal_idx], rsi_curr)
                break
    
    # Bullish Divergence - 최근 봉만
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
                        log(f"🟢 Bullish Divergence 감지! RSI: {rsi_prev:.1f}→{rsi_curr:.1f}, Price: {price_prev:.2f}→{price_curr:.2f}")
                        
                        # 텔레그램 알림
                        send_divergence_alert('bullish', df['close'].iloc[signal_idx], rsi_curr)
                break
    
    return signals

# ============================================================================
# 주문 실행 함수 - 🔧 수정됨
# ============================================================================

def execute_entry(signal_type, amount=POSITION_SIZE):
    """진입 주문 실행 - division by zero 방지"""
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
        
        # 🔧 진입 가격 가져오기 (여러 방법 시도)
        entry_price = 0.0
        
        # 방법 1: avgPrice 사용
        if 'avgPrice' in order and order['avgPrice']:
            entry_price = float(order['avgPrice'])
        
        # 방법 2: avgPrice가 없으면 fills 사용
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
        
        # 방법 3: 둘 다 없으면 현재 시장가 사용
        if entry_price == 0.0:
            ticker = client.futures_symbol_ticker(symbol=SYMBOL)
            entry_price = float(ticker['price'])
            log(f"⚠️ 주문 응답에 가격 없음, 현재 시장가 사용: {entry_price}")
        
        # 🔧 가격 유효성 검증
        if entry_price <= 0:
            log(f"❌ 진입 가격이 유효하지 않음: {entry_price}")
            return None
        
        log(f"✅ {'숏' if signal_type == 'bearish' else '롱'} 진입 성공! 가격: {entry_price:,.2f}, 수량: {amount}")
        
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
    """스탑로스 설정 - 가격 유효성 검증 추가"""
    try:
        entry_price = position['entry_price']
        signal_type = position['type']
        
        # 🔧 진입 가격 유효성 검증
        if entry_price <= 0:
            log(f"❌ 진입 가격이 유효하지 않아 스탑로스 설정 불가: {entry_price}")
            return None
        
        # 스탑로스 가격 계산
        if signal_type == 'bearish':
            # 숏: 진입가보다 위
            stop_price = entry_price * (1 + STOP_LOSS_BEAR / 100)
            side = SIDE_BUY  # 숏 청산 = 매수
        else:
            # 롱: 진입가보다 아래
            stop_price = entry_price * (1 - STOP_LOSS_BULL / 100)
            side = SIDE_SELL  # 롱 청산 = 매도
        
        # 🔧 스탑 가격 유효성 검증
        if stop_price <= 0:
            log(f"❌ 스탑로스 가격이 음수: {stop_price}")
            return None
        
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
    """부분 청산 - 정확한 수량 계산"""
    try:
        # 🔧 현재 포지션의 정확한 수량으로 계산
        close_amount = position['amount'] * ratio
        
        # 🔧 소수점 처리 (바이낸스 최소 단위에 맞춤)
        close_amount = round(close_amount, 3)
        
        side = SIDE_BUY if position['side'] == SIDE_SELL else SIDE_SELL
        
        log(f"💰 부분 익절 시도: 전체 {position['amount']:.4f} BTC 중 {close_amount:.4f} BTC 청산")
        
        order = client.futures_create_order(
            symbol=SYMBOL,
            side=side,
            type=ORDER_TYPE_MARKET,
            quantity=close_amount,
            reduceOnly=True
        )
        
        log(f"✅ 부분 익절 ({ratio*100}%) 성공! 청산: {close_amount:.4f} BTC")
        return order
        
    except Exception as e:
        log(f"❌ 부분 청산 실패: {e}")
        return None

def execute_full_close(position):
    """전체 청산 - 남은 수량만 청산"""
    try:
        # 🔧 현재 포지션에 남아있는 정확한 수량
        close_amount = position['amount']
        
        # 🔧 소수점 처리
        close_amount = round(close_amount, 3)
        
        side = SIDE_BUY if position['side'] == SIDE_SELL else SIDE_SELL
        
        log(f"🏁 전체 청산 시도: {close_amount:.4f} BTC")
        
        order = client.futures_create_order(
            symbol=SYMBOL,
            side=side,
            type=ORDER_TYPE_MARKET,
            quantity=close_amount,
            reduceOnly=True
        )
        
        log(f"✅ 전체 청산 성공! 수량: {close_amount:.4f} BTC")
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
    """현재 수익률 계산 (종가 기준) - division by zero 방지"""
    entry_price = position['entry_price']
    signal_type = position['type']
    
    # 🔧 division by zero 방지
    if entry_price <= 0:
        log(f"⚠️ 진입 가격이 0이어서 수익률 계산 불가")
        return 0.0
    
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
    """현재 봉에서 도달 가능한 최대 수익률 계산 (고가/저가 기준) - division by zero 방지"""
    if candle is None:
        return 0
    
    entry_price = position['entry_price']
    signal_type = position['type']
    
    # 🔧 division by zero 방지
    if entry_price <= 0:
        log(f"⚠️ 진입 가격이 0이어서 최대 수익률 계산 불가")
        return 0.0
    
    if signal_type == 'bearish':
        # 숏: 저가에서 최대 이익
        max_profit = ((entry_price - candle['low']) / entry_price) * 100
    else:
        # 롱: 고가에서 최대 이익
        max_profit = ((candle['high'] - entry_price) / entry_price) * 100
    
    return max_profit

# ============================================================================
# 메인 봇 로직 - 🔧 신호 중복 방지 추가
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
    # 🔧 진입한 신호 인덱스 기록 (중복 방지)
    entered_signals = set()
    
    while True:
        try:
            log(f"\n{'='*60}")
            log(f"📊 데이터 업데이트 중... ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
            
            # 1. 최신 데이터 가져오기
            df = get_historical_data(SYMBOL, TIMEFRAME, limit=CANDLES_TO_LOAD)
            
            if df is None:
                log("❌ 데이터 로드 실패, 60초 후 재시도...")
                time.sleep(60)
                continue
            
            log(f"✅ 데이터 로드: {len(df)}개 캔들")
            
            # 2. RSI 계산
            df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
            df = df.dropna().reset_index(drop=True)
            
            log(f"✅ RSI 계산 후: {len(df)}개 캔들")
            
            # 필요한 최소 데이터 체크
            required_candles = RSI_PERIOD + LOOKBACK_LEFT + RANGE_UPPER
            
            if len(df) < required_candles:
                log(f"⚠️ 데이터 부족: {len(df)}개 < {required_candles}개 필요")
                log(f"   (RSI={RSI_PERIOD} + LOOKBACK={LOOKBACK_LEFT} + RANGE={RANGE_UPPER})")
                log(f"   📌 CANDLES_TO_LOAD를 {CANDLES_TO_LOAD + 50}로 증가 권장")
                time.sleep(60)
                continue
            
            current_price = df['close'].iloc[-1]
            current_rsi = df['rsi'].iloc[-1]
            log(f"현재 가격: ${current_price:,.2f}, RSI: {current_rsi:.2f}")
            
            # 3. 다이버전스 신호 감지
            if len(active_positions) < MAX_POSITIONS:
                signals = detect_regular_divergence(df)
                
                if signals:
                    for signal in signals:
                        signal_index = signal['index']
                        
                        # 🔧 이미 진입한 신호는 건너뛰기
                        if signal_index in entered_signals:
                            log(f"⚠️ 신호 #{signal_index}는 이미 진입함, 건너뜀")
                            continue
                        
                        # 진입 시도
                        position = execute_entry(signal['type'], POSITION_SIZE)
                        
                        if position:
                            # 진입 성공
                            # 🔧 포지션 ID 생성
                            position_id = get_next_position_id()
                            
                            # 스탑로스 설정
                            stop_order_id = set_stop_loss(position)
                            
                            # 포지션 기록
                            position['position_id'] = position_id  # 내부 추적 ID
                            position['stop_order_id'] = stop_order_id
                            position['partial_closed'] = False
                            position['signal_index'] = signal_index
                            position['initial_amount'] = POSITION_SIZE  # 초기 진입 수량 기록
                            
                            active_positions[position_id] = position  # 🔧 position_id를 키로 사용
                            entered_signals.add(signal_index)
                            
                            log(f"✅ 포지션 오픈 완료: ID={position_id}, {signal['type'].upper()}, 수량={POSITION_SIZE} BTC (총 {len(active_positions)}개)")
                            
                            # 최대 포지션 도달 시 중단
                            if len(active_positions) >= MAX_POSITIONS:
                                break
                        else:
                            # 🔧 진입 실패 (잔고 부족 등)
                            # 신호는 기록하되 포지션은 열지 않음
                            entered_signals.add(signal_index)
                            log(f"⚠️ 진입 실패했지만 신호 #{signal_index} 기록 (중복 방지)")
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
                
                log(f"📍 포지션 ID={pos_id}: {position['type'].upper()}, "
                    f"진입가: ${position['entry_price']:,.2f}, "
                    f"현재: ${current_price:,.2f}, "
                    f"현재수량: {position['amount']:.4f} BTC, "
                    f"수익(종가): {profit:+.2f}%, "
                    f"최대수익(봉내): {max_profit_in_candle:+.2f}%, "
                    f"보유: {bars_held}봉 ({minutes_held:.0f}분)")
                
                # 부분 익절 체크 (고가/저가 기준으로 0.4% 도달 확인)
                if not position['partial_closed'] and max_profit_in_candle >= PARTIAL_PROFIT_TARGET:
                    log(f"🎯 포지션 ID={pos_id} 부분 익절 조건 달성! (최대 {max_profit_in_candle:.2f}% >= {PARTIAL_PROFIT_TARGET}%)")
                    
                    result = execute_partial_close(position, PARTIAL_PROFIT_RATIO)
                    
                    if result:
                        # 🔧 남은 수량 정확히 계산
                        closed_amount = position['amount'] * PARTIAL_PROFIT_RATIO
                        position['amount'] = position['amount'] - closed_amount
                        position['partial_closed'] = True
                        
                        log(f"✅ 포지션 ID={pos_id} 부분 익절 완료, 남은 수량: {position['amount']:.4f} BTC")
                        
                        # 텔레그램 알림
                        send_partial_close_alert(position, max_profit_in_candle)
                
                # 15봉 도달 체크 (실제 시간 기준)
                if bars_held >= HOLD_BARS:
                    log(f"⏰ 포지션 ID={pos_id} {HOLD_BARS}봉 도달! ({minutes_held:.0f}분 경과) 전체 청산 실행")
                    
                    # 전체 청산 (남은 수량만)
                    result = execute_full_close(position)
                    
                    if result:
                        # 스탑로스 취소
                        if position.get('stop_order_id'):
                            cancel_stop_loss(position['stop_order_id'])
                        
                        # 최종 수익 계산
                        final_price = get_current_price()
                        final_profit = calculate_profit(position, final_price)
                        
                        log(f"🏁 포지션 ID={pos_id} 종료: 최종 수익률 {final_profit:+.2f}%")
                        
                        # 텔레그램 알림
                        send_final_close_alert(position, final_profit, final_price)
                        
                        # 포지션 제거
                        del active_positions[pos_id]
                        # 진입 신호도 제거 (나중에 다시 진입 가능하도록)
                        if 'signal_index' in position:
                            entered_signals.discard(position['signal_index'])
            
            # 5. 다음 봉까지 대기
            current_time = datetime.now()
            
            # 🔧 매 시간마다 포지션 상태 전송 (예: 매시 00분)
            if active_positions and current_time.minute == 0:
                send_positions_status(active_positions)
            
            log(f"\n⏳ 다음 봉까지 대기 중... (15분) - 현재: {current_time.strftime('%H:%M:%S')}")
            time.sleep(900)  # 15분 = 900초
            
        except KeyboardInterrupt:
            log("\n🛑 봇 종료 (사용자 중단)")
            break
            
        except Exception as e:
            import traceback
            log(f"❌ 오류 발생: {e}")
            log(f"📋 상세 오류:\n{traceback.format_exc()}")
            log("⏳ 60초 후 재시도...")
            time.sleep(60)

# ============================================================================
# 실행
# ============================================================================

if __name__ == "__main__":
    main()