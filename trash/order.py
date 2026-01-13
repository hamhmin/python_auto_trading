import os
from dotenv import load_dotenv
from binance.client import Client
from binance.enums import *

# .env 로드 및 클라이언트 생성
load_dotenv()
client = Client(os.getenv('API_KEY'), os.getenv('SECRET_KEY'))

def execute_smart_order(symbol="BTCUSDT", side=SIDE_BUY, amount=0.001, leverage=10):
    """
    레버리지 설정부터 주문까지 한 번에 처리하는 함수
    """
    try:
        # 1. 마진 모드 설정 (ISOLATED: 격리모드 추천)
        # 이미 설정되어 있더라도 다시 실행해도 무방하지만, 
        # 포지션이 이미 있을 때는 변경이 안 되므로 예외처리를 합니다.
        try:
            client.futures_change_margin_type(symbol=symbol, marginType='ISOLATED')
            print(f"[{symbol}] 격리 모드 설정 완료")
        except:
            pass # 이미 설정된 경우 무시

        # 2. 레버리지 설정
        client.futures_change_leverage(symbol=symbol, leverage=leverage)
        print(f"[{symbol}] 레버리지 {leverage}배 설정 완료")

        # 3. 시장가 주문 실행
        order = client.futures_create_order(
            symbol=symbol,
            side=side,
            type=ORDER_TYPE_MARKET,
            quantity=amount
        )
        
        print(f"✅ 주문 성공! 방향: {side}, 수량: {amount}")
        return order

    except Exception as e:
        print(f"❌ 주문 중 오류 발생: {e}")
        return None

# --- 사용 예시 ---

# 1. 롱 진입 (Bullish Divergence 발생 시)
# execute_smart_order(symbol="BTCUSDT", side=SIDE_BUY, amount=0.002, leverage=20)

# 2. 숏 진입 (Bearish Divergence 발생 시)
execute_smart_order(symbol="BTCUSDT", side=SIDE_SELL, amount=0.002, leverage=20)