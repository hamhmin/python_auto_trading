import requests
import json
from datetime import datetime

def save_btc_futures_to_json(filename="btc_15m_data.json"):
    # 1. 바이낸스 선물 API 호출 (최대 1500개)
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {
        "symbol": "BTCUSDT",
        "interval": "15m",
        "limit": 1500
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status() # 오류 발생 시 예외 처리
        data = response.json()
        
        # 2. 데이터 가공 (리스트 형태를 딕셔너리 리스트로 변환)
        json_list = []
        for d in data:
            candle = {
                "open_time": datetime.fromtimestamp(d[0]/1000).strftime('%Y-%m-%d %H:%M:%S'),
                "open": float(d[1]),
                "high": float(d[2]),
                "low": float(d[3]),
                "close": float(d[4]),
                "volume": float(d[5]),
                "close_time": datetime.fromtimestamp(d[6]/1000).strftime('%Y-%m-%d %H:%M:%S')
            }
            json_list.append(candle)
        
        # 3. 파일로 저장
        with open(filename, 'w', encoding='utf-8') as f:
            # indent=4로 저장하면 파일 내용이 사람이 보기 좋게 정렬됩니다.
            json.dump(json_list, f, ensure_ascii=False, indent=4)
            
        print(f"성공: {len(json_list)}개의 데이터를 '{filename}' 파일로 저장했습니다.")

    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    save_btc_futures_to_json()
# RSI Regular Divergence는 트레이딩뷰 {} 에서 가져오고 lookback_right를 1로 설정, 15분봉,  1개 봉이 지난 후 다이버전스 발생시 포지션 진입, 수익률 0.4% 도달시 50% 부분 익절 ,15봉 후 포지션 종료
# 캔들 1500개 데이터로 최대 손실 2%. 스탑로스는 2.5~3%로.수익은 +8.61% (수수료 포함)

