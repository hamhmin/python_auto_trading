import requests
import json
import time
from datetime import datetime

def get_binance_futures_json(symbol="BTCUSDT", interval="15m", total_calls=1):
    url = "https://fapi.binance.com/fapi/v1/klines"
    limit = 1500
    all_formatted_data = []
    last_timestamp = None

    print(f"{symbol} 데이터를 가져오는 중...")

    for i in range(total_calls):
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        if last_timestamp:
            params["endTime"] = last_timestamp - 1

        try:
            response = requests.get(url, params=params)
            data = response.json()

            if not data:
                break

            # 데이터를 역순으로(최신순) 가져오므로 변환 후 리스트 앞에 추가
            formatted_batch = []
            for d in data:
                # 데이터 매핑 (인덱스 순서: 0:OpenTime, 1:Open, 2:High, 3:Low, 4:Close, 5:Volume, 6:CloseTime)
                candle = {
                    "open_time": datetime.fromtimestamp(d[0] / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                    "open": float(d[1]),
                    "high": float(d[2]),
                    "low": float(d[3]),
                    "close": float(d[4]),
                    "volume": float(d[5]),
                    "close_time": datetime.fromtimestamp(d[6] / 1000).strftime('%Y-%m-%d %H:%M:%S')
                }
                formatted_batch.append(candle)

            # 전체 리스트에 병합 (과거 데이터가 앞에 오도록)
            all_formatted_data = formatted_batch + all_formatted_data
            
            # 다음 호출을 위해 가장 과거 데이터의 타임스탬프 갱신
            last_timestamp = data[0][0]
            
            print(f"[{i+1}/{total_calls}] 호출 완료 (현재까지 {len(all_formatted_data)}개 확보)")
            time.sleep(0.2)

        except Exception as e:
            print(f"에러 발생: {e}")
            break

    # JSON 파일 저장
    file_name = f"btc_15m_data.json"
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(all_formatted_data, f, indent=4, ensure_ascii=False)

    print("-" * 30)
    print(f"저장 완료: {file_name}")

if __name__ == "__main__":
    get_binance_futures_json()