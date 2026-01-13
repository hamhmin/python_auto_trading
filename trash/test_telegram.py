import os
import requests
from dotenv import load_dotenv

# .env 로드
load_dotenv()

BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

print("="*60)
print("텔레그램 연결 테스트")
print("="*60)

# 1. 설정 확인
if not BOT_TOKEN:
    print("❌ TELEGRAM_BOT_TOKEN이 설정되지 않았습니다!")
    print("   .env 파일에 TELEGRAM_BOT_TOKEN을 추가하세요.")
    exit(1)

if not CHAT_ID:
    print("❌ TELEGRAM_CHAT_ID가 설정되지 않았습니다!")
    print("   .env 파일에 TELEGRAM_CHAT_ID를 추가하세요.")
    print("\n📌 Chat ID 얻는 방법:")
    print(f"   1. 텔레그램 봇과 대화 시작 (/start)")
    print(f"   2. 브라우저에서 접속:")
    print(f"      https://api.telegram.org/bot{BOT_TOKEN}/getUpdates")
    print(f"   3. 'chat':{{{'id':숫자}}} 확인")
    exit(1)

print(f"✅ BOT_TOKEN: {BOT_TOKEN[:20]}...")
print(f"✅ CHAT_ID: {CHAT_ID}")

# 2. 메시지 전송 테스트
print("\n📤 테스트 메시지 전송 중...")

url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
data = {
    "chat_id": CHAT_ID,
    "text": """🤖 <b>연결 테스트 성공!</b>

텔레그램 봇이 정상적으로 작동합니다.

✅ 이제 자동매매 봇을 실행하면
   모든 거래 알림을 받을 수 있습니다!

📊 알림 종류:
  • 다이버전스 신호 감지
  • 진입 체결
  • 부분 익절
  • 최종 청산 (수익률 포함)

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""".replace("{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
            __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
    "parse_mode": "HTML"
}

try:
    response = requests.post(url, data=data, timeout=10)
    
    if response.status_code == 200:
        print("✅ 메시지 전송 성공!")
        print("\n📱 텔레그램을 확인하세요.")
        print("   메시지가 도착했다면 설정 완료입니다!")
    else:
        print(f"❌ 전송 실패: HTTP {response.status_code}")
        print(f"   응답: {response.text}")
        
        if response.status_code == 400:
            print("\n💡 가능한 원인:")
            print("   - Chat ID가 잘못되었습니다.")
            print("   - 봇과 대화를 시작하지 않았습니다. (/start 입력)")
        elif response.status_code == 401:
            print("\n💡 가능한 원인:")
            print("   - BOT_TOKEN이 잘못되었습니다.")
            
except Exception as e:
    print(f"❌ 오류 발생: {e}")

print("\n" + "="*60)
print("테스트 완료")
print("="*60)