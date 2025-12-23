import requests

# 1. 데이터를 가져올 주소(URL) 입력
url = "https://jsonplaceholder.typicode.com/todos"

# 2. 웹사이트에 데이터 요청 (GET 방식)
response = requests.get(url)

# 3. 가져온 데이터 상태 확인 (200이면 성공)
if response.status_code == 200:
    # 4. JSON 데이터를 파이썬 리스트로 변환
    data_list = response.json()
    
    # 5. 결과 확인 (리스트의 첫 번째, 두 번째 데이터만 출력해보기)
    print(f"총 데이터 개수: {len(data_list)}개")
    print(data_list[0]) # 첫 번째 데이터
    print(data_list[1]) # 두 번째 데이터
else:
    print("데이터를 가져오는 데 실패했습니다.")
