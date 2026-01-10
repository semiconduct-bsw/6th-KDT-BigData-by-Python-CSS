import requests
from bs4 import BeautifulSoup
import pymysql

# DB 연결 (사용자 환경에 맞게 수정 필요)
conn = pymysql.connect(
    host='localhost',
    user='root',      # DB 아이디
    password='1234',      # DB 비밀번호 (여기에 입력하세요)
    db='my_db',       # DB 이름
    charset='utf8'
)
cur = conn.cursor()

end_page = 3  # 크롤링할 마지막 페이지 설정

for page in range(1, end_page + 1):
    print(f"\n[{page} 페이지 크롤링 중...]")
    url = f'https://zdnet.co.kr/news/?lstcode=0050&page={page}'
    response = requests.get(url)

    # HTML 모양 파싱
    soup = BeautifulSoup(response.content, 'html.parser')

    # 뉴스 박스들 모두 찾기 (인기 뉴스, 최신 뉴스 등)
    news_boxes = soup.find_all('div', class_='news_box')

    target_box = None

    # 1페이지는 '인기 뉴스'가 맨 위에 있어서 두 번째 박스(인덱스 1)가 '최신 뉴스'
    # 2페이지부터는 '인기 뉴스'가 없어서 첫 번째 박스(인덱스 0)가 '최신 뉴스'
    if page == 1 and len(news_boxes) >= 2:
        target_box = news_boxes[1]
    elif len(news_boxes) > 0:
        target_box = news_boxes[0]

    if target_box:
        # 뉴스 리스트(div)를 순회하며 제목(h3) 추출
        articles = target_box.find_all('div', recursive=False)
        
        for article in articles:
            # 각 기사 안에서 제목 태그(h3) 찾기
            title_tag = article.find('h3')
            if title_tag:
                title = title_tag.text.strip()
                print(title)
                
                # DB 저장
                sql = "INSERT INTO news (title, created_at) VALUES (%s, NOW())"
                cur.execute(sql, (title,))
                conn.commit()
    else:
        print("뉴스 박스를 찾을 수 없습니다.")

# 연결 종료
conn.close()
