# 여러 개 받는 데이터 : 리스트, Dictionary
user_list = ['홍길동','김우현','김민지'] #리스트

woohyun = user_list[1]
user_list[2] = '박민지'
print(woohyun)
print(user_list)

# 튜플
user_tuple = ('홍길동','김우현','김민지')

# 딕셔너리
user_dict = {
    'name':'홍길동',
    'age':20,
    'email':'hong@naver.com'
    }

city = user_dict['city'] #city 키에 해당하는 값을 가져옴
