users =[
   {
      'id':'jm90hong@naver.com',
      'pw':'1234',
      'nick':'닉네임홍길동',
      'created_at':'2025-12-23 10:00:00'
   },
   {
      'id':'user2@example.com',
      'pw':'abcd1234',
      'nick':'바나나러버',
      'created_at':'2025-12-24 09:30:00'
   },
   {
      'id':'grape99@example.com',
      'pw':'pw9999',
      'nick':'포도농장주',
      'created_at':'2025-12-25 18:45:00'
   },
   {
      'id':'strawberry@naver.com',
      'pw':'berry2025',
      'nick':'딸기조아',
      'created_at':'2025-12-26 07:20:00'
   },
]

# for idx in range(len(users)):
#   user = users[idx]
#   print(user)

# user는 Dictionary, users는 수(Dictionary 등록된 수)
for user in users:
    print(user['nick'])
