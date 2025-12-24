import pandas as pd

# CSV 파일에서 데이터 읽기
df = pd.read_csv('data_res/employee_data.csv')
print(df.head())  # 데이터의 처음 5행을 출력

# 결측치 확인
print(df.isnull().sum())

# 결측치 채우기 fillna 함수 이용(예: 나이의 결측치를 평균 나이로 채우기)
df['나이'].fillna(df['나이'].mean(), inplace=True)

# 결측치가 있는 행 제거
df_dropped = df.dropna()

# csv 저장
df.to_csv('data_res/modified_employee_data.csv', index=False)
