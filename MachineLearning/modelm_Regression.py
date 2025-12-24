import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 데이터셋 로드
data = fetch_california_housing(as_frame=True)
df = data.frame

# 특성과 타겟 분리
X = df.drop(columns=["MedHouseVal"])  # 특징 (특성)
y = df["MedHouseVal"]  # 목표 변수 (집값)

# 학습 및 테스트 데이터 분리 train:공부, test:시험
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 모델 예측
y_pred = model.predict(X_test)

# 모델 평가
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)

# 회귀 계수 및 절편
print("회귀 계수 (Coefficients):", model.coef_)
print("절편 (Intercept):", model.intercept_)

# 모델 저장(중요)
import joblib
joblib.dump(model, 'house_price_model.pkl')

# 모델 불러오기
house_model = joblib.load('house_price_model.pkl')

# 임의의 값으로 예측
sample_data = np.array([[8.3252, 41.0, 6.984127, 1.023810, 322.0, 2.555556, 37.88, -122.23]])
predicted_price = house_model.predict(sample_data)
print(f"Predicted House Price for sample data: {predicted_price[0]}")
