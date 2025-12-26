import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. 데이터 로드 및 전처리
iris = load_iris() # 제공되는 함수로 아이리스 데이터셋 로드
X = iris.data
y = iris.target

# 이진 분류 문제를 위해 클래스 0과 1만 선택
binary_mask = y < 2  # 클래스 0과 1만 선택
X_binary = X[binary_mask]
y_binary = y[binary_mask]

# 학습용 데이터와 테스트용 데이터로 분할
X_train, X_test, y_train, y_test = train_test_split(X_binary, y_binary, test_size=0.2, random_state=42)

# 2. 로지스틱 회귀 모델 생성 및 학습
model = LogisticRegression()
model.fit(X_train, y_train)

# 3. 모델 평가
y_pred = model.predict(X_test)

# 정확도 출력
accuracy = accuracy_score(y_test, y_pred)
print(f"정확도: {accuracy:.2f}")

# 모델 저장
joblib.dump(model, 'model_pkl/iris_logi_model.pkl')

# 모델 로드
loaded_model = joblib.load('model_pkl/iris_logi_model.pkl')

# 예측
test_data = np.array([[5.1, 3.5, 1.4, 0.2]])
predicted_class = model.predict(test_data)
print(f"Predicted Class: {predicted_class[0]}")
