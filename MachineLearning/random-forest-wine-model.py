import joblib
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# 데이터셋 로드
wine = load_wine()
X = wine.data
y = wine.target

# 데이터셋 나누기 (Train/Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 데이터 정규화
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 랜덤 포레스트 모델 학습
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 모델 예측
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cfr = classification_report(y_test, y_pred)

# 평가 결과 출력
print("정확도:", accuracy)
print("\n결과 보고서:\n", cfr)

# 모델 저장
if accuracy > 0.85:
    joblib.dump(model, f'model_pkl/wine_model_accuracy_{accuracy:.2f}.pkl')

# 모델 로드
model = joblib.load(f'model_pkl/wine_model_accuracy_{accuracy:.2f}.pkl')

# 임의의 값으로 예축, 정규화된 값으로 예측
X_new = np.array([[13.2, 2.7, 2.5, 15.6, 98, 3.3, 2.5, 15.6, 98, 3.3, 2.5, 15.6, 98]])
X_new = scaler.transform(X_new)
y_pred = model.predict(X_new)
print("예측 결과:", y_pred)
