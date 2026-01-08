# 필수 라이브러리 가져오기
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# 데이터 불러오기
df = pd.read_csv('https://raw.githubusercontent.com/zzhining/python_ml_dl/main/dataset/concrete.csv')

# 데이터 확인
print(df.shape)
print(df.head())

# 특성과 레이블 분리
X = df.drop(columns=['CompressiveStrength'])  # 입력 데이터 (특성)
y = df['CompressiveStrength']  # 출력 데이터 (레이블)

# 훈련과 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 정규화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 인공신경 모델망 만들기
model = tf.keras.Sequential([
    # 첫 번째 은닉층, 두 번째 은닉층
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),

    # 출력층 (회귀 문제 = 출력 뉴런은 1개, 분류 문제라면 타겟 개수만큼 설정)
    tf.keras.layers.Dense(1)
])

# 모델 컴파일
model.compile(optimizer='adam', loss='mse')

# 모델 학습
history = model.fit(X_train_scaled, y_train, epochs=100, validation_split=0.2)

# 학습 결과 시각화
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss (MSE)')
plt.xlabel('Epochs')
plt.legend()
plt.show()

# 모델 평가
y_pred = model.predict(X_test_scaled)

# MSE(평균 제곱 오차) 계산
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# R2(결정계수) 계산
r2 = r2_score(y_test, y_pred)
print(f'R2 Score: {r2}')

# 실제값과 예측값 비교
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()})
print(comparison_df.head())

# 모델 저장
# model.save('models/concrete_model.h5')

# 모델 로드
# model = tf.keras.models.load_model('models/concrete_model.h5')
