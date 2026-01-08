from tensorflow.keras.datasets import mnist
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2

# 데이터 불러오기
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# 데이터 구조 확인
print("X_train:{}, Y_train:{}\nX_test:{}, Y_test:{}"\
      .format(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape))

# 스케일링
X_train = X_train / 255
X_test = X_test / 255

# 차원 변경
X_train = X_train.reshape(X_train.shape[0],
                          X_train.shape[1],
                          X_train.shape[2],
                          1)
X_test = X_test.reshape(X_test.shape[0],
                        X_test.shape[1],
                        X_test.shape[2], 
                        1)
print(X_train.shape)
print(X_test.shape)

# 모델 구성
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # 2D 이미지 데이터를 1D 벡터로 변환
    layers.Dense(128, activation='relu'),  # 은닉층
    layers.Dropout(0.2),                   # 과적합 방지를 위한 Dropout 20%뉴런 Dropout
    layers.Dense(10, activation='softmax') # 출력층: 10개의 클래스 - 숫자가 10개 0,1..9까지
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 학습
model.fit(X_train, Y_train, epochs=5)
print('===모델 평가 결과==')

# 모델 평가
model.evaluate(X_train, Y_train, verbose=2)
model.summary() #모델 설명 요약본

# # 첫 번째 X_train 데이터로 예측
# first_prediction = model.predict(X_train[0:1])  # 첫 번째 데이터만 사용
# print("첫 번째 데이터에 대한 예측 결과 (확률 분포):", first_prediction)

# # 가장 높은 확률을 가진 클래스 확인
# predicted_class = np.argmax(first_prediction)  # 확률이 가장 높은 클래스 인덱스
# print("첫 번째 데이터에 대한 예측 클래스:", predicted_class)

# 모델 저장
model.save('models/mnist_model.keras')

# 모델 로드
loaded_model = tf.keras.models.load_model('models/mnist_model.keras')

# 임의의 이미지 예측 - png 파일을 이미지로 불러와 예측
image = cv2.imread('test1.png', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (28, 28))
image = image.reshape(1, 28, 28, 1)
prediction = loaded_model.predict(image)
predicted_class = np.argmax(prediction)
print("예측된 클래스:", predicted_class)    