import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# 1. IMDB 데이터셋 로드
# num_words=10000 : 가장 빈도 높은 10,000개의 단어만 사용
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)

# 2. 데이터 전처리 (패딩)
max_sequence_length = 300  # 리뷰의 최대 길이 설정
X_train_pad = pad_sequences(X_train, maxlen=max_sequence_length)
X_test_pad = pad_sequences(X_test, maxlen=max_sequence_length)

# 3. RNN 모델 구성
model = Sequential()

# 임베딩 레이어: 단어 인덱스를 고정된 크기의 밀집 벡터로 변환
model.add(Embedding(input_dim=10000, output_dim=128, input_length=max_sequence_length))

# LSTM 레이어 추가
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))

# 출력층 (이진 분류를 위한 sigmoid 활성화 함수)
model.add(Dense(1, activation='sigmoid'))

# 4. 모델 컴파일
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 5. 모델 학습
model.fit(X_train_pad, y_train, epochs=5, batch_size=64, validation_data=(X_test_pad, y_test))

# 6. 모델 평가
loss, accuracy = model.evaluate(X_test_pad, y_test)
print(f'Test Accuracy: {accuracy:.4f}')