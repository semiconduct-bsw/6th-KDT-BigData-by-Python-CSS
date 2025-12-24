import numpy as np
import joblib

# 모델 불러오기
house_model = joblib.load('model_pkl/house_price_model.pkl')

# 임의의 값으로 예측
sample_data = np.array([[8.3252, 41.0, 6.984127, 1.023810, 322.0, 2.555556, 37.88, -122.23]])
predicted_price = house_model.predict(sample_data)
print(f"Predicted House Price for sample data: {predicted_price[0]}")
