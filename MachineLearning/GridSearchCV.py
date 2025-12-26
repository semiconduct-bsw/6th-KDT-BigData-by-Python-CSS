from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# 데이터 준비
X, y = make_regression(n_samples=100, n_features=1, noise=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# GridSearchCV 설정
k_values = [1, 3, 5, 7, 9, 11, 15, 20]
param_grid = {'n_neighbors': k_values}

# 모델 준비
knn = KNeighborsRegressor()

# GridSearch 생성 (cv=5: 5번 교차 검증)
# scoring='neg_mean_squared_error': 사이킷런은 점수가 높을수록 좋다고 판단하므로 
# MSE에 음수를 붙여 사용
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_squared_error')

print("=== GridSearchCV 실행 (교차 검증 수행 중...) ===")
grid_search.fit(X_train, y_train)

# 결과 확인
# 음수로 계산된 점수를 다시 양수로 변환하여 저장
cv_mse_scores = -grid_search.cv_results_['mean_test_score']
best_k = grid_search.best_params_['n_neighbors']
best_mse = -grid_search.best_score_  # 교차 검증에서의 평균 MSE

print(f"\n최적 K값: {best_k}")
print(f"교차 검증 평균 MSE: {best_mse:.2f}")

# 최적 모델로 테스트 데이터 예측
# GridSearchCV는 학습이 끝나면 자동으로 최적의 모델(best_estimator_)로 재학습
y_pred_best = grid_search.predict(X_test)
final_mse = mean_squared_error(y_test, y_pred_best)

print(f"\n=== 최종 테스트 세트 성능 (Test Set) ===")
print(f"최종 MSE: {final_mse:.2f}")

# 시각화
plt.figure(figsize=(12, 4))

# 1. K값에 따른 교차 검증 MSE 변화
plt.subplot(1, 2, 1)
plt.plot(k_values, cv_mse_scores, 'bo-')
plt.xlabel('K Value')
plt.ylabel('Cross-Validated MSE')
plt.title('MSE vs K Value (5-Fold CV)')
plt.grid(True)

# 2. 최적 모델 예측 결과 시각화
plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, color='blue', alpha=0.6, label='Actual')
plt.scatter(X_test, y_pred_best, color='red', alpha=0.6, label='Prediction')
plt.xlabel('X')
plt.ylabel('y')
plt.title(f'Prediction Result (Best K={best_k})')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
