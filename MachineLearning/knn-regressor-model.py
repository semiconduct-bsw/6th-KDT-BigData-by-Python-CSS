from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 데이터 준비
X, y = make_regression(n_samples=100, n_features=1, noise=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# KNN 회귀기 생성
knn_reg = KNeighborsRegressor(n_neighbors=5)  # K=5
knn_reg.fit(X_train, y_train)

# 예측
y_pred = knn_reg.predict(X_test)

# 성능 평가
print("MSE:", mean_squared_error(y_test, y_pred))

# r2 점수
print("R2 Score:", r2_score(y_test, y_pred))
