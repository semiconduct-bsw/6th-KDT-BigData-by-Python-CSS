from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 데이터 준비
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# KNN 분류기 생성
knn = KNeighborsClassifier(n_neighbors=3)  # K=3
knn.fit(X_train, y_train)

# 예측
y_pred = knn.predict(X_test)

# 정확도 평가
print("Accuracy:", accuracy_score(y_test, y_pred))
