import xgboost as xgb
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. 데이터 로드
iris = load_iris()
X = iris.data
y = iris.target

# 2. 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. XGBoost 모델 생성
model = xgb.XGBClassifier(
    objective='multi:softmax',  # 다중 클래스 분류
    num_class=3,                # 클래스 개수
    max_depth=3,
    learning_rate=0.1,
    n_estimators=100,
    use_label_encoder=False,    # 경고 방지용
    eval_metric='mlogloss'      # 다중 클래스 로스
)

# 4. 학습
model.fit(X_train, y_train)

# 5. 예측
y_pred = model.predict(X_test)

# 6. 평가
print("정확도:", accuracy_score(y_test, y_pred))
print("\n분류 리포트:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# 7. 모델 저장
joblib.dump(model, 'model/iris_xgb_model.pkl')

# 8. 모델 로드
model = joblib.load('model/iris_xgb_model.pkl')

# 9. 임의의 값으로 예측
X_new = np.array([[5.1, 3.5, 1.4, 0.2]])
y_pred = model.predict(X_new)
print("예측 결과:", y_pred)
