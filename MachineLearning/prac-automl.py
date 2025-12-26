import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

# 1. 데이터 로드
iris = load_iris()
X = iris.data
y = iris.target

# 2. DataFrame으로 변환 (X와 y를 합치기)
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y

# 3. train_test_split (DataFrame으로)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])

# 4. automl 모델로 학습 (label은 'target' 컬럼 이름)
predictor = TabularPredictor(label='target', eval_metric='accuracy').fit(train_data=train_df, time_limit=60)

# 학습된 모든 모델 이름 확인
print("학습된 모델들:", predictor.model_names)

# 리더보드 확인 (성능 순위)
leaderboard = predictor.leaderboard(train_df, silent=True)
print(leaderboard)

# 최고 성능 단일 모델 확인
best_model = predictor.model_best
print(f"최고 성능 모델: {best_model}")


# 5. 예측 (test_df에서 target 제외하고 예측)
y_pred = predictor.predict(test_df.drop(columns=['target']))

# 6. 평가, 정확도
y_test = test_df['target']
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
