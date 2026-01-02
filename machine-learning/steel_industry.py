import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1) 데이터 로딩
df = pd.read_csv("data_res/Steel_industry_data.csv")
print("=== columns ===")
print(df.columns)

# 2) 타겟, 입력 분리
TARGET_COL = "Usage_kWh" # 예측할 값
# 컬럼명이 다를 시 여기서 TARGET_COL 맞춰주기
# date 컬럼은 학습에 불필요하거나 문자열이므로 제외
X = df.drop(columns=[TARGET_COL, "date"], errors="ignore")
y = df[TARGET_COL]

# 3) 범주형/수치형 컬럼 정의
categorical_cols = ["Day_of_week", "WeekStatus", "Load_Type"]
missing = [c for c in categorical_cols if c not in X.columns]
if missing:
    print(f"[경고] 아래 범주형 컬럼이 데이터에 없습니다: {missing}")
    # 필요하면 여기서 categorical_cols를 실제 컬럼명으로 수정하세요.
numerical_cols = [c for c in X.columns if c not in categorical_cols]

print("\n=== categorical_cols ===")
print(categorical_cols)
print("\n=== numerical_cols ===")
print(numerical_cols)

# 4) 데이터 전처리
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numerical_cols),
    ],
    remainder="drop"
)

# 5) 모델 : RandomForestRegressor
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

# 6) 파이프라인 : 전처리 + 모델
pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ]
)

# 7) Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 8) 학습, 예측 및 평가
pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)

# RMSE 계산 방법 (sklearn 버전 호환)
try:
    rmse = root_mean_squared_error(y_test, pred)
except:
    # root_mean_squared_error가 없으면 직접 계산
    rmse = np.sqrt(mean_squared_error(y_test, pred))

r2 = r2_score(y_test, pred)

print("\n=== Evaluation ===")
print(f"RMSE: {rmse:.4f}")
print(f"R2  : {r2:.4f}")

# 9) Feature Importance (원-핫 포함 전체 중요도)
# OneHotEncoder로 생성된 컬럼명 가져오기
ohe = pipeline.named_steps["preprocess"].named_transformers_["cat"]
ohe_feature_names = ohe.get_feature_names_out(categorical_cols)

all_feature_names = np.concatenate([ohe_feature_names, np.array(numerical_cols)])
importances = pipeline.named_steps["model"].feature_importances_

fi = pd.DataFrame({
    "feature": all_feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

print("\n=== Top 20 Feature Importances ===")
print(fi.head(20))

# 필요하면 csv로 저장
fi.to_csv("data_res/feature_importance_steel.csv", index=False)
print("\n[저장] feature_importance_steel.csv 생성 완료")