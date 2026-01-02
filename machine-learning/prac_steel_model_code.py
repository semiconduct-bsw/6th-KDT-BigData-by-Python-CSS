# 모델 저장
joblib.dump(pipeline, "model/steel_industry_model.pkl")

# 모델 로드
steel_model = joblib.load("model/steel_industry_model.pkl")

# 임의의 값 예측
# DataFrame 형태로 제공해야 함 (X와 동일한 컬럼 순서)
test_data = pd.DataFrame({
    'Lagging_Current_Reactive.Power_kVarh': [2.95],
    'Leading_Current_Reactive_Power_kVarh': [0],
    'CO2(tCO2)': [0],
    'Lagging_Current_Power_Factor': [73.21],
    'Leading_Current_Power_Factor': [100],
    'NSM': [900],
    'WeekStatus': ['Weekday'],  # 범주형: 실제 문자열 값
    'Day_of_week': ['Monday'],  # 범주형: 실제 문자열 값
    'Load_Type': ['Light_Load']  # 범주형: 실제 문자열 값
})

# 파이프라인이 자동으로 전처리(one-hot encoding 포함) 수행
predicted_usage = steel_model.predict(test_data)
print(f"Predicted Usage: {predicted_usage[0]}")