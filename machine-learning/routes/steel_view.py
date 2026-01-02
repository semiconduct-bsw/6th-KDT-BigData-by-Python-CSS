from flask import Blueprint, request, jsonify
import joblib
import numpy as np
import os
import pandas as pd

bp = Blueprint('steel', __name__, url_prefix='/steel')

# 모델 로드
steel_model = joblib.load('model/steel_industry_model.pkl')

# Feature Importance
@bp.route("/importance", methods=["GET"])
def importance_steel():
    if steel_model is None:
        return jsonify({'error': '모델이 로드되지 않았습니다.'}), 500
    try:
        preprocessor = steel_model.named_steps["preprocess"]
        rf_model = steel_model.named_steps["model"]
        categorical_cols = ["Day_of_week", "WeekStatus", "Load_Type"]
        numerical_cols = [
            'Lagging_Current_Reactive.Power_kVarh', 
            'Leading_Current_Reactive_Power_kVarh', 
            'CO2(tCO2)', 
            'Lagging_Current_Power_Factor', 
            'Leading_Current_Power_Factor', 
            'NSM'
        ]

        ohe = preprocessor.named_transformers_["cat"]
        ohe_feature_names = ohe.get_feature_names_out(categorical_cols)
        all_feature_names = np.concatenate([ohe_feature_names, np.array(numerical_cols)])

        importances = rf_model.feature_importances_

        fi_df = pd.DataFrame({
            "feature": all_feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)

        data = fi_df.to_dict(orient='records')
        return jsonify({
            'success': True,
            'source': 'Computed dynamically from model',
            'data': data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route("/predict-steel-usage", methods=["POST"])
def predict_steel_usage():
    if steel_model is None:
        return jsonify({'error': '모델이 로드되지 않았습니다.'}), 500
    try:
        data = request.json
        if not data:
            return jsonify({'error': '데이터가 없습니다.'}), 400
        
        # 데이터프레임으로 변환
        df = pd.DataFrame([data])
        
        # 전처리 및 예측
        pred = steel_model.predict(df)
        
        return jsonify({
            'success': True,
            'predicted_usage': pred[0]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500