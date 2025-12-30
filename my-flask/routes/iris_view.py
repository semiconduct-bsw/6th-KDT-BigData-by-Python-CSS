from flask import Blueprint, request, jsonify
import joblib
import numpy as np
from db import get_conn

bp = Blueprint('ai', __name__, url_prefix='/ai')

# 모델 로드
iris_model = joblib.load('models/iris_xgb_model.pkl')

@bp.route("/predict", methods=['GET', 'POST'])
def predict_iris():
    sepal_length = request.json.get("sepal_length")
    sepal_width = request.json.get("sepal_width")
    petal_length = request.json.get("petal_length")
    petal_width = request.json.get("petal_width")

    # 유효성 체크
    if sepal_length is None or sepal_width is None or petal_length is None or petal_width is None:
        return jsonify({
            'success': False,
            'message': '모든 값이 입력되어야 합니다.'
        }), 400

    # 임의의 값 예측
    test_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    predict_class = iris_model.predict(test_data)

    class_names = ['setosa', 'versicolor', 'virginica']
    predicted_class_name = class_names[predict_class[0]]

    # 2. DB에 데이터 결과 및 저장
    conn = None
    try:
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute("insert into `iris` (sepal_length, sepal_width, petal_length, petal_width, class_name, created_at) values (%s, %s, %s, %s, %s, now())", 
        (sepal_length, sepal_width, petal_length, petal_width, predicted_class_name))

        conn.commit()
        conn.close()
        return jsonify({
            "success": True,
            "message": "예측 완료, 클래스 종류는 setosa, versicolor, virginica 중 하나입니다.",
            "예측된 클래스 종류": predicted_class_name
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": "예측 실패",
            "error": str(e)
        }), 500