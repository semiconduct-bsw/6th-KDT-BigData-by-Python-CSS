@app.route("/predict-iris", methods=['POST'])
def predict_iris():
    # 모델 로드
    model = joblib.load('model/iris_xgb_model.pkl')
    # 임의의 값 예측
    sepal_length = request.json.get("sepal_length")
    sepal_width = request.json.get("sepal_width")
    petal_length = request.json.get("petal_length")
    petal_width = request.json.get("petal_width")
    test_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    predict_class = model.predict(test_data)

    label_map = {
        0: 'setosa',
        1: 'versicolor',
        2: 'virginica'
    }
    result_name = label_map.get(predict_class[0], "Unknown")

    print(predict_class)
    print(f"Predicted class: {predict_class[0]}")
    return result_name
