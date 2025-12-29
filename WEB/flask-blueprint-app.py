from flask import Flask
# 1. 분리한 뷰 파일들 임포트
from views import iris_view, main_view

app = Flask(__name__)

# 2. 블루프린트 등록
# app.register_blueprint(블루프린트_객체)
app.register_blueprint(iris_view.bp)
app.register_blueprint(main_view.bp)

# 3. 서버 실행
if __name__ == '__main__':
    app.run(debug=True)
