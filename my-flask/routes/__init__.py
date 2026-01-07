from flask import Flask
from . import iris_view, main_view, test, user, board, front_js_view

def register_routes(app: Flask):
    app.register_blueprint(iris_view.bp)
    app.register_blueprint(main_view.bp)
    app.register_blueprint(test.bp)
    app.register_blueprint(user.bp)
    app.register_blueprint(board.bp)
    app.register_blueprint(front_js_view.bp)