from flask import Flask
from . import review_dl

def register_routes(app: Flask):
    app.register_blueprint(review_dl.bp)