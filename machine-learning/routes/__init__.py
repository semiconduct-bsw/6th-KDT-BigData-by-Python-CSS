from flask import Flask
from . import steel_view

def register_routes(app):
    app.register_blueprint(steel_view.bp)