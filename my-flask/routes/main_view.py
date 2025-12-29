from flask import Blueprint, render_template

# Blueprint 생성
bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    return render_template('index.html')