from flask import Blueprint, render_template

# Blueprint 생성
bp = Blueprint('main', __name__)

@bp.route('/layout')
def layout():
    return render_template('layout.html')

@bp.route('/id-class')
def id_class():
    return render_template('id-class.html')

@bp.route('/')
def index():
    return render_template('index.html')