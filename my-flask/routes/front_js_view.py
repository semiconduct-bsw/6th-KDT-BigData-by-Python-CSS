from flask import Blueprint, render_template

# Blueprint 생성
bp = Blueprint('front_js_view', __name__)

@bp.route('/front1')
def front1():
    return render_template('front1.html')
