from flask import Blueprint, jsonify, request

bp = Blueprint('user', __name__, url_prefix='/user')

@bp.route('/list', methods=['GET'])
def user_list():
    return jsonify({'users': ['user1', 'user2', 'user3']})
