from flask import Blueprint, jsonify, request
from db import get_conn

bp = Blueprint('user', __name__, url_prefix='/user')

@bp.route('/users', methods=['GET'])
def get_users():
    conn = None
    try:
        # 1. 함수를 호출해서 연결 객체를 받습니다.
        conn = get_conn()
        
        # 2. 커서를 열고 SQL을 실행합니다.
        with conn.cursor() as cur:
            sql = "SELECT * FROM users"
            cur.execute(sql)
            result = cur.fetchall() # DictCursor 덕분에 딕셔너리 리스트가 됩니다.

        # 3. JSON으로 응답합니다.
        return jsonify(result)

    except Exception as e:
        print(e)
        return jsonify({"error": "DB 연결 실패 또는 쿼리 오류"}), 500
        
    finally:
        # 4. 사용이 끝난 연결은 반드시 닫아줍니다. (중요)
        if conn:
            conn.close()