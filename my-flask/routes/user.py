from flask import Blueprint, jsonify, request
from db import get_conn

bp = Blueprint('user', __name__, url_prefix='/user')

#회원 가입
@bp.route("/save",methods=['POST'])
def save_user():
   try:
      id = request.json['id']
      pw = request.json['pw']
      nick = request.json['nick']
      conn = get_conn()
      cursor = conn.cursor()
      cursor.execute("INSERT INTO `user` (id, pw, nick, created_at) VALUES (%s, %s, %s, now())", (id, pw, nick))
      conn.commit()
      conn.close()
      return jsonify({
         "success": True,
         "message": "회원 가입 완료",
         "user": {
            "id": id,
            "pw": pw,
            "nick": nick,
            "created_at": created_at
         }
      })
   # 유효성 체크
   except Exception as e:
      return jsonify({
         "success": False,
         "message": "회원 가입 실패",
         "error": str(e)
      }), 500

# 회원 전체 조회
@bp.route('/all', methods=['GET'])
def all_users():
    conn = None
    try:
        # 1. 함수를 호출해서 연결 객체를 받습니다.
        conn = get_conn()
        
        # 2. 커서를 열고 SQL을 실행합니다.
        with conn.cursor() as cur:
            sql = "SELECT * FROM `user`"
            cur.execute(sql)
            result = cur.fetchall() # DictCursor 덕분에 딕셔너리 리스트가 됩니다.

        # 3. JSON으로 응답합니다.
        return jsonify({
            "success": True,
            "message": "모든 사용자 조회 완료",
            "users": result
        })

    except Exception as e:
        print(e)
        return jsonify({"error": "DB 연결 실패 또는 쿼리 오류"}), 500
        
    finally:
        # 4. 사용이 끝난 연결은 반드시 닫아줍니다. (중요)
        if conn:
            conn.close()