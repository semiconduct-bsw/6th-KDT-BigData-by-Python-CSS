from flask import Blueprint, jsonify, request
from db import get_conn

bp = Blueprint('board', __name__, url_prefix='/board')

#게시글 전체 조회
@bp.route("/all",methods=['GET'])
def all_boards():
   try:
      conn = get_conn()
      cursor = conn.cursor()
      cursor.execute("SELECT * FROM `board` inner join `user` on `board`.user_idx = `user`.idx")
      boards = cursor.fetchall()
      conn.close()
      return jsonify({
         "success": True,
         "message": "게시글 전체 조회 완료",
         "boards": boards
      })
   except Exception as e:
      return jsonify({
         "success": False,
         "message": "게시글 전체 조회 실패",
         "error": str(e)
      }), 500

#게시글 추가
@bp.route("/save",methods=['POST'])
def save_board():
   try:
      user_idx = request.json.get('user_idx')
      title = request.json.get('title')
      content = request.json.get('content')
      conn = get_conn()
      cursor = conn.cursor()
      cursor.execute("INSERT INTO `board` (user_idx, title, content, created_at) VALUES (%s, %s, %s, now())", (user_idx, title, content))
      conn.commit()
      conn.close()
      return jsonify({
         "success": True,
         "message": "게시글 추가 완료"
      })
   except Exception as e:
      return jsonify({
         "success": False,
         "message": "게시글 추가 실패",
         "error": str(e)
      }), 500
    