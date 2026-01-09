from flask import Blueprint, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
import tensorflow as tf

# Mecab/Okt import try-except block
try:
    from konlpy.tag import Mecab
except ImportError:
    Mecab = None
from konlpy.tag import Okt

bp = Blueprint('review_dl', __name__, url_prefix='/review')

# 전역 변수
model = None
tokenizer = None
MAX_LEN = 100

def get_tagger():
    """Mecab이 있으면 사용하고, 없으면 Okt를 반환"""
    if Mecab:
        try:
            return Mecab()
        except Exception:
            pass
    return Okt()

def init_resources():
    """모델과 토크나이저를 로드"""
    global model, tokenizer
    
    # 이미 로드되었으면 패스
    if model is not None and tokenizer is not None:
        return

    print("리소스 로딩 시작...")
    try:
        # 경로 설정 (절대 경로가 안전할 수 있음 -> 상대 경로로 시도)
        # 현재 실행 위치(app.py 있는 곳) 기준
        model_path = os.path.join('models', 'naver-review-model.keras')
        token_path = os.path.join('token', 'tokenizer.pkl')

        if os.path.exists(model_path):
            model = load_model(model_path)
            print(f"모델 로드 완료: {model_path}")
        else:
            print(f"모델 파일 없음: {model_path}")

        if os.path.exists(token_path):
            with open(token_path, 'rb') as f:
                tokenizer = pickle.load(f)
            print(f"토크나이저 로드 완료: {token_path}")
        else:
            print(f"토크나이저 파일 없음: {token_path}")
            
    except Exception as e:
        print(f"리소스 로딩 중 에러 발생: {e}")

# 블루프린트 임포트 시점에 로딩 시도 (서버 시작 시)
init_resources()

@bp.route('/naver', methods=['GET', 'POST'])
def naver_review_page():
    # POST 요청: 리뷰 분석 (AJAX)
    if request.method == 'POST':
        # 리소스 로드 확인
        if model is None or tokenizer is None:
            init_resources()
            if model is None or tokenizer is None:
                return jsonify({'error': 'Server Error: Model not loaded'}), 500

        data = request.get_json()
        content = data.get('content', '')

        if not content.strip():
            return jsonify({'error': '내용을 입력해주세요.'}), 400

        try:
            # 1. 형태소 분석
            tagger = get_tagger()
            morphs = tagger.morphs(content)
            processed_text = ' '.join(morphs)

            # 2. 시퀀스 변환
            sequence = tokenizer.texts_to_sequences([processed_text])

            # 3. 패딩
            padded = pad_sequences(sequence, maxlen=MAX_LEN)

            # 4. 예측
            pred = model.predict(padded)
            score = float(pred[0][0])
            
            # 결과 판정
            result = "긍정" if score > 0.5 else "부정"
            probability = score * 100 if score > 0.5 else (1 - score) * 100

            return jsonify({
                'result': result,
                'probability': round(probability, 2),
                'raw_score': score
            })

        except Exception as e:
            print(f"예측 중 에러: {e}")
            return jsonify({'error': str(e)}), 500

    # GET 요청: 페이지 렌더링
    return render_template('naver-review.html')