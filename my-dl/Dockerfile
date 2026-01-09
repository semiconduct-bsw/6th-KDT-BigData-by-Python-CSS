# Python 3.11 베이스 이미지 사용
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필요한 패키지 설치
RUN apt-get update && apt-get install -y \
    gcc \
    default-libmysqlclient-dev \
    pkg-config \
    default-jdk \
    mecab \
    mecab-ipadic-utf8 \
    libmecab-dev \
    && rm -rf /var/lib/apt/lists/*

# JAVA_HOME 설정 (Konlpy 실행을 위해 필요)
ENV JAVA_HOME /usr/lib/jvm/default-java

# requirements.txt 복사 및 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 파일 복사
COPY app.py .
COPY db.py .
COPY routes/ ./routes/
COPY templates/ ./templates/
COPY models/ ./models/
COPY token/ ./token/
COPY static/ ./static/

# 포트 노출 (Flask 기본 포트)
EXPOSE 5000

# 환경 변수 설정 (프로덕션 환경)
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Flask 애플리케이션 실행
CMD ["python", "app.py"]