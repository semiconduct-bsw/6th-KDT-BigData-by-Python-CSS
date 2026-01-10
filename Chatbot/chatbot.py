import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os

# Gemini API 설정
# https://aistudio.google.com/ 에서 키를 발급받아 아래에 입력하세요.
GEMINI_API_KEY = "."
genai.configure(api_key=GEMINI_API_KEY)

# 모델과 인덱스 로드
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index("faiss_index.bin")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    print("💡 먼저 'python embbeding.py'를 실행하세요.")
    exit(1)

def search_context(question, top_k=3):
    """질문과 관련된 문서 컨텍스트를 검색합니다."""
    q_embedding = model.encode([question])
    distances, indices = index.search(np.array(q_embedding), top_k)
    return [chunks[i] for i in indices[0]]

def ask_gemini(context, question):
    """Gemini를 사용하여 질문에 답변합니다."""
    prompt = f"""
    다음 문서를 참고하여 질문에 답해주세요:
    
    문서 내용:
    {context}
    
    질문:
    {question}
    
    답변은 한국어로 작성해주세요.
    """
    
    # Gemini 모델 선택
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(prompt)
    
    return response.text.strip()

# 메인 루프
if __name__ == "__main__":
    print("📄 사내 문서 기반 Gemini 챗봇")
    print("💡 종료하려면 'exit'를 입력하세요.\n")
    
    while True:
        try:
            question = input("질문 > ").strip()
            if question.lower() in ["exit", "quit"]:
                print("👋 안녕히 가세요!")
                break
            if not question:
                continue
                
            context = "\n".join(search_context(question))
            answer = ask_gemini(context, question)
            print(f"\n💬 답변: {answer}\n")
            
        except KeyboardInterrupt:
            print("\n👋 안녕히 가세요!")
            break
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
