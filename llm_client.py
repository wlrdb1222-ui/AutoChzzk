"""
llm_client.py
Gemini(genai) 클라이언트 공통 생성 모듈.
API 키 조회 로직을 한 곳에 모아서, 여러 모듈에서 중복 생성하지 않도록 한다.
"""

from google import genai
from google.colab import userdata


def get_llm_client() -> genai.Client:
    """환경에 저장된 API 키로 genai 클라이언트를 생성해 반환한다."""
    api_key = userdata.get("LLM_KEY")
    if not api_key:
        raise RuntimeError("LLM_KEY를 찾을 수 없습니다. Colab Secrets에 등록되어 있는지 확인하세요.")
    return genai.Client(api_key=api_key)
