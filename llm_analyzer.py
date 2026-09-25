"""
llm_analyzer.py
전사된 자막(json/텍스트)을 Gemini로 분석해 하이라이트를 추출하는 모듈.
"""

import os
from pathlib import Path
from typing import Optional

from .llm_client import get_llm_client


# --------------------------------------------------
# 설정값 (거의 안 바뀌는 값들 - 필요하면 여기서 직접 수정)
# --------------------------------------------------

MODEL_NAME = "gemini-3.5-flash"

PROMPT_TEMPLATE = """
너는 생방송 전사 스크립트를 분석해서 타임라인 하이라이트를 뽑는 어시스턴트다.
주어진 발화 구간(시간, 텍스트)을 순서대로 읽으면서, 화제나 상황이 바뀌는 지점마다 포인트를 하나씩 찍어라.
같은 화제가 계속 이어지는 동안에는 억지로 찍지 않아도 된다.

각 포인트마다 아래 항목을 채워서 JSON 배열로만 응답하라. 다른 설명은 절대 추가하지 마라.

- start: 그 순간이 시작된 시각(초, 숫자)
- fun_score: 1~5 사이의 재미/임팩트 점수
- comment: 짧은 한줄 코멘트 (구어체)

주의사항:
- 화제 전환 기준으로만 포인트 생성
- 중복 구간은 다시 생성하지 말 것

[자막 데이터]
{subtitle_text}
"""


# --------------------------------------------------
# 1. 자막 로드
# --------------------------------------------------

def load_subtitle(subtitle_path: Path) -> str:
    """자막 파일(json/txt 등)을 텍스트로 읽어 반환한다."""
    if not os.path.exists(subtitle_path):
        raise FileNotFoundError(f"자막 파일을 찾을 수 없습니다: {subtitle_path}")

    try:
        with open(subtitle_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(subtitle_path, "r", encoding="cp949") as f:
            return f.read()


# --------------------------------------------------
# 2. LLM 분석 실행
# --------------------------------------------------

def run_llm_analysis(subtitle_text: str) -> str:
    """자막 텍스트를 프롬프트에 담아 LLM에 분석을 요청하고 결과 텍스트를 반환한다."""
    client = get_llm_client()
    prompt = PROMPT_TEMPLATE.format(subtitle_text=subtitle_text)

    print(f"\n{MODEL_NAME} 모델이 자막을 분석 중입니다...")
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )
    return response.text


# --------------------------------------------------
# 3. 조립 함수 (수동/자동 겸용)
# --------------------------------------------------

def analyze_subtitle(subtitle_path: Optional[str] = None) -> str:
    """
    자막 파일을 읽어 LLM 분석 결과를 반환한다.
    - subtitle_path가 없으면 input()으로 받는다.
    - 모델명/프롬프트는 모듈 상단 상수를 직접 참조한다.
    """
    if subtitle_path is None:
        subtitle_path = input("자막 파일 경로: ").strip()

    subtitle_text = load_subtitle(subtitle_path)
    return run_llm_analysis(subtitle_text)


if __name__ == "__main__":
    result = analyze_subtitle()
    print("\n" + "=" * 30 + " 분석 결과 " + "=" * 30)
    print(result)
    print("=" * 71)
