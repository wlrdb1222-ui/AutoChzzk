"""
llm_analyzer.py
전사된 자막(및 선택적으로 채팅 통계)을 Gemini로 분석해
키리누키용 세밀한 하이라이트 이벤트를 추출하는 모듈.
"""

import os
import csv
import json
import re
from pathlib import Path
from typing import Optional

from .llm_client import get_llm_client


# --------------------------------------------------
# 설정값
# --------------------------------------------------

MODEL_NAME = "gemini-3.8-flash"
OUTPUT_DIR = "/content/analysis"

# 평균 채팅량 대비 몇 배부터 채팅 급증으로 판단할지
CHAT_SPIKE_THRESHOLD = 1.5

# 하나의 이벤트를 지나치게 길게 묶지 않기 위한 참고 기준
MAX_EVENT_INTERVAL_SEC = 300


# --------------------------------------------------
# 프롬프트
# --------------------------------------------------
PROMPT_PATH = Path(__file__).parent / "prompts" / "prompt.txt"

def load_prompt():
    return PROMPT_PATH.read_text(encoding="utf-8")
# --------------------------------------------------
# 이벤트 밀도 지침
# --------------------------------------------------

def build_density_instruction(duration_sec: Optional[float]) -> str:
    """
    전체 방송 길이를 참고하여 이벤트가 지나치게 크게 묶이지 않도록
    세밀한 이벤트 분리 지침을 만든다.

    주의:
    특정 시간마다 이벤트를 강제로 생성하지 않는다.
    """

    if not duration_sec or duration_sec <= 0:
        return ""

    duration_min = round(duration_sec / 60, 1)

    return f"""
전체 방송 길이는 약 {duration_min}분이다.

방송 전체를 몇 개의 큰 주제로 요약하지 말고,
개별 사건과 장면을 가능한 한 세밀하게 분리하라.

하나의 이벤트가 특별한 이유 없이
{MAX_EVENT_INTERVAL_SEC}초를 크게 넘지 않도록 하라.

단, {MAX_EVENT_INTERVAL_SEC}초 이내라도
새로운 사건, 강한 리액션, 재미있는 발언, 채팅과의 상호작용 등이 발생하면
별도의 이벤트로 분리하라.

반대로 하나의 사건이 계속 진행되는 경우에는
시간 기준을 지키기 위해 억지로 분리하지 마라.

{MAX_EVENT_INTERVAL_SEC}초는 이벤트를 강제로 생성하는 기준이 아니다.
단지 지나치게 긴 하나의 이벤트로 묶는 것을 방지하기 위한 참고 기준이다.

특별한 사건이 없는 구간을 단순히 방송 전체를 채우기 위해
강제로 이벤트로 만들 필요는 없다.
"""


# --------------------------------------------------
# 채팅 분석 지침
# --------------------------------------------------

CHAT_INSTRUCTION = """
채팅 반응 데이터를 이벤트 탐지의 보조 정보로 사용하라.

채팅 급증 기준:
- 평균 채팅량 대비 CHAT_SPIKE_THRESHOLD배 이상 증가한 구간을
  채팅 급증 구간으로 본다.

채팅 급증은 하이라이트 후보를 찾는 중요한 신호지만,
채팅 급증 = 재미있음으로 판단하지 마라.

다음 세 가지를 구분해서 판단하라.

1. 발화도 재미있고 채팅도 급증
   → fun_score를 높일 수 있다.

2. 발화는 평범하지만 채팅만 급증
   → fun_score는 그대로 두고 value_score를 높이거나
     시청자 반응이 중요한 별도 이벤트로 만들 수 있다.

3. 발화도 평범하고 채팅도 특별한 반응이 없음
   → 일반적인 구간으로 처리한다.

채팅 급증 시점이 기존 이벤트의 중간에 있더라도,
그 시점에서 새로운 사건이나 강한 반응이 발생했다면
기존 이벤트와 분리하여 별도의 이벤트로 만들어라.
"""


# --------------------------------------------------
# 1. 자막 로드
# --------------------------------------------------

def load_subtitle(subtitle_path: Path) -> str:
    """자막 파일(json/txt 등)을 텍스트로 읽어 반환한다."""

    if not os.path.exists(subtitle_path):
        raise FileNotFoundError(
            f"자막 파일을 찾을 수 없습니다: {subtitle_path}"
        )

    try:
        with open(subtitle_path, "r", encoding="utf-8") as f:
            return f.read()

    except UnicodeDecodeError:
        with open(subtitle_path, "r", encoding="cp949") as f:
            return f.read()


def get_subtitle_duration(
    subtitle_text: str
) -> Optional[float]:
    """
    transcriber.py가 만든 자막 json에서
    전체 길이(초)를 추정한다.

    segments 리스트의 마지막 end 값을 사용한다.
    실패하면 None을 반환한다.
    """

    try:
        data = json.loads(subtitle_text)
        segments = data.get("segments")

        if not segments:
            return None

        return max(
            seg["end"]
            for seg in segments
            if "end" in seg
        )

    except (
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError
    ):
        return None


# --------------------------------------------------
# 2. 채팅 통계 로드 + 요약
# --------------------------------------------------

def load_chat_data(chat_data_path: Path) -> list[dict]:
    """CSV로 저장된 채팅 구간 통계를 로드한다."""

    if not os.path.exists(chat_data_path):
        raise FileNotFoundError(
            f"채팅 데이터 파일을 찾을 수 없습니다: {chat_data_path}"
        )

    with open(
        chat_data_path,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        return [
            {
                "start": int(row["start"]),
                "end": int(row["end"]),
                "count": int(row["count"])
            }
            for row in reader
        ]


def build_chat_summary_text(
    buckets: list[dict],
    mode: str = "spike"
) -> str:
    """
    채팅 구간 통계를 프롬프트용 텍스트로 변환한다.

    mode="full":
        전체 구간 수치를 그대로 나열

    mode="spike":
        평균 대비 급증한 구간만 요약해서 전달
    """

    if not buckets:
        return ""

    if mode == "full":

        lines = [
            f"{b['start']}~{b['end']}초: {b['count']}개"
            for b in buckets
        ]

        return "\n".join(lines)

    if mode == "spike":

        avg = sum(
            b["count"]
            for b in buckets
        ) / len(buckets)

        spikes = [
            b
            for b in buckets
            if b["count"] >= avg * CHAT_SPIKE_THRESHOLD
        ]

        if not spikes:
            return (
                "채팅 반응이 평소와 비슷한 수준으로 유지되어 "
                "특별한 급증 구간이 없습니다."
            )

        lines = [
            (
                f"{s['start']}~{s['end']}초: "
                f"평균 대비 {s['count'] / avg:.1f}배 "
                f"({s['count']}개)"
            )
            for s in spikes
        ]

        return (
            "채팅 급증 구간:\n"
            + "\n".join(lines)
        )

    raise ValueError(
        f"알 수 없는 mode: {mode} "
        "(full 또는 spike만 지원)"
    )


# --------------------------------------------------
# 3. LLM 분석 실행
# --------------------------------------------------

def run_llm_analysis(
    subtitle_text: str,
    chat_summary_text: str = "",
    density_instruction: str = "",
) -> str:
    """
    자막 + 채팅 요약 + 밀도 지침을 프롬프트에 담아
    LLM에 분석을 요청하고 원본 응답 텍스트를 반환한다.
    """

    client = get_llm_client()

    chat_section = (
        f"\n[채팅 반응 데이터]\n{chat_summary_text}"
        if chat_summary_text
        else ""
    )

    chat_instruction = (
        CHAT_INSTRUCTION
        if chat_summary_text
        else ""
    )

    prompt_template = load_prompt()

    prompt = prompt_template.format(
        subtitle_text=subtitle_text,
        chat_section=chat_section,
        chat_instruction=chat_instruction,
        density_instruction=density_instruction,
    )

    print(
        f"\n{MODEL_NAME} 모델이 자막을 분석 중입니다..."
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    return response.text


# --------------------------------------------------
# 4. 응답 파싱
# --------------------------------------------------

def parse_analysis_result(
    raw_text: str
) -> list[dict]:
    """
    LLM 원본 응답 텍스트를 JSON으로 파싱한다.
    코드블록/여분 텍스트 방어 포함.
    """

    text = raw_text.strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    start = text.find("[")
    end = text.rfind("]")

    if start == -1 or end == -1:
        raise ValueError(
            "응답에서 JSON 배열을 찾을 수 없습니다:\n"
            f"{raw_text}"
        )

    json_text = text[start:end + 1]

    try:
        return json.loads(json_text)

    except json.JSONDecodeError as e:
        raise ValueError(
            f"JSON 파싱에 실패했습니다: {e}\n"
            f"원본 텍스트:\n{raw_text}"
        )


# --------------------------------------------------
# 5. 결과 저장
# --------------------------------------------------

def save_analysis(
    audio_name: str,
    results: list[dict]
) -> Path:
    """분석 결과를 JSON으로 저장하고 저장 경로를 반환한다."""

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    payload = {
        "audio_source": audio_name,
        "model": MODEL_NAME,
        "num_points": len(results),
        "highlights": results,
    }

    out_path = (
        Path(OUTPUT_DIR)
        / f"{audio_name}_analysis.json"
    )

    with open(
        out_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"[저장 완료] {out_path}"
    )

    return out_path


# --------------------------------------------------
# 6. 조립 함수
# --------------------------------------------------

def analyze_subtitle(
    subtitle_path: Optional[str] = None,
    chat_data_path: Optional[str] = None,
    chat_summary_mode: str = "spike",
) -> Path:
    """
    자막 파일을 읽어 LLM 분석 결과를 JSON 파일로 저장하고
    그 경로를 반환한다.

    - subtitle_path가 없으면 input()으로 받는다.
    - chat_data_path가 있으면 채팅 통계를 함께 고려한다.
    - chat_summary_mode:
        "spike" = 급증 구간만 전달
        "full"  = 전체 채팅 수치 전달
    - 자막 전체 길이를 계산해 이벤트가 지나치게
      큰 구간으로 묶이지 않도록 지침을 추가한다.
    """

    if subtitle_path is None:
        subtitle_path = input(
            "자막 파일 경로: "
        ).strip()

    subtitle_path = Path(
        subtitle_path
    )

    audio_name = subtitle_path.stem

    # 자막 로드
    subtitle_text = load_subtitle(
        subtitle_path
    )

    # 전체 방송 길이 확인
    duration_sec = get_subtitle_duration(
        subtitle_text
    )

    # 이벤트 밀도 지침 생성
    density_instruction = build_density_instruction(
        duration_sec
    )

    # 채팅 데이터 로드
    chat_summary_text = ""

    if chat_data_path is not None:

        buckets = load_chat_data(
            Path(chat_data_path)
        )

        chat_summary_text = build_chat_summary_text(
            buckets,
            mode=chat_summary_mode
        )

    # LLM 분석
    raw_result = run_llm_analysis(
        subtitle_text,
        chat_summary_text,
        density_instruction
    )

    # JSON 파싱
    parsed_result = parse_analysis_result(
        raw_result
    )

    # 결과 저장
    return save_analysis(
        audio_name,
        parsed_result
    )


# --------------------------------------------------
# 실행
# --------------------------------------------------

if __name__ == "__main__":

    result_path = analyze_subtitle()

    print(
        f"\n분석 결과 저장 위치: {result_path}"
    )
