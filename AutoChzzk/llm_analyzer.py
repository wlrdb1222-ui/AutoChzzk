"""
llm_analyzer.py
전사된 자막(및 선택적으로 채팅 통계)을 Gemini로 분석해 하이라이트를 추출하는 모듈.
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


CHAT_SPIKE_THRESHOLD = 1.5
MIN_INTERVAL_SEC = 300

PROMPT_TEMPLATE = """
너는 생방송 전사 스크립트를 분석해서 방송 전체의 타임라인 이벤트를 만드는 어시스턴트다.

발화를 시간 순서대로 읽으면서 화제나 상황이 바뀌는 '구간'을 하나의 이벤트로 정의하라.

반드시 JSON 배열만 출력하고 다른 설명은 절대 하지 마라.

각 이벤트는 아래 필드를 가진다.

- start : 이벤트 시작 시각(초)
- end : 이벤트 종료 시각(초)
- fun_score : 1~5 (얼마나 웃기거나 임팩트 있는가)
- value_score : 1~5 (시청자가 알아야 할 중요도)
- title : 15~25자 정도의 짧은 제목
- comment : 한 줄 설명 (구어체)

fun_score 기준
1 = 평범한 잡담
2 = 정보 전달 / 일상 대화
3 = 흥미로운 화제 전환
4 = 웃음, 당황, 큰 리액션
5 = 방송 대표 장면 수준의 클라이맥스

value_score 기준
1 = 가벼운 대화
2 = 일반적인 잡담
3 = 의미 있는 화제
4 = 방송 흐름에 중요한 사건
5 = 공지, 큰 반응, 대표 이벤트

주의사항
- 방송 전체를 빠짐없이 고르게 커버한다.
- 이벤트는 반드시 연속적인 구간(start~end)으로 작성한다.
- end는 다음 이벤트 시작 직전 시각으로 설정한다.
- 중복되는 이벤트는 만들지 않는다.
- 재미가 낮은 일상 구간도 포함하여 전체 흐름이 이어지도록 작성한다.

{density_instruction}
{chat_instruction}

[자막 데이터]
{subtitle_text}

{chat_section}
"""

CHAT_INSTRUCTION = """
채팅 반응 데이터는 value_score 판단에 활용한다.

- 채팅이 급증한 구간은 중요한 사건일 가능성이 높다.
- 단, 채팅이 많다고 fun_score를 무조건 올리지 말 것.
- 실제 발화 내용과 채팅 반응을 함께 고려하여 value_score를 결정한다.
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


def get_subtitle_duration(subtitle_text: str) -> Optional[float]:
    """
    transcriber.py가 만든 자막 json에서 전체 길이(초)를 추정한다.
    segments 리스트의 마지막 end 값을 사용한다. 실패하면 None을 반환한다.
    """
    try:
        data = json.loads(subtitle_text)
        segments = data.get("segments")
        if not segments:
            return None
        return max(seg["end"] for seg in segments if "end" in seg)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def build_density_instruction(duration_sec: Optional[float]) -> str:
    """전체 길이를 바탕으로 '최소 몇 개 이상 만들어라'는 지침 문장을 만든다."""
    if not duration_sec or duration_sec <= 0:
        return ""

    min_points = max(1, int(duration_sec // MIN_INTERVAL_SEC))
    duration_min = round(duration_sec / 60, 1)

    return (
        f"- 전체 방송 길이는 약 {duration_min}분이다. "
        f"{MIN_INTERVAL_SEC // 60}분마다 최소 한 개씩, 총 {min_points}개 이상의 포인트를 만들어라. "
        f"이 기준을 지키기 위해 필요하다면 임팩트가 낮은 구간도 fun_score 1~2로 포함시켜라."
    )


# --------------------------------------------------
# 2. 채팅 통계 로드 + 요약
# --------------------------------------------------

def load_chat_data(chat_data_path: Path) -> list[dict]:
    """csv로 저장된 채팅 구간 통계를 로드한다."""
    if not os.path.exists(chat_data_path):
        raise FileNotFoundError(f"채팅 데이터 파일을 찾을 수 없습니다: {chat_data_path}")

    with open(chat_data_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [
            {"start": int(row["start"]), "end": int(row["end"]), "count": int(row["count"])}
            for row in reader
        ]


def build_chat_summary_text(buckets: list[dict], mode: str = "spike") -> str:
    """
    채팅 구간 통계를 프롬프트용 텍스트로 변환한다.
    - mode="full": 전체 구간 수치를 그대로 나열 (토큰 많이 씀, LLM이 직접 판단)
    - mode="spike": 평균 대비 급증한 구간만 요약해서 전달 (토큰 절약, 우리가 미리 판단)
    """
    if not buckets:
        return ""

    if mode == "full":
        lines = [f"{b['start']}~{b['end']}초: {b['count']}개" for b in buckets]
        return "\n".join(lines)

    if mode == "spike":
        avg = sum(b["count"] for b in buckets) / len(buckets)
        spikes = [b for b in buckets if b["count"] >= avg * CHAT_SPIKE_THRESHOLD]

        if not spikes:
            return "채팅 반응이 평소와 비슷한 수준으로 유지되어 특별한 급증 구간이 없습니다."

        lines = [
            f"{s['start']}~{s['end']}초: 평균 대비 {s['count'] / avg:.1f}배 ({s['count']}개)"
            for s in spikes
        ]
        return "채팅 급증 구간:\n" + "\n".join(lines)

    raise ValueError(f"알 수 없는 mode: {mode} (full 또는 spike만 지원)")


# --------------------------------------------------
# 3. LLM 분석 실행
# --------------------------------------------------

def run_llm_analysis(
    subtitle_text: str,
    chat_summary_text: str = "",
    density_instruction: str = "",
) -> str:
    """자막(+채팅 요약, +밀도 지침)을 프롬프트에 담아 LLM에 분석을 요청하고 원본 응답 텍스트를 반환한다."""
    client = get_llm_client()

    chat_section = f"\n[채팅 반응 데이터]\n{chat_summary_text}" if chat_summary_text else ""
    chat_instruction = CHAT_INSTRUCTION if chat_summary_text else ""

    prompt = PROMPT_TEMPLATE.format(
        subtitle_text=subtitle_text,
        chat_section=chat_section,
        chat_instruction=chat_instruction,
        density_instruction=density_instruction,
    )

    print(f"\n{MODEL_NAME} 모델이 자막을 분석 중입니다...")
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )
    return response.text


# --------------------------------------------------
# 4. 응답 파싱
# --------------------------------------------------

def parse_analysis_result(raw_text: str) -> list[dict]:
    """LLM 원본 응답 텍스트를 JSON으로 파싱한다 (코드블록/여분 텍스트 방어 포함)."""
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"응답에서 JSON 배열을 찾을 수 없습니다:\n{raw_text}")

    json_text = text[start:end + 1]

    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 파싱에 실패했습니다: {e}\n원본 텍스트:\n{raw_text}")


# --------------------------------------------------
# 5. 결과 저장
# --------------------------------------------------

def save_analysis(audio_name: str, results: list[dict]) -> Path:
    """분석 결과를 json으로 저장하고 저장 경로를 반환한다."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    payload = {
        "audio_source": audio_name,
        "model": MODEL_NAME,
        "num_points": len(results),
        "highlights": results,
    }

    out_path = Path(OUTPUT_DIR) / f"{audio_name}_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[저장 완료] {out_path}")
    return out_path


# --------------------------------------------------
# 6. 조립 함수 (수동/자동 겸용)
# --------------------------------------------------

def analyze_subtitle(
    subtitle_path: Optional[str] = None,
    chat_data_path: Optional[str] = None,
    chat_summary_mode: str = "spike",
) -> Path:
    """
    자막 파일을 읽어 LLM 분석 결과를 json 파일로 저장하고, 그 경로를 반환한다.
    - subtitle_path가 없으면 input()으로 받는다.
    - chat_data_path가 있으면 채팅 통계를 함께 고려해 분석한다 (없으면 기존처럼 자막만 사용).
    - chat_summary_mode: "spike"(기본, 급증 구간만 요약) 또는 "full"(전체 수치 나열)
    - 자막의 전체 길이를 계산해, 최소 밀도(MIN_INTERVAL_SEC)를 만족하도록 지침을 자동 추가한다.
    """
    if subtitle_path is None:
        subtitle_path = input("자막 파일 경로: ").strip()

    subtitle_path = Path(subtitle_path)
    audio_name = subtitle_path.stem

    subtitle_text = load_subtitle(subtitle_path)

    duration_sec = get_subtitle_duration(subtitle_text)
    density_instruction = build_density_instruction(duration_sec)

    chat_summary_text = ""
    if chat_data_path is not None:
        buckets = load_chat_data(Path(chat_data_path))
        chat_summary_text = build_chat_summary_text(buckets, mode=chat_summary_mode)

    raw_result = run_llm_analysis(subtitle_text, chat_summary_text, density_instruction)
    parsed_result = parse_analysis_result(raw_result)

    return save_analysis(audio_name, parsed_result)


if __name__ == "__main__":
    result_path = analyze_subtitle()
    print(f"\n분석 결과 저장 위치: {result_path}")
