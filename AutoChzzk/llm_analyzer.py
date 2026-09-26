"""
llm_analyzer.py

전사된 자막 및 선택적으로 채팅 통계를 Gemini로 분석해
키리누키용 세밀한 하이라이트 이벤트를 추출하는 모듈.

LLM의 분석 규칙과 프롬프트 내용은 prompts/prompt.txt에서 관리한다.
이 모듈은 파일 처리, 데이터 계산, 프롬프트 조립, LLM 호출,
결과 파싱 및 저장을 담당한다.
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

PROMPT_PATH = (
    Path(__file__).parent
    / "prompts"
    / "prompt.txt"
)


def load_prompt() -> str:
    """
    prompts/prompt.txt에서 LLM 분석 프롬프트를 읽는다.
    """

    if not PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"프롬프트 파일을 찾을 수 없습니다: {PROMPT_PATH}"
        )

    return PROMPT_PATH.read_text(
        encoding="utf-8"
    )


# --------------------------------------------------
# 1. 자막 로드
# --------------------------------------------------

def load_subtitle(
    subtitle_path: Path
) -> str:
    """
    자막 파일을 텍스트로 읽어 반환한다.

    UTF-8을 우선 사용하고,
    실패하면 CP949로 다시 읽는다.
    """

    if not subtitle_path.exists():
        raise FileNotFoundError(
            f"자막 파일을 찾을 수 없습니다: {subtitle_path}"
        )

    try:
        with open(
            subtitle_path,
            "r",
            encoding="utf-8"
        ) as f:
            return f.read()

    except UnicodeDecodeError:
        with open(
            subtitle_path,
            "r",
            encoding="cp949"
        ) as f:
            return f.read()


def get_subtitle_duration(
    subtitle_text: str
) -> Optional[float]:
    """
    transcriber.py가 만든 자막 JSON에서
    전체 방송 길이를 추정한다.

    segments 리스트의 가장 마지막 end 값을 사용한다.

    실패하면 None을 반환한다.
    """

    try:
        data = json.loads(
            subtitle_text
        )

        segments = data.get(
            "segments"
        )

        if not segments:
            return None

        end_times = [
            seg["end"]
            for seg in segments
            if "end" in seg
        ]

        if not end_times:
            return None

        return max(end_times)

    except (
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError
    ):
        return None


# --------------------------------------------------
# 2. 채팅 통계 로드
# --------------------------------------------------

def load_chat_data(
    chat_data_path: Path
) -> list[dict]:
    """
    CSV로 저장된 채팅 구간 통계를 로드한다.

    CSV 형식:

    start,end,count
    """

    if not chat_data_path.exists():
        raise FileNotFoundError(
            f"채팅 데이터 파일을 찾을 수 없습니다: "
            f"{chat_data_path}"
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
    채팅 구간 통계를 LLM 프롬프트에 전달할 텍스트로 변환한다.

    mode="full":
        전체 구간의 채팅 수치를 전달한다.

    mode="spike":
        평균 대비 급증한 구간만 전달한다.
    """

    if not buckets:
        return ""

    if mode == "full":

        lines = [
            f"{bucket['start']}~{bucket['end']}초: "
            f"{bucket['count']}개"
            for bucket in buckets
        ]

        return "\n".join(lines)

    if mode == "spike":

        average_count = (
            sum(
                bucket["count"]
                for bucket in buckets
            )
            / len(buckets)
        )

        if average_count <= 0:
            return (
                "채팅 데이터는 존재하지만 "
                "평균 채팅량을 계산할 수 없습니다."
            )

        spikes = [
            bucket
            for bucket in buckets
            if bucket["count"]
            >= average_count * CHAT_SPIKE_THRESHOLD
        ]

        if not spikes:
            return (
                "채팅 반응이 평소와 비슷한 수준으로 유지되어 "
                "특별한 급증 구간이 없습니다."
            )

        lines = [
            (
                f"{spike['start']}~{spike['end']}초: "
                f"평균 대비 "
                f"{spike['count'] / average_count:.1f}배 "
                f"({spike['count']}개)"
            )
            for spike in spikes
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
    duration_sec: Optional[float] = None,
) -> str:
    """
    자막, 채팅 통계, 방송 길이를 프롬프트에 삽입하여
    LLM에 분석을 요청한다.

    LLM 분석 규칙은 모두 prompts/prompt.txt에서 관리한다.
    """

    client = get_llm_client()

    prompt_template = load_prompt()

    # 방송 길이
    if duration_sec and duration_sec > 0:
        duration_min = round(
            duration_sec / 60,
            1
        )
    else:
        duration_min = "알 수 없음"

    # 채팅 데이터가 없는 경우에도
    # 프롬프트의 해당 영역이 자연스럽게 유지되도록 한다.
    if chat_summary_text:
        chat_section = chat_summary_text
    else:
        chat_section = (
            "채팅 데이터가 제공되지 않았습니다. "
            "자막과 문맥만을 기준으로 분석하십시오."
        )

    # prompt.txt의 변수에 실제 데이터를 삽입
    prompt = prompt_template.format(
        subtitle_text=subtitle_text,
        chat_section=chat_section,
        chat_spike_threshold=CHAT_SPIKE_THRESHOLD,
        duration_min=duration_min,
        max_event_interval_sec=MAX_EVENT_INTERVAL_SEC,
    )

    print(
        f"\n{MODEL_NAME} 모델이 "
        f"자막을 분석 중입니다..."
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
    LLM 원본 응답 텍스트를 JSON 배열로 파싱한다.

    코드블록이나 JSON 앞뒤의 불필요한 텍스트가
    포함된 경우에도 최대한 파싱을 시도한다.
    """

    text = raw_text.strip()

    # Markdown 코드블록 제거
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

    # JSON 배열의 시작과 끝 탐색
    start = text.find("[")
    end = text.rfind("]")

    if start == -1 or end == -1:
        raise ValueError(
            "응답에서 JSON 배열을 찾을 수 없습니다:\n"
            f"{raw_text}"
        )

    json_text = text[
        start:end + 1
    ]

    try:
        result = json.loads(
            json_text
        )

    except json.JSONDecodeError as e:
        raise ValueError(
            f"JSON 파싱에 실패했습니다: {e}\n"
            f"원본 텍스트:\n{raw_text}"
        )

    if not isinstance(result, list):
        raise ValueError(
            "LLM 응답 JSON이 배열 형식이 아닙니다."
        )

    return result


# --------------------------------------------------
# 5. 결과 저장
# --------------------------------------------------

def save_analysis(
    audio_name: str,
    results: list[dict]
) -> Path:
    """
    분석 결과를 JSON으로 저장하고
    저장 경로를 반환한다.
    """

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
# 6. 전체 분석
# --------------------------------------------------

def analyze_subtitle(
    subtitle_path: Optional[str] = None,
    chat_data_path: Optional[str] = None,
    chat_summary_mode: str = "spike",
) -> Path:
    """
    자막 파일을 읽어 LLM 분석 결과를 JSON 파일로 저장하고
    그 경로를 반환한다.

    Parameters
    ----------
    subtitle_path:
        자막 파일 경로.
        지정하지 않으면 input()으로 받는다.

    chat_data_path:
        채팅 통계 CSV 경로.
        지정하지 않으면 자막만 분석한다.

    chat_summary_mode:
        "spike" = 채팅 급증 구간만 전달
        "full"  = 전체 채팅 수치 전달
    """

    if subtitle_path is None:
        subtitle_path = input(
            "자막 파일 경로: "
        ).strip()

    subtitle_path = Path(
        subtitle_path
    )

    audio_name = subtitle_path.stem

    # --------------------------------------------------
    # 자막 로드
    # --------------------------------------------------

    subtitle_text = load_subtitle(
        subtitle_path
    )

    # --------------------------------------------------
    # 방송 길이 계산
    # --------------------------------------------------

    duration_sec = get_subtitle_duration(
        subtitle_text
    )

    # --------------------------------------------------
    # 채팅 데이터
    # --------------------------------------------------

    chat_summary_text = ""

    if chat_data_path is not None:

        buckets = load_chat_data(
            Path(chat_data_path)
        )

        chat_summary_text = (
            build_chat_summary_text(
                buckets,
                mode=chat_summary_mode
            )
        )

    # --------------------------------------------------
    # LLM 분석
    # --------------------------------------------------

    raw_result = run_llm_analysis(
        subtitle_text=subtitle_text,
        chat_summary_text=chat_summary_text,
        duration_sec=duration_sec,
    )

    # --------------------------------------------------
    # JSON 파싱
    # --------------------------------------------------

    parsed_result = parse_analysis_result(
        raw_result
    )

    # --------------------------------------------------
    # 결과 저장
    # --------------------------------------------------

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
