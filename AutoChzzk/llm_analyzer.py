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

CHAT_SPIKE_THRESHOLD = 1.5

# 이벤트가 지나치게 길어지는 것을 방지하기 위한 참고값
MAX_EVENT_INTERVAL_SEC = 300

PROMPT_PATH = Path(__file__).parent / "prompts" / "prompt.txt"


# --------------------------------------------------
# 프롬프트
# --------------------------------------------------

def load_prompt() -> str:
    """
    prompts/prompt.txt의 전체 내용을 읽는다.
    """

    if not PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"프롬프트 파일을 찾을 수 없습니다: {PROMPT_PATH}"
        )

    return PROMPT_PATH.read_text(encoding="utf-8")


def build_prompt(
    subtitle_text: str,
    chat_section: str = "",
    duration_sec: Optional[float] = None,
) -> str:
    """
    prompt.txt에 필요한 값을 삽입한다.

    str.format()을 사용하지 않는 이유:
    prompt.txt 안에 JSON 예시처럼 중괄호가 포함될 수 있기 때문이다.

    따라서 실제로 사용하는 변수만 명시적으로 치환한다.
    """

    prompt = load_prompt()

    duration_min = 0

    if duration_sec is not None:
        duration_min = duration_sec / 60

    replacements = {
        "{subtitle_text}": subtitle_text,
        "{chat_section}": chat_section,
        "{chat_spike_threshold}": str(CHAT_SPIKE_THRESHOLD),
        "{duration_min}": f"{duration_min:.2f}",
        "{max_event_interval_sec}": str(MAX_EVENT_INTERVAL_SEC),
    }

    for placeholder, value in replacements.items():
        prompt = prompt.replace(placeholder, value)

    return prompt


# --------------------------------------------------
# 자막
# --------------------------------------------------

def load_subtitle(subtitle_path: Path) -> str:
    """
    자막 JSON 파일을 문자열 그대로 읽는다.
    """

    try:
        return subtitle_path.read_text(encoding="utf-8")

    except UnicodeDecodeError:
        return subtitle_path.read_text(encoding="cp949")


def get_subtitle_duration(subtitle_text: str) -> Optional[float]:
    """
    자막 JSON에서 가장 마지막 segment의 end 시간을 가져온다.
    """

    try:
        data = json.loads(subtitle_text)

    except json.JSONDecodeError:
        return None

    segments = data.get("segments", [])

    if not segments:
        return None

    end_times = []

    for segment in segments:
        end = segment.get("end")

        if isinstance(end, (int, float)):
            end_times.append(end)

    if not end_times:
        return None

    return max(end_times)


# --------------------------------------------------
# 채팅
# --------------------------------------------------

def load_chat_data(chat_data_path: Path) -> list[dict]:
    """
    채팅 통계 CSV를 읽는다.

    CSV 형식:
        start,end,count
    """

    buckets = []

    with chat_data_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            try:
                start = int(float(row["start"]))
                end = int(float(row["end"]))
                count = int(float(row["count"]))

            except (ValueError, KeyError):
                continue

            buckets.append(
                {
                    "start": start,
                    "end": end,
                    "count": count,
                }
            )

    return buckets


def build_chat_summary_text(
    buckets: list[dict],
    mode: str = "spike",
) -> str:
    """
    채팅 데이터를 LLM에 전달하기 위한 텍스트로 변환한다.

    mode:
        full  -> 전체 채팅 통계
        spike -> 평균 대비 급증 구간만 전달
    """

    if not buckets:
        return "채팅 데이터가 없습니다."

    total_count = sum(
        bucket["count"]
        for bucket in buckets
    )

    average = total_count / len(buckets)

    # ----------------------------------------------
    # 전체 데이터
    # ----------------------------------------------

    if mode == "full":

        lines = [
            f"전체 구간 평균 채팅 수: {average:.2f}",
            "",
            "시간대별 채팅 수:",
        ]

        for bucket in buckets:
            lines.append(
                f'{bucket["start"]}~{bucket["end"]}초: '
                f'{bucket["count"]}개'
            )

        return "\n".join(lines)

    # ----------------------------------------------
    # 급증 구간
    # ----------------------------------------------

    if mode == "spike":

        if average <= 0:
            return (
                "채팅 평균을 계산할 수 없습니다. "
                "채팅 급증 분석을 사용하지 마세요."
            )

        spike_lines = []

        for bucket in buckets:

            ratio = bucket["count"] / average

            if ratio >= CHAT_SPIKE_THRESHOLD:
                spike_lines.append(
                    f'{bucket["start"]}~{bucket["end"]}초: '
                    f'평균 대비 {ratio:.1f}배 '
                    f'({bucket["count"]}개)'
                )

        if not spike_lines:
            return (
                f"채팅 급증 구간이 없습니다. "
                f"(급증 기준: 평균의 {CHAT_SPIKE_THRESHOLD:.1f}배)"
            )

        return (
            f"전체 구간 평균 채팅 수: {average:.2f}\n"
            f"채팅 급증 기준: 평균의 "
            f"{CHAT_SPIKE_THRESHOLD:.1f}배 이상\n\n"
            "채팅 급증 구간:\n"
            + "\n".join(spike_lines)
        )

    raise ValueError(
        f"지원하지 않는 chat summary mode입니다: {mode}"
    )


# --------------------------------------------------
# LLM
# --------------------------------------------------

def run_llm_analysis(
    subtitle_text: str,
    chat_summary_text: str = "",
    duration_sec: Optional[float] = None,
) -> str:
    """
    자막과 채팅 데이터를 Gemini에 전달해 분석한다.
    """

    prompt = build_prompt(
        subtitle_text=subtitle_text,
        chat_section=chat_summary_text,
        duration_sec=duration_sec,
    )

    client = get_llm_client()

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    return response.text


# --------------------------------------------------
# 결과 파싱
# --------------------------------------------------

def parse_analysis_result(raw_text: str) -> list:
    """
    LLM 응답에서 JSON 배열을 추출한다.
    """

    text = raw_text.strip()

    # Markdown 코드블록 제거
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    # JSON 배열 시작/끝 찾기
    start = text.find("[")
    end = text.rfind("]")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(
            "LLM 응답에서 JSON 배열을 찾을 수 없습니다."
        )

    json_text = text[start:end + 1]

    try:
        result = json.loads(json_text)

    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM 응답 JSON 파싱 실패: {e}\n\n"
            f"응답 내용:\n{raw_text}"
        ) from e

    if not isinstance(result, list):
        raise ValueError(
            "LLM 결과가 JSON 배열이 아닙니다."
        )

    return result


# --------------------------------------------------
# 저장
# --------------------------------------------------

def save_analysis(
    audio_name: str,
    results: list,
) -> Path:
    """
    분석 결과를 JSON으로 저장한다.
    """

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"{audio_name}_analysis.json"
    )

    data = {
        "audio_source": audio_name,
        "model": MODEL_NAME,
        "num_points": len(results),
        "highlights": results,
    }

    output_path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return output_path


# --------------------------------------------------
# 전체 분석
# --------------------------------------------------

def analyze_subtitle(
    subtitle_path=None,
    chat_data_path=None,
    chat_summary_mode="spike",
):
    """
    자막 + 선택적 채팅 데이터를 분석한다.

    subtitle_path:
        전사 JSON 파일 경로

    chat_data_path:
        채팅 통계 CSV 경로

    chat_summary_mode:
        "spike" 또는 "full"
    """

    # ----------------------------------------------
    # 자막 경로
    # ----------------------------------------------

    if subtitle_path is None:
        subtitle_path = input(
            "자막 JSON 파일 경로를 입력하세요: "
        ).strip()

    subtitle_path = Path(subtitle_path)

    if not subtitle_path.exists():
        raise FileNotFoundError(
            f"자막 파일을 찾을 수 없습니다: {subtitle_path}"
        )

    # ----------------------------------------------
    # 자막 읽기
    # ----------------------------------------------

    subtitle_text = load_subtitle(
        subtitle_path
    )

    duration_sec = get_subtitle_duration(
        subtitle_text
    )

    # ----------------------------------------------
    # 채팅
    # ----------------------------------------------

    chat_summary_text = ""

    if chat_data_path is not None:

        chat_data_path = Path(
            chat_data_path
        )

        if not chat_data_path.exists():
            raise FileNotFoundError(
                f"채팅 파일을 찾을 수 없습니다: "
                f"{chat_data_path}"
            )

        chat_buckets = load_chat_data(
            chat_data_path
        )

        chat_summary_text = build_chat_summary_text(
            chat_buckets,
            mode=chat_summary_mode,
        )

    # ----------------------------------------------
    # LLM 분석
    # ----------------------------------------------

    raw_result = run_llm_analysis(
        subtitle_text=subtitle_text,
        chat_summary_text=chat_summary_text,
        duration_sec=duration_sec,
    )

    # ----------------------------------------------
    # 결과 파싱
    # ----------------------------------------------

    results = parse_analysis_result(
        raw_result
    )

    # ----------------------------------------------
    # 저장
    # ----------------------------------------------

    audio_name = subtitle_path.stem

    output_path = save_analysis(
        audio_name=audio_name,
        results=results,
    )

    print(
        f"분석 완료: {output_path}"
    )

    return output_path


# --------------------------------------------------
# CLI
# --------------------------------------------------

if __name__ == "__main__":
    analyze_subtitle()
