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

PROMPT_TEMPLATE = """
너는 생방송 전사 스크립트를 분석해서 키리누키(하이라이트 영상)로 활용할 수 있는
세밀한 타임라인 이벤트를 찾아내는 어시스턴트다.

방송 전체를 큰 주제별로 요약하지 마라.
하나의 주제가 오래 이어지더라도 그 안에서 실제로 재미있거나,
흥미롭거나, 반응이 발생한 순간이 있다면 별도의 이벤트로 분리하라.

목표는 방송 요약이 아니라,
"이 부분을 잘라서 키리누키 영상으로 만들 수 있는가?"
를 판단할 수 있는 세밀한 이벤트 목록을 만드는 것이다.

발화를 시간 순서대로 읽으면서 다음과 같은 순간을 찾아라.

- 재미있는 말이나 행동
- 갑작스러운 반응이나 감정 변화
- 시청자 채팅에 대한 반응
- 예상하지 못한 상황
- 재미있는 에피소드나 썰
- 스트리머의 독특한 표현이나 발언
- 시청자가 흥미를 느낄 만한 정보
- 방송에서 중요한 공지나 사건
- 채팅과 상호작용하면서 만들어진 재미있는 상황
- 하나의 주제 안에서 새롭게 발생한 사건이나 화제

중요:
큰 주제 하나를 하나의 긴 이벤트로 묶지 마라.

예를 들어 10분 동안 네일아트 이야기를 하더라도,

"네일아트 이야기"

하나로 끝내지 말고,

- 네일샵 경험을 이야기하는 부분
- 사진을 보여주는 부분
- 강아지 이야기가 나오는 부분
- 시청자에게 설명하기 위해 그림을 그리는 부분
- 특정 발언에 화내거나 웃는 부분

처럼 실제로 서로 다른 클립으로 사용할 수 있는 순간은 별도의 이벤트로 분리하라.

각 이벤트는 하나의 명확한 사건이나 장면을 중심으로 만들어라.

단순히 주제가 바뀌었다는 이유만으로 이벤트를 생성하지 마라.
반대로 재미있는 사건이 발생했다면 같은 주제 안에 있더라도 반드시 별도의 이벤트로 분리하라.

반드시 JSON 배열만 출력하고 다른 설명은 절대 하지 마라.

각 이벤트는 아래 필드를 가진다.

- start : 이벤트가 실제로 시작되는 시각(초)
- end : 해당 이벤트가 끝나는 시각(초)
- fun_score : 1~5
- value_score : 1~5
- title : 실제 키리누키 제목으로 사용할 수 있는 짧은 제목
- comment : 해당 장면을 설명하는 한 줄 설명

fun_score 기준

1 = 평범한 잡담이나 특별한 반응이 없음
2 = 약간 흥미롭지만 키리누키로서 임팩트가 약함
3 = 흥미로운 이야기, 재미있는 상황, 볼 만한 장면
4 = 웃음, 당황, 강한 리액션, 재미있는 사건
5 = 방송에서 대표 키리누키로 사용할 수 있을 정도의 강한 장면

value_score 기준

1 = 단순한 일상 대화
2 = 일반적인 잡담이나 개인적인 이야기
3 = 흥미롭거나 알아둘 만한 내용
4 = 방송 흐름에서 중요한 사건이나 정보
5 = 중요한 공지, 핵심 사건, 방송을 이해하는 데 중요한 내용

fun_score와 value_score는 서로 독립적으로 판단하라.

재미있는 장면이라고 해서 반드시 value_score가 높을 필요는 없다.
반대로 재미는 없더라도 중요한 공지나 사건이라면 value_score를 높게 줄 수 있다.

이벤트 구간 기준

- 이벤트는 하나의 명확한 사건이나 장면을 포함해야 한다.
- 너무 긴 구간으로 만들지 마라.
- 특별한 이유 없이 수 분 이상 이어지는 긴 이벤트를 만들지 마라.
- 하나의 이벤트 안에서 새로운 사건이나 강한 반응이 발생하면 별도의 이벤트로 분리하라.
- 이벤트 사이에 약간의 일반적인 대화가 있더라도 서로 다른 사건이라면 분리할 수 있다.
- 이벤트가 실제로 시작되기 전의 불필요한 발화를 start에 포함하지 마라.
- 이벤트의 핵심 반응이나 사건이 끝나는 시점을 기준으로 end를 설정하라.
- 서로 겹치는 이벤트를 만들지 마라.
- 단순한 화제 전환 자체는 하이라이트로 취급하지 마라.

방송 전체를 분석해야 하지만 모든 시간대를 이벤트로 채울 필요는 없다.

특별한 사건이나 장면이 없는 구간은 과감하게 건너뛰어도 된다.
단순히 방송 전체를 빠짐없이 요약하기 위해 일반적인 잡담을 이벤트로 만들지 마라.

이 작업의 우선순위는 방송 전체의 요약이 아니라
키리누키로 사용할 가능성이 있는 개별 장면을 놓치지 않는 것이다.

특히 중요한 점:

"방송 전체를 요약하는 것"이 아니라
"방송에서 잘라낼 수 있는 장면을 발견하는 것"을 우선하라.

채팅 반응 데이터는 이벤트의 중요성과 시청자 반응을 판단하는 보조 정보로 사용하라.

채팅이 급증한 경우:
- 해당 시점의 발화와 상황을 먼저 확인한다.
- 채팅 급증 자체만으로 fun_score를 올리지 않는다.
- 발화 내용은 평범하지만 시청자 반응이 크게 증가했다면
  value_score를 높이거나 별도의 이벤트로 분리할 수 있다.
- 채팅 반응과 실제 재미있는 상황이 동시에 발생했다면 fun_score를 높일 수 있다.
- 단순히 채팅이 많다는 이유만으로 높은 점수를 주지 마라.

{density_instruction}
{chat_instruction}

[자막 데이터]
{subtitle_text}

{chat_section}
"""


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

    prompt = PROMPT_TEMPLATE.format(
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
