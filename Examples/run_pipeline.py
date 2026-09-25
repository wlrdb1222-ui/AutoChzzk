"""
run_pipeline.py
AutoChzzk 패키지 사용 예제 - 전체 파이프라인 조립 스크립트.

다운로드 -> 전사 -> LLM 분석 -> csv 변환까지 이어지는 흐름을 보여준다.

- 값을 다 채워서 호출하면 input() 없이 자동 실행된다.
- 값을 비워두면 각 함수 내부에서 필요한 값을 input()으로 물어본다.
"""

import sys
from pathlib import Path

# 패키지가 설치되어 있지 않은 상태에서 로컬 실행할 경우를 대비한 경로 등록
sys.path.append(str(Path(__file__).resolve().parent.parent))

from AutoChzzk import audio_down, transcribe_audio, analyze_subtitle, json_to_csv


def run(vod_url: str = None, output_dir: str = ".") -> None:
    # 1. 오디오 다운로드
    audio_path = audio_down(vod_url=vod_url, output_dir=output_dir)

    # 2. 전사
    transcript_json_path = transcribe_audio(audio_path)

    # 3. csv 변환 (선택)
    json_to_csv(transcript_json_path)

    # 4. LLM 분석
    analysis_result = analyze_subtitle(transcript_json_path)

    print("\n" + "=" * 30 + " 분석 결과 " + "=" * 30)
    print(analysis_result)
    print("=" * 71)


if __name__ == "__main__":
    # 자동 실행 예시: run(vod_url="https://chzzk.naver.com/video/12345")
    # 수동 실행 예시: run()  -> 각 단계에서 필요한 값을 물어봄
    run()
