# AutoChzzk

치지직(CHZZK) VOD를 다운로드하고, faster-whisper로 전사한 뒤,
Gemini로 하이라이트를 분석하는 파이프라인 프로젝트입니다.

## 구조

```
AutoChzzk/
├── AutoChzzk/            # 패키지
│   ├── __init__.py
│   ├── downloader.py     # VOD 다운로드 (download, audio_down)
│   ├── transcriber.py    # 오디오 전사 (transcribe_audio)
│   ├── llm_analyzer.py   # LLM 분석 (analyze_subtitle)
│   ├── llm_client.py     # genai 클라이언트 공통 생성
│   └── utils.py          # json_to_csv, count_file_tokens
├── examples/
│   └── run_pipeline.py   # 전체 파이프라인 실행 예제
├── requirements.txt
└── .gitignore
```

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

각 함수는 값을 인자로 넘기면 자동 실행되고, 비워두면 필요한 값을 `input()`으로 물어봅니다.

```python
from AutoChzzk import audio_down, transcribe_audio, analyze_subtitle, json_to_csv

# 자동 (파라미터를 다 채움)
audio_path = audio_down(vod_url="https://chzzk.naver.com/video/12345")
transcript_json_path = transcribe_audio(audio_path)
json_to_csv(transcript_json_path)
result = analyze_subtitle(transcript_json_path)

# 수동 (비워두면 input()으로 물어봄)
audio_path = audio_down()
```

전체 파이프라인 실행 예제는 `examples/run_pipeline.py`를 참고하세요.

## 참고

- 현재 `llm_client.py`는 Google Colab 환경(`google.colab.userdata`)을 기준으로
  API 키를 조회합니다. 다른 환경(Kaggle 등)으로 옮길 경우 이 부분만 수정하면 됩니다.
- `downloader.py`의 `fetch_and_save()`는 실제 다운로드 실행 로직만 담당하도록
  분리되어 있어, 추후 `requests` 대신 `yt-dlp`나 `aria2c` 등으로 교체할 수 있습니다.
