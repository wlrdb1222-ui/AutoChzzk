"""
transcriber.py
faster-whisper 기반 오디오 전사 모듈.
"""

import os
import time
import json
from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel, BatchedInferencePipeline
from tqdm import tqdm


# --------------------------------------------------
# 설정값 (거의 안 바뀌는 값들 - 필요하면 여기서 직접 수정)
# --------------------------------------------------

MODEL_NAME = "large-v3"
COMPUTE_TYPE = "float16"
LANGUAGE = "ko"
OUTPUT_DIR = "/content/transcripts"


# --------------------------------------------------
# 1. 모델 로드
# --------------------------------------------------

def load_model() -> BatchedInferencePipeline:
    """faster-whisper 모델을 로드한다."""
    print(f"[로드] {MODEL_NAME} ({COMPUTE_TYPE}) 준비 중...")
    t0 = time.time()

    base_model = WhisperModel(
        MODEL_NAME,
        device="cuda",
        compute_type=COMPUTE_TYPE,
    )
    model = BatchedInferencePipeline(model=base_model)

    print(f"[로드 완료] {MODEL_NAME} - {time.time() - t0:.1f}초")
    return model


# --------------------------------------------------
# 2. 전사 실행
# --------------------------------------------------

def run_transcription(
    model: BatchedInferencePipeline,
    audio_path: Path,
    initial_prompt: Optional[str] = None,
) -> tuple[list[dict], float]:
    """오디오를 전사하고 (구간 리스트, 소요시간)을 반환한다."""
    t0 = time.time()

    segments, info = model.transcribe(
            str(audio_path),
            language=LANGUAGE,
            initial_prompt=initial_prompt,
        
            condition_on_previous_text=False,
        
            beam_size=10,
            temperature=0.0,
        
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=300,
            ),
        
            word_timestamps=False,
            batch_size=16,
        ) batch_size=16,
    )

    results = []
    with tqdm(total=round(info.duration, 1), unit="초", desc="전사 진행률") as pbar:
        last_end = 0.0
        for seg in segments:
            results.append({
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
            })
            pbar.update(round(seg.end - last_end, 2))
            last_end = seg.end

    elapsed = time.time() - t0
    print(f"[전사 완료] {len(results)}개 구간 - {elapsed:.1f}초")
    return results, elapsed


# --------------------------------------------------
# 3. 결과 저장
# --------------------------------------------------

def save_result(
    audio_name: str,
    segments: list[dict],
    elapsed_sec: float,
) -> Path:
    """전사 결과를 json으로 저장하고 저장 경로를 반환한다."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    payload = {
        "audio_source": audio_name,
        "model": MODEL_NAME,
        "elapsed_sec": round(elapsed_sec, 1),
        "num_segments": len(segments),
        "segments": segments,
    }

    out_path = Path(OUTPUT_DIR) / f"{audio_name}_{MODEL_NAME}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[저장 완료] {out_path}")
    return out_path


# --------------------------------------------------
# 4. 조립 함수 (수동/자동 겸용)
# --------------------------------------------------

def transcribe_audio(
    audio_path: Optional[str] = None,
    initial_prompt: Optional[str] = None,
) -> Path:
    """
    오디오 파일을 전사하고 결과 json 경로를 반환한다.
    - audio_path가 없으면 input()으로 받는다.
    - initial_prompt는 실험적으로 바뀔 수 있어 인자로 남겨둠.
    - 그 외 설정값(모델명, compute_type, language, output_dir)은
      모듈 상단 상수를 직접 참조한다. 바꾸고 싶으면 상수를 수정할 것.
    """
    if audio_path is None:
        audio_path = input("전사할 오디오 파일 경로를 입력하세요: ").strip()

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {audio_path}")

    audio_name = audio_path.stem
    print(f"입력된 파일: {audio_name}")

    model = load_model()
    segments, elapsed_sec = run_transcription(model, audio_path, initial_prompt)

    return save_result(audio_name, segments, elapsed_sec)


if __name__ == "__main__":
    transcribe_audio()
