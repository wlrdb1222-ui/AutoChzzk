"""
transcriber.py
faster-whisper 기반 오디오 전사 모듈.
"""

import os
import time
import json
from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel
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

def load_model() -> WhisperModel:
    """faster-whisper 모델을 로드한다."""
    print(f"[로드] {MODEL_NAME} ({COMPUTE_TYPE}) 준비 중...")
    t0 = time.time()

    model = WhisperModel(
        MODEL_NAME,
        device="cuda",
        compute_type=COMPUTE_TYPE,
    )

    print(f"[로드 완료] {MODEL_NAME} - {time.time() - t0:.1f}초")
    return model


# --------------------------------------------------
# 2. 전사 실행
# --------------------------------------------------

MAX_CHUNK_SEC = 1800   # 조각 최대 길이 (30분)
PRE_PAD_SEC = 10       # 조각 앞쪽 문맥용 여유
POST_PAD_SEC = 20      # 조각 뒤쪽 여유 (VAD 최대 구간 15초보다 크게)


def get_duration(path: Path) -> float:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ])
    return float(out.decode().strip())


def cut_chunk(src: Path, dst: Path, start: float, length: float):
    """구간을 잘라 16kHz mono wav로 저장"""
    subprocess.run([
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{start:.3f}", "-t", f"{length:.3f}",
        "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", str(dst),
    ], check=True)


def run_transcription_chunked(
    model,
    audio_path: Path,
    work_dir: Path,
    initial_prompt=None,
):
    audio_path, work_dir = Path(audio_path), Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    duration = get_duration(audio_path)
    n = max(1, math.ceil(duration / MAX_CHUNK_SEC))
    chunk_len = duration / n          # 균등 분할
    print(f"[분할] 총 {duration/60:.1f}분 -> {n}개 조각 (조각당 약 {chunk_len/60:.1f}분)")

    all_segments, total_elapsed = [], 0.0

    for i in range(n):
        core_start = i * chunk_len
        core_end = duration if i == n - 1 else (i + 1) * chunk_len
        is_last = (i == n - 1)

        cache = work_dir / f"chunk_{i:03d}.json"
        if cache.exists():                      # 이어하기
            kept = json.loads(cache.read_text(encoding="utf-8"))
            print(f"[{i+1}/{n}] 캐시 사용")
            all_segments.extend(kept)
            continue

        # 실제로 자르는 범위 = 담당 구간 + 앞뒤 여유
        cut_start = max(0.0, core_start - PRE_PAD_SEC)
        cut_end = min(duration, core_end + POST_PAD_SEC)
        wav = work_dir / f"chunk_{i:03d}.wav"
        cut_chunk(audio_path, wav, cut_start, cut_end - cut_start)

        print(f"[{i+1}/{n}] {core_start/60:.1f}~{core_end/60:.1f}분 전사 중")
        segs, elapsed = run_transcription(model, wav, initial_prompt)
        total_elapsed += elapsed
        wav.unlink(missing_ok=True)

        # 전체 타임라인으로 변환 후, "시작 시각이 담당 구간 안인 것"만 채택
        kept = []
        for s in segs:
            g_start = s["start"] + cut_start
            g_end = s["end"] + cut_start
            if core_start <= g_start < core_end or (is_last and g_start >= core_start):
                kept.append({
                    "start": round(g_start, 2),
                    "end": round(g_end, 2),
                    "text": s["text"],
                })

        cache.write_text(json.dumps(kept, ensure_ascii=False), encoding="utf-8")
        all_segments.extend(kept)

    all_segments.sort(key=lambda x: x["start"])
    print(f"[전체 완료] {len(all_segments)}개 구간 - 전사 합계 {total_elapsed:.1f}초")
    return all_segments, total_elapsed
def run_transcription(
    model: WhisperModel,
    audio_path: Path,
    initial_prompt: Optional[str] = None,
) -> tuple[list[dict], float]:
    """오디오를 전사하고 (구간 리스트, 소요시간)을 반환한다."""
    t0 = time.time()

    segments, elapsed = run_transcription_chunked(
        model, audio_path, "/content/chunks", initial_prompt
    )
    eTime = time.time() - t0
    print(f"[전처리 완료]{eTime:.1f}초")
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
