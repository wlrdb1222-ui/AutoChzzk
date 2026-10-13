
from pathlib import Path
import subprocess
import re
from .segment_maker import SegmentInfo 
import shutil
import time
from tqdm.auto import tqdm
DEFAULT_CONNECTIONS = 16
DEFAULT_RETRIES = 5
CHUNK_SIZE = 1024 * 1024

def download_mp4(url: str, output_path: str | Path) -> Path:
    """MP4 파일을 URL에서 다운로드한다."""
    print("downloading")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    connections = DEFAULT_CONNECTIONS
    retries = DEFAULT_RETRIES
    command = [
        "aria2c",
        "--allow-overwrite=true",
        "--auto-file-renaming=false",
        "--file-allocation=none",
        f"--max-connection-per-server={connections}",
        f"--split={connections}",
        f"--max-tries={retries}",
        "--summary-interval=1",
        "--human-readable=true",
        "--console-log-level=notice",
        "--show-console-readout=false",
        "--download-result=hide",
        "--dir", str(output_path.parent),
        "--out", output_path.name,
        url,
    ]

    returncode = _run_aria2c(command)

    if returncode != 0:
        raise RuntimeError(f"MP4 download failed: aria2c exited with code {returncode}")

    return output_path

def download_segments(segments: list[str], output_path: str | Path, connections: int = DEFAULT_CONNECTIONS, retries: int = DEFAULT_RETRIES) -> Path:
    """세그먼트를 병렬 다운로드하고 하나의 파일로 합친다."""
    if not segments:
        raise ValueError("segments must not be empty")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    segment_dir = output_path.parent / f".{output_path.stem}_segments"
    segment_dir.mkdir(parents=True, exist_ok=True)

    input_file = segment_dir / "segments.txt"
    segment_paths = [segment_dir / f"{index:06d}.part" for index in range(len(segments))]

    try:
        _create_aria2_input(segments, input_file)

        command = [
            "aria2c",
            "--allow-overwrite=true",
            "--auto-file-renaming=false",
            "--file-allocation=none",
            f"--max-concurrent-downloads={connections}",
            f"--max-tries={retries}",
            "--summary-interval=1",
            "--human-readable=true",
            "--console-log-level=notice",
            "--show-console-readout=false",
            "--download-result=hide",
            "--dir", str(segment_dir),
            "--input-file", str(input_file),
        ]

        print(f"세그먼트 다운로드 시작: {len(segments)}개")
        print(f"병렬 다운로드: {connections}개")

        returncode = _run_aria2c(command, total_segments=len(segments))

        if returncode != 0:
            raise RuntimeError(f"Segment download failed: aria2c exited with code {returncode}")

        print("세그먼트 다운로드 완료")
        print("세그먼트 병합 중...")

        _merge_segments(segment_paths, output_path)

        print(f"병합 완료: {output_path}")

    finally:
        input_file.unlink(missing_ok=True)

        for segment_path in segment_paths:
            segment_path.unlink(missing_ok=True)

        try:
            segment_dir.rmdir()
        except OSError:
            pass

    return remux_mp4(output_path)

def _remux_mp4(input_path: str | Path, output_path: str | Path | None = None) -> Path:
    """fMP4를 재인코딩 없이 일반 MP4로 리먹싱한다."""
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else input_path

    # 입력과 출력이 같으면 임시 파일을 거쳐 덮어쓴다
    tmp_path = output_path.with_name(f".{output_path.stem}.remux.mp4")

    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(input_path),
        "-c", "copy",
        "-movflags", "+faststart",
        str(tmp_path),
    ]

    result = subprocess.run(command)

    if result.returncode != 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg remux failed: exit code {result.returncode}")

    tmp_path.replace(output_path)
    return output_path
def _create_aria2_input(segments: list[str], input_file: Path) -> None:
    """aria2c 입력 파일을 생성한다."""
    lines = []

    for index, url in enumerate(segments):
        lines.append(f"{url}\n  out={index:06d}.part")

    input_file.write_text("\n".join(lines), encoding="utf-8")
def _merge_segments(segment_paths: list[Path], output_path: Path) -> None:
    """다운로드된 세그먼트를 순서대로 하나의 파일로 합친다."""
    with output_path.open("wb") as output_file:
        for segment_path in segment_paths:
            with segment_path.open("rb") as segment_file:
                while chunk := segment_file.read(CHUNK_SIZE):
                    output_file.write(chunk)
PERCENT_PATTERN = re.compile(r"\((\d+)%\)")
SPEED_PATTERN = re.compile(r"DL:(\S+?)[\s\]]")
_SIZE_UNITS = {"B": 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3}
_SINGLE_RE = re.compile(
    r"\[#\w+ (\S+?)/(\S+?)\((\d+)%\).*?DL:(\S+?)(?: ETA:(\S+?))?\]"
)
_SPEED_RE = re.compile(r"DL:([\d.]+)(GiB|MiB|KiB|B)")


def _bar(ratio: float, width: int = 30) -> str:
    ratio = max(0.0, min(1.0, ratio))
    filled = int(width * ratio)
    return "█" * filled + "░" * (width - filled)


def _total_speed(line: str) -> str:
    """요약 줄에 있는 모든 DL 값을 합산해서 보기 좋게 반환."""
    total = sum(float(v) * _SIZE_UNITS[u] for v, u in _SPEED_RE.findall(line))
    if total >= 1024**2:
        return f"{total / 1024**2:.1f}MiB/s"
    return f"{total / 1024:.0f}KiB/s"

def _fmt_time(sec: float) -> str:
    sec = int(max(sec, 0))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _run_aria2c(command: list[str], total_segments: int | None = None) -> int:
    """aria2c를 실행하고 진행 바를 한 줄로 그린다.

    total_segments가 None이면 단일 파일(퍼센트 기준),
    값이 있으면 세그먼트 완료 개수 기준으로 바를 그린다.
    ERROR/WARN 줄 외의 출력은 모두 무시한다.
    """
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    done = 0
    error_count = 0
    start = time.monotonic()
    last_draw = 0.0
    in_progress = False

    def draw(text: str) -> None:
        nonlocal in_progress
        width = min(shutil.get_terminal_size(fallback=(80, 20)).columns - 1, 100)
        print("\r" + text[:width].ljust(width), end="", flush=True)
        in_progress = True

    def segment_text() -> str:
        elapsed = time.monotonic() - start
        eta = (elapsed / done) * (total_segments - done) if done else 0
        eta_text = _fmt_time(eta) if done else "-"
        return (
            f"{_bar(done / total_segments)} {done}/{total_segments} "
            f"· 경과 {_fmt_time(elapsed)} · ETA {eta_text}"
        )

    for raw in proc.stdout:
        line = raw.strip()
        if not line:
            continue

        # 오류/경고만 바를 끊고 출력
        if "ERROR" in line or "WARN" in line:
            error_count += 1
            if in_progress:
                print()
                in_progress = False
            print(line, flush=True)
            continue

        # 세그먼트 모드: 완료 알림만 센다 (0.1초 간격으로만 다시 그림)
        if total_segments is not None:
            if "Download complete" in line:
                done += 1
                now = time.monotonic()
                if now - last_draw >= 0.1 or done == total_segments:
                    draw(segment_text())
                    last_draw = now
            continue

        # 단일 파일 모드: [# 진행 줄만 사용
        if line.startswith("[#"):
            m = _SINGLE_RE.search(line)
            if m:
                cur, total, pct, speed, eta = m.groups()
                draw(
                    f"{_bar(int(pct) / 100)} {pct}% "
                    f"{cur}/{total} {speed}/s ETA:{eta or '-'}"
                )

    proc.wait()

    # 마지막 상태를 한 번 더 그려서 최종 값이 보이게 함
    if total_segments is not None and done:
        draw(segment_text())
    if in_progress:
        print()
    if error_count:
        print(f"경고/오류 로그 {error_count}줄 발생")

    return proc.returncode

# def _run_aria2c(command: list[str], total_files: int | None = None) -> int:
#     """aria2c를 실행하며 진행률을 표시하고 종료 코드를 반환한다.

#     total_files가 있으면 완료된 파일 수로, 없으면 출력의 퍼센트 값으로 진행률을 갱신한다.
#     """
#     process = subprocess.Popen(
#         command,
#         stdout=subprocess.PIPE,
#         stderr=subprocess.STDOUT,
#         text=True,
#         bufsize=1,
#     )

#     if total_files:
#         bar = tqdm(total=total_files, unit="seg", desc="세그먼트")
#     else:
#         bar = tqdm(
#             total=100,
#             desc="다운로드",
#             bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}% [{elapsed}<{remaining}]{postfix}",
#         )

#     log_tail: list[str] = []

#     try:
#         for line in process.stdout:
#             line = line.strip()

#             if not line:
#                 continue

#             log_tail = (log_tail + [line])[-20:]

#             if total_files:
#                 if "Download complete" in line:
#                     bar.update(1)
#                 continue

#             percent = PERCENT_PATTERN.search(line)

#             if percent:
#                 bar.n = int(percent.group(1))
#                 bar.refresh()

#             speed = SPEED_PATTERN.search(line)

#             if speed:
#                 bar.set_postfix_str(f"{speed.group(1)}/s")

#         returncode = process.wait()

#         if returncode == 0:
#             bar.n = bar.total
#             bar.refresh()
#     finally:
#         bar.close()

#     if returncode != 0:
#         print("\n".join(log_tail))

#     return returncode
def convert_to_mp3(mp4_path: str) -> str:
    mp3_path = Path(mp4_path).with_suffix(".mp3")

    subprocess.run(
        ["ffmpeg", "-y", "-i", str(mp4_path), "-vn", "-acodec", "libmp3lame", "-q:a", "2", str(mp3_path)],
        check=True,
    )

    return str(mp3_path)
