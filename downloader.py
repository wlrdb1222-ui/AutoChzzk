"""
downloader.py
치지직(CHZZK) VOD 다운로드 모듈.
- 오디오/비디오 공용
- 실제 다운로드 실행 로직(fetch_and_save)은 별도 분리되어 있어
  나중에 requests -> yt-dlp/aria2c 등으로 교체 가능
"""

import re
import time
from pathlib import Path
from typing import Optional

import requests


DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
}


# --------------------------------------------------
# 1. VOD 정보 조회
# --------------------------------------------------

def get_video_info(vod_url: str) -> dict:
    """VOD URL로부터 video_no, video_id, in_key, title 등을 조회한다."""
    match = re.search(r"/video/(\d+)", vod_url)
    if not match:
        raise ValueError(f"VOD URL에서 video_no를 찾을 수 없습니다: {vod_url}")

    video_no = match.group(1)
    info_url = f"https://api.chzzk.naver.com/service/v2/videos/{video_no}"

    response = requests.get(info_url, headers=DEFAULT_HEADERS, timeout=10)
    response.raise_for_status()
    content = response.json()["content"]

    return {
        "video_no": video_no,
        "video_id": content["videoId"],
        "in_key": content["inKey"],
        "title": content["videoTitle"],
    }


# --------------------------------------------------
# 2. 사용 가능한 포맷 조회
# --------------------------------------------------

def get_available_formats(video_id: str, in_key: str) -> list[dict]:
    """playback API를 호출해 다운로드 가능한 포맷 목록을 반환한다."""
    playback_url = (
        f"https://apis.naver.com/neonplayer/vodplay/v2/"
        f"playback/{video_id}?key={in_key}"
    )

    response = requests.get(playback_url, headers=DEFAULT_HEADERS, timeout=10)
    response.raise_for_status()
    data = response.json()

    formats = []
    for period in data.get("period", []):
        for adaptation in period.get("adaptationSet", []):
            for rep in adaptation.get("representation", []):

                # HLS(UUID) 등 mimeType 없는 항목 제외
                if rep.get("mimeType") is None:
                    continue

                base_urls = rep.get("baseURL", [])
                if not base_urls:
                    continue

                url = base_urls[0].get("value")
                if not url:
                    continue

                formats.append({
                    "id": rep.get("id"),
                    "width": rep.get("width"),
                    "height": rep.get("height"),
                    "bitrate": rep.get("bandwidth"),
                    "codec": rep.get("codecs"),
                    "mime": rep.get("mimeType"),
                    "url": url,
                })

    if not formats:
        raise RuntimeError("사용 가능한 포맷을 찾을 수 없습니다.")

    return formats


# --------------------------------------------------
# 3. 포맷 선택
# --------------------------------------------------

def _print_format_list(formats: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("사용 가능한 포맷")
    print("=" * 80)

    for i, f in enumerate(formats, 1):
        name = f"{f['height']}p" if f["height"] else "Audio"
        bitrate = f"{f['bitrate'] / 1000:.0f} kbps" if f["bitrate"] else "-"
        resolution = (
            f"{f['width']}x{f['height']}"
            if f["width"] and f["height"]
            else "-"
        )
        print(
            f"{i:2}. "
            f"{name:>6} | "
            f"{resolution:<11} | "
            f"{bitrate:>10} | "
            f"{f['mime'] or '-':<10} | "
            f"{f['id']}"
        )


def select_format(
    formats: list[dict],
    target_height: Optional[int] = None,
    audio_only: bool = False,
) -> dict:
    """
    포맷을 선택한다.
    - audio_only=True: height가 없는(오디오) 포맷 중 첫 번째를 선택
    - target_height 지정: 해당 height와 일치하는 포맷 탐색 (없으면 에러)
    - 둘 다 없으면: 목록을 출력하고 input()으로 수동 선택
    """
    if audio_only:
        audio_formats = [f for f in formats if f["height"] is None]
        if not audio_formats:
            raise RuntimeError("오디오 전용 포맷을 찾을 수 없습니다.")
        return audio_formats[0]

    if target_height is not None:
        matched = [f for f in formats if f["height"] == target_height]
        if not matched:
            available = sorted({f["height"] for f in formats if f["height"]})
            raise RuntimeError(
                f"{target_height}p 포맷을 찾을 수 없습니다. "
                f"사용 가능한 해상도: {available}"
            )
        return matched[0]

    # 수동 선택
    _print_format_list(formats)
    while True:
        try:
            choice = int(input("\n다운로드할 포맷 번호: "))
            if 1 <= choice <= len(formats):
                return formats[choice - 1]
            print("올바른 번호를 입력하세요.")
        except ValueError:
            print("숫자를 입력하세요.")


# --------------------------------------------------
# 4. 출력 경로 결정
# --------------------------------------------------

def build_output_path(title: str, selected_format: dict, output_dir: str = ".") -> Path:
    """파일명(특수문자 제거)과 확장자를 결정해 최종 저장 경로를 반환한다."""
    clean_title = re.sub(r'[\\/:*?"<>|]', '', title)

    if selected_format["height"]:
        format_name = f"{selected_format['height']}p"
        extension = ".mp4"
    else:
        format_name = "audio"
        extension = ".m4a"

    filename = f"{clean_title}_{format_name}{extension}"
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    return output_path


# --------------------------------------------------
# 5. 실제 다운로드 실행 (교체 대상)
# --------------------------------------------------

def fetch_and_save(
    url: str,
    output_path: Path,
    headers: Optional[dict] = None,
) -> Path:
    """
    실제 파일을 다운로드해서 저장한다.
    이 함수만 나중에 requests -> yt-dlp/aria2c 등으로 교체하면 된다.
    계약: url과 저장 경로를 받아서, 최종 저장된 Path를 반환한다.
    """
    if headers is None:
        headers = DEFAULT_HEADERS

    with requests.get(url, headers=headers, stream=True, timeout=30) as r:
        r.raise_for_status()

        total = int(r.headers.get("Content-Length", 0))
        downloaded = 0
        start_time = time.time()

        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue

                f.write(chunk)
                downloaded += len(chunk)

                elapsed = time.time() - start_time
                speed_mb = (downloaded / elapsed / 1024 / 1024) if elapsed > 0 else 0
                downloaded_mb = downloaded / 1024 / 1024

                if total:
                    total_mb = total / 1024 / 1024
                    percent = downloaded / total * 100
                    print(
                        f"\r진행률: {percent:6.2f}% | "
                        f"{downloaded_mb:8.2f} / {total_mb:8.2f} MB | "
                        f"{speed_mb:6.2f} MB/s",
                        end=""
                    )
                else:
                    print(
                        f"\r다운로드: {downloaded_mb:.2f} MB | "
                        f"속도: {speed_mb:.2f} MB/s",
                        end=""
                    )

    print(f"\n\n다운로드 완료: {output_path}")
    return output_path


# --------------------------------------------------
# 6. 범용 조립 함수 (수동/자동 겸용)
# --------------------------------------------------

def download(
    vod_url: Optional[str] = None,
    target_height: Optional[int] = None,
    audio_only: bool = False,
    output_dir: str = ".",
) -> Path:
    """
    치지직 VOD를 다운로드한다. (오디오/비디오 공용)
    - vod_url이 없으면 input()으로 받는다.
    - target_height/audio_only가 둘 다 없으면 포맷을 수동 선택한다.
    """
    if vod_url is None:
        vod_url = input("VOD URL: ").strip()

    info = get_video_info(vod_url)
    print(f"VOD: {info['video_no']}")
    print(f"제목: {info['title']}")

    formats = get_available_formats(info["video_id"], info["in_key"])
    selected = select_format(formats, target_height=target_height, audio_only=audio_only)

    print("\n선택한 포맷")
    print("-" * 40)
    print("ID:", selected["id"])
    print("해상도:", selected["width"], "x", selected["height"])
    print("코덱:", selected["codec"])

    output_path = build_output_path(info["title"], selected, output_dir)
    print("저장 위치:", output_path)

    return fetch_and_save(selected["url"], output_path)


# --------------------------------------------------
# 7. 파이프라인 전용 진입점 (오디오 자동 다운로드)
# --------------------------------------------------

def audio_down(vod_url: Optional[str] = None, output_dir: str = ".") -> Path:
    """오디오만 자동으로 다운로드하는 파이프라인 진입점."""
    return download(vod_url=vod_url, audio_only=True, output_dir=output_dir)


if __name__ == "__main__":
    download()
