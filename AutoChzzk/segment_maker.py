from dataclasses import dataclass
from urllib.parse import urljoin

import requests


@dataclass
class SegmentInfo:
    index: int
    url: str
    start_time: float
    duration: float


def make_segments_from_m3u8(m3u8_url: str, quality: str = "1080p",start_time: float | None = None,end_time: float | None = None,) -> list[SegmentInfo]:
    """Master m3u8에서 지정한 화질의 세그먼트 정보를 생성한다.

    start_time, end_time은 영상 시작 기준 초 단위이며, 둘 다 없으면 전체를 반환한다.
    """
    response = requests.get(m3u8_url, timeout=10)
    response.raise_for_status()

    media_url = _find_quality_playlist(response.text, m3u8_url, quality)

    response = requests.get(media_url, timeout=10)
    response.raise_for_status()

    segments = _parse_media_playlist(response.text, media_url)

    return filter_segments(segments, start_time, end_time)


def filter_segments(segments: list[SegmentInfo],start_time: float | None = None,end_time: float | None = None) -> list[SegmentInfo]:
    """지정한 구간(초)과 겹치는 세그먼트만 남긴다. init 세그먼트(index=-1)는 항상 유지한다."""
    if start_time is None and end_time is None:
        return segments

    start = 0.0 if start_time is None else start_time
    end = float("inf") if end_time is None else end_time

    if start < 0:
        raise ValueError("start_time must not be negative")
    if start >= end:
        raise ValueError("start_time must be less than end_time")

    selected = [
        segment
        for segment in segments
        if segment.index == -1
        or (segment.start_time + segment.duration > start and segment.start_time < end)
    ]

    if not any(segment.index != -1 for segment in selected):
        raise ValueError(f"No segments in range: {start_time} ~ {end_time}")

    return selected
def _find_quality_playlist(content: str, base_url: str, quality: str) -> str:
    """Master m3u8에서 지정한 화질의 Media Playlist URL을 찾는다."""
    lines = [line.strip() for line in content.splitlines() if line.strip()]

    for index, line in enumerate(lines):
        if line.startswith("#EXT-X-STREAM-INF:") and "RESOLUTION=" in line:
            resolution = line.split("RESOLUTION=", 1)[1].split(",", 1)[0]
            height = resolution.split("x")[1]

            if f"{height}p" == quality:
                return urljoin(base_url, lines[index + 1])

    raise ValueError(f"Quality not found: {quality}")


def _parse_media_playlist(content: str, base_url: str) -> list[SegmentInfo]:
    """Media Playlist에서 세그먼트 정보를 생성한다."""
    segments = []
    init_url = None
    current_time = 0.0
    duration = None

    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#EXT-X-MAP:"):
            uri = line.split('URI="', 1)[1].split('"', 1)[0]
            init_url = urljoin(base_url, uri)
            continue
        if line.startswith("#EXTINF:"):
            duration = float(line.split(":", 1)[1].split(",", 1)[0])
            continue
        if line.startswith("#"):
            continue
        if duration is None:
            continue
        segments.append(
            SegmentInfo(
                index=len(segments),
                url=urljoin(base_url, line),
                start_time=current_time,
                duration=duration,
            )
        )

        current_time += duration
        duration = None

    if init_url:
        segments.insert(0, SegmentInfo(index=-1, url=init_url, start_time=0.0, duration=0.0))

    return segments
import re


def make_segments_from_mpd(playback_url: str,quality: str = "1080p",start_time: float | None = None,end_time: float | None = None) -> list[SegmentInfo]:
    """플레이백 URL에서 지정한 화질의 TS 세그먼트 정보를 생성한다.

    start_time, end_time은 영상 시작 기준 초 단위이며, 둘 다 없으면 전체를 반환한다.
    """
    response = requests.get(playback_url, timeout=10)
    response.raise_for_status()

    adaptation, representation = _find_ts_representation(response.json(), quality)
    segments = _expand_segment_template(adaptation, representation)

    return filter_segments(segments, start_time, end_time)


def _find_ts_representation(mpd: dict, quality: str) -> tuple[dict, dict]:
    """video/mp2t AdaptationSet에서 지정한 화질의 Representation을 찾는다."""
    for period in mpd.get("period", []):
        for adaptation in period.get("adaptationSet", []):
            if adaptation.get("mimeType") != "video/mp2t":
                continue

            for representation in adaptation.get("representation", []):
                if f"{representation.get('height')}p" == quality:
                    return adaptation, representation

    raise ValueError(f"TS representation not found: {quality}")


def _expand_segment_template(adaptation: dict, representation: dict) -> list[SegmentInfo]:
    """SegmentTemplate + SegmentTimeline을 세그먼트 목록으로 풀어낸다."""
    template = representation.get("segmentTemplate") or adaptation.get("segmentTemplate")

    if not template or not template.get("segmentTimeline"):
        raise ValueError("segmentTemplate with segmentTimeline not found")

    base_urls = representation.get("baseURL") or adaptation.get("baseURL")

    if not base_urls:
        raise ValueError("baseURL not found")

    base_url = base_urls[0]["value"]
    timescale = template.get("timescale") or 1
    offset = template.get("presentationTimeOffset") or 0
    number = template["startNumber"] if template.get("startNumber") is not None else 1

    segments = []
    ticks = 0

    for entry in template["segmentTimeline"]["s"]:
        if entry.get("t") is not None:
            ticks = entry["t"]

        duration_ticks = entry["d"]
        repeat = entry.get("r") or 0

        for _ in range(repeat + 1):
            path = _fill_template(template["media"], representation["id"], number, ticks)

            segments.append(
                SegmentInfo(
                    index=len(segments),
                    url=urljoin(base_url, path),
                    start_time=(ticks - offset) / timescale,
                    duration=duration_ticks / timescale,
                )
            )

            ticks += duration_ticks
            number += 1

    return segments
def convert_to_mp3(mp4_path: str) -> str:
    mp3_path = Path(mp4_path).with_suffix(".mp3")

    subprocess.run(
        ["ffmpeg", "-y", "-i", str(mp4_path), "-vn", "-acodec", "libmp3lame", "-q:a", "2", str(mp3_path)],
        check=True,
    )

    return str(mp3_path)

def _fill_template(template: str, representation_id: str, number: int, ticks: int) -> str:
    """미디어 템플릿의 $RepresentationID$, $Number$, $Time$ 변수를 치환한다."""
    def replace_variable(match: re.Match) -> str:
        name, width = match.group(1), match.group(2)
        value = {"Number": number, "Time": ticks}[name]

        return str(value).zfill(int(width)) if width else str(value)

    result = template.replace("$RepresentationID$", representation_id)

    return re.sub(r"\$(Number|Time)(?:%0(\d+)d)?\$", replace_variable, result)
# m3u8_url = "https://ex-nlive-slitvod-streaming.navercdn.com/chzzk/kr/live_rewind/c/live_rewind_kr/bsk1anfuynnyahygv6i7izl5im2sti/vod_playlist.m3u8?hdnts=st=1791012102~exp=1791073312~acl=*/kr/*~hmac=0281628c532187f3d1df771e12df6f895a64de5372576150abc1125a704f15bf"

# source = make_segments_from_mpd(playback_url, "144p",start_time=0,end_time=600)


# download_segments(
#     [segment.url for segment in source],
#     "/content/downloader_test/test_60_120122.mp4",
# )
