"""
chat_collector.py
치지직 VOD의 실시간 채팅을 수집해 구간별 통계로 집계하는 모듈.
- 채팅 원문은 저장하지 않고, 구간별 개수(count)만 csv로 저장한다.
"""

import csv
import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional
from .downloader import get_video_info_by_id

# --------------------------------------------------
# 설정값
# --------------------------------------------------

API_BASE = "https://api.chzzk.naver.com/service"
USER_AGENT = "Mozilla/5.0"
BUCKET_SEC = 30  # 집계 단위(초)
OUTPUT_DIR = "/content/chat_data"


# --------------------------------------------------
# 1. video_id 파싱
# --------------------------------------------------

def parse_video_id(vod_url_or_id) -> int:
    """URL이든 순수 ID든 받아서 정수 video_id로 변환한다."""
    try:
        return int(str(vod_url_or_id).split("?")[0].rstrip("/").split("/")[-1])
    except ValueError:
        raise ValueError(f"유효한 영상 ID/URL이 아닙니다: {vod_url_or_id}")


# --------------------------------------------------
# 2. 채팅 API 조회
# --------------------------------------------------

def _fetch_chat_page(video_id: int, player_message_time: int) -> dict:
    """(내부 전용) 채팅 API 한 페이지를 호출한다."""
    url = (
        f"{API_BASE}/v1/videos/{video_id}/chats"
        f"?playerMessageTime={player_message_time}&previousVideoChatSize=50"
    )
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Referer": "https://chzzk.naver.com/",
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as res:
            return json.load(res)
    except urllib.error.HTTPError as e:
        try:
            body = json.load(e)
        except ValueError:
            body = {"code": e.code, "message": str(e)}
        print("에러 응답:", body)
        return body


def fetch_all_chats(video_id: int) -> list[dict]:
    """playerMessageTime -> nextPlayerMessageTime 반복 호출로 전체 채팅을 수집한다."""
    chats = []
    player_message_time = 0

    while True:
        data = _fetch_chat_page(video_id, player_message_time)
        if data["code"] != 200:
            raise RuntimeError(f"채팅 조회 실패: {data.get('message')}")

        content = data["content"]
        chats.extend(content["videoChats"])

        if content["nextPlayerMessageTime"] is None:
            return chats
        player_message_time = content["nextPlayerMessageTime"]


# --------------------------------------------------
# 3. 구간별 집계
# --------------------------------------------------

def aggregate_by_bucket(chats: list[dict], bucket_sec: int = BUCKET_SEC) -> list[dict]:
    """bucket_sec 구간으로 잘라 원문은 버리고 구간별 채팅 개수만 남긴다."""
    counts = {}
    for chat in chats:
        ms = chat.get("playerMessageTime")
        if ms is None:
            continue
        bucket_start = (ms // 1000) // bucket_sec * bucket_sec
        counts[bucket_start] = counts.get(bucket_start, 0) + 1

    return [
        {"start": s, "end": s + bucket_sec, "count": counts[s]}
        for s in sorted(counts)
    ]


# --------------------------------------------------
# 4. 결과 저장 (csv - 토큰/용량 절약)
# --------------------------------------------------

def save_chat_data(video_id: int, video_title: str, buckets: list[dict]) -> Path:
    """구간별 채팅 통계를 csv로 저장하고 저장 경로를 반환한다."""
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    clean_title = video_title.replace("/", "_").replace("\\", "_")
    out_path = Path(OUTPUT_DIR) / f"{video_id}_{clean_title}_chatData.csv"

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["start", "end", "count"])
        writer.writeheader()
        writer.writerows(buckets)

    print(f"저장 완료: {out_path} (구간 수: {len(buckets)})")
    return out_path


# --------------------------------------------------
# 5. 조립 함수 (수동/자동 겸용)
# --------------------------------------------------




def collect_chat_data(
    video_id: Optional[str] = None,
    video_title: Optional[str] = None,
) -> Path:
    """
    VOD의 채팅 통계를 수집해 csv로 저장하고 그 경로를 반환한다.
    - video_id가 없으면 input()으로 받는다 (URL도 허용).
    - video_title이 없으면 video_id로 VOD 정보를 조회해 자동으로 채운다.
    """
    if video_id is None:
        video_id = input("영상 ID 또는 URL: ").strip()

    parsed_id = parse_video_id(video_id)

    if video_title is None:
        info = get_video_info_by_id(parsed_id)
        video_title = info["title"]
        print(f"제목 자동 조회: {video_title}")

    chats = fetch_all_chats(parsed_id)
    buckets = aggregate_by_bucket(chats)

    return save_chat_data(parsed_id, video_title, buckets)


if __name__ == "__main__":
    collect_chat_data()
