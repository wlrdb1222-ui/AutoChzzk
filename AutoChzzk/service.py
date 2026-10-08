# @title Service.py

import requests
import json
import re
from .new_downloader import download_mp4,download_segments,convert_to_mp3
from .segment_maker import make_segments_from_mpd,make_segments_from_m3u8,SegmentInfo



# video_no = 15491709 # 신
# video_no = 12070461 # 구
# video_no = 14962519 # m3u8

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/140.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
}
def parse_m3u8(data):
    liveRewindPlaybackJson = json.loads(data['content']['liveRewindPlaybackJson'])
    hls_url = liveRewindPlaybackJson['media'][0]['path']
    res = requests.get(hls_url, headers=headers, timeout=10)
    _RESOLUTION_PATTERN = re.compile(r"RESOLUTION=\d+x(\d+)")
    video_title = data['content']['videoTitle']

    heights = set()

    for line in res.text.splitlines():
        line = line.strip()

        if not line.startswith("#EXT-X-STREAM-INF:"):
            continue

        match = _RESOLUTION_PATTERN.search(line)
        if match:
            heights.add(int(match.group(1)))

    representations = []

    for height in sorted(heights, reverse=True):
        quality = f"{height}p"
        representations.append({
            "quality": quality,
            "id": quality,
            "url" : hls_url,
            "title": video_title
        })

    return representations

def parse_HLS(data):
    video_id = data['content']['videoId']
    in_key = data['content']['inKey']
    video_title = data['content']['videoTitle']
    playback_url = (
        f"https://apis.naver.com/neonplayer/vodplay/v2/"
        f"playback/{video_id}?key={in_key}"
    )
    response = requests.get(playback_url, headers=headers, timeout=10)
    response.raise_for_status()
    data = response.json()


    period = data['period'][0]
    adaptation_sets = period['adaptationSet']

    representations = []

    for adaptation_set in adaptation_sets:
        for rep in adaptation_set['representation']:
            representations.append({
                "quality": rep['any'][0]['value'],
                "id": rep['id'],
                "url": rep["baseURL"][0]["value"],
                "title":video_title
            })

    return representations

def normalize_formats(formats: list[dict]) -> list[dict]:
    options = []
    for item in formats:
        quality, item_id, url = item["quality"], item["id"], item["url"]

        if quality.startswith("MP4A"):
            media_type, normalized = "audio", None
        elif item_id.startswith("PD_"):
            media_type, normalized = "mp4", quality.split("P")[0] + "p"
        elif "vod_playlist.m3u8" in url:
            media_type, normalized = "m3u8", quality.lower()
        else:
            media_type, normalized = "hls", quality.split("P")[0] + "p"

        options.append({"type": media_type, "quality": normalized, "url": url})
    return options


def find_option(options: list[dict], quality: str, is_range: bool) -> dict:
    preferred = ("hls", "m3u8") if is_range else ("mp4", "m3u8")
    for media_type in preferred:
        for option in options:
            if option["type"] == media_type and option["quality"] == quality:
                return option
    raise ValueError(f"{quality} 화질을 {'구간' if is_range else '전체'} 다운로드할 수 없습니다.")


def make_file_name(title: str, quality: str = None, start=None, end=None) -> str:
    title = re.sub(r'[\\/:*?"<>|]', "_", title).strip()

    if quality is None:
        return f"{title}_오디오"

    name = f"{title}_{quality}"
    if start is not None or end is not None:
        name += f"_{int(start or 0)}_{'끝' if end is None else int(end)}"
    return name


def build_audio_selection(options: list[dict], title: str) -> dict:
    file_name = make_file_name(title)

    for option in options:
        if option["type"] == "audio":
            return {"type": "audio", "title": file_name, "url": option["url"]}

    for source_type, audio_type in (("m3u8", "audio_m3u8"), ("mp4", "audio_hls")):
        for option in options:
            if option["type"] == source_type and option["quality"] == "144p":
                return {"type": audio_type, "title": file_name, "quality": "144p", "url": option["url"]}

    raise ValueError("오디오 소스를 찾을 수 없습니다.")


def ask_range() -> dict:
    while True:
        start = input("startTime(sec, 빈칸=처음부터) : ").strip()
        end = input("endTime(sec, 빈칸=끝까지) : ").strip()

        time_range = {}
        if start:
            time_range["start"] = float(start)
        if end:
            time_range["end"] = float(end)

        if not time_range:
            print("start, end 중 하나는 입력해야 합니다.")
            continue
        if time_range.get("start", 0) < 0 or time_range.get("end", float("inf")) <= time_range.get("start", 0):
            print("범위가 올바르지 않습니다.")
            continue
        return time_range

def ask_selection(formats: list[dict]) -> dict:
    title = formats[0]["title"]
    options = normalize_formats(formats)

    qualities = sorted(
        {option["quality"] for option in options if option["quality"]},
        key=lambda quality: int(quality[:-1]),
        reverse=True,
    )
    for number, quality in enumerate(qualities, 1):
        print(f"{number}. {quality}")
    audio_number = len(qualities) + 1
    print(f"{audio_number}. 오디오")

    choice = int(input("번호 : "))
    if choice == audio_number:
        return build_audio_selection(options, title)

    quality = qualities[choice - 1]
    time_range = ask_range() if input("구간 다운로드? (y/n) : ") == "y" else {}

    option = find_option(options, quality, is_range=bool(time_range))
    media_type = "hls" if option["type"] == "mp4" else option["type"]
    return {
        "type": media_type,
        "title": make_file_name(title, quality, **time_range),
        "quality": quality,
        "url": option["url"],
        **time_range,
    }

def process_selection(selection: dict) -> str:
    type_ = selection["type"]
    url = selection["url"]
    title = selection["title"]
    quality = selection.get("quality")
    start = selection.get("start")
    end = selection.get("end")
    print("process")
    match type_:
        case "audio":
            return download_mp4(url, title+".mp3")

        case "audio_hls":
            return download_mp4(url, title+".mp3")

        case "audio_m3u8":
            return download_segments(make_segments_from_m3u8(url, quality), title+".mp3")

        case "hls":
            if start is None and end is None:
                return download_mp4(url, title+".mp4")
            return download_segments(make_segments_from_mpd(url, quality, start, end), title+".mp4")

        case "m3u8":
            return download_segments(make_segments_from_m3u8(url, quality, start, end), title+".mp4")

        case _:
            raise ValueError(f"알 수 없는 type: {type_}")
def extract_qualities(items: list[dict]) -> list[str]:
    """파서 결과(dict 리스트)에서 화질만 뽑아 '1080p' 형식으로, 높이 내림차순으로 반환한다."""
    heights = set()

    for item in items:
        match = re.match(r"(\d+)[pP]", item["quality"])
        if match:
            heights.add(int(match.group(1)))

    return [f"{height}p" for height in sorted(heights, reverse=True)]


def service():

    video_no = input("video_no 입력:")
    url = f"https://api.chzzk.naver.com/service/v2/videos/{video_no}"
    res = requests.get(url, headers=headers)
    data = res.json()
    # print(json.dumps(data, ensure_ascii=False, indent=2))
    vod_status = data['content']['vodStatus']
    formats = []
    if vod_status == "ABR_HLS" :
        formats = parse_HLS(data)
    else :
        formats = parse_m3u8(data)

    qualities = extract_qualities(formats)   # items = 파서가 돌려준 dict 리스트(mpd면 mp4+TS+오디오 전부)
    selection = ask_selection(formats)
    print(process_selection(selection))

# service(15491709)
# 15491709
# 14962519
# 13139369
