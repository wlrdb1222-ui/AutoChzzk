"""
utils.py
파이프라인 보조용 유틸 모음.
- csv 변환
- 토큰 수 카운트
"""

import json
import csv
import os
from pathlib import Path
from typing import Optional

from .llm_client import get_llm_client


# --------------------------------------------------
# 설정값
# --------------------------------------------------

TOKEN_COUNT_MODEL = "gemini-3.5-flash"  # 실제 분석에 쓰는 모델과 동일하게 맞춤


# --------------------------------------------------
# 1. JSON -> CSV 변환
# --------------------------------------------------

def _extract_records(data, list_key: Optional[str] = None):
    """JSON 데이터에서 레코드 리스트를 추출한다."""
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        if list_key:
            return data.get(list_key)

        list_keys = [k for k, v in data.items() if isinstance(v, list)]
        if len(list_keys) == 1:
            print(f"'{list_keys[0]}' 키에서 리스트를 찾았습니다.")
            return data[list_keys[0]]
        elif len(list_keys) > 1:
            raise ValueError(f"리스트 키가 여러 개입니다: {list_keys}. list_key 인자로 지정해주세요.")
        else:
            raise ValueError("리스트 형태의 데이터를 찾을 수 없습니다.")

    raise ValueError("지원하지 않는 JSON 구조입니다.")


def json_to_csv(
    json_path: Optional[str] = None,
    csv_path: Optional[str] = None,
    list_key: Optional[str] = None,
) -> Path:
    """
    JSON 파일을 CSV로 변환한다.
    - json_path가 없으면 input()으로 받는다.
    - csv_path가 없으면 json_path와 같은 이름으로 자동 생성한다.
    """
    if json_path is None:
        json_path = input("변환할 JSON 파일 경로를 입력하세요: ").strip()

    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = _extract_records(data, list_key)
    if not records:
        raise ValueError("리스트가 비어있습니다.")

    if csv_path is None:
        csv_path = json_path.with_suffix(".csv")
    else:
        csv_path = Path(csv_path)

    fieldnames = list(records[0].keys())

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(row)

    print(f"변환 완료: {json_path} -> {csv_path}")
    print(f"컬럼: {fieldnames}")
    print(f"행 수: {len(records)}")
    return csv_path


# --------------------------------------------------
# 2. 파일 토큰 수 카운트
# --------------------------------------------------

def count_file_tokens(path: Optional[str] = None) -> dict:
    """
    파일의 글자 수/토큰 수를 계산해 반환한다.
    - path가 없으면 input()으로 받는다.
    - 모델명은 모듈 상단 상수(TOKEN_COUNT_MODEL)를 직접 참조한다.
    """
    if path is None:
        path = input("토큰 수를 측정할 파일 경로를 입력하세요: ").strip()

    if not os.path.exists(path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="cp949") as f:
            text = f.read()

    client = get_llm_client()
    tokens = client.models.count_tokens(model=TOKEN_COUNT_MODEL, contents=text).total_tokens

    print(f"파일          : {path}")
    print(f"글자 수       : {len(text):,}")
    print(f"토큰 수       : {tokens:,}")

    return {"path": path, "chars": len(text), "tokens": tokens}


if __name__ == "__main__":
    print("1. json -> csv 변환")
    print("2. 파일 토큰 수 카운트")
    choice = input("실행할 작업 번호: ").strip()

    if choice == "1":
        json_to_csv()
    elif choice == "2":
        count_file_tokens()
    else:
        print("올바른 번호를 입력하세요.")
