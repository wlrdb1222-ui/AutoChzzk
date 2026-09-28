"""
api.py
Google Drive API 최소 인터페이스

Public
------
download_file(drive_path, local_path=None) -> str
upload(local_path, drive_path) -> str
exists(drive_path) -> bool
"""

import os
from pathlib import Path
from io import BytesIO

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

# --------------------------------------------------
# 설정
# --------------------------------------------------

SCOPES = ["https://www.googleapis.com/auth/drive"]
ROOT_FOLDER_ID = "YOUR_DRIVE_ROOT_FOLDER_ID"

# Colab / Kaggle 공용 임시 폴더
LOCAL_ROOT = Path(
    "/content" if os.path.exists("/content") else "/kaggle/working"
)


# --------------------------------------------------
# 인증
# --------------------------------------------------

def _get_service():
    try:
        # Colab
        from google.colab import userdata
        import json

        info = json.loads(userdata.get("GOOGLE_SERVICE_ACCOUNT"))

    except Exception:
        # Kaggle
        from kaggle_secrets import UserSecretsClient
        import json

        info = json.loads(
            UserSecretsClient().get_secret("GOOGLE_SERVICE_ACCOUNT")
        )

    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    return build("drive", "v3", credentials=creds)


# --------------------------------------------------
# 내부 유틸
# --------------------------------------------------

def _find(parent_id: str, name: str, mime_type=None):
    service = _get_service()

    query = (
        f"'{parent_id}' in parents and "
        f"name='{name}' and trashed=false"
    )

    if mime_type:
        query += f" and mimeType='{mime_type}'"

    result = service.files().list(
        q=query,
        fields="files(id,name,mimeType)"
    ).execute()

    files = result.get("files", [])
    return files[0] if files else None


def _resolve_path(drive_path: str):
    """Drive 상대경로 -> File 정보"""
    parts = Path(drive_path).parts

    current = ROOT_FOLDER_ID
    info = None

    for name in parts:
        info = _find(current, name)
        if not info:
            return None
        current = info["id"]

    return info


def _mkdirs(folder_path: str):
    """중간 폴더 생성"""
    service = _get_service()

    current = ROOT_FOLDER_ID

    if folder_path in ("", "."):
        return current

    for name in Path(folder_path).parts:
        folder = _find(
            current,
            name,
            "application/vnd.google-apps.folder"
        )

        if folder:
            current = folder["id"]
            continue

        meta = {
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [current],
        }

        folder = service.files().create(
            body=meta,
            fields="id"
        ).execute()

        current = folder["id"]

    return current


# --------------------------------------------------
# Public API
# --------------------------------------------------

def exists(drive_path: str) -> bool:
    """Drive 경로 존재 여부"""
    return _resolve_path(drive_path) is not None


def download_file(drive_path: str, local_path: str | None = None) -> str:
    """
    Drive -> Local

    Returns
    -------
    저장된 로컬 파일 경로
    """
    service = _get_service()

    info = _resolve_path(drive_path)

    if info is None:
        raise FileNotFoundError(drive_path)

    if local_path is None:
        local_path = str(LOCAL_ROOT / Path(drive_path).name)

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    request = service.files().get_media(fileId=info["id"])

    with open(local_path, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)

        done = False
        while not done:
            _, done = downloader.next_chunk()

    return local_path


def upload(local_path: str, drive_path: str) -> str:
    """
    Local -> Drive

    중간 폴더는 자동 생성
    기존 파일은 자동 덮어쓰기
    """
    service = _get_service()

    parent = _mkdirs(str(Path(drive_path).parent))
    filename = Path(drive_path).name

    media = MediaFileUpload(local_path, resumable=True)

    file = _find(parent, filename)

    if file:
        service.files().update(
            fileId=file["id"],
            media_body=media
        ).execute()
        return file["id"]

    created = service.files().create(
        body={
            "name": filename,
            "parents": [parent]
        },
        media_body=media,
        fields="id"
    ).execute()

    return created["id"]
