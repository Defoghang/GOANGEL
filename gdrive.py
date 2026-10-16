"""GOANGEL — Client Google Drive (upload/download/resumable)."""
import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import (
    MediaFileUpload,
    MediaIoBaseDownload,
    MediaIoBaseUpload,
)

logger = logging.getLogger("goangel.gdrive")

SCOPES = ["https://www.googleapis.com/auth/drive"]
DEFAULT_CHUNK_SIZE = 10 * 1024 * 1024
MAX_RETRIES = 5
RETRY_BACKOFF = 2.0
DRIVE_FOLDER_MIME = "application/vnd.google-apps.folder"


class GDriveError(Exception):
    pass


class GDriveAuthError(GDriveError):
    pass


class GDriveUploadError(GDriveError):
    pass


class GDriveDownloadError(GDriveError):
    pass


def _load_credentials() -> Credentials:
    raw = os.environ.get("GDRIVE_CREDENTIALS_JSON", "").strip()
    if not raw:
        raw_b64 = os.environ.get("GDRIVE_CREDENTIALS_B64", "").strip()
        if raw_b64:
            import base64
            raw = base64.b64decode(raw_b64).decode("utf-8")
    if not raw:
        path = os.environ.get("GDRIVE_CREDENTIALS_PATH", "").strip()
        if path and Path(path).exists():
            raw = Path(path).read_text(encoding="utf-8")
    if not raw:
        raise GDriveAuthError(
            "Aucune credential Google Drive trouvée. "
            "Définir GDRIVE_CREDENTIALS_JSON ou GDRIVE_CREDENTIALS_B64 "
            "ou GDRIVE_CREDENTIALS_PATH."
        )
    try:
        info = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GDriveAuthError(f"JSON credentials invalide : {exc}") from exc
    try:
        return Credentials.from_service_account_info(info, scopes=SCOPES)
    except Exception as exc:
        raise GDriveAuthError(f"Credentials invalides : {exc}") from exc


def get_service():
    return build("drive", "v3", credentials=_load_credentials(), cache_discovery=False)


def ensure_folder(service, name: str, parent_id: Optional[str] = None) -> str:
    safe_name = name.replace("'", "\\'")
    query = f"mimeType='{DRIVE_FOLDER_MIME}' and name='{safe_name}' and trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    try:
        results = service.files().list(
            q=query,
            spaces="drive",
            fields="files(id, name)",
            pageSize=1,
        ).execute()
    except HttpError as exc:
        raise GDriveError(f"Recherche dossier échouée : {exc}") from exc
    files = results.get("files", [])
    if files:
        return files[0]["id"]
    metadata: Dict[str, Any] = {
        "name": name,
        "mimeType": DRIVE_FOLDER_MIME,
    }
    if parent_id:
        metadata["parents"] = [parent_id]
    try:
        folder = service.files().create(
            body=metadata, fields="id",
        ).execute()
    except HttpError as exc:
        raise GDriveError(f"Création dossier échouée : {exc}") from exc
    return folder["id"]


def find_file(
    service,
    name: str,
    parent_id: Optional[str] = None,
) -> Optional[str]:
    safe_name = name.replace("'", "\\'")
    query = f"name='{safe_name}' and trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    try:
        results = service.files().list(
            q=query,
            spaces="drive",
            fields="files(id, name, size, modifiedTime)",
            pageSize=10,
        ).execute()
    except HttpError as exc:
        raise GDriveError(f"Recherche fichier échouée : {exc}") from exc
    files = results.get("files", [])
    return files[0]["id"] if files else None


def _retry(fn, *args, **kwargs):
    last_exc: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except (HttpError, GDriveError, IOError) as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                sleep_s = RETRY_BACKOFF ** attempt
                logger.warning(
                    "gdrive_retry attempt=%d/%d err=%s sleep=%.1fs",
                    attempt, MAX_RETRIES, exc, sleep_s,
                )
                time.sleep(sleep_s)
    if last_exc:
        raise last_exc
    raise GDriveError("retry exhausted")


def upload_file(
    local_path: str,
    drive_name: Optional[str] = None,
    parent_id: Optional[str] = None,
    mimetype: Optional[str] = None,
    resumable: bool = True,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overwrite: bool = True,
) -> str:
    p = Path(local_path)
    if not p.exists():
        raise GDriveUploadError(f"Fichier introuvable : {local_path}")
    if not p.is_file():
        raise GDriveUploadError(f"Pas un fichier : {local_path}")
    service = get_service()
    name = drive_name or p.name
    existing_id = find_file(service, name, parent_id) if overwrite else None
    if mimetype is None:
        import mimetypes
        mimetype, _ = mimetypes.guess_type(str(p))
        mimetype = mimetype or "application/octet-stream"
    size_mb = p.stat().st_size / (1024 * 1024)
    logger.info(
        "gdrive_upload file=%s size_mb=%.2f resumable=%s parent=%s",
        name, size_mb, resumable, parent_id or "root",
    )
    media = MediaFileUpload(
        str(p), mimetype=mimetype, resumable=resumable, chunksize=chunk_size,
    )
    if existing_id:
        request = service.files().update(
            fileId=existing_id, media_body=media, fields="id",
        )
    else:
        metadata: Dict[str, Any] = {"name": name}
        if parent_id:
            metadata["parents"] = [parent_id]
        request = service.files().create(
            body=metadata, media_body=media, fields="id",
        )
    response = None
    attempts = 0
    while response is None:
        attempts += 1
        try:
            status, response = request.next_chunk()
            if status:
                logger.info(
                    "gdrive_upload_progress file=%s pct=%d%%",
                    name, int(status.progress() * 100),
                )
        except HttpError as exc:
            if attempts >= MAX_RETRIES:
                raise GDriveUploadError(
                    f"Upload échoué après {MAX_RETRIES} tentatives : {exc}"
                ) from exc
            sleep_s = RETRY_BACKOFF ** attempts
            logger.warning(
                "gdrive_upload_retry file=%s attempt=%d err=%s",
                name, attempts, exc,
            )
            time.sleep(sleep_s)
    file_id = response.get("id")
    logger.info("gdrive_upload_done file=%s id=%s", name, file_id)
    return file_id


def upload_bytes(
    data: bytes,
    drive_name: str,
    parent_id: Optional[str] = None,
    mimetype: str = "application/octet-stream",
    overwrite: bool = True,
) -> str:
    service = get_service()
    existing_id = find_file(service, drive_name, parent_id) if overwrite else None
    media = MediaIoBaseUpload(io.BytesIO(data), mimetype=mimetype, resumable=False)
    if existing_id:
        request = service.files().update(
            fileId=existing_id, media_body=media, fields="id",
        )
    else:
        metadata: Dict[str, Any] = {"name": drive_name}
        if parent_id:
            metadata["parents"] = [parent_id]
        request = service.files().create(
            body=metadata, media_body=media, fields="id",
        )
    response = request.execute()
    return response["id"]


def download_file(
    drive_id: str,
    local_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> str:
    service = get_service()
    p = Path(local_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    request = service.files().get_media(fileId=drive_id)
    fh = io.FileIO(str(p), "wb")
    downloader = MediaIoBaseDownload(fh, request, chunksize=chunk_size)
    done = False
    attempts = 0
    while not done:
        attempts += 1
        try:
            status, done = downloader.next_chunk()
            if status:
                logger.info(
                    "gdrive_download_progress id=%s pct=%d%%",
                    drive_id, int(status.progress() * 100),
                )
        except HttpError as exc:
            if attempts >= MAX_RETRIES:
                fh.close()
                raise GDriveDownloadError(
                    f"Download échoué après {MAX_RETRIES} : {exc}"
                ) from exc
            sleep_s = RETRY_BACKOFF ** attempts
            time.sleep(sleep_s)
    fh.close()
    logger.info("gdrive_download_done id=%s path=%s", drive_id, local_path)
    return str(p)


def download_by_name(
    drive_name: str,
    local_path: str,
    parent_id: Optional[str] = None,
) -> str:
    service = get_service()
    fid = find_file(service, drive_name, parent_id)
    if fid is None:
        raise GDriveDownloadError(f"Fichier introuvable : {drive_name}")
    return download_file(fid, local_path)


def list_files(parent_id: Optional[str] = None) -> List[Dict[str, Any]]:
    service = get_service()
    query = "trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    results = service.files().list(
        q=query,
        spaces="drive",
        fields="files(id, name, size, mimeType, modifiedTime, md5Checksum)",
        pageSize=200,
    ).execute()
    return results.get("files", [])


def delete_file(drive_id: str) -> None:
    service = get_service()
    try:
        service.files().delete(fileId=drive_id).execute()
    except HttpError as exc:
        raise GDriveError(f"Suppression échouée : {exc}") from exc


def file_exists(name: str, parent_id: Optional[str] = None) -> bool:
    return find_file(get_service(), name, parent_id) is not None


def get_folder_id() -> Optional[str]:
    return os.environ.get("GDRIVE_FOLDER_ID", "").strip() or None


def upload_release_artifacts(
    files: List[str],
    subfolder: Optional[str] = None,
) -> Dict[str, str]:
    parent = get_folder_id()
    if subfolder and parent:
        parent = ensure_folder(get_service(), subfolder, parent)
    results: Dict[str, str] = {}
    for f in files:
        p = Path(f)
        if not p.exists():
            logger.warning("gdrive_skip_missing file=%s", f)
            continue
        try:
            fid = upload_file(str(p), parent_id=parent)
            results[p.name] = fid
        except GDriveError as exc:
            logger.error("gdrive_upload_failed file=%s err=%s", f, exc)
    return results


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("usage: python3 gdrive.py <upload|list|download> ...", file=sys.stderr)
        sys.exit(2)
    cmd = sys.argv[1]
    if cmd == "upload" and len(sys.argv) >= 3:
        fid = upload_file(sys.argv[2], parent_id=get_folder_id())
        print(fid)
    elif cmd == "list":
        for f in list_files(get_folder_id()):
            print(f"{f['name']}\t{f.get('size','?')}\t{f['id']}")
    elif cmd == "download" and len(sys.argv) >= 4:
        p = download_by_name(sys.argv[2], sys.argv[3], get_folder_id())
        print(p)
    else:
        print("commande inconnue", file=sys.stderr)
        sys.exit(2)
