import base64
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("goangel.gdrive_sync")

RCLONE_REMOTE = os.environ.get("RCLONE_REMOTE", "gdrive")
RCLONE_FOLDER = os.environ.get("RCLONE_FOLDER", "GOANGEL")
RCLONE_CONF_PATH = Path("/tmp/rclone.conf")
RCLONE_BIN = "rclone"
SYNC_TIMEOUT = 3600
COPY_RETRIES = 3
RETRY_BACKOFF = 5.0


class GDriveSyncError(Exception):
    pass


def _ensure_rclone() -> str:
    if shutil.which(RCLONE_BIN):
        return RCLONE_BIN
    install_script = "/tmp/install-rclone.sh"
    try:
        subprocess.run(
            ["bash", "-c", f"curl -sS https://rclone.org/install.sh > {install_script}"],
            check=True, timeout=60,
        )
        subprocess.run(
            ["bash", "-c", f"sudo bash {install_script} >/dev/null 2>&1 || bash {install_script} >/dev/null 2>&1"],
            check=True, timeout=300,
        )
    except Exception as exc:
        raise GDriveSyncError(f"rclone install échoué : {exc}") from exc
    if not shutil.which(RCLONE_BIN):
        raise GDriveSyncError("rclone introuvable après installation")
    return RCLONE_BIN


def _ensure_config() -> Path:
    if RCLONE_CONF_PATH.exists():
        return RCLONE_CONF_PATH

    b64 = os.environ.get("RCLONE_CONFIG", "").strip()
    if b64:
        try:
            data = base64.b64decode(b64)
            RCLONE_CONF_PATH.parent.mkdir(parents=True, exist_ok=True)
            RCLONE_CONF_PATH.write_bytes(data)
            RCLONE_CONF_PATH.chmod(0o600)
            logger.info("rclone config décodée : %s bytes", len(data))
            return RCLONE_CONF_PATH
        except Exception as exc:
            raise GDriveSyncError(f"Décodage RCLONE_CONFIG échoué : {exc}") from exc

    env_cfg = os.environ.get("RCLONE_CONFIG_RAW", "").strip()
    if env_cfg:
        RCLONE_CONF_PATH.parent.mkdir(parents=True, exist_ok=True)
        RCLONE_CONF_PATH.write_text(env_cfg, encoding="utf-8")
        RCLONE_CONF_PATH.chmod(0o600)
        return RCLONE_CONF_PATH

    default_path = Path.home() / ".config" / "rclone" / "rclone.conf"
    if default_path.exists():
        shutil.copy2(default_path, RCLONE_CONF_PATH)
        RCLONE_CONF_PATH.chmod(0o600)
        return RCLONE_CONF_PATH

    raise GDriveSyncError(
        "Aucune config rclone. Définir RCLONE_CONFIG (base64) ou RCLONE_CONFIG_RAW."
    )


def _run(args: List[str], timeout: int = SYNC_TIMEOUT, check: bool = True) -> subprocess.CompletedProcess:
    bin_path = _ensure_rclone()
    conf = _ensure_config()
    full_args = [bin_path, "--config", str(conf)] + args
    logger.info("rclone_cmd : %s", " ".join(full_args[:6]) + " ...")
    result = subprocess.run(full_args, capture_output=True, text=True, timeout=timeout)
    if check and result.returncode != 0:
        raise GDriveSyncError(
            f"rclone failed (RC={result.returncode}) : {result.stderr[:400]}"
        )
    return result


def remote_path(sub: str = "") -> str:
    base = f"{RCLONE_REMOTE}:{RCLONE_FOLDER}"
    if sub:
        return f"{base}/{sub.lstrip('/')}"
    return base


def ping() -> bool:
    try:
        _run(["lsd", remote_path(), "--max-depth", "1"], timeout=60)
        return True
    except Exception as exc:
        logger.warning("rclone ping échoué : %s", exc)
        return False


def ensure_remote() -> None:
    try:
        _run(["mkdir", remote_path(), "--parents"], check=False)
    except Exception:
        pass


def upload_file(local: str, remote_sub: str = "", retries: int = COPY_RETRIES) -> bool:
    p = Path(local)
    if not p.exists():
        logger.warning("upload_file : introuvable %s", local)
        return False
    dest = remote_path(remote_sub) if remote_sub else remote_path()
    for attempt in range(1, retries + 1):
        try:
            _run([
                "copy", str(p), dest,
                "--progress", "--stats-one-line",
                "--transfers", "4", "--checkers", "8",
            ], timeout=SYNC_TIMEOUT)
            logger.info("upload OK : %s -> %s", p.name, dest)
            return True
        except Exception as exc:
            logger.warning("upload tentative %d échouée : %s", attempt, exc)
            if attempt < retries:
                time.sleep(RETRY_BACKOFF * attempt)
    return False


def download_file(remote_name: str, local: str, retries: int = COPY_RETRIES) -> bool:
    src = remote_path(remote_name)
    p = Path(local)
    p.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            _run([
                "copy", src, str(p.parent),
                "--progress", "--stats-one-line",
                "--transfers", "4", "--checkers", "8",
            ], timeout=SYNC_TIMEOUT)
            if p.exists():
                logger.info("download OK : %s (%d bytes)", p.name, p.stat().st_size)
                return True
            return False
        except Exception as exc:
            logger.warning("download tentative %d échouée : %s", attempt, exc)
            if attempt < retries:
                time.sleep(RETRY_BACKOFF * attempt)
    return False


def download_dir(remote_sub: str, local_dir: str, retries: int = COPY_RETRIES) -> bool:
    src = remote_path(remote_sub)
    p = Path(local_dir)
    p.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            _run([
                "copy", src, str(p),
                "--progress", "--stats-one-line",
                "--transfers", "4", "--checkers", "8",
            ], timeout=SYNC_TIMEOUT)
            logger.info("download_dir OK : %s -> %s", src, local_dir)
            return True
        except Exception as exc:
            logger.warning("download_dir tentative %d échouée : %s", attempt, exc)
            if attempt < retries:
                time.sleep(RETRY_BACKOFF * attempt)
    return False


def upload_dir(local_dir: str, remote_sub: str, retries: int = COPY_RETRIES) -> bool:
    p = Path(local_dir)
    if not p.exists():
        logger.warning("upload_dir : introuvable %s", local_dir)
        return False
    dest = remote_path(remote_sub)
    for attempt in range(1, retries + 1):
        try:
            _run([
                "copy", str(p), dest,
                "--progress", "--stats-one-line",
                "--transfers", "4", "--checkers", "8",
                "--exclude", "*.tmp",
                "--exclude", "goangel.log",
                "--exclude", "training.log",
            ], timeout=SYNC_TIMEOUT)
            logger.info("upload_dir OK : %s -> %s", local_dir, dest)
            return True
        except Exception as exc:
            logger.warning("upload_dir tentative %d échouée : %s", attempt, exc)
            if attempt < retries:
                time.sleep(RETRY_BACKOFF * attempt)
    return False


def sync_up(local_dir: str, remote_sub: str = "cache", delete: bool = False) -> bool:
    p = Path(local_dir)
    if not p.exists():
        logger.warning("sync_up : introuvable %s", local_dir)
        return False
    dest = remote_path(remote_sub)
    args = [
        "sync", str(p), dest,
        "--progress", "--stats-one-line",
        "--transfers", "4", "--checkers", "8",
        "--exclude", "*.tmp",
        "--exclude", "goangel.log",
        "--exclude", "training.log",
        "--exclude", "checkpoints/**",
    ]
    if not delete:
        args.append("--ignore-existing")
    try:
        _run(args, timeout=SYNC_TIMEOUT)
        logger.info("sync_up OK : %s -> %s", local_dir, dest)
        return True
    except Exception as exc:
        logger.warning("sync_up échoué : %s", exc)
        return False


def sync_down(remote_sub: str, local_dir: str, delete: bool = False) -> bool:
    p = Path(local_dir)
    p.mkdir(parents=True, exist_ok=True)
    src = remote_path(remote_sub)
    args = [
        "sync", src, str(p),
        "--progress", "--stats-one-line",
        "--transfers", "4", "--checkers", "8",
    ]
    if not delete:
        args.append("--ignore-existing")
    try:
        _run(args, timeout=SYNC_TIMEOUT)
        logger.info("sync_down OK : %s -> %s", src, local_dir)
        return True
    except Exception as exc:
        logger.warning("sync_down échoué : %s", exc)
        return False


def list_remote(remote_sub: str = "") -> List[Dict[str, Any]]:
    path = remote_path(remote_sub)
    try:
        result = _run([
            "lsjson", path,
            "--files-only", "--no-mimetype", "--no-modtime",
        ], timeout=120, check=False)
        import json
        try:
            return json.loads(result.stdout)
        except Exception:
            return []
    except Exception as exc:
        logger.warning("list_remote échoué : %s", exc)
        return []


def remote_file_exists(remote_sub: str) -> bool:
    path = remote_path(remote_sub)
    try:
        result = _run(["lsf", path], timeout=60, check=False)
        return bool(result.stdout.strip())
    except Exception:
        return False


def delete_remote(remote_sub: str) -> bool:
    path = remote_path(remote_sub)
    try:
        _run(["delete", path], timeout=300, check=False)
        return True
    except Exception:
        return False


def get_remote_size_bytes(remote_sub: str = "") -> int:
    path = remote_path(remote_sub)
    try:
        result = _run(["size", path, "--json"], timeout=120, check=False)
        import json
        data = json.loads(result.stdout)
        return int(data.get("bytes", 0))
    except Exception:
        return 0


def restore_cache(local_cache: str = "./cache", verbose: bool = True) -> bool:
    p = Path(local_cache)
    p.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"  📥 Restauration cache depuis Drive ({remote_path('cache')})...", flush=True)
    ok_models = download_dir("cache/stages/models", str(p / "stages" / "models"))
    ok_final = download_dir("cache/stages/finalize", str(p / "stages" / "finalize"))
    ok_pb = download_dir("cache/stages/pb_convergence", str(p / "stages" / "pb_convergence"))
    ok_top = download_dir("cache", str(p))
    return any([ok_models, ok_final, ok_pb, ok_top])


def backup_cache(local_cache: str = "./cache", verbose: bool = True) -> bool:
    p = Path(local_cache)
    if not p.exists():
        logger.warning("backup_cache : cache absent")
        return False
    if verbose:
        total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        print(f"  📤 Upload cache vers Drive ({total/1024/1024:.1f} MB)...", flush=True)
    ensure_remote()
    ok = sync_up(str(p), "cache", delete=False)
    if verbose and ok:
        size = get_remote_size_bytes("cache")
        print(f"  ✅ Drive : {size/1024/1024:.1f} MB total", flush=True)
    return ok


def cleanup_drive_old(keep_days: int = 7) -> int:
    logger.info("cleanup_drive_old : %d jours", keep_days)
    return 0


def status() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "rclone_available": bool(shutil.which(RCLONE_BIN)),
        "config_path": str(RCLONE_CONF_PATH),
        "config_exists": RCLONE_CONF_PATH.exists(),
        "remote_path": remote_path(),
        "connected": False,
        "size_bytes": 0,
    }
    try:
        info["connected"] = ping()
        if info["connected"]:
            info["size_bytes"] = get_remote_size_bytes()
    except Exception:
        pass
    return info


def cli_main(argv: List[str]) -> int:
    logging.basicConfig(level=logging.INFO)
    if len(argv) < 2:
        print("usage: gdrive_sync.py <status|restore|backup|upload|download|list> [args]", file=sys.stderr)
        return 2
    cmd = argv[1]
    if cmd == "status":
        import json
        print(json.dumps(status(), indent=2))
        return 0
    if cmd == "restore":
        return 0 if restore_cache() else 1
    if cmd == "backup":
        return 0 if backup_cache() else 1
    if cmd == "upload" and len(argv) >= 4:
        return 0 if upload_file(argv[2], argv[3]) else 1
    if cmd == "download" and len(argv) >= 4:
        return 0 if download_file(argv[2], argv[3]) else 1
    if cmd == "list":
        import json
        print(json.dumps(list_remote(argv[2] if len(argv) > 2 else ""), indent=2))
        return 0
    print(f"commande inconnue : {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(cli_main(sys.argv))
