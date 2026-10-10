import os
import time
import pickle
import lzma
import logging
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("goangel.checkpoint")

DEFAULT_CHECKPOINT_DIR = "checkpoints"
DEFAULT_GITHUB_REPO = "Angeloda444/GOANGELCLOUD"
DEFAULT_GITHUB_USER = "Angeloda444"
_PUSH_INTERVAL_SECONDS = 30.0
_COMPRESS_THRESHOLD_MB = 20.0
_MAX_FILE_MB = 900.0
_GIT_ADD_TIMEOUT = 600
_GIT_PUSH_TIMEOUT = 1200


class CheckpointManager:
    def __init__(
        self,
        checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
        github_token: Optional[str] = None,
        github_user: str = DEFAULT_GITHUB_USER,
        github_repo: str = DEFAULT_GITHUB_REPO,
        auto_push: bool = True,
        verbose: bool = True,
    ):
        self.dir = Path(checkpoint_dir).resolve()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.github_token = github_token
        self.github_user = github_user
        self.github_repo = github_repo
        self.auto_push = auto_push and bool(github_token)
        self.verbose = verbose
        self._lock = threading.Lock()
        self._pending_push = False
        self._push_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._push_count = 0
        self._skip_save_count = 0
        self._push_in_progress = False
        self._push_lock = threading.Lock()

        if self.auto_push:
            self._start_push_worker()
            if self.verbose:
                print("☁️ Push GitHub automatique activé", flush=True)

    def _file_path(self, key: str) -> Path:
        plain = self.dir / f"{key}.pkl"
        compressed = self.dir / f"{key}.pkl.xz"
        if plain.exists():
            return plain
        if compressed.exists():
            return compressed
        return plain

    def save(self, key: str, value: Any, sync_push: bool = False) -> bool:
        plain_path = self.dir / f"{key}.pkl"
        compressed_path = self.dir / f"{key}.pkl.xz"
        tmp_path = self.dir / f"{key}.tmp"

        try:
            with self._lock:
                with open(tmp_path, "wb") as f:
                    pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
                    f.flush()
                    try:
                        os.fsync(f.fileno())
                    except OSError:
                        pass

            raw_size_mb = tmp_path.stat().st_size / (1024 * 1024)

            if raw_size_mb > _MAX_FILE_MB:
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
                self._skip_save_count += 1
                if self.verbose:
                    print(
                        f"⚠️ {key} trop gros ({raw_size_mb:.0f} MB > "
                        f"{_MAX_FILE_MB:.0f} MB) — non sauvegardé",
                        flush=True,
                    )
                return False

            if raw_size_mb > _COMPRESS_THRESHOLD_MB:
                with lzma.open(compressed_path, "wb", preset=6) as f:
                    pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
                tmp_path.unlink()
                if plain_path.exists():
                    plain_path.unlink()
                final_size_mb = compressed_path.stat().st_size / (1024 * 1024)
                if self.verbose:
                    print(
                        f"💾 Checkpoint : {key}.xz "
                        f"({raw_size_mb:.0f} MB → {final_size_mb:.0f} MB)",
                        flush=True,
                    )
            else:
                os.replace(tmp_path, plain_path)
                if compressed_path.exists():
                    compressed_path.unlink()
                if self.verbose:
                    print(
                        f"💾 Checkpoint : {key} ({raw_size_mb:.1f} MB)",
                        flush=True,
                    )

            logger.info("Checkpoint saved: %s", key)

            if self.auto_push:
                self._pending_push = True
                if sync_push:
                    self._push_to_github()
            return True

        except Exception as exc:
            logger.error("Erreur sauvegarde %s : %s", key, exc)
            if self.verbose:
                print(f"❌ Checkpoint {key} : {exc}", flush=True)
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            return False

    def load(self, key: str) -> Optional[Any]:
        plain_path = self.dir / f"{key}.pkl"
        compressed_path = self.dir / f"{key}.pkl.xz"

        if plain_path.exists():
            try:
                with open(plain_path, "rb") as f:
                    return pickle.load(f)
            except Exception as exc:
                logger.warning("Checkpoint %s illisible : %s", key, exc)

        if compressed_path.exists():
            try:
                with lzma.open(compressed_path, "rb") as f:
                    return pickle.load(f)
            except Exception as exc:
                logger.warning("Checkpoint %s.xz illisible : %s", key, exc)

        return None

    def clear(self) -> None:
        count = 0
        for f in self.dir.glob("*"):
            try:
                f.unlink()
                count += 1
            except OSError:
                pass
        if self.verbose:
            print(f"🗑️ {count} checkpoint(s) effacé(s)", flush=True)

    def _start_push_worker(self):
        def worker():
            while not self._stop.is_set():
                time.sleep(_PUSH_INTERVAL_SECONDS)
                if self._pending_push:
                    with self._lock:
                        self._pending_push = False
                    try:
                        self._push_to_github()
                    except Exception as exc:
                        logger.warning("Push error : %s", exc)

        self._push_thread = threading.Thread(target=worker, daemon=True)
        self._push_thread.start()

    def _push_to_github(self) -> bool:
        if not self.github_token:
            return False
        if self._push_in_progress:
            return False
        self._push_in_progress = True
        try:
            try:
                subprocess.run(
                    ["git", "lfs", "install", "--local"],
                    check=False, capture_output=True, timeout=30,
                )
            except Exception:
                pass

            try:
                subprocess.run(
                    ["git", "add", "-A", str(self.dir)],
                    check=False, capture_output=True,
                    timeout=_GIT_ADD_TIMEOUT,
                )
            except subprocess.TimeoutExpired:
                if self.verbose:
                    print("⚠️ git add timeout", flush=True)
                return False

            r = subprocess.run(
                ["git", "diff", "--cached", "--name-only"],
                check=False, capture_output=True, text=True,
            )
            staged = [x for x in r.stdout.split("\n") if x.strip()]
            if not staged:
                return False

            subprocess.run(
                ["git", "commit", "-m", f"checkpoint {int(time.time())}"],
                check=False, capture_output=True, timeout=120,
            )

            url = (
                f"https://{self.github_user}:{self.github_token}"
                f"@github.com/{self.github_repo}.git"
            )
            try:
                result = subprocess.run(
                    ["git", "push", url, "main"],
                    check=False, capture_output=True, text=True,
                    timeout=_GIT_PUSH_TIMEOUT,
                )
            except subprocess.TimeoutExpired:
                if self.verbose:
                    print("⚠️ git push timeout", flush=True)
                return False

            if result.returncode == 0:
                self._push_count += 1
                if self.verbose:
                    print(
                        f"☁️ Push OK ({len(staged)} fichier(s))",
                        flush=True,
                    )
                return True
            if self.verbose:
                print(f"⚠️ Push : {result.stderr[:200]}", flush=True)
            return False
        except Exception as exc:
            logger.warning("Push exception : %s", exc)
            return False
        finally:
            self._push_in_progress = False

    def finalize(self) -> None:
        self._stop.set()
        if self._pending_push and self.auto_push:
            self._push_to_github()
        if self.verbose:
            print(
                f"☁️ {self._push_count} push(s) GitHub, "
                f"{self._skip_save_count} fichier(s) ignoré(s)",
                flush=True,
            )

    def status(self) -> Dict[str, Any]:
        files = []
        for f in sorted(self.dir.glob("*")):
            try:
                size_mb = f.stat().st_size / (1024 * 1024)
            except OSError:
                size_mb = 0
            files.append({"name": f.name, "size_mb": round(size_mb, 2)})
        return {
            "dir": str(self.dir),
            "auto_push": self.auto_push,
            "push_count": self._push_count,
            "files": files,
        }


def list_checkpoints(checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR) -> Dict[str, Any]:
    mgr = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        auto_push=False, verbose=False,
    )
    return mgr.status()


def clear_checkpoints(checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR) -> None:
    mgr = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        auto_push=False, verbose=True,
    )
    mgr.clear()
