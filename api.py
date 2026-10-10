import os
import time
import json
import pickle
import shutil
import logging
import hashlib
import threading
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from itertools import cycle
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import (
    tokens,
    TOP_60_LEAGUES,
    API_BASE_URL,
    API_TIMEOUT_SECONDS,
    API_MAX_RETRIES,
    API_RETRY_BACKOFF_SECONDS,
    API_RATE_LIMIT_SLEEP_SECONDS,
    CACHE_DIR,
    HISTORICAL_SEASONS_LOOKBACK,
    stage_print,
)


logger = logging.getLogger("goangel.api")


class BzzoiroAPIError(Exception):
    pass


_RATE_LIMIT_PER_MINUTE: int = 10
_RATE_LIMIT_SAFETY_MARGIN: int = 1
_RATE_LIMIT_WINDOW_SECONDS: float = 60.0
_RATE_LIMIT_POLL_INTERVAL_SECONDS: float = 0.1
_MAX_REQUESTS_HISTORY_PER_TOKEN: int = 512

_API_PAGE_LIMIT: int = 100
_API_MAX_PAGE_LIMIT: int = 200
_API_PAGE_SLEEP_SECONDS: float = 0.15

_MAX_PARALLEL_WORKERS: int = max(1, min(len(tokens), 8))

_CHECKPOINT_DIR: Path = CACHE_DIR / "checkpoints"
_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
_CHECKPOINT_STATE_FILE: Path = _CHECKPOINT_DIR / "state.json"
_CHECKPOINT_RAW_FILE: Path = _CHECKPOINT_DIR / "raw_matches.jsonl"
_CHECKPOINT_SAVE_INTERVAL_SECONDS: float = 30.0
_CHECKPOINT_PUSH_INTERVAL_SECONDS: float = 300.0
_CHECKPOINT_MAX_FILE_MB: float = 200.0

_GITHUB_PUSH_ENABLED: bool = os.getenv("GOANGEL_GITHUB_PUSH", "false").lower() in ("1", "true", "yes", "on")
_GITHUB_REPO: str = os.getenv("GOANGEL_GITHUB_REPO", "Angeloda444/GOANGELCLOUD")
_GITHUB_USER: str = os.getenv("GOANGEL_GITHUB_USER", "Angeloda444")
_GITHUB_TOKEN: str = os.getenv("GH_PAT", "") or os.getenv("GITHUB_TOKEN", "")
_GITHUB_BRANCH: str = os.getenv("GOANGEL_GITHUB_BRANCH", "cache-auto")

_STATUS_MAP: Dict[str, str] = {
    "finished": "FINISHED",
    "ft": "FINISHED",
    "match_finished": "FINISHED",
    "live": "IN_PLAY",
    "in_play": "IN_PLAY",
    "upcoming": "SCHEDULED",
    "scheduled": "SCHEDULED",
    "notstarted": "NOT_STARTED",
    "not_started": "NOT_STARTED",
    "not started": "NOT_STARTED",
    "notstart": "NOT_STARTED",
    "ns": "NOT_STARTED",
    "timed": "TIMED",
    "tbd": "TBD",
    "postponed": "POSTPONED",
    "cancelled": "CANCELLED",
    "canceled": "CANCELLED",
    "suspended": "SUSPENDED",
    "awarded": "AWARDED",
}

STAT_KEYS: Tuple[str, ...] = (
    "xg",
    "ball_possession",
    "total_shots",
    "shots_on_target",
    "corner_kicks",
    "yellow_cards",
    "red_cards",
    "fouls",
    "pass_accuracy_pct",
    "big_chances",
    "big_chances_missed",
    "goalkeeper_saves",
    "duels",
    "aerial_duels:pct",
    "ground_duels:pct",
    "crosses:pct",
    "long_balls:pct",
    "tackles_won",
    "dribbles:pct",
    "interceptions",
    "clearances",
    "recoveries",
    "goals_prevented",
    "shots_inside_box",
    "blocked_shots",
    "dangerous_attack_pct",
    "final_third_entries",
    "touches_in_penalty_area",
    "big_chances_scored",
    "expected_goals_on_target",
    "offsides",
    "throw_ins",
    "free_kicks",
    "goal_kicks",
    "high_claims",
    "total_saves",
    "big_saves",
    "punches",
    "dispossessed",
    "hit_woodwork",
    "through_balls",
    "total_tackles",
    "tackles",
    "passes",
    "accurate_passes",
    "shots_off_target",
    "shots_outside_box",
    "number_of_sprints",
    "attack",
    "ball_safe",
    "attack_pct",
    "ball_safe_pct",
    "errors_lead_to_a_shot",
    "fouled_in_final_third",
    "crosses:value",
    "crosses:total",
    "long_balls:value",
    "long_balls:total",
    "dribbles:value",
    "dribbles:total",
    "aerial_duels:value",
    "aerial_duels:total",
    "ground_duels:value",
    "ground_duels:total",
    "final_third_phase:value",
    "final_third_phase:pct",
)

STAT_KEY_MAP: Dict[str, str] = {
    "xg": "xg",
    "ball_possession": "ball_possession",
    "total_shots": "total_shots",
    "shots_on_target": "shots_on_target",
    "corner_kicks": "corner_kicks",
    "yellow_cards": "yellow_cards",
    "red_cards": "red_cards",
    "fouls": "fouls",
    "pass_accuracy_pct": "pass_accuracy_pct",
    "big_chances": "big_chances",
    "big_chances_missed": "big_chances_missed",
    "goalkeeper_saves": "goalkeeper_saves",
    "duels": "duels",
    "aerial_duels:pct": "aerial_pct",
    "ground_duels:pct": "ground_pct",
    "crosses:pct": "crosses_pct",
    "long_balls:pct": "long_pct",
    "tackles_won": "tackles_won",
    "dribbles:pct": "dribbles_pct",
    "interceptions": "interceptions",
    "clearances": "clearances",
    "recoveries": "recoveries",
    "goals_prevented": "goals_prev",
    "shots_inside_box": "shots_inside",
    "blocked_shots": "blocked",
    "dangerous_attack_pct": "danger_pct",
    "final_third_entries": "final_third",
    "touches_in_penalty_area": "pen_touches",
    "big_chances_scored": "big_scored",
    "expected_goals_on_target": "xg_target",
    "offsides": "offsides",
    "throw_ins": "throw_ins",
    "free_kicks": "free_kicks",
    "goal_kicks": "goal_kicks",
    "high_claims": "high_claims",
    "total_saves": "total_saves",
    "big_saves": "big_saves",
    "punches": "punches",
    "dispossessed": "dispossessed",
    "hit_woodwork": "hit_woodwork",
    "through_balls": "through_balls",
    "total_tackles": "total_tackles",
    "tackles": "tackles",
    "passes": "passes",
    "accurate_passes": "accurate_passes",
    "shots_off_target": "shots_off_target",
    "shots_outside_box": "shots_outside_box",
    "number_of_sprints": "number_of_sprints",
    "attack": "attack",
    "ball_safe": "ball_safe",
    "attack_pct": "attack_pct",
    "ball_safe_pct": "ball_safe_pct",
    "errors_lead_to_a_shot": "errors_lead_to_shot",
    "fouled_in_final_third": "fouled_final_third",
    "crosses:value": "crosses_value",
    "crosses:total": "crosses_total",
    "long_balls:value": "long_balls_value",
    "long_balls:total": "long_balls_total",
    "dribbles:value": "dribbles_value",
    "dribbles:total": "dribbles_total",
    "aerial_duels:value": "aerial_duels_value",
    "aerial_duels:total": "aerial_duels_total",
    "ground_duels:value": "ground_duels_value",
    "ground_duels:total": "ground_duels_total",
    "final_third_phase:value": "final_third_value",
    "final_third_phase:pct": "final_third_pct",
}


def _safe_float_or_none(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v:
        return None
    if v in (float("inf"), float("-inf")):
        return None
    return v


def _safe_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        if isinstance(value, float):
            if value != value or not value.is_integer():
                return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_goals(value: Any) -> Optional[int]:
    v = _safe_int(value)
    if v is None:
        return None
    if v < 0 or v > 30:
        return None
    return v


def _extract_team_name(team: Any) -> Optional[str]:
    if isinstance(team, str):
        s = team.strip()
        return s if s else None
    if isinstance(team, dict):
        for field in ("name", "shortName", "tla"):
            v = team.get(field)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None


def _normalize_status(value: Any) -> str:
    if not isinstance(value, str):
        return "UNKNOWN"
    key = value.strip().lower()
    return _STATUS_MAP.get(key, value.strip().upper())


def _read_stat(side_data: Any, key: str) -> Optional[float]:
    if not isinstance(side_data, dict):
        return None
    sub_key = "value"
    if ":" in key:
        key, sub_key = key.split(":", 1)
    v = side_data.get(key)
    if isinstance(v, dict):
        if sub_key in v:
            return _safe_float_or_none(v[sub_key])
        for sub in ("actual", "value", "pct"):
            if sub in v:
                return _safe_float_or_none(v[sub])
        return None
    if isinstance(v, (int, float)):
        return _safe_float_or_none(v)
    return None


class CheckpointManager:
    def __init__(
        self,
        checkpoint_dir: Path = _CHECKPOINT_DIR,
        save_interval: float = _CHECKPOINT_SAVE_INTERVAL_SECONDS,
        push_interval: float = _CHECKPOINT_PUSH_INTERVAL_SECONDS,
        github_enabled: bool = _GITHUB_PUSH_ENABLED,
        github_repo: str = _GITHUB_REPO,
        github_user: str = _GITHUB_USER,
        github_token: str = _GITHUB_TOKEN,
        github_branch: str = _GITHUB_BRANCH,
    ):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.dir / "state.json"
        self.raw_file = self.dir / "raw_matches.jsonl"
        self.meta_file = self.dir / "meta.json"
        self.save_interval = float(save_interval)
        self.push_interval = float(push_interval)
        self.github_enabled = bool(github_enabled and github_token)
        self.github_repo = github_repo
        self.github_user = github_user
        self.github_token = github_token
        self.github_branch = github_branch
        self._lock = threading.Lock()
        self._last_save = 0.0
        self._last_push = 0.0
        self._push_count = 0
        self._save_count = 0
        self._stop = threading.Event()
        self._pending_push = False
        if self.github_enabled:
            t = threading.Thread(target=self._push_worker, daemon=True)
            t.start()
            logger.info("Checkpoint GitHub push activé : %s/%s", github_user, github_repo)

    def _safe_write_json(self, path: Path, data: Dict[str, Any]) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp, path)

    def _load_json(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Checkpoint %s illisible : %s", path, exc)
            return None

    def load_state(self) -> Dict[str, Any]:
        data = self._load_json(self.state_file)
        if not isinstance(data, dict):
            return {}
        return data

    def load_meta(self) -> Dict[str, Any]:
        data = self._load_json(self.meta_file)
        if not isinstance(data, dict):
            return {}
        return data

    def load_raw_matches(self) -> List[Dict[str, Any]]:
        matches: List[Dict[str, Any]] = []
        if not self.raw_file.exists():
            return matches
        try:
            with open(self.raw_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            matches.append(obj)
                    except json.JSONDecodeError:
                        continue
        except Exception as exc:
            logger.warning("Erreur lecture raw_matches : %s", exc)
        return matches

    def get_done_seasons(self) -> Set[Tuple[int, int]]:
        state = self.load_state()
        done = state.get("done_seasons", [])
        result: Set[Tuple[int, int]] = set()
        if isinstance(done, list):
            for item in done:
                if isinstance(item, list) and len(item) == 2:
                    lid = _safe_int(item[0])
                    sid = _safe_int(item[1])
                    if lid is not None and sid is not None:
                        result.add((lid, sid))
        return result

    def append_season(
        self,
        lid: int,
        sid: int,
        yr: int,
        matches: List[Dict[str, Any]],
    ) -> None:
        with self._lock:
            if not matches:
                return
            try:
                with open(self.raw_file, "a", encoding="utf-8") as f:
                    for m in matches:
                        if isinstance(m, dict):
                            f.write(json.dumps(m, ensure_ascii=False, default=str) + "\n")
                    f.flush()
                    try:
                        os.fsync(f.fileno())
                    except OSError:
                        pass
            except Exception as exc:
                logger.error("Erreur append season %d/%d : %s", lid, sid, exc)
                return

            state = self.load_state()
            done = state.get("done_seasons", [])
            if not isinstance(done, list):
                done = []
            entry = [int(lid), int(sid), int(yr)]
            done.append(entry)
            state["done_seasons"] = done
            state["last_update"] = datetime.now(timezone.utc).isoformat()
            state["total_seasons_done"] = len(done)
            self._safe_write_json(self.state_file, state)

            self._save_count += 1
            self._last_save = time.monotonic()
            self._pending_push = True

    def save_meta(self, meta: Dict[str, Any]) -> None:
        with self._lock:
            try:
                self._safe_write_json(self.meta_file, meta)
            except Exception as exc:
                logger.warning("Erreur save meta : %s", exc)

    def clear(self) -> None:
        for f in (self.state_file, self.raw_file, self.meta_file):
            try:
                if f.exists():
                    f.unlink()
            except OSError:
                pass

    def _push_worker(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(self.push_interval)
            if self._stop.is_set():
                break
            if self._pending_push:
                self._pending_push = False
                try:
                    self._push_to_github()
                except Exception as exc:
                    logger.warning("Push worker erreur : %s", exc)

    def _push_to_github(self) -> bool:
        if not self.github_enabled:
            return False
        if not self.github_token:
            return False
        repo_url = f"https://{self.github_user}:{self.github_token}@github.com/{self.github_repo}.git"
        try:
            tmp = Path("/tmp/goangel-checkpoint-push")
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True, exist_ok=True)

            env = os.environ.copy()
            env["GIT_LFS_SKIP_SMUDGE"] = "1"

            subprocess.run(
                ["git", "init", "-q", "-b", self.github_branch],
                cwd=str(tmp), check=False, env=env,
                capture_output=True, timeout=60,
            )
            subprocess.run(
                ["git", "config", "user.email", "kaggle@goangel.local"],
                cwd=str(tmp), check=False, env=env,
                capture_output=True, timeout=30,
            )
            subprocess.run(
                ["git", "config", "user.name", "GOANGEL Auto"],
                cwd=str(tmp), check=False, env=env,
                capture_output=True, timeout=30,
            )

            dest = tmp / "checkpoints"
            dest.mkdir(parents=True, exist_ok=True)
            for f in (self.state_file, self.raw_file, self.meta_file):
                if f.exists():
                    shutil.copy2(f, dest / f.name)

            subprocess.run(
                ["git", "add", "-A"],
                cwd=str(tmp), check=False, env=env,
                capture_output=True, timeout=120,
            )
            subprocess.run(
                ["git", "commit", "-m", f"checkpoint {int(time.time())}"],
                cwd=str(tmp), check=False, env=env,
                capture_output=True, timeout=120,
            )
            r = subprocess.run(
                ["git", "push", "-f", repo_url, f"{self.github_branch}:{self.github_branch}"],
                cwd=str(tmp), check=False, env=env,
                capture_output=True, text=True, timeout=600,
            )
            if r.returncode == 0:
                self._push_count += 1
                logger.info("Checkpoint poussé sur GitHub (%d)", self._push_count)
                return True
            logger.warning("Push GitHub échoué : %s", (r.stderr or "")[:200])
            return False
        except subprocess.TimeoutExpired:
            logger.warning("Push GitHub timeout")
            return False
        except Exception as exc:
            logger.warning("Push GitHub exception : %s", exc)
            return False

    def finalize(self) -> None:
        self._stop.set()
        if self._pending_push and self.github_enabled:
            try:
                self._push_to_github()
            except Exception:
                pass
        logger.info(
            "Checkpoint bilan : %d save, %d push",
            self._save_count, self._push_count,
        )

    def status(self) -> Dict[str, Any]:
        state = self.load_state()
        meta = self.load_meta()
        return {
            "dir": str(self.dir),
            "save_count": self._save_count,
            "push_count": self._push_count,
            "github_enabled": self.github_enabled,
            "done_seasons": len(state.get("done_seasons", [])),
            "last_update": state.get("last_update"),
            "raw_file_exists": self.raw_file.exists(),
            "raw_file_mb": (
                round(self.raw_file.stat().st_size / (1024 * 1024), 2)
                if self.raw_file.exists() else 0
            ),
            "meta": meta,
        }


class TokenManager:
    def __init__(self, token_list: List[str]):
        if not token_list:
            raise BzzoiroAPIError("Aucun token disponible")
        self._lock = threading.Lock()
        self.tokens = list(token_list)
        self._pool = cycle(self.tokens)
        self._current = next(self._pool)
        self._cooldowns: Dict[str, float] = {t: 0.0 for t in self.tokens}
        self._request_counts: Dict[str, List[float]] = {t: [] for t in self.tokens}

    def get_next(self) -> str:
        with self._lock:
            attempts = 0
            now = time.monotonic()
            token = next(self._pool)
            while self._cooldowns.get(token, 0.0) > now and attempts < len(self.tokens):
                token = next(self._pool)
                attempts += 1
            self._current = token
            return self._current

    def mark_rate_limited(self, token: str, retry_after_seconds: float) -> None:
        with self._lock:
            self._cooldowns[token] = time.monotonic() + max(float(retry_after_seconds), 1.0)

    def register_request(self, token: str) -> None:
        with self._lock:
            now = time.monotonic()
            history = self._request_counts.setdefault(token, [])
            history.append(now)
            cutoff = now - _RATE_LIMIT_WINDOW_SECONDS
            pruned = [t for t in history if t >= cutoff]
            if len(pruned) > _MAX_REQUESTS_HISTORY_PER_TOKEN:
                pruned = pruned[-_MAX_REQUESTS_HISTORY_PER_TOKEN:]
            self._request_counts[token] = pruned

    def requests_last_minute(self, token: str) -> int:
        with self._lock:
            now = time.monotonic()
            history = self._request_counts.get(token, [])
            cutoff = now - _RATE_LIMIT_WINDOW_SECONDS
            return sum(1 for t in history if t >= cutoff)

    def wait_if_needed(self, token: str, max_per_minute: int) -> None:
        if max_per_minute <= 0:
            return
        effective_limit = max(1, max_per_minute - _RATE_LIMIT_SAFETY_MARGIN)
        while True:
            if self._cooldowns.get(token, 0.0) > time.monotonic():
                time.sleep(_RATE_LIMIT_POLL_INTERVAL_SECONDS)
                continue
            if self.requests_last_minute(token) < effective_limit:
                return
            time.sleep(_RATE_LIMIT_POLL_INTERVAL_SECONDS)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            cutoff = now - _RATE_LIMIT_WINDOW_SECONDS
            return {
                "count": len(self.tokens),
                "cooldowns": {
                    f"{t[:6]}***": max(0.0, self._cooldowns.get(t, 0.0) - now)
                    for t in self.tokens
                },
                "requests_last_minute": {
                    f"{t[:6]}***": sum(
                        1 for r in self._request_counts.get(t, []) if r >= cutoff
                    )
                    for t in self.tokens
                },
            }


token_manager = TokenManager(tokens)

_session_local = threading.local()


def _get_session() -> requests.Session:
    session = getattr(_session_local, "session", None)
    if session is None:
        session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=32,
            pool_maxsize=32,
            max_retries=Retry(total=0, connect=0, read=0, redirect=2, status=0),
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        _session_local.session = session
    return session


def _parse_retry_after(resp: requests.Response, default_seconds: float) -> float:
    header_value = (
        resp.headers.get("Retry-After")
        or resp.headers.get("X-RequestCounter-Reset")
    )
    if header_value is None:
        return default_seconds
    try:
        return max(float(header_value), 0.0)
    except (TypeError, ValueError):
        return default_seconds


def make_request(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    retries: int = API_MAX_RETRIES,
    respect_rate_limit: bool = True,
) -> Optional[Dict[str, Any]]:
    if retries < 1:
        retries = 1
    last_error: Optional[str] = None

    for attempt in range(retries):
        current_token = token_manager.get_next()
        if respect_rate_limit:
            token_manager.wait_if_needed(current_token, _RATE_LIMIT_PER_MINUTE)
        headers = {"Authorization": f"Token {current_token}"}
        session = _get_session()

        try:
            resp = session.get(
                url,
                headers=headers,
                params=params,
                timeout=API_TIMEOUT_SECONDS,
            )
            token_manager.register_request(current_token)
        except requests.exceptions.Timeout as exc:
            last_error = f"Timeout: {exc}"
            logger.warning("Timeout sur %s (tentative %d/%d)", url, attempt + 1, retries)
            if attempt < retries - 1:
                time.sleep(API_RETRY_BACKOFF_SECONDS * (attempt + 1))
                continue
            break
        except requests.exceptions.ConnectionError as exc:
            last_error = f"ConnectionError: {exc}"
            logger.warning(
                "Erreur de connexion sur %s (tentative %d/%d)",
                url, attempt + 1, retries,
            )
            if attempt < retries - 1:
                time.sleep(API_RETRY_BACKOFF_SECONDS * (attempt + 1))
                continue
            break
        except requests.exceptions.RequestException as exc:
            last_error = f"RequestException: {exc}"
            logger.error("Erreur requête sur %s : %s", url, exc)
            if attempt < retries - 1:
                time.sleep(API_RETRY_BACKOFF_SECONDS * (attempt + 1))
                continue
            break

        if resp.status_code == 200:
            try:
                return resp.json()
            except ValueError as exc:
                last_error = f"JSONDecodeError: {exc}"
                logger.error("Réponse JSON invalide depuis %s : %s", url, exc)
                if attempt < retries - 1:
                    time.sleep(API_RETRY_BACKOFF_SECONDS * (attempt + 1))
                    continue
                break

        if resp.status_code == 429:
            retry_after = _parse_retry_after(resp, API_RATE_LIMIT_SLEEP_SECONDS)
            token_manager.mark_rate_limited(current_token, retry_after)
            logger.warning(
                "Rate limit sur %s, attente %.1fs (tentative %d/%d)",
                url, retry_after, attempt + 1, retries,
            )
            time.sleep(retry_after)
            continue

        if resp.status_code in (401, 403):
            last_error = f"Auth error {resp.status_code}: {resp.text[:200]}"
            logger.error(
                "Erreur d'authentification (%d) sur %s avec token %s***",
                resp.status_code, url, current_token[:6],
            )
            if attempt < retries - 1:
                time.sleep(API_RETRY_BACKOFF_SECONDS * (attempt + 1))
                continue
            break

        if resp.status_code == 404:
            logger.info("Ressource introuvable (404) : %s", url)
            return None

        if 500 <= resp.status_code < 600:
            last_error = f"ServerError {resp.status_code}: {resp.text[:200]}"
            logger.warning(
                "Erreur serveur %d sur %s (tentative %d/%d)",
                resp.status_code, url, attempt + 1, retries,
            )
            if attempt < retries - 1:
                time.sleep(API_RETRY_BACKOFF_SECONDS * (attempt + 1))
                continue
            break

        last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
        logger.error(
            "Réponse inattendue %d sur %s : %s",
            resp.status_code, url, resp.text[:200],
        )
        if attempt < retries - 1:
            time.sleep(API_RETRY_BACKOFF_SECONDS * (attempt + 1))
            continue
        break

    logger.error(
        "Échec définitif de la requête vers %s après %d tentatives : %s",
        url, retries, last_error,
    )
    return None


def _validate_date_str(date_str: str) -> bool:
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except (TypeError, ValueError):
        return False


def _paginate(
    url: str,
    base_params: Dict[str, Any],
    per_page: int = _API_PAGE_LIMIT,
) -> List[Dict[str, Any]]:
    per_page = max(1, min(int(per_page), _API_MAX_PAGE_LIMIT))
    all_items: List[Dict[str, Any]] = []
    seen_ids: Set[int] = set()
    offset = 0
    total_count: Optional[int] = None
    page_index = 0

    while True:
        params = dict(base_params)
        params["limit"] = per_page
        params["offset"] = offset

        data = make_request(url, params=params)
        if not isinstance(data, dict):
            break

        chunk = data.get("results", [])
        if not isinstance(chunk, list) or not chunk:
            break

        added = 0
        for m in chunk:
            if not isinstance(m, dict):
                continue
            mid = _safe_int(m.get("id"))
            if mid is not None:
                if mid in seen_ids:
                    continue
                seen_ids.add(mid)
            all_items.append(m)
            added += 1

        if total_count is None:
            c = data.get("count")
            if isinstance(c, int) and c > 0:
                total_count = c

        if added == 0:
            break
        if len(chunk) < per_page:
            break
        if total_count is not None and len(all_items) >= total_count:
            break

        offset += per_page
        page_index += 1
        if page_index > 0:
            time.sleep(_API_PAGE_SLEEP_SECONDS)

    return all_items


def transform_bzzoiro_match(raw: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    lid = _safe_int(raw.get("league_id"))
    league_name = TOP_60_LEAGUES.get(lid, f"League {lid}") if lid is not None else "Unknown"

    hs = _safe_goals(raw.get("home_score"))
    a_s = _safe_goals(raw.get("away_score"))
    hs_ht = _safe_goals(raw.get("home_score_ht"))
    as_ht = _safe_goals(raw.get("away_score_ht"))

    home_xg = raw.get("stat_home_xg")
    away_xg = raw.get("stat_away_xg")
    if isinstance(home_xg, dict):
        home_xg = home_xg.get("actual")
    if isinstance(away_xg, dict):
        away_xg = away_xg.get("actual")

    home_name = _extract_team_name(raw.get("home_team"))
    away_name = _extract_team_name(raw.get("away_team"))

    out = {
        "id": _safe_int(raw.get("id")),
        "event_id": _safe_int(raw.get("id")),
        "utcDate": raw.get("event_date") or raw.get("date") or "",
        "status": _normalize_status(raw.get("status")),
        "homeTeam": {"id": _safe_int(raw.get("home_team_id")), "name": home_name},
        "awayTeam": {"id": _safe_int(raw.get("away_team_id")), "name": away_name},
        "score": {
            "fullTime": {"home": hs, "away": a_s},
            "halfTime": {"home": hs_ht, "away": as_ht},
        },
        "competition": {
            "code": str(lid) if lid is not None else None,
            "name": league_name,
        },
        "home_xg": _safe_float_or_none(home_xg),
        "away_xg": _safe_float_or_none(away_xg),
        "has_xg": raw.get("has_xg"),
        "is_derby": raw.get("is_local_derby"),
        "is_neutral": raw.get("is_neutral_ground"),
        "travel_km": raw.get("travel_distance_km"),
        "attendance": raw.get("attendance"),
        "referee_id": raw.get("referee_id"),
        "venue_id": raw.get("venue_id"),
        "weather": raw.get("weather"),
        "pitch": raw.get("pitch_condition"),
        "round_number": raw.get("round_number"),
        "stage": raw.get("stage"),
    }

    for key in STAT_KEYS:
        clean_key = STAT_KEY_MAP.get(key, key)
        h = raw.get(f"stat_home_{clean_key}")
        a = raw.get(f"stat_away_{clean_key}")
        if h is not None:
            out[f"home_{clean_key}"] = _safe_float_or_none(h)
        if a is not None:
            out[f"away_{clean_key}"] = _safe_float_or_none(a)

    return out


def get_league_seasons(league_id: int) -> List[Dict[str, Any]]:
    url = f"{API_BASE_URL}/leagues/{int(league_id)}/seasons/"
    data = make_request(url)
    if not isinstance(data, dict):
        return []
    seasons = data.get("seasons", [])
    if not isinstance(seasons, list):
        return []
    return [s for s in seasons if isinstance(s, dict)]


def get_league_info(league_id: int) -> Optional[Dict[str, Any]]:
    url = f"{API_BASE_URL}/leagues/{int(league_id)}/"
    return make_request(url)


def get_all_leagues(per_page: int = _API_PAGE_LIMIT) -> List[Dict[str, Any]]:
    url = f"{API_BASE_URL}/leagues/"
    return _paginate(url, {}, per_page=per_page)


def get_season_matches(
    league_id: int,
    season_id: int,
    per_page: int = _API_PAGE_LIMIT,
    only_finished: bool = True,
) -> List[Dict[str, Any]]:
    url = f"{API_BASE_URL}/events/"
    params = {
        "league_id": int(league_id),
        "season_id": int(season_id),
        "date_from": "2000-01-01",
    }
    raw_matches = _paginate(url, params, per_page=per_page)
    if not only_finished:
        for m in raw_matches:
            if not isinstance(m, dict):
                continue
            if _safe_int(m.get("league_id")) is None:
                m["league_id"] = int(league_id)
            if _safe_int(m.get("season_id")) is None:
                m["season_id"] = int(season_id)
        return raw_matches
    finished: List[Dict[str, Any]] = []
    non_finished_count = 0
    missing_score_count = 0
    for m in raw_matches:
        if not isinstance(m, dict):
            continue
        if _safe_int(m.get("league_id")) is None:
            m["league_id"] = int(league_id)
        if _safe_int(m.get("season_id")) is None:
            m["season_id"] = int(season_id)
        status = str(m.get("status", "")).strip().lower()
        if status != "finished":
            non_finished_count += 1
            continue
        if m.get("home_score") is None or m.get("away_score") is None:
            missing_score_count += 1
            continue
        finished.append(m)
    if non_finished_count > 0 or missing_score_count > 0:
        logger.info(
            "Ligue %d saison %d : %d non-joues + %d sans score exclus "
            "(%d FINISHED retenus)",
            league_id, season_id, non_finished_count, missing_score_count,
            len(finished),
        )
    return finished

def get_matches_by_date(
    date_iso: str,
    per_page: int = _API_PAGE_LIMIT,
) -> List[Dict[str, Any]]:
    if not _validate_date_str(date_iso):
        raise BzzoiroAPIError(f"Format de date invalide : {date_iso}")
    url = f"{API_BASE_URL}/events/"
    params = {"date_from": date_iso, "date_to": date_iso}
    matches = _paginate(url, params, per_page=per_page)
    for m in matches:
        if isinstance(m, dict):
            if not m.get("event_date") and m.get("date"):
                m["event_date"] = m["date"]
    return matches


def get_event_detail(event_id: int) -> Optional[Dict[str, Any]]:
    eid = _safe_int(event_id)
    if eid is None or eid <= 0:
        return None
    return make_request(f"{API_BASE_URL}/events/{eid}/")


def get_event_stats(event_id: int) -> Optional[Dict[str, Any]]:
    eid = _safe_int(event_id)
    if eid is None or eid <= 0:
        return None
    return make_request(f"{API_BASE_URL}/events/{eid}/stats/")


def get_event_odds(event_id: int) -> Optional[Dict[str, Any]]:
    eid = _safe_int(event_id)
    if eid is None or eid <= 0:
        return None
    return make_request(f"{API_BASE_URL}/events/{eid}/odds/")


def get_event_h2h(event_id: int) -> Optional[Dict[str, Any]]:
    eid = _safe_int(event_id)
    if eid is None or eid <= 0:
        return None
    return make_request(f"{API_BASE_URL}/events/{eid}/h2h/")


def get_event_lineups(event_id: int) -> Optional[Dict[str, Any]]:
    eid = _safe_int(event_id)
    if eid is None or eid <= 0:
        return None
    return make_request(f"{API_BASE_URL}/events/{eid}/lineups/")


def _enrich_single(raw_match: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw_match, dict):
        return raw_match
    eid = _safe_int(raw_match.get("id"))
    if eid is None or eid <= 0:
        return raw_match
    stats = get_event_stats(eid)
    if not isinstance(stats, dict):
        return raw_match
    st = stats.get("stats")
    if not isinstance(st, dict):
        return raw_match
    for side in ("home", "away"):
        side_data = st.get(side) or {}
        if not isinstance(side_data, dict):
            continue
        for key in STAT_KEYS:
            clean_key = STAT_KEY_MAP.get(key, key)
            val = _read_stat(side_data, key)
            if val is not None:
                raw_match[f"stat_{side}_{clean_key}"] = val
    return raw_match


def enrich_matches_with_stats(
    matches: List[Dict[str, Any]],
    max_workers: int = 8,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    total = len(matches)
    if total == 0:
        return []

    enriched: List[Dict[str, Any]] = []
    done = 0
    with_xg = 0
    with_poss = 0
    with_shots = 0
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_enrich_single, m): m for m in matches}
        for fut in as_completed(futures):
            done += 1
            try:
                result = fut.result()
            except Exception:
                result = futures[fut]
            enriched.append(result)
            if result.get("stat_home_xg") is not None:
                with_xg += 1
            if result.get("stat_home_ball_possession") is not None:
                with_poss += 1
            if result.get("stat_home_total_shots") is not None:
                with_shots += 1

            if verbose and (done % 100 == 0 or done == total):
                elapsed = time.time() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                print(
                    f"   [{done:>5}/{total}] "
                    f"xG={with_xg} | poss={with_poss} | tirs={with_shots} | "
                    f"{rate:.1f} matchs/s | ETA {eta:.0f}s",
                    flush=True,
                )

    return enriched


def _fetch_seasons_for_league(league_id: int) -> Tuple[int, List[Dict[str, Any]]]:
    try:
        seasons = get_league_seasons(league_id)
        return (league_id, seasons)
    except Exception as exc:
        logger.warning("Saisons ligue %d indisponibles : %s", league_id, exc)
        return (league_id, [])


def _fetch_season_task(
    league_id: int,
    season_id: int,
    season_year: int,
    max_retries: int = 3,
) -> Tuple[int, int, int, List[Dict[str, Any]]]:
    last_error: Optional[str] = None
    for attempt in range(max_retries):
        try:
            matches = get_season_matches(league_id, season_id)
            if matches:
                if attempt > 0:
                    logger.info(
                        "Ligue %d saison %d : %d matchs apres %d tentative(s)",
                        league_id, season_id, len(matches), attempt + 1,
                    )
                return (league_id, season_id, season_year, matches)
            logger.info(
                "Ligue %d saison %d : 0 matchs (saison terminee ou archive incomplete)",
                league_id, season_id,
            )
            return (league_id, season_id, season_year, [])
        except Exception as exc:
            last_error = str(exc)
            logger.warning(
                "Erreur ligue %d saison %d (tentative %d/%d) : %s",
                league_id, season_id, attempt + 1, max_retries, exc,
            )
            if attempt < max_retries - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
    logger.error(
        "Ligue %d saison %d : echec reseau definitif apres %d tentatives (%s)",
        league_id, season_id, max_retries, last_error,
    )
    return (league_id, season_id, season_year, [])


def get_historical_matches(
    seasons_lookback: Optional[int] = None,
    per_page: int = _API_PAGE_LIMIT,
    enrich_xg: Optional[bool] = None,
    use_checkpoint: bool = True,
    checkpoint_manager: Optional[CheckpointManager] = None,
) -> List[Dict[str, Any]]:
    if seasons_lookback is None:
        seasons_lookback = HISTORICAL_SEASONS_LOOKBACK
    if enrich_xg is None:
        enrich_xg = os.getenv("BZZOIRO_ENRICH_XG", "true").lower() in ("1", "true", "yes", "on")

    mgr = checkpoint_manager
    if use_checkpoint and mgr is None:
        mgr = CheckpointManager()

    league_ids = list(TOP_60_LEAGUES.keys())
    total_leagues = len(league_ids)
    n_tokens = max(1, len(token_manager.tokens))
    n_workers = max(1, min(n_tokens, _MAX_PARALLEL_WORKERS))

    stage_print("♻️ RÉCUPÉRATION DE L'HISTORIQUE Bzzoiro")
    print(
        f"🚀 {n_tokens} token(s) — {n_workers} workers — {total_leagues} ligues",
        flush=True,
    )

    if mgr is not None:
        st = mgr.status()
        print(
            f"💾 Checkpoint : {st['done_seasons']} saisons déjà traitées, "
            f"raw={st['raw_file_mb']} MB",
            flush=True,
        )

    seasons_by_league: Dict[int, List[Dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_fetch_seasons_for_league, lid): lid
            for lid in league_ids
        }
        for fut in as_completed(futures):
            try:
                lid, seasons = fut.result()
                seasons_by_league[lid] = seasons
            except Exception:
                continue

    tasks: List[Tuple[int, int, int]] = []
    seen_lid_year: Set[Tuple[int, int]] = set()
    duplicates_removed = 0
    for lid, seasons in seasons_by_league.items():
        valid = [s for s in seasons if isinstance(s.get("year"), int)]
        if not valid:
            continue
        by_year: Dict[int, List[Dict[str, Any]]] = {}
        for s in valid:
            yr_s = _safe_int(s.get("year"))
            if yr_s is None:
                continue
            by_year.setdefault(yr_s, []).append(s)
        sorted_years = sorted(by_year.keys())
        if seasons_lookback > 0:
            picked_years = sorted_years[-seasons_lookback:]
        else:
            picked_years = sorted_years
        for yr in picked_years:
            candidates = by_year.get(yr, [])
            if not candidates:
                continue
            best: Optional[Dict[str, Any]] = None
            for c in candidates:
                if c.get("is_current") is True:
                    best = c
                    break
            if best is None:
                best = candidates[0]
            sid = _safe_int(best.get("id"))
            if sid is None:
                continue
            key = (int(lid), int(yr))
            if key in seen_lid_year:
                duplicates_removed += 1
                continue
            seen_lid_year.add(key)
            tasks.append((lid, sid, yr))
    if duplicates_removed > 0:
        print(f"🧹 {duplicates_removed} saisons dupliquees supprimees", flush=True)

    print(f"📊 {len(tasks)} tâches (ligue × saison)", flush=True)

    done_seasons: Set[Tuple[int, int]] = set()
    if mgr is not None:
        done_seasons = mgr.get_done_seasons()
    if done_seasons:
        print(
            f"♻️  Reprise : {len(done_seasons)} saisons déjà traitées "
            f"seront ignorées",
            flush=True,
        )

    pending = [
        (lid, sid, yr)
        for (lid, sid, yr) in tasks
        if (lid, sid) not in done_seasons
    ]

    if not pending:
        print("✅ Toutes les saisons déjà traitées", flush=True)
    else:
        print(f"▶️  {len(pending)} saisons restantes à récupérer", flush=True)

    season_stats: Dict[int, Dict[int, int]] = {}
    completed = 0
    t0 = time.time()

    if pending:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_fetch_season_task, lid, sid, yr): (lid, sid, yr)
                for (lid, sid, yr) in pending
            }
            for fut in as_completed(futures):
                completed += 1
                try:
                    lid, sid, yr, matches = fut.result()
                except Exception:
                    continue

                if mgr is not None:
                    mgr.append_season(lid, sid, yr, matches)

                season_stats.setdefault(lid, {})[yr] = len(matches)
                league_name = TOP_60_LEAGUES.get(lid, f"Ligue {lid}")
                print(
                    f"✅ [{completed:>3}/{len(pending)}] "
                    f"{league_name} {yr} — Total : {len(matches)} matchs",
                    flush=True,
                )
                time.sleep(_API_PAGE_SLEEP_SECONDS)

    all_raw: List[Dict[str, Any]] = []
    seen_ids: Set[int] = set()

    if mgr is not None:
        cached_raw = mgr.load_raw_matches()
        for m in cached_raw:
            mid = _safe_int(m.get("id"))
            if mid is not None:
                if mid in seen_ids:
                    continue
                seen_ids.add(mid)
            all_raw.append(m)
        print(
            f"📥 {len(all_raw)} matchs bruts chargés depuis le checkpoint",
            flush=True,
        )

    print()
    print("=" * 75)
    print("📊 DÉTAIL COMPLET PAR LIGUE (cette session)")
    print("=" * 75)
    for lid in sorted(season_stats.keys(), key=lambda x: TOP_60_LEAGUES.get(x, f"Ligue {x}")):
        league_name = TOP_60_LEAGUES.get(lid, f"Ligue {lid}")
        seasons = season_stats[lid]
        total_league = sum(seasons.values())
        print()
        print(f"🏆 {league_name} (id={lid}) — TOTAL {total_league} matchs")
        for year in sorted(seasons.keys(), reverse=True):
            print(f"   📅 {league_name} {year} — Total : {seasons[year]} matchs")
    print()

    if all_raw:
        _before = len(all_raw)
        _kept: List[Dict[str, Any]] = []
        _non_finished = 0
        _no_score = 0
        for m in all_raw:
            if not isinstance(m, dict):
                continue
            status = str(m.get("status", "")).strip().lower()
            if status != "finished":
                _non_finished += 1
                continue
            if m.get("home_score") is None or m.get("away_score") is None:
                _no_score += 1
                continue
            _kept.append(m)
        all_raw = _kept
        _excluded = _before - len(all_raw)
        if _excluded > 0:
            print(
                f"🎯 {_excluded} matchs non-jouables exclus AVANT enrichissement "
                f"({_non_finished} non-finished + {_no_score} sans score) "
                f"→ {len(all_raw)} FINISHED retenus",
                flush=True,
            )

    if enrich_xg and all_raw:
        to_enrich = [m for m in all_raw if m.get("stat_home_xg") is None]
        already = len(all_raw) - len(to_enrich)
        print(
            f"📊 Enrichissement xG+stats : {len(to_enrich)} à traiter "
            f"({already} déjà enrichis)",
            flush=True,
        )
        if to_enrich:
            enriched_subset = enrich_matches_with_stats(
                to_enrich, max_workers=n_workers * 2, verbose=True,
            )
            by_id: Dict[int, Dict[str, Any]] = {}
            for m in enriched_subset:
                mid = _safe_int(m.get("id"))
                if mid is not None:
                    by_id[mid] = m
            new_all: List[Dict[str, Any]] = []
            for m in all_raw:
                mid = _safe_int(m.get("id"))
                if mid is not None and mid in by_id:
                    new_all.append(by_id[mid])
                else:
                    new_all.append(m)
            all_raw = new_all

        with_xg = sum(1 for m in all_raw if m.get("stat_home_xg") is not None)
        with_poss = sum(1 for m in all_raw if m.get("stat_home_ball_possession") is not None)
        with_shots = sum(1 for m in all_raw if m.get("stat_home_total_shots") is not None)
        print(
            f"   ✅ xG: {with_xg}/{len(all_raw)} | "
            f"possession: {with_poss}/{len(all_raw)} | "
            f"tirs: {with_shots}/{len(all_raw)}",
            flush=True,
        )

    if all_raw:
        before_filter = len(all_raw)
        all_raw = [
            m for m in all_raw
            if isinstance(m, dict)
            and str(m.get("status", "")).strip().lower() in (
                "finished", "ft", "match_finished",
            )
        ]
        filtered_out = before_filter - len(all_raw)
        if filtered_out > 0:
            print(
                f"🎯 {filtered_out} matchs non-FINISHED exclus "
                f"({len(all_raw)} matchs joues conserves)",
                flush=True,
            )

    if all_raw:
        def _date_sort_key(m: Dict[str, Any]) -> str:
            d = m.get("event_date") or m.get("date") or ""
            s = str(d) if d else "9999-12-31T23:59:59+00:00"
            return s
        all_raw = sorted(all_raw, key=_date_sort_key)
        print(f"📅 {len(all_raw)} matchs tries chronologiquement", flush=True)

    transformed: List[Dict[str, Any]] = []
    for m in all_raw:
        t = transform_bzzoiro_match(m)
        if t and t.get("id") is not None:
            transformed.append(t)

    if mgr is not None:
        mgr.save_meta({
            "total_raw": len(all_raw),
            "total_transformed": len(transformed),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "seasons_lookback": seasons_lookback,
            "enrich_xg": enrich_xg,
        })
        mgr.finalize()
        st = mgr.status()
        print(
            f"💾 Checkpoint final : {st['save_count']} saves, "
            f"{st['push_count']} push GitHub",
            flush=True,
        )

    seen_ids_final: Set[int] = set()
    unique_transformed: List[Dict[str, Any]] = []
    for t in transformed:
        tid = t.get("id")
        if tid is None:
            continue
        if tid in seen_ids_final:
            continue
        seen_ids_final.add(tid)
        unique_transformed.append(t)
    dedup_removed = len(transformed) - len(unique_transformed)
    if dedup_removed > 0:
        print(
            f"🧹 {dedup_removed} doublons d'ID match supprimes",
            flush=True,
        )
        transformed = unique_transformed

    finished_count = sum(1 for t in transformed if t.get("status") == "FINISHED")
    non_finished = len(transformed) - finished_count
    print(
        f"📊 Stats : {finished_count} FINISHED | {non_finished} autres "
        f"(total {len(transformed)})",
        flush=True,
    )
    if non_finished > 0:
        logger.info(
            "%d matchs non-finished presents dans le dataset final",
            non_finished,
        )

    duration = time.time() - t0
    stage_print(f"🏆 TOTAL MATCHS HISTORIQUES : {len(transformed)}")
    print(f"⏱️ Durée : {duration:.1f}s", flush=True)

    _fin = sum(1 for t in transformed if t.get("status") == "FINISHED")
    _other = len(transformed) - _fin
    print(
        f"📊 Bilan final : {_fin} FINISHED utilisables | "
        f"{_other} non-joués (ignorés pour l'entraînement)",
        flush=True,
    )

    _seen: Set[int] = set()
    _uniq: List[Dict[str, Any]] = []
    for t in transformed:
        tid = t.get("id")
        if tid is None or tid in _seen:
            continue
        _seen.add(tid)
        _uniq.append(t)
    if len(_uniq) != len(transformed):
        print(f"🧹 {len(transformed) - len(_uniq)} doublons ID supprimés", flush=True)
        transformed = _uniq

    return transformed


def get_all_matches_for_date(
    date_str: str,
    per_page: int = _API_PAGE_LIMIT,
) -> Tuple[List[Dict[str, Any]], str]:
    raw = get_matches_by_date(date_str, per_page=per_page)
    transformed: List[Dict[str, Any]] = []
    for m in raw:
        t = transform_bzzoiro_match(m)
        if t and t.get("id") is not None:
            transformed.append(t)
    return transformed, "bzzoiro"


def get_api_diagnostics() -> Dict[str, Any]:
    return {
        "tokens": token_manager.snapshot(),
        "base_url": API_BASE_URL,
        "timeout_seconds": API_TIMEOUT_SECONDS,
        "max_retries": API_MAX_RETRIES,
        "chunk_days": 0,
        "page_limit": _API_PAGE_LIMIT,
        "page_limit_max": _API_MAX_PAGE_LIMIT,
        "max_parallel_workers": _MAX_PARALLEL_WORKERS,
        "rate_limit_per_minute": _RATE_LIMIT_PER_MINUTE,
        "rate_limit_safety_margin": _RATE_LIMIT_SAFETY_MARGIN,
        "leagues_followed": len(TOP_60_LEAGUES),
        "checkpoint_dir": str(_CHECKPOINT_DIR),
        "github_push_enabled": _GITHUB_PUSH_ENABLED,
    }
