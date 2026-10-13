#!/usr/bin/env python3
"""GOANGEL — Mise à jour incrémentale de l'historique Bzzoiro.

Récupère UNIQUEMENT les nouveaux matchs FINISHED depuis la dernière date
connue, les enrichit avec xG + stats, puis fusionne avec l'existant.

Usage:
    python3 update_historical.py                    # depuis la dernière date
    python3 update_historical.py --since 2026-09-01 # date explicite
    python3 update_historical.py --dry-run          # simulation
    python3 update_historical.py --limit-days 7     # 7 derniers jours max
"""
import argparse
import logging
import pickle
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from config import CACHE_DIR, HISTORICAL_FILE, BZZOIRO_ENRICH_XG, stage_print
from api import (
    get_matches_by_date,
    enrich_matches_with_stats,
    transform_bzzoiro_match,
)
from features import process_matches

try:
    from predict import _atomic_read_pickle, _atomic_write_pickle
except ImportError:
    # Fallbacks si predict importe des trucs lourds
    import lzma

    def _atomic_read_pickle(path):
        with open(path, "rb") as f:
            magic = f.read(6)
        if magic[:6] == b"\xfd7zXZ\x00":
            with open(path, "rb") as f:
                return pickle.loads(lzma.decompress(f.read()))
        with open(path, "rb") as f:
            return pickle.load(f)

    def _atomic_write_pickle(path, obj):
        raw = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        if len(raw) > 10 * 1024 * 1024:
            import lzma
            data = lzma.compress(raw, preset=6)
        else:
            data = raw
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
        tmp.replace(path)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("goangel.update_historical")


def _parse_date(value):
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def _extract_existing():
    """Charge l'historique existant, retourne (df, ids_set, max_date)."""
    if not HISTORICAL_FILE.exists():
        logger.warning("Aucun historical.pkl existant → construction vierge")
        return None, set(), None

    try:
        df = _atomic_read_pickle(HISTORICAL_FILE)
    except Exception as exc:
        logger.error("Lecture historical.pkl échouée : %s", exc)
        return None, set(), None

    if not isinstance(df, pd.DataFrame) or df.empty:
        return None, set(), None

    ids = set()
    if "match_id" in df.columns:
        for mid in df["match_id"].dropna().tolist():
            try:
                ids.add(int(mid))
            except (TypeError, ValueError):
                continue

    max_date = None
    if "date_parsed" in df.columns:
        try:
            dates = pd.to_datetime(df["date_parsed"], errors="coerce", utc=True)
            dates = dates.dropna()
            if not dates.empty:
                max_date = dates.max().date()
        except Exception as exc:
            logger.warning("Impossible de lire la date max : %s", exc)

    logger.info(
        "Historique chargé : %d matchs, %d IDs, dernière date = %s",
        len(df), len(ids), max_date,
    )
    return df, ids, max_date


def _iter_dates(start_date, end_date, limit_days=None):
    """Génère les dates entre start et end (inclus)."""
    current = start_date
    count = 0
    while current <= end_date:
        if limit_days is not None and count >= limit_days:
            break
        yield current
        current += timedelta(days=1)
        count += 1


def _fetch_new_raw(existing_ids, since_date, until_date, limit_days=None):
    """Récupère les matchs FINISHED non présents dans existing_ids."""
    new_raw = []
    seen_ids = set()
    dates_scanned = 0
    dates_with_matches = 0
    total_api_matches = 0
    skipped_status = 0
    skipped_dup = 0

    stage_print(f"📅 Scan du {since_date} au {until_date}")
    for d in _iter_dates(since_date, until_date, limit_days):
        date_iso = d.strftime("%Y-%m-%d")
        dates_scanned += 1
        try:
            matches = get_matches_by_date(date_iso)
        except Exception as exc:
            logger.warning("Erreur API %s : %s", date_iso, exc)
            continue

        if not matches:
            continue

        dates_with_matches += 1
        total_api_matches += len(matches)

        for m in matches:
            if not isinstance(m, dict):
                continue
            mid = m.get("id")
            if mid is None:
                continue
            try:
                mid_int = int(mid)
            except (TypeError, ValueError):
                continue

            status = str(m.get("status", "")).strip().lower()
            if status != "finished":
                skipped_status += 1
                continue

            if mid_int in existing_ids or mid_int in seen_ids:
                skipped_dup += 1
                continue

            seen_ids.add(mid_int)
            new_raw.append(m)

        if dates_scanned % 10 == 0:
            stage_print(
                f"   {dates_scanned} dates scannées, "
                f"{len(new_raw)} nouveaux FINISHED"
            )

    logger.info(
        "Scan terminé : %d dates, %d avec matchs, %d matchs API, "
        "%d FINISHED nouveaux, %d non-finished, %d doublons",
        dates_scanned, dates_with_matches, total_api_matches,
        len(new_raw), skipped_status, skipped_dup,
    )
    return new_raw


def _enrich(matches):
    """Enrichit avec xG + stats si activé."""
    if not matches:
        return []
    if not BZZOIRO_ENRICH_XG:
        stage_print("⏭  Enrichissement xG désactivé (BZZOIRO_ENRICH_XG=false)")
        return matches

    to_enrich = [m for m in matches if m.get("stat_home_xg") is None]
    already = len(matches) - len(to_enrich)
    if not to_enrich:
        stage_print(f"✅ Tous déjà enrichis ({already} matchs)")
        return matches

    stage_print(f"📊 Enrichissement {len(to_enrich)} matchs ({already} déjà enrichis)")
    enriched_subset = enrich_matches_with_stats(
        to_enrich, max_workers=4, verbose=True,
    )

    by_id = {}
    for m in enriched_subset:
        try:
            by_id[int(m.get("id"))] = m
        except (TypeError, ValueError):
            continue

    out = []
    for m in matches:
        try:
            mid = int(m.get("id"))
        except (TypeError, ValueError):
            out.append(m)
            continue
        out.append(by_id.get(mid, m))
    return out


def _merge(existing_df, new_transformed):
    """Fusionne l'historique existant avec les nouveaux matchs transformés."""
    new_df = process_matches(
        new_transformed,
        allowed_statuses=("FINISHED",),
        include_awarded=False,
    )
    if new_df is None or new_df.empty:
        return existing_df

    if existing_df is None or existing_df.empty:
        return new_df

    merged = pd.concat([existing_df, new_df], ignore_index=True)

    if "match_id" in merged.columns:
        before = len(merged)
        merged = merged.drop_duplicates(subset=["match_id"], keep="first")
        dropped = before - len(merged)
        if dropped > 0:
            logger.info("Doublons supprimés après merge : %d", dropped)

    if "date_parsed" in merged.columns:
        merged = (
            merged.dropna(subset=["date_parsed"])
            .sort_values("date_parsed", kind="mergesort")
            .reset_index(drop=True)
        )
    return merged


def main():
    parser = argparse.ArgumentParser(description="GOANGEL — Update historique")
    parser.add_argument("--since", type=str, default=None,
                        help="Date de départ (YYYY-MM-DD). Sinon : max historique + 1j")
    parser.add_argument("--until", type=str, default=None,
                        help="Date de fin (YYYY-MM-DD). Sinon : aujourd'hui UTC")
    parser.add_argument("--dry-run", action="store_true",
                        help="Ne pas sauvegarder, juste afficher")
    parser.add_argument("--limit-days", type=int, default=None,
                        help="Limiter à N jours en partant de --since")
    args = parser.parse_args()

    t0 = time.time()
    stage_print("═══════════════════════════════════════════════════════════")
    stage_print("  GOANGEL — UPDATE HISTORICAL (incrémental)")
    stage_print("═══════════════════════════════════════════════════════════")

    existing_df, existing_ids, max_date = _extract_existing()

    today = datetime.now(timezone.utc).date()
    since = _parse_date(args.since) if args.since else None
    until = _parse_date(args.until) if args.until else today

    if since is None:
        if max_date is None:
            since = today - timedelta(days=30)
            stage_print(f"⚠️  Aucune date connue → scan des 30 derniers jours")
        else:
            since = max_date + timedelta(days=1)

    if since > until:
        stage_print(f"✅ Historique déjà à jour (max={max_date}, today={today})")
        return 0

    stage_print(f"📆 Période : {since} → {until}")
    if args.limit_days:
        stage_print(f"   Limité à {args.limit_days} jours")

    new_raw = _fetch_new_raw(
        existing_ids, since, until, limit_days=args.limit_days,
    )

    stage_print(f"✅ {len(new_raw)} nouveaux matchs FINISHED détectés")

    if not new_raw:
        stage_print("✅ Aucun nouveau match → historique inchangé")
        return 0

    # Enrichissement xG + stats
    new_enriched = _enrich(new_raw)

    # Transform au format GOANGEL
    new_transformed = []
    for raw in new_enriched:
        try:
            t = transform_bzzoiro_match(raw)
            if t and t.get("id") is not None:
                new_transformed.append(t)
        except Exception as exc:
            logger.debug("Transform échoué pour id=%s : %s", raw.get("id"), exc)

    stage_print(f"✅ {len(new_transformed)} matchs transformés")

    if not new_transformed:
        stage_print("⚠️  Aucun match transformé → historique inchangé")
        return 0

    merged = _merge(existing_df, new_transformed)
    if merged is None or merged.empty:
        stage_print("❌ Merge échoué → historique inchangé")
        return 1

    old_n = len(existing_df) if existing_df is not None else 0
    new_n = len(merged)
    delta = new_n - old_n
    stage_print(f"📊 Historique : {old_n} → {new_n} (+{delta} matchs)")

    if args.dry_run:
        stage_print("🧪 DRY-RUN → pas de sauvegarde")
        return 0

    try:
        _atomic_write_pickle(HISTORICAL_FILE, merged)
        size_mb = HISTORICAL_FILE.stat().st_size / (1024 * 1024)
        stage_print(f"💾 Sauvegardé : {HISTORICAL_FILE} ({size_mb:.1f} MB)")
    except Exception as exc:
        logger.error("Sauvegarde échouée : %s", exc)
        return 1

    stage_print(f"⏱  Durée totale : {time.time() - t0:.1f}s")
    stage_print("🎉 Update terminé — prêt à réentraîner")
    return 0


if __name__ == "__main__":
    sys.exit(main())
