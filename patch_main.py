import os
import shutil
import sys
from pathlib import Path

MAIN_FILE = Path("main.py")
BACKUP_SUFFIX = ".bak_pre_pb_patch"

IMPORT_ANCHOR = "from api import (\n    CheckpointManager,\n    get_historical_matches,\n)\n"
IMPORT_ADDITION = IMPORT_ANCHOR + "\nfrom pb_convergence import PBIncompleteError\n"

NEW_TRAIN_PB = '''def _stage_train_pb_model(class_name: str) -> None:
    from pb_convergence import fit_pb_attempt, PBIncompleteError

    payload = _stage_load_features()
    X = payload["X"]
    home_keys = payload["home_keys"]
    away_keys = payload["away_keys"]
    gh_all = payload["gh_all"]
    ga_all = payload["ga_all"]
    n = len(X)
    cut_train = int(n * TRAIN_RATIO)
    cut_train = max(1, min(cut_train, n - 4))
    gh_train = gh_all[:cut_train]
    ga_train = ga_all[:cut_train]
    th_train = home_keys[:cut_train]
    ta_train = away_keys[:cut_train]

    model_cls = _PB_MODEL_CLASSES.get(class_name)
    if model_cls is None:
        raise RuntimeError(f"Modèle penaltyblog inconnu : {class_name}")
    is_bayesian = class_name in _PB_BAYESIAN_MODELS

    stage_print(
        f"🧠 Tentative {class_name} (bayesian={is_bayesian}) "
        f"sur {len(gh_train)} matchs..."
    )
    status, model, meta = fit_pb_attempt(
        class_name=class_name,
        model_class=model_cls,
        gh=gh_train,
        ga=ga_train,
        th=th_train,
        ta=ta_train,
        dates=None,
        is_bayesian=is_bayesian,
    )
    stage_print(
        f"   → attempt={meta.get('attempt')} filter={meta.get('filter')} "
        f"n_kept={meta.get('n_kept')} status={status}"
    )

    if status == "ok" and model is not None:
        model_path = MODELS_DIR / f"pb_{class_name}.pkl"
        if model_path.exists():
            model_path.unlink()
        _atomic_write_pickle(
            model_path, {"model": model, "type": class_name, "meta": meta}
        )
        stage_print(f"💾 {class_name} sauvegardé : {model_path}")
        return

    if status == "incomplete":
        raise PBIncompleteError(
            f"{class_name} : non convergé ({meta.get('error', 'unknown')})"
        )

    raise RuntimeError(
        f"Erreur réelle sur {class_name} : {meta.get('error', 'unknown')}"
    )

'''

OLD_STAGE_RUN_EXCEPT = '''    except Exception as exc:
        logger.exception("Stage %s échoué", stage_name)
        print(f"{RED}❌ Stage {stage_name} échoué : {exc}{RESET}")
        return 1
'''

NEW_STAGE_RUN_EXCEPT = '''    except PBIncompleteError as exc:
        logger.warning("Stage %s incomplet : %s", stage_name, exc)
        print(
            f"{YELLOW}⏸️ Stage {stage_name} incomplet "
            f"(retry next attempt) : {exc}{RESET}"
        )
        return 42
    except Exception as exc:
        logger.exception("Stage %s échoué", stage_name)
        print(f"{RED}❌ Stage {stage_name} échoué : {exc}{RESET}")
        return 1
'''

MARKER_IMPORT = "from pb_convergence import PBIncompleteError"
MARKER_CATCH = "except PBIncompleteError as exc:"
MARKER_TRAIN = "status, model, meta = fit_pb_attempt("


def _atomic_write(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp, path)


def _validate_structure(src: str) -> None:
    required = [
        "def _stage_train_pb_model(",
        "def _stage_finalize(",
        "def _stage_run(",
        "from api import (",
        "    except Exception as exc:\n        logger.exception(\"Stage %s échoué\", stage_name)",
    ]
    for marker in required:
        if marker not in src:
            raise SystemExit(f"main.py : marqueur introuvable : {marker!r}")


def _patch_import(src: str) -> str:
    if MARKER_IMPORT in src:
        return src
    if IMPORT_ANCHOR not in src:
        raise SystemExit("main.py : bloc import api introuvable")
    return src.replace(IMPORT_ANCHOR, IMPORT_ADDITION, 1)


def _patch_train_pb(src: str) -> str:
    start_marker = "def _stage_train_pb_model("
    end_marker = "\n\ndef _stage_finalize("
    start = src.find(start_marker)
    if start < 0:
        raise SystemExit("main.py : _stage_train_pb_model introuvable")
    end = src.find(end_marker, start)
    if end < 0:
        raise SystemExit("main.py : _stage_finalize introuvable après _stage_train_pb_model")
    return src[:start] + NEW_TRAIN_PB + src[end + 2:]


def _patch_stage_run(src: str) -> str:
    if MARKER_CATCH in src:
        return src
    if OLD_STAGE_RUN_EXCEPT not in src:
        raise SystemExit("main.py : bloc except de _stage_run introuvable")
    return src.replace(OLD_STAGE_RUN_EXCEPT, NEW_STAGE_RUN_EXCEPT, 1)


def main() -> int:
    if not MAIN_FILE.exists():
        print("❌ main.py introuvable dans le répertoire courant")
        return 2

    src = MAIN_FILE.read_text(encoding="utf-8")

    already_full = (
        MARKER_IMPORT in src
        and MARKER_CATCH in src
        and MARKER_TRAIN in src
    )
    if already_full:
        print("⏭  main.py : déjà entièrement patché — rien à faire")
        return 0

    _validate_structure(src)

    try:
        backup = MAIN_FILE.with_suffix(MAIN_FILE.suffix + BACKUP_SUFFIX)
        shutil.copy2(MAIN_FILE, backup)
    except Exception as exc:
        print(f"⚠️  Backup non créé : {exc}")

    src = _patch_import(src)
    src = _patch_train_pb(src)
    src = _patch_stage_run(src)

    _atomic_write(MAIN_FILE, src)

    check = MAIN_FILE.read_text(encoding="utf-8")

    if MARKER_IMPORT not in check:
        print("❌ Import PBIncompleteError absent après patch")
        return 3
    if MARKER_CATCH not in check:
        print("❌ Handler PBIncompleteError absent après patch")
        return 4
    if MARKER_TRAIN not in check:
        print("❌ _stage_train_pb_model non remplacé (fit_pb_attempt absent)")
        return 5
    if "from pb_convergence import PBIncompleteError" not in check:
        print("❌ Import top-level manquant")
        return 7

    try:
        compile(check, str(MAIN_FILE), "exec")
    except SyntaxError as exc:
        print(f"❌ Syntaxe invalide après patch : {exc}")
        return 6

    print("✅ main.py patché :")
    print("   - import top-level : from pb_convergence import PBIncompleteError")
    print("   - _stage_train_pb_model → fit_pb_attempt() avec filtres progressifs")
    print("   - _stage_run → catch PBIncompleteError → return 42 (soft-retry)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
