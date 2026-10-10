import os
import shutil
import sys
from pathlib import Path

FEATURES_FILE = Path("features.py")
BACKUP_SUFFIX = ".bak_pre_pb_patch"

NEW_FIT_PB_MODEL = '''def _fit_pb_model(
    class_name: str,
    gh: List[int],
    ga: List[int],
    th: List[str],
    ta: List[str],
    weights: Optional[np.ndarray] = None,
) -> Tuple[str, Any]:
    cls = _PB_MODEL_CLASSES.get(class_name)
    if cls is None:
        return "error", f"Modèle penaltyblog inconnu : {class_name}"

    def _new_model():
        if weights is not None:
            try:
                return cls(gh, ga, th, ta, weights=weights)
            except TypeError:
                pass
            except Exception as exc:
                logger.debug(
                    "Poids rejetés pour %s : %s", class_name, exc,
                )
        return cls(gh, ga, th, ta)

    if class_name in _PB_BAYESIAN_MODELS:
        try:
            m = _new_model()
            fit_kwargs: Dict[str, Any] = {
                "n_samples": BAYESIAN_N_SAMPLES,
                "burn": BAYESIAN_BURN,
                "n_chains": BAYESIAN_N_CHAINS,
                "thin": BAYESIAN_THIN,
            }
            try:
                import inspect as _inspect
                sig = _inspect.signature(m.fit)
                if "n_cores" in sig.parameters:
                    fit_kwargs["n_cores"] = 1
            except Exception:
                pass
            m.fit(**fit_kwargs)
            return "ok", m
        except Exception as exc:
            return (
                "error",
                f"Bayesian {class_name} : {type(exc).__name__}: {exc}",
            )

    try:
        m = _new_model()
        m.fit(minimizer_options={"maxiter": 5000})
        return "ok", m
    except Exception as exc:
        return (
            "error",
            f"{class_name} : {type(exc).__name__}: {exc}",
        )
'''

MARKER_PATCHED = "weights: Optional[np.ndarray] = None,\n) -> Tuple[str, Any]:\n    cls = _PB_MODEL_CLASSES.get(class_name)"


def _verify_target_structure(src: str) -> None:
    start_idx = src.find("def _fit_pb_model(")
    if start_idx < 0:
        raise SystemExit("features.py : _fit_pb_model introuvable")
    calib_idx = src.find("def _calibrate_or_keep(")
    if calib_idx < 0:
        raise SystemExit("features.py : _calibrate_or_keep introuvable")
    if calib_idx <= start_idx:
        raise SystemExit(
            "features.py : _calibrate_or_keep apparaît avant _fit_pb_model"
        )
    return start_idx, calib_idx


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


def main() -> int:
    if not FEATURES_FILE.exists():
        print("❌ features.py introuvable dans le répertoire courant")
        return 2

    src = FEATURES_FILE.read_text(encoding="utf-8")

    if MARKER_PATCHED in src:
        print("⏭  features.py : _fit_pb_model déjà patché — rien à faire")
        return 0

    start_idx, calib_idx = _verify_target_structure(src)

    prefix = src[:start_idx]
    suffix = src[calib_idx:]

    new_src = prefix + NEW_FIT_PB_MODEL + "\n\n" + suffix

    try:
        backup_path = FEATURES_FILE.with_suffix(
            FEATURES_FILE.suffix + BACKUP_SUFFIX
        )
        shutil.copy2(FEATURES_FILE, backup_path)
    except Exception as exc:
        print(f"⚠️  Backup non créé : {exc}")

    _atomic_write(FEATURES_FILE, new_src)

    check = FEATURES_FILE.read_text(encoding="utf-8")
    if MARKER_PATCHED not in check:
        print("❌ Patch non appliqué : vérification post-écriture échouée")
        return 3

    if "def _calibrate_or_keep(" not in check:
        print("❌ _calibrate_or_keep disparu après patch")
        return 4

    if "strategies: List[Tuple[str, Dict[str, Any]]] = [" in check:
        print("❌ Ancienne boucle multi-optimizers toujours présente")
        return 5

    try:
        compile(check, str(FEATURES_FILE), "exec")
    except SyntaxError as exc:
        print(f"❌ Syntaxe invalide après patch : {exc}")
        return 6

    print("✅ features.py patché : _fit_pb_model remplacé")
    print("   - signature étendue : weights: Optional[np.ndarray] = None")
    print("   - bayesian : n_cores=1 si supporté")
    print("   - fallback trust-constr géré par penaltyblog_method_patch")
    print("   - ancienne boucle multi-optimizers supprimée")
    return 0


if __name__ == "__main__":
    sys.exit(main())
