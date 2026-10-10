import os
import time
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logger = logging.getLogger("goangel.training_runner")


def _train_dl_model(
    Xs_train,
    y_train,
    Xs_val,
    y_val,
    Xs_calib,
    y_calib,
    input_dim,
    stage_print,
):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    import random as _random

    from config import (
        DL_EPOCHS,
        DL_BATCH_SIZE,
        DL_LEARNING_RATE,
        DL_WEIGHT_DECAY,
        DL_EARLY_STOPPING_PATIENCE,
        DL_GRAD_CLIP,
        DL_DROPOUT,
        MODEL_RANDOM_SEED,
    )
    from features import _TorchNet, _TorchProbaWrapper, _MIN_VAL_SAMPLES_FOR_DL

    if len(Xs_val) < _MIN_VAL_SAMPLES_FOR_DL or len(np.unique(y_val)) < 2:
        return None, "validation_insufficient"

    lr_candidates = [
        float(DL_LEARNING_RATE),
        float(DL_LEARNING_RATE) * 0.3,
        float(DL_LEARNING_RATE) * 0.1,
        float(DL_LEARNING_RATE) * 0.03,
    ]

    best_model = None
    best_val_loss = float("inf")
    best_lr = None

    for lr_try in lr_candidates:
        try:
            torch.manual_seed(MODEL_RANDOM_SEED)
            np.random.seed(MODEL_RANDOM_SEED)
            _random.seed(MODEL_RANDOM_SEED)

            model = _TorchNet(input_dim)
            opt = torch.optim.AdamW(
                model.parameters(),
                lr=lr_try,
                weight_decay=DL_WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=0.5, patience=3
            )
            crit = nn.CrossEntropyLoss()

            Xt = torch.FloatTensor(Xs_train)
            yt = torch.LongTensor(y_train)
            Xv = torch.FloatTensor(Xs_val)
            yv = torch.LongTensor(y_val)

            if not torch.all(torch.isfinite(Xt)):
                return None, "input_nan"
            if not torch.all(torch.isfinite(Xv)):
                return None, "input_nan"

            use_drop_last = len(Xt) > DL_BATCH_SIZE
            generator = torch.Generator().manual_seed(MODEL_RANDOM_SEED)
            loader = DataLoader(
                TensorDataset(Xt, yt),
                batch_size=DL_BATCH_SIZE,
                shuffle=True,
                drop_last=use_drop_last,
                generator=generator,
            )

            best_val_loss_local = float("inf")
            best_state_local = None
            patience_counter = 0
            nan_hit = False

            for epoch in range(DL_EPOCHS):
                model.train()
                for bx, by in loader:
                    opt.zero_grad()
                    logits = model(bx)
                    if not torch.all(torch.isfinite(logits)):
                        nan_hit = True
                        break
                    loss = crit(logits, by)
                    if not torch.isfinite(loss):
                        nan_hit = True
                        break
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), DL_GRAD_CLIP
                    )
                    opt.step()
                if nan_hit:
                    break

                model.eval()
                with torch.no_grad():
                    val_logits = model(Xv)
                    if not torch.all(torch.isfinite(val_logits)):
                        nan_hit = True
                        break
                    val_loss = crit(val_logits, yv).item()
                    if not np.isfinite(val_loss):
                        nan_hit = True
                        break

                scheduler.step(val_loss)
                if val_loss < best_val_loss_local:
                    best_val_loss_local = val_loss
                    best_state_local = {
                        k: v.clone() for k, v in model.state_dict().items()
                    }
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= DL_EARLY_STOPPING_PATIENCE:
                        break

            if nan_hit or best_state_local is None:
                continue

            if best_val_loss_local < best_val_loss:
                best_val_loss = best_val_loss_local
                best_lr = lr_try
                if best_state_local is not None:
                    model.load_state_dict(best_state_local)
                    best_model = model

        except Exception as exc:
            logger.warning("DL tentative lr=%.5f échouée : %s", lr_try, exc)
            continue

    if best_model is None:
        return None, "all_lr_failed"

    try:
        from features import (
            _calibrate_or_keep,
            _model_produces_three_classes,
            _compute_calibration_log_loss,
            _evaluate_dl_walk_forward,
            _TorchProbaWrapper,
            _CALIBRATION_STATUS_RAW_KEPT,
        )
        from config import MODEL_WALK_FORWARD_FOLDS

        calibrated, calib_status = _calibrate_or_keep(
            "DL", best_model, Xs_calib, y_calib
        )
        if not _model_produces_three_classes(calibrated, Xs_calib, "DL"):
            calibrated = _TorchProbaWrapper(best_model)
            calib_status = _CALIBRATION_STATUS_RAW_KEPT

        if not _model_produces_three_classes(calibrated, Xs_calib, "DL"):
            return None, "invalid_proba"

        ll_calib = _compute_calibration_log_loss(
            calibrated, Xs_calib, y_calib
        )
        if not np.isfinite(ll_calib) or ll_calib <= 0:
            return None, "invalid_logloss"

        ll_wf, wf_eval, wf_attempt = _evaluate_dl_walk_forward(
            Xs_train,
            y_train,
            MODEL_WALK_FORWARD_FOLDS,
            input_dim,
        )

        return {
            "model": calibrated,
            "calibration_status": calib_status,
            "ll_calib": float(ll_calib),
            "ll_wf": float(ll_wf),
            "wf_eval": wf_eval,
            "wf_attempt": wf_attempt,
            "lr": best_lr,
        }, "ok"

    except Exception as exc:
        logger.error("DL post-traitement : %s", exc)
        return None, "post_process_failed"


def run_resilient_training(
    df_past: pd.DataFrame,
    checkpoint_mgr: Any,
) -> Tuple[
    Optional[Dict[str, Any]],
    Optional[List[Tuple[str, Any]]],
    Any,
    Dict[str, float],
]:
    from config import (
        TRAIN_RATIO,
        VAL_RATIO,
        CALIB_RATIO,
        MIN_MATCHES_FOR_TRAINING,
        MODEL_WALK_FORWARD_FOLDS,
        FEATURE_OUTLIER_ZSCORE,
        TEMPERATURE_SCALING_ENABLED,
        TEMPERATURE_SCALING_DEFAULT,
        ENSEMBLE_MIN_WEIGHT,
        ENSEMBLE_MAX_WEIGHT,
        DL_ENABLED,
        stage_print,
    )

    from features import (
        create_features_with_context,
        _STRICT_COMPETITION_DEFAULT,
        _validate_splits,
        _clip_outliers_signed,
        _make_xgb_factory,
        _make_lgbm_factory,
        _make_catboost_factory,
        _make_rf_factory,
        _calibrate_or_keep,
        _compute_optimal_temperature,
        _compute_ensemble_probas,
        _evaluate_ml_model_walk_forward,
        _evaluate_ensemble_on_set,
        _fit_pb_model,
        _normalize_weights,
        _filter_valid_weights,
        _select_fallback_uniform_models,
        _apply_quality_filter_on_weights,
        _compute_calibration_log_loss,
        _model_produces_three_classes,
        _PB_MODEL_ORDER,
        _CALIBRATION_STATUS_NOT_CALIBRATED,
        _CALIBRATION_STATUS_RAW_KEPT,
        _LL_SOURCE_WALK_FORWARD,
        _LL_SOURCE_CALIBRATION_FALLBACK,
        EXPECTED_FEATURE_COUNT,
        FEATURE_NAMES,
        _extract_model_probabilities,
    )

    XGBClassifier = None
    LGBMClassifier = None
    CatBoostClassifier = None
    RandomForestClassifier = None
    try:
        from xgboost import XGBClassifier
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        pass
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        pass
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        pass

    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import log_loss

    if df_past is None or len(df_past) < MIN_MATCHES_FOR_TRAINING:
        stage_print("❌ Données insuffisantes pour l'entraînement")
        return None, None, None, {}

    features_key = "stage_1_features"
    cached = checkpoint_mgr.load(features_key)
    if cached is not None and len(cached) == 8:
        (
            X, y, home_keys, away_keys, gh_all, ga_all,
            comp_codes, h2h_sources_all,
        ) = cached
        stage_print("📦 Features chargées depuis checkpoint")
    else:
        stage_print("🔧 Construction des features...")
        (
            X, y, home_keys, away_keys, gh_all, ga_all,
            comp_codes, h2h_sources_all,
        ) = create_features_with_context(
            df_past, strict_competition=_STRICT_COMPETITION_DEFAULT
        )
        if X.shape[0] < MIN_MATCHES_FOR_TRAINING:
            stage_print("❌ Échec de la construction des features")
            return None, None, None, {}
        checkpoint_mgr.save(
            features_key,
            (
                X, y, home_keys, away_keys, gh_all, ga_all,
                comp_codes, h2h_sources_all,
            ),
        )

    splits_key = "stage_2_splits"
    cached = checkpoint_mgr.load(splits_key)
    if cached is not None and len(cached) == 16:
        (
            X_train_raw, X_val_raw, X_calib_raw, X_test_raw,
            y_train, y_val, y_calib, y_test,
            home_keys_train, away_keys_train,
            home_keys_calib, away_keys_calib,
            home_keys_test, away_keys_test,
            gh_train, ga_train,
        ) = cached
        stage_print("📦 Splits chargés depuis checkpoint")
    else:
        n = len(X)
        cut_train = max(1, min(int(n * TRAIN_RATIO), n - 4))
        cut_val = max(
            cut_train + 1,
            min(int(n * (TRAIN_RATIO + VAL_RATIO)), n - 3),
        )
        cut_calib = max(
            cut_val + 1,
            min(
                int(n * (TRAIN_RATIO + VAL_RATIO + CALIB_RATIO)),
                n - 2,
            ),
        )

        X_train_raw = X[:cut_train].astype(np.float64, copy=True)
        X_val_raw = X[cut_train:cut_val].astype(np.float64, copy=True)
        X_calib_raw = X[cut_val:cut_calib].astype(np.float64, copy=True)
        X_test_raw = X[cut_calib:].astype(np.float64, copy=True)

        y_train = y[:cut_train]
        y_val = y[cut_train:cut_val]
        y_calib = y[cut_val:cut_calib]
        y_test = y[cut_calib:]

        if not _validate_splits(y_train, y_val, y_calib, y_test):
            stage_print("❌ Splits invalides")
            return None, None, None, {}

        home_keys_train = home_keys[:cut_train]
        away_keys_train = away_keys[:cut_train]
        home_keys_calib = home_keys[cut_val:cut_calib]
        away_keys_calib = away_keys[cut_val:cut_calib]
        home_keys_test = home_keys[cut_calib:]
        away_keys_test = away_keys[cut_calib:]
        gh_train = gh_all[:cut_train]
        ga_train = ga_all[:cut_train]

        checkpoint_mgr.save(
            splits_key,
            (
                X_train_raw, X_val_raw, X_calib_raw, X_test_raw,
                y_train, y_val, y_calib, y_test,
                home_keys_train, away_keys_train,
                home_keys_calib, away_keys_calib,
                home_keys_test, away_keys_test,
                gh_train, ga_train,
            ),
        )

    scaler_key = "stage_3_scaler"
    cached = checkpoint_mgr.load(scaler_key)
    if cached is not None and len(cached) == 5:
        scaler, Xs_train, Xs_val, Xs_calib, Xs_test = cached
        stage_print("📦 Scaler chargé depuis checkpoint")
    else:
        nan_count_before = int(np.isnan(X_train_raw).sum())
        inf_count_before = int(np.isinf(X_train_raw).sum())
        if nan_count_before > 0 or inf_count_before > 0:
            stage_print(
                f"🔧 Features brutes : {nan_count_before} NaN, "
                f"{inf_count_before} inf — imputation médiane"
            )

        X_train_raw = np.where(
            np.isfinite(X_train_raw), X_train_raw, np.nan
        )
        X_val_raw = np.where(
            np.isfinite(X_val_raw), X_val_raw, np.nan
        )
        X_calib_raw = np.where(
            np.isfinite(X_calib_raw), X_calib_raw, np.nan
        )
        X_test_raw = np.where(
            np.isfinite(X_test_raw), X_test_raw, np.nan
        )

        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline

        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train_raw)
        X_val_imp = imputer.transform(X_val_raw)
        X_calib_imp = imputer.transform(X_calib_raw)
        X_test_imp = imputer.transform(X_test_raw)

        if FEATURE_OUTLIER_ZSCORE > 0:
            (
                X_train_imp,
                [X_val_imp, X_calib_imp, X_test_imp],
            ) = _clip_outliers_signed(
                X_train_imp,
                [X_val_imp, X_calib_imp, X_test_imp],
                float(FEATURE_OUTLIER_ZSCORE),
            )

        scaler_only = StandardScaler()
        Xs_train = scaler_only.fit_transform(X_train_imp)
        Xs_val = scaler_only.transform(X_val_imp)
        Xs_calib = scaler_only.transform(X_calib_imp)
        Xs_test = scaler_only.transform(X_test_imp)

        scaler = Pipeline([
            ("imputer", imputer),
            ("scaler", scaler_only),
        ])

        nan_count_after = int(np.isnan(Xs_train).sum())
        stage_print(
            f"✅ Preprocessing terminé — NaN restants : {nan_count_after}"
        )

        checkpoint_mgr.save(
            scaler_key,
            (scaler, Xs_train, Xs_val, Xs_calib, Xs_test),
        )

    metadata_key = "stage_4_metadata"
    metadata = checkpoint_mgr.load(metadata_key)
    if not isinstance(metadata, dict):
        metadata = {
            "wf_ll_by_key": {},
            "calib_ll_by_key": {},
            "ll_source_by_key": {},
            "calibration_status_by_key": {},
            "wf_fold_coverage_by_key": {},
        }
    else:
        for k in (
            "wf_ll_by_key", "calib_ll_by_key", "ll_source_by_key",
            "calibration_status_by_key", "wf_fold_coverage_by_key",
        ):
            metadata.setdefault(k, {})

    ml_models: Dict[str, Any] = {}
    factories: List[Tuple[str, Any, str]] = []
    if XGBClassifier is not None:
        factories.append(("XGB", _make_xgb_factory(), "XGBOOST"))
    if LGBMClassifier is not None:
        factories.append(("LGBM", _make_lgbm_factory(), "LIGHTGBM"))
    if CatBoostClassifier is not None:
        factories.append(("CatBoost", _make_catboost_factory(), "CATBOOST"))
    if RandomForestClassifier is not None:
        factories.append(("RF", _make_rf_factory(), "RANDOM FOREST"))

    stage_print("🧠 ENTRAÎNEMENT")

    for name, factory, display in factories:
        model_key = f"ml_{name}"
        cached_model = checkpoint_mgr.load(model_key)
        if cached_model is not None:
            ml_models[name] = cached_model
            stage_print(f"📦 {display} chargé depuis checkpoint")
            continue

        try:
            m = factory()
            m.fit(Xs_train, y_train)
            if not _model_produces_three_classes(m, Xs_train, name):
                stage_print(f"❌ {display} : probabilités invalides")
                continue

            calibrated, calib_status = _calibrate_or_keep(
                name, m, Xs_calib, y_calib
            )
            if not _model_produces_three_classes(
                calibrated, Xs_calib, name
            ):
                calibrated = m
                calib_status = _CALIBRATION_STATUS_RAW_KEPT
            ml_models[name] = calibrated
            metadata["calibration_status_by_key"][name] = calib_status

            ll_calib = _compute_calibration_log_loss(
                calibrated, Xs_calib, y_calib
            )
            metadata["calib_ll_by_key"][name] = float(ll_calib)

            ll_wf, wf_eval, wf_attempt = _evaluate_ml_model_walk_forward(
                factory, X_train_raw, y_train, MODEL_WALK_FORWARD_FOLDS
            )
            metadata["wf_ll_by_key"][name] = float(ll_wf)
            metadata["ll_source_by_key"][name] = _LL_SOURCE_WALK_FORWARD
            coverage = wf_eval / wf_attempt if wf_attempt > 0 else 0.0
            metadata["wf_fold_coverage_by_key"][name] = float(coverage)

            checkpoint_mgr.save(model_key, calibrated)
            checkpoint_mgr.save(metadata_key, metadata, sync_push=False)

            if np.isfinite(ll_calib):
                stage_print(
                    f"✅ {display} : LogLoss {float(ll_calib):.4f} "
                    f"(WF {wf_eval}/{wf_attempt} folds)"
                )
            else:
                stage_print(f"⚠️ {display} : LogLoss invalide")
        except Exception as exc:
            logger.error("%s : %s", display, exc)
            stage_print(f"❌ {display} : échec")

    # === DEEP LEARNING avec retry LR ===
    dl_key = "ml_DL"
    if DL_ENABLED:
        cached_dl = checkpoint_mgr.load(dl_key)
        if cached_dl is not None:
            ml_models["DL"] = cached_dl
            stage_print("📦 DEEP LEARNING chargé depuis checkpoint")
        else:
            try:
                import torch as _torch_check
                dl_available = _torch_check is not None
            except ImportError:
                dl_available = False

            if not dl_available:
                stage_print("⚠️ DEEP LEARNING : torch indisponible")
            else:
                stage_print("🧠 DEEP LEARNING : entraînement avec retry LR...")
                dl_result, dl_status = _train_dl_model(
                    Xs_train,
                    y_train,
                    Xs_val,
                    y_val,
                    Xs_calib,
                    y_calib,
                    Xs_train.shape[1],
                    stage_print,
                )

                if dl_status == "ok" and dl_result is not None:
                    ml_models["DL"] = dl_result["model"]
                    metadata["calibration_status_by_key"]["DL"] = (
                        dl_result["calibration_status"]
                    )
                    metadata["calib_ll_by_key"]["DL"] = dl_result["ll_calib"]
                    metadata["wf_ll_by_key"]["DL"] = dl_result["ll_wf"]
                    metadata["ll_source_by_key"]["DL"] = _LL_SOURCE_WALK_FORWARD
                    coverage = (
                        dl_result["wf_eval"] / dl_result["wf_attempt"]
                        if dl_result["wf_attempt"] > 0 else 0.0
                    )
                    metadata["wf_fold_coverage_by_key"]["DL"] = float(coverage)

                    checkpoint_mgr.save(dl_key, dl_result["model"])
                    checkpoint_mgr.save(metadata_key, metadata, sync_push=False)

                    stage_print(
                        f"✅ DEEP LEARNING : LogLoss "
                        f"{dl_result['ll_calib']:.4f} "
                        f"(WF {dl_result['wf_eval']}/{dl_result['wf_attempt']} "
                        f"folds, lr={dl_result['lr']:.5f})"
                    )
                else:
                    stage_print(
                        f"❌ DEEP LEARNING : échec ({dl_status})"
                    )

    stage_print("🤖 PENALTY BLOG MODELS")

    pb_models: List[Tuple[str, Any]] = []

    for class_name in _PB_MODEL_ORDER:
        pb_key = f"pb_{class_name}"
        cached_pb = checkpoint_mgr.load(pb_key)
        if cached_pb is not None:
            pb_models.append((class_name, cached_pb))
            stage_print(f"📦 {class_name.upper()} chargé depuis checkpoint")
            continue

        status, model = _fit_pb_model(
            class_name,
            gh_train,
            ga_train,
            home_keys_train,
            away_keys_train,
        )
        if status != "ok":
            stage_print(f"❌ {class_name.upper()} : échec")
            continue

        checkpoint_mgr.save(pb_key, model)

        try:
            probas: List[List[float]] = []
            y_kept: List[int] = []
            for h, a, yv in zip(
                home_keys_calib, away_keys_calib, y_calib
            ):
                try:
                    grid = model.predict(h, a)
                    markets = _extract_model_probabilities(grid)
                    hda = markets.get("home_draw_away")
                    if hda is None or len(hda) != 3:
                        continue
                    probas.append(hda)
                    y_kept.append(int(yv))
                except Exception:
                    continue

            if len(probas) < 30:
                continue

            probas_arr = np.array(probas, dtype=np.float64)
            y_arr = np.array(y_kept, dtype=np.int64)
            ll_calib = float(
                log_loss(y_arr, probas_arr, labels=[0, 1, 2])
            )

            from features import _evaluate_pb_model_walk_forward
            ll_wf_pb, wf_eval, wf_attempt = (
                _evaluate_pb_model_walk_forward(
                    class_name,
                    gh_train,
                    ga_train,
                    home_keys_train,
                    away_keys_train,
                    MODEL_WALK_FORWARD_FOLDS,
                )
            )
            ll_source = _LL_SOURCE_WALK_FORWARD
            if not np.isfinite(ll_wf_pb) or ll_wf_pb <= 0:
                ll_wf_pb = ll_calib
                ll_source = _LL_SOURCE_CALIBRATION_FALLBACK

            pb_index = len(pb_models)
            pb_models.append((class_name, model))
            metadata["calib_ll_by_key"][f"pb_{pb_index}"] = ll_calib
            metadata["wf_ll_by_key"][f"pb_{pb_index}"] = float(ll_wf_pb)
            metadata["ll_source_by_key"][f"pb_{pb_index}"] = ll_source
            metadata["calibration_status_by_key"][
                f"pb_{pb_index}"
            ] = _CALIBRATION_STATUS_NOT_CALIBRATED
            coverage = wf_eval / wf_attempt if wf_attempt > 0 else 0.0
            metadata["wf_fold_coverage_by_key"][
                f"pb_{pb_index}"
            ] = float(coverage)

            checkpoint_mgr.save(metadata_key, metadata, sync_push=False)

            stage_print(
                f"✅ {class_name.upper()} : CALIB {ll_calib:.4f} | "
                f"WF {ll_wf_pb:.4f} (wf {wf_eval}/{wf_attempt})"
            )
        except Exception as exc:
            logger.error("PB %s : %s", class_name, exc)

    weights_key = "stage_6_weights"
    cached_weights = checkpoint_mgr.load(weights_key)
    if isinstance(cached_weights, dict) and cached_weights:
        stage_print("📦 Poids chargés depuis checkpoint")
        return ml_models, pb_models, scaler, cached_weights

    weights: Dict[str, float] = {}
    eps_weight = 1e-6
    max_acceptable_logloss = 1.35

    for k, ll in metadata["wf_ll_by_key"].items():
        try:
            ll_f = float(ll)
        except (TypeError, ValueError):
            ll_f = float("inf")
        if (
            np.isfinite(ll_f)
            and ll_f > 0
            and ll_f <= max_acceptable_logloss
        ):
            weights[k] = 1.0 / (ll_f + eps_weight)
        else:
            weights[k] = 0.0

    if not ml_models and not pb_models:
        stage_print("❌ Aucun modèle entraîné")
        return None, None, None, {}

    filtered_weights = _filter_valid_weights(weights)
    valid_keys = [k for k in filtered_weights if not k.startswith("__")]
    if not valid_keys:
        fallback_keys = _select_fallback_uniform_models(
            ml_models, pb_models, metadata["wf_ll_by_key"]
        )
        if not fallback_keys:
            stage_print("❌ Aucun modèle valide disponible")
            return None, None, None, {}
        uniform = 1.0 / len(fallback_keys)
        for k in fallback_keys:
            filtered_weights[k] = uniform
    else:
        filtered_weights = _apply_quality_filter_on_weights(
            filtered_weights, metadata["wf_ll_by_key"]
        )

    weights = _normalize_weights(
        filtered_weights, ENSEMBLE_MIN_WEIGHT, ENSEMBLE_MAX_WEIGHT
    )

    temperature_validated = False
    if TEMPERATURE_SCALING_ENABLED:
        try:
            calib_blend = _compute_ensemble_probas(
                ml_models,
                pb_models,
                Xs_calib,
                home_keys_calib,
                away_keys_calib,
                weights,
            )
            if calib_blend is not None:
                temp, temp_ok = _compute_optimal_temperature(
                    calib_blend, y_calib
                )
                weights["__temperature__"] = float(temp)
                temperature_validated = bool(temp_ok)
            else:
                weights["__temperature__"] = float(
                    TEMPERATURE_SCALING_DEFAULT
                )
        except Exception:
            weights["__temperature__"] = float(
                TEMPERATURE_SCALING_DEFAULT
            )

    weights["__temperature_validated__"] = bool(temperature_validated)
    weights["__ll_source_map__"] = metadata["ll_source_by_key"]
    weights["__calibration_status_map__"] = metadata[
        "calibration_status_by_key"
    ]
    weights["__walk_forward_ll_map__"] = metadata["wf_ll_by_key"]
    weights["__calibration_ll_map__"] = metadata["calib_ll_by_key"]
    weights["__walk_forward_fold_coverage_map__"] = metadata[
        "wf_fold_coverage_by_key"
    ]
    weights["__strict_competition__"] = bool(_STRICT_COMPETITION_DEFAULT)
    weights["__feature_names__"] = list(FEATURE_NAMES)
    weights["__feature_count__"] = int(EXPECTED_FEATURE_COUNT)

    if metadata["calib_ll_by_key"]:
        try:
            finite_lls = [
                float(v)
                for v in metadata["calib_ll_by_key"].values()
                if np.isfinite(float(v)) and float(v) > 0
            ]
            if finite_lls:
                weights["__calibration_logloss__"] = float(
                    np.mean(finite_lls)
                )
        except Exception:
            pass

    final_temp = float(weights.get("__temperature__", 1.0))
    test_metrics = _evaluate_ensemble_on_set(
        ml_models,
        pb_models,
        Xs_test,
        y_test,
        home_keys_test,
        away_keys_test,
        weights,
        final_temp,
    )
    if test_metrics is not None:
        ll_test, acc_test = test_metrics
        stage_print(
            f"📊 TEST FINAL : LogLoss {ll_test:.4f} | "
            f"Accuracy {acc_test * 100:.1f}%"
        )
        weights["__test_logloss__"] = float(ll_test)
        weights["__test_accuracy__"] = float(acc_test)

    checkpoint_mgr.save(weights_key, weights)
    checkpoint_mgr.save(metadata_key, metadata, sync_push=False)

    return ml_models, pb_models, scaler, weights


def save_cache_from_checkpoints(
    checkpoint_mgr: Any,
    ml_models: Dict[str, Any],
    pb_models: List[Tuple[str, Any]],
    pipeline: Any,
    df_past: pd.DataFrame,
    weights: Dict[str, float],
) -> None:
    from config import (
        CACHE_DIR,
        HISTORICAL_FILE,
        MODELS_FILE,
        SCALER_FILE,
        WEIGHTS_FILE,
    )
    import pickle

    def _atomic(path, obj):
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp, path)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _atomic(HISTORICAL_FILE, df_past)
    _atomic(MODELS_FILE, (ml_models, pb_models))
    _atomic(SCALER_FILE, pipeline)
    _atomic(WEIGHTS_FILE, weights)
