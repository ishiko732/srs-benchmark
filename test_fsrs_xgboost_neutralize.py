"""--xgboost_neutralize: named feature sets held at 0, recorded in result names and booster meta."""

import pytest

from config import load_config
from models.fsrs_xgboost import (
    FEATURE_NAMES,
    NEUTRALIZE_SETS,
    check_global_meta,
    neutralized_features,
)


def test_sets_name_real_features():
    for names in NEUTRALIZE_SETS.values():
        assert set(names) <= set(FEATURE_NAMES)
    assert set(NEUTRALIZE_SETS["user_recall"]) < set(NEUTRALIZE_SETS["user_ewma"])


def test_result_name_and_validation():
    cfg = load_config(
        ["--algo", "FSRS-7", "--xgboost", "per-user", "--xgboost_neutralize", "user_recall"]
    )
    assert cfg.get_evaluation_file_name() == "FSRS-7-xgboost-nz_user_recall"
    assert neutralized_features(cfg) == NEUTRALIZE_SETS["user_recall"]
    plain = load_config(["--algo", "FSRS-7", "--xgboost", "per-user"])
    assert neutralized_features(plain) == ()
    with pytest.raises(ValueError, match="requires --xgboost"):
        load_config(["--algo", "FSRS-7", "--xgboost_neutralize", "user_recall"])


def test_global_meta_must_match_neutralization():
    cfg = load_config(
        ["--algo", "FSRS-7", "--xgboost", "per-user", "--xgboost_neutralize", "user_ewma"]
    )
    cfg.xgboost = "global"  # check_global_meta only reads these fields
    meta = {"scope": "global", "baseModel": "FSRS-7", "priorMode": "trained"}
    with pytest.raises(ValueError, match="neutralized features"):
        check_global_meta(meta, cfg)  # boosters trained without neutralization
    check_global_meta({**meta, "neutralizedFeatures": list(NEUTRALIZE_SETS["user_ewma"])}, cfg)
