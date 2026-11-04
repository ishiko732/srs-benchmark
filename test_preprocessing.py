"""Comprehensive unit tests for data preprocessing and feature engineering.

Verifies 100% bit-exact equivalence (values, dtypes, indices, tensor shapes/dtypes,
and on-demand column rules) across all model feature engineers, CLI flags, and
UserDataLoader / partition discovery / pretraining IPC helpers.
"""

import sys
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import torch
from fsrs_optimizer import remove_non_continuous_rows, remove_outliers
from sklearn.model_selection import TimeSeriesSplit

from config import Config, create_parser
from data_loader import InsufficientUserDataError, UserDataLoader
from features import create_features
from features.dash_engineer import DashACTRFeatureEngineer, DashFeatureEngineer
from features.neural_engineer import NN17FeatureEngineer
from utils import cum_concat, get_bin


def _make_config(*cli_args: str, data_path: Path | None = None) -> Config:
    parser = create_parser()
    full_args = list(cli_args)
    if data_path is not None:
        full_args.extend(["--data", str(data_path)])
    return Config(parser.parse_args(full_args))


def _build_synthetic_revlogs(sort_index_by_card: bool = True) -> pd.DataFrame:
    """Build a rich multi-card synthetic revlog covering edge cases:
    - Interleaved chronological reviews across 35 cards
    - Initial reviews (elapsed_days=-1, elapsed_seconds=-1)
    - Same-day short-term reviews (elapsed_days=0, elapsed_seconds=0 and >0)
    - Lapses (rating=1), all ratings 1..4, plus an invalid rating (5) row
    - Outlier i==2 intervals (>100 for rating 3, >365 for rating 4) and
      subsequent i==3, 4 reviews to exercise both remove_outliers and
      remove_non_continuous_rows.
    """
    rng: Any = np.random.default_rng(42)
    rows: list[dict[str, int]] = []

    # 100 normal cards (>=25 per first_rating, above remove_outliers min threshold of 20)
    for cid in range(1, 101):
        first_r = int((cid % 4) + 1)
        day = cid % 5
        rows.append(
            {
                "card_id": cid,
                "day_offset": day,
                "rating": first_r,
                "state": 0,
                "duration": rng.integers(1500, 12000).item(),
                "elapsed_days": -1,
                "elapsed_seconds": -1,
            }
        )
        # Same-day short-term learning step on half the cards
        if cid % 2 == 0:
            sec = 0 if cid % 6 == 0 else rng.integers(60, 1800).item()
            rows.append(
                {
                    "card_id": cid,
                    "day_offset": day,
                    "rating": int(rng.choice([1, 3, 4])),
                    "state": 1,
                    "duration": rng.integers(1500, 9000).item(),
                    "elapsed_days": 0,
                    "elapsed_seconds": sec,
                }
            )
        # Common i==2 interval (delta_t = 3 days, >= 6 cards per rating so not removed as rare)
        day += 3
        rows.append(
            {
                "card_id": cid,
                "day_offset": day,
                "rating": 1 if cid % 5 == 0 else int(rng.choice([2, 3, 4])),
                "state": 2,
                "duration": rng.integers(2000, 15000).item(),
                "elapsed_days": 3,
                "elapsed_seconds": 3 * 86400 + rng.integers(0, 3600).item(),
            }
        )
        # Same-day relearning step after lapse
        if cid % 5 == 0:
            rows.append(
                {
                    "card_id": cid,
                    "day_offset": day,
                    "rating": 3,
                    "state": 3,
                    "duration": 3500,
                    "elapsed_days": 0,
                    "elapsed_seconds": 600,
                }
            )
        # Subsequent multi-day reviews
        for step in range(4):
            ivl = int(rng.choice([1, 4, 9, 20, 45]))
            day += ivl
            r_val = 1 if (step == 1 and cid % 7 == 0) else int(rng.choice([2, 3, 4]))
            rows.append(
                {
                    "card_id": cid,
                    "day_offset": day,
                    "rating": r_val,
                    "state": 2,
                    "duration": rng.integers(1200, 14000).item(),
                    "elapsed_days": ivl,
                    "elapsed_seconds": ivl * 86400 + rng.integers(10, 7200).item(),
                }
            )

    # Card 101: outlier i==2 (delta_t=250 > 100 for first_rating=3), followed by i=3, 4
    rows.extend(
        [
            {
                "card_id": 101,
                "day_offset": 1,
                "rating": 3,
                "state": 0,
                "duration": 4000,
                "elapsed_days": -1,
                "elapsed_seconds": -1,
            },
            {
                "card_id": 101,
                "day_offset": 251,
                "rating": 3,
                "state": 2,
                "duration": 4200,
                "elapsed_days": 250,
                "elapsed_seconds": 250 * 86400,
            },
            {
                "card_id": 101,
                "day_offset": 260,
                "rating": 4,
                "state": 2,
                "duration": 3800,
                "elapsed_days": 9,
                "elapsed_seconds": 9 * 86400,
            },
            {
                "card_id": 101,
                "day_offset": 280,
                "rating": 3,
                "state": 2,
                "duration": 3600,
                "elapsed_days": 20,
                "elapsed_seconds": 20 * 86400,
            },
        ]
    )

    # Card 102: outlier i==2 for rating 4 (delta_t=500 > 365) with a short-term review
    rows.extend(
        [
            {
                "card_id": 102,
                "day_offset": 2,
                "rating": 4,
                "state": 0,
                "duration": 3000,
                "elapsed_days": -1,
                "elapsed_seconds": -1,
            },
            {
                "card_id": 102,
                "day_offset": 2,
                "rating": 3,
                "state": 1,
                "duration": 3100,
                "elapsed_days": 0,
                "elapsed_seconds": 900,
            },
            {
                "card_id": 102,
                "day_offset": 502,
                "rating": 2,
                "state": 2,
                "duration": 5000,
                "elapsed_days": 500,
                "elapsed_seconds": 500 * 86400,
            },
            {
                "card_id": 102,
                "day_offset": 510,
                "rating": 3,
                "state": 2,
                "duration": 4100,
                "elapsed_days": 8,
                "elapsed_seconds": 8 * 86400,
            },
        ]
    )

    # Card 103: invalid rating row (rating=5) mixed with valid reviews
    rows.extend(
        [
            {
                "card_id": 103,
                "day_offset": 0,
                "rating": 5,
                "state": 0,
                "duration": 1000,
                "elapsed_days": -1,
                "elapsed_seconds": -1,
            },
            {
                "card_id": 103,
                "day_offset": 0,
                "rating": 2,
                "state": 0,
                "duration": 2500,
                "elapsed_days": -1,
                "elapsed_seconds": -1,
            },
            {
                "card_id": 103,
                "day_offset": 3,
                "rating": 3,
                "state": 2,
                "duration": 2600,
                "elapsed_days": 3,
                "elapsed_seconds": 3 * 86400,
            },
        ]
    )

    # Sort chronologically by day_offset (preserving per-card order within same day)
    df = pd.DataFrame(rows)
    df["_ord"] = np.arange(len(df))
    df = df.sort_values(by=["day_offset", "_ord"]).drop(columns=["_ord"])
    if sort_index_by_card:
        # Match real anki-revlogs-10k parquet index layout (index assigned when sorted by card_id)
        by_card = df.sort_values(by=["card_id"], kind="stable").reset_index(drop=True)
        df = by_card.sort_values(by=["day_offset"], kind="stable")
    else:
        df = df.reset_index(drop=True)
    return df


def _reference_base_pipeline(df_in: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Reference unoptimized pandas groupby implementation from main branch."""
    df = df_in.copy()
    df["review_th"] = range(1, df.shape[0] + 1)
    if "day_offset" in df.columns:
        df["nth_today"] = df.groupby("day_offset").cumcount() + 1
    df.sort_values(by=["card_id", "review_th"], inplace=True)
    df = df[df["rating"].isin([1, 2, 3, 4])].copy()
    if config.two_buttons:
        df["rating"] = df["rating"].replace({2: 3, 4: 3})
    df["i"] = df.groupby("card_id").cumcount() + 1
    df = df[df["i"] <= config.max_seq_len * 2].copy()

    df["delta_t"] = df["elapsed_days"].map(lambda x: max(0, x))
    if config.use_secs_intervals:
        df["delta_t_secs"] = df["elapsed_seconds"].map(lambda x: max(0.0, x / 86400))
    df["delta_t_int"] = df["elapsed_days"].map(lambda x: max(0, x))

    if not config.include_short_term:
        df = df[df["elapsed_days"] != 0].copy()
        df["i"] = df.groupby("card_id").cumcount() + 1

    t_non_secs = (
        df.groupby("card_id", group_keys=False)["delta_t"]
        .apply(lambda x: cum_concat([[v] for v in x.values]))
        .tolist()
    )
    r_list = (
        df.groupby("card_id", group_keys=False)["rating"]
        .apply(lambda x: cum_concat([[v] for v in x.values]))
        .tolist()
    )

    last_rating = []
    for t_sub, r_sub in zip(t_non_secs, r_list):
        for t_h, r_h in zip(t_sub, r_sub):
            flag = True
            for t, r in zip(reversed(t_h[:-1]), reversed(r_h[:-1])):
                if t > 0:
                    last_rating.append(r)
                    flag = False
                    break
            if flag:
                last_rating.append(r_h[0])
    df["last_rating"] = last_rating
    df["r_history"] = [",".join(map(str, item[:-1])) for sub in r_list for item in sub]

    if config.use_secs_intervals:
        t_secs = (
            df.groupby("card_id", group_keys=False)["delta_t_secs"]
            .apply(lambda x: cum_concat([[v] for v in x.values]))
            .tolist()
        )
        if config.equalize_test_with_non_secs:
            df["t_history"] = [
                ",".join(map(str, item[:-1])) for sub in t_non_secs for item in sub
            ]
            df["t_history_secs"] = [
                ",".join(map(str, item[:-1])) for sub in t_secs for item in sub
            ]
        else:
            df["t_history"] = [
                ",".join(map(str, item[:-1])) for sub in t_secs for item in sub
            ]
        df["delta_t"] = df["delta_t_secs"]
        t_active = t_secs
    else:
        df["t_history"] = [
            ",".join(map(str, item[:-1])) for sub in t_non_secs for item in sub
        ]
        t_active = t_non_secs

    # Model-specific features
    mname = config.model_name
    if mname in ("FSRS-6", "FSRS-7", "SBD", "RNN", "Transformer", "FSRS-rs"):
        cast(Any, df)["tensor"] = [
            torch.tensor((t_i[:-1], r_i[:-1]), dtype=torch.float32).transpose(0, 1)
            for t_sub, r_sub in zip(t_active, r_list)
            for t_i, r_i in zip(t_sub, r_sub)
        ]
    elif mname == "FSRS-6-one-step":
        cast(Any, df)["tensor"] = [
            torch.tensor((t_i[:-1], r_i[:-1]), dtype=torch.float32).transpose(0, 1)
            for t_sub, r_sub in zip(t_active, r_list)
            for t_i, r_i in zip(t_sub, r_sub)
        ]
        cast(Any, df)["inputs"] = [
            tuple(zip(t_i[:-1], r_i[:-1]))
            for t_sub, r_sub in zip(t_active, r_list)
            for t_i, r_i in zip(t_sub, r_sub)
        ]
    elif mname in ("LSTM", "GRU"):
        df["is_new_card"] = (df["card_id"] != df["card_id"].shift(1)).astype(int)
        df["cum_new_cards"] = df["is_new_card"].cumsum()
        df["diff_new_cards"] = df.groupby("card_id")["cum_new_cards"].diff().fillna(0)
        df["diff_reviews"] = np.maximum(
            0, -1 + df.groupby("card_id")["review_th"].diff().fillna(0)
        )
        df["cum_new_cards_today"] = df.groupby("day_offset")["is_new_card"].cumsum()
        df["cum_reviews_today"] = df.groupby("day_offset").cumcount()
        df["delta_t_days"] = df["elapsed_days"].map(lambda x: max(0, x))
        feat_cols = config.get_lstm_tensor_feature_names()
        cast(Any, df)["tensor"] = [
            torch.tensor(vals[:-1], dtype=torch.float32)
            for _, grp in df.groupby("card_id", sort=False)
            for vals in cum_concat(
                [[row] for row in grp[feat_cols].to_numpy().tolist()]
            )
        ]
    elif mname in ("DASH", "DASH[MCM]"):
        eng = DashFeatureEngineer(config)
        decay = mname == "DASH[MCM]"
        cast(Any, df)["tensor"] = [
            torch.tensor(
                eng._dash_tw_features(r_i[:-1], t_i[1:], enable_decay=decay),
                dtype=torch.float32,
            )
            for t_sub, r_sub in zip(t_active, r_list)
            for t_i, r_i in zip(t_sub, r_sub)
        ]
    elif mname == "DASH[ACT-R]":
        eng_actr = DashACTRFeatureEngineer(config)
        cast(Any, df)["tensor"] = [
            eng_actr._dash_actr_features(r_i[:-1], t_i[1:])
            for t_sub, r_sub in zip(t_active, r_list)
            for t_i, r_i in zip(t_sub, r_sub)
        ]
    elif mname == "HLR":
        cast(Any, df)["tensor"] = [
            torch.tensor(
                [
                    np.sqrt(sum(1 for r in r_i[:-1] if r > 1)),
                    np.sqrt(sum(1 for r in r_i[:-1] if r == 1)),
                ],
                dtype=torch.float32,
            )
            for r_sub in r_list
            for r_i in r_sub
        ]
    elif mname == "ACT-R":
        cast(Any, df)["tensor"] = [
            torch.cumsum(torch.tensor([t_i]), dim=1).T
            for t_sub in t_active
            for t_i in t_sub
        ]
    elif mname == "NN-17":
        nn_eng = NN17FeatureEngineer(config)
        cast(Any, df)["tensor"] = [
            torch.tensor(
                (t_i[:-1], r_i[:-1], nn_eng._r_history_to_l_history(r_i[:-1]))
            ).T
            for t_sub, r_sub in zip(t_active, r_list)
            for t_i, r_i in zip(t_sub, r_sub)
        ]
    elif mname == "Ebisu-v2":
        cast(Any, df)["sequence"] = [
            tuple(zip(t_i[:-1], r_i[:-1]))
            for t_sub, r_sub in zip(t_active, r_list)
            for t_i, r_i in zip(t_sub, r_sub)
        ]
    elif mname == "SM2":
        df["sequence"] = df["r_history"]

    # Postprocessing
    df["first_rating"] = df.groupby("card_id")["rating"].transform("first").astype(str)
    df["y"] = df["rating"].map(lambda x: {1: 0, 2: 1, 3: 1, 4: 1}[x])
    is_lapse = ((df["rating"] == 1) & (df["delta_t"].astype(str) != "0")).astype(int)
    df["rmse_bins_lapse"] = is_lapse.groupby(df["card_id"]).cumsum() - is_lapse
    if config.include_short_term:
        df = df[(df["delta_t"] != 0) | (df["i"] == 1)].copy()
    df["i"] = df["elapsed_days"].gt(0).groupby(df["card_id"]).cumsum().add(1)

    if not config.use_secs_intervals:
        filtered = (
            # pyrefly: ignore [bad-index]
            df[df["i"] == 2]
            .groupby(by=["first_rating"], as_index=False, group_keys=False)[df.columns]
            .apply(remove_outliers)
        )
        if filtered.empty:
            raise ValueError("No data after handling outliers and non-continuous rows")
        df[df["i"] == 2] = filtered
        df.dropna(inplace=True)
        # pyrefly: ignore [bad-index]
        df = df.groupby("card_id", as_index=False, group_keys=False)[df.columns].apply(
            remove_non_continuous_rows
        )
        if df.empty:
            raise ValueError("No data after handling outliers and non-continuous rows")

    return df[df["delta_t"] > 0].sort_values(by=["review_th"])


def _reference_create_features(df_in: pd.DataFrame, config: Config) -> pd.DataFrame:
    if config.use_secs_intervals and config.equalize_test_with_non_secs:
        import copy

        cfg_ns = copy.copy(config)
        cfg_ns.use_secs_intervals = False
        df_ns = _reference_base_pipeline(df_in, cfg_ns)

        cfg_s = copy.copy(config)
        cfg_s.use_secs_intervals = True
        df_s = _reference_base_pipeline(df_in, cfg_s)
        df_s["rmse_bins_lapse"] = df_ns["rmse_bins_lapse"]

        tscv = TimeSeriesSplit(n_splits=config.n_splits)
        for split_i, (_, ns_test_idx) in enumerate(tscv.split(df_ns)):
            ns_test = df_ns.iloc[ns_test_idx]
            df_s[f"{split_i}_train"] = df_s["review_th"] < ns_test["review_th"].min()
            df_s[f"{split_i}_test"] = df_s["review_th"].isin(ns_test["review_th"])
        return df_s
    return _reference_base_pipeline(df_in, config)


def _assert_frames_bit_exact(
    df_opt: pd.DataFrame, df_ref: pd.DataFrame, full_columns: bool = False
) -> None:
    assert df_opt.index.equals(df_ref.index)
    if full_columns:
        assert set(df_opt.columns) == set(df_ref.columns)
    else:
        assert set(df_opt.columns).issubset(set(df_ref.columns))

    for col in df_opt.columns:
        if col == "tensor":
            list_o = df_opt[col].tolist()
            list_r = df_ref[col].tolist()
            assert len(list_o) == len(list_r)
            for t_o, t_r in zip(list_o, list_r):
                assert t_o.dtype == t_r.dtype
                assert t_o.shape == t_r.shape
                assert torch.equal(t_o, t_r)
        elif col in ("inputs", "sequence"):
            assert df_opt[col].tolist() == df_ref[col].tolist()
        else:
            s_o = df_opt[col]
            s_r = df_ref[col]
            if not ({str(s_o.dtype), str(s_r.dtype)} <= {"object", "str", "string"}):
                assert s_o.dtype == s_r.dtype, f"{col}: {s_o.dtype} != {s_r.dtype}"
            assert s_o.equals(s_r), f"Column {col} mismatch"


@pytest.mark.parametrize(
    "cli_args",
    [
        ["--algo", "FSRS-6"],
        ["--algo", "FSRS-6", "--short"],
        ["--algo", "FSRS-6", "--two_buttons"],
        ["--algo", "FSRS-6", "--file"],
        ["--algo", "FSRS-7", "--short", "--secs"],
        ["--algo", "FSRS-7", "--short", "--secs", "--equalize_test_with_non_secs"],
        [
            "--algo",
            "FSRS-7",
            "--short",
            "--secs",
            "--equalize_test_with_non_secs",
            "--file",
        ],
        ["--algo", "SBD", "--short", "--secs", "--equalize_test_with_non_secs"],
        ["--algo", "LSTM", "--short", "--secs", "--duration"],
        ["--algo", "LSTM", "--short", "--secs", "--duration", "--file"],
        ["--algo", "GRU", "--short", "--secs"],
        ["--algo", "DASH", "--short", "--secs"],
        ["--algo", "DASH[MCM]", "--short", "--secs"],
        ["--algo", "DASH[ACT-R]", "--short", "--secs"],
        ["--algo", "HLR", "--short", "--secs"],
        ["--algo", "ACT-R"],
        ["--algo", "ACT-R", "--short", "--secs"],
        ["--algo", "NN-17"],
        ["--algo", "NN-17", "--short", "--secs"],
        ["--algo", "Ebisu-v2", "--short", "--secs"],
        ["--algo", "SM2", "--short", "--secs"],
        ["--algo", "FSRS-rs", "--short"],
        ["--algo", "RMSE-BINS-EXPLOIT", "--short"],
        ["--algo", "AVG", "--short"],
        ["--algo", "FSRS-6-one-step", "--short", "--secs"],
    ],
)
def test_all_engineers_bit_exact_against_reference(cli_args: list[str]) -> None:
    raw_df = _build_synthetic_revlogs(sort_index_by_card=True)
    cfg = _make_config(*cli_args)
    df_opt = create_features(raw_df, config=cfg)
    df_ref = _reference_create_features(raw_df, config=cfg)
    _assert_frames_bit_exact(df_opt, df_ref, full_columns=("--file" in cli_args))


def test_non_monotonic_index_outlier_continuity_fallback() -> None:
    raw_df = _build_synthetic_revlogs(sort_index_by_card=False)
    cfg = _make_config("--algo", "FSRS-6")
    df_opt = create_features(raw_df, config=cfg)
    df_ref = _reference_create_features(raw_df, config=cfg)
    _assert_frames_bit_exact(df_opt, df_ref)


def test_on_demand_history_columns_and_get_bin() -> None:
    raw_df = _build_synthetic_revlogs()

    # Default FSRS-7 skips unused string histories and auxiliary columns
    cfg_default = _make_config("--algo", "FSRS-7", "--short", "--secs")
    df_default = create_features(raw_df, config=cfg_default)
    for skipped_col in ("r_history", "t_history", "last_rating", "nth_today"):
        assert skipped_col not in df_default.columns

    # include_history_strings=True populates r_history and t_history for get_bin()
    cfg_hist = _make_config("--algo", "FSRS-6", "--short")
    cfg_hist.include_history_strings = True
    df_hist = create_features(raw_df, config=cfg_hist)
    df_ref = _reference_create_features(raw_df, config=cfg_hist)
    assert "r_history" in df_hist.columns and "t_history" in df_hist.columns
    bins_opt = [get_bin(df_hist.iloc[k]) for k in range(len(df_hist))]
    bins_ref = [get_bin(df_ref.iloc[k]) for k in range(len(df_ref))]
    assert bins_opt == bins_ref


def _reference_logreg_features(df_in: pd.DataFrame) -> pd.DataFrame:
    from models.logistic_regression import transform_elapsed_days_real_np

    df = df_in.copy()
    g = df.groupby("card_id", sort=False)
    df["feat_elapsed_real"] = g["delta_t_secs"].shift(1).fillna(0)
    df["feat_elapsed_int"] = g["delta_t_int"].shift(1).fillna(0)
    r = df["feature_rating"]

    same_day = (df["feat_elapsed_int"] == 0).astype(np.float32)
    success = (r > 1).astype(np.float32)
    fail = (r == 1).astype(np.float32)
    is_hard = (r == 2).astype(np.float32)
    better_than_hard = (r > 2).astype(np.float32)
    label_int = df["delta_t_int"].fillna(0)
    label_real = df["delta_t_secs"].fillna(0)
    label_is_same_day = (label_int == 0).astype(np.float32)

    is_first = np.zeros(len(df), dtype=np.float32)
    is_first[g.cumcount() == 0] = 1.0
    not_first = 1.0 - is_first

    r_clipped = np.clip(r.astype(np.int32) - 2, 0, 2)
    rating_onehot = np.zeros((len(df), 3), dtype=np.float32)
    rating_onehot[np.arange(len(df)), r_clipped] = 1.0
    rating_onehot *= (r > 1).astype(np.float32).to_numpy()[:, None]

    df["_r"] = r
    first_r = g["_r"].transform("first").astype(np.float32)
    first_r_clipped = np.clip(first_r.astype(np.int32) - 2, 0, 2)
    first_rating_onehot = np.zeros((len(df), 3), dtype=np.float32)
    first_rating_onehot[np.arange(len(df)), first_r_clipped] = 1.0
    first_rating_onehot *= (first_r > 1).astype(np.float32).to_numpy()[:, None]

    df["_same_day"] = same_day
    df["_non_same_day"] = 1 - same_day
    # pyrefly: ignore [unsupported-operation]
    df["_sd_fail"] = same_day * not_first * fail
    # pyrefly: ignore [unsupported-operation]
    df["_nsd_fail"] = (1 - same_day) * not_first * fail
    # pyrefly: ignore [unsupported-operation]
    df["_sd_pass"] = same_day * not_first * success
    # pyrefly: ignore [unsupported-operation]
    df["_nsd_pass"] = (1 - same_day) * not_first * success
    df["_pass"] = success

    num_same_day_fail = g["_sd_fail"].cumsum()
    num_nsd_fail = g["_nsd_fail"].cumsum()
    num_same_day_pass = g["_sd_pass"].cumsum()
    num_nsd_pass = g["_nsd_pass"].cumsum()
    num_pass = g["_pass"].cumsum()
    num_same_day = g["_same_day"].cumsum()
    num_non_same_day = g["_non_same_day"].cumsum()
    has_passed = (num_pass > 0).astype(np.float32)

    df["_elapsed_real"] = df["feat_elapsed_real"]
    times = g["_elapsed_real"].cumsum().astype(np.float32)
    fl_mask = (is_first.astype(bool)) | (success == 0)
    df["_flt"] = np.where(fl_mask, times, 0.0)
    running_max_flt = g["_flt"].cummax().astype(np.float32)
    time_since_first_or_lapse = times - running_max_flt

    t_elapsed_real = transform_elapsed_days_real_np(
        df["feat_elapsed_real"].astype(np.float32)
    )
    t_label_real = transform_elapsed_days_real_np(label_real.astype(np.float32))
    t_time_since_lapse = transform_elapsed_days_real_np(time_since_first_or_lapse)
    t_times = transform_elapsed_days_real_np(times)

    deg1 = np.stack(
        [
            np.ones(len(df), dtype=np.float32),
            rating_onehot[:, 0],
            rating_onehot[:, 1],
            rating_onehot[:, 2],
            # pyrefly: ignore [unsupported-operation]
            is_hard * is_first,
            # pyrefly: ignore [unsupported-operation]
            better_than_hard * is_first,
            np.log1p(num_same_day_fail),
            np.log1p(num_nsd_pass * (1 - label_is_same_day)),
            np.log1p(num_nsd_fail * (1 - label_is_same_day)),
            np.log1p(num_same_day_pass),
        ],
        axis=1,
    )
    deg0 = np.stack(
        [
            np.ones(len(df), dtype=np.float32),
            rating_onehot[:, 0],
            rating_onehot[:, 1],
            rating_onehot[:, 2],
            first_rating_onehot[:, 0],
            first_rating_onehot[:, 1],
            first_rating_onehot[:, 2],
            first_rating_onehot[:, 0] * is_first,
            first_rating_onehot[:, 1] * is_first,
            first_rating_onehot[:, 2] * is_first,
            np.log1p(num_same_day_fail),
            np.log1p(num_nsd_pass * (1 - label_is_same_day)),
            t_elapsed_real,
            np.log1p(num_nsd_fail),
            np.log1p(num_nsd_pass),
            np.log1p(num_same_day_pass * label_is_same_day),
            has_passed,
            t_time_since_lapse,
            label_is_same_day,
            t_times,
            (first_r > 1) * np.log1p(num_same_day_fail),
            (first_r > 1) * np.log1p(num_nsd_fail),
            (first_r > 1) * np.log1p(num_same_day),
            (first_r > 1) * np.log1p(num_non_same_day),
        ],
        axis=1,
    )
    v = t_label_real.to_numpy()[:, None]
    y = np.concatenate([deg1 * v, deg0], axis=1)
    cols = [f"feat_{i}" for i in range(y.shape[1])]
    return pd.DataFrame(y, index=df.index, columns=cols)


def test_logistic_regression_features() -> None:
    raw_df = _build_synthetic_revlogs()
    # Without --secs, LogisticRegressionEngineer skips feat_0..feat_33 (used for equalize split pass)
    cfg_non_secs = _make_config("--algo", "LogisticRegression")
    df_non_secs = create_features(raw_df, config=cfg_non_secs)
    assert [c for c in df_non_secs.columns if c.startswith("feat_")] == []

    for cli_args in [
        ["--algo", "LogisticRegression", "--short", "--secs"],
        [
            "--algo",
            "LogisticRegression",
            "--short",
            "--secs",
            "--equalize_test_with_non_secs",
        ],
    ]:
        cfg = _make_config(*cli_args)
        df_lr = create_features(raw_df, config=cfg)
        feat_cols = [c for c in df_lr.columns if c.startswith("feat_")]
        assert len(feat_cols) == 34
        feat_mat = df_lr[feat_cols].to_numpy()
        assert np.isfinite(feat_mat).all()
        ref_feats = _reference_logreg_features(df_lr.drop(columns=feat_cols))
        assert df_lr[feat_cols].equals(ref_feats)


def test_user_data_loader_partitions_and_missing_cards(tmp_path: Path) -> None:
    raw_df = _build_synthetic_revlogs()
    # User 1: normal cards + decks (leaving card 103 unmapped to test -1 fallback)
    u1_rev = tmp_path / "revlogs" / "user_id=1"
    u1_cards = tmp_path / "cards" / "user_id=1"
    u1_decks = tmp_path / "decks" / "user_id=1"
    u1_rev.mkdir(parents=True)
    u1_cards.mkdir(parents=True)
    u1_decks.mkdir(parents=True)

    raw_df.to_parquet(u1_rev / "data.parquet")
    pd.DataFrame(
        {
            "card_id": list(range(1, 103)),
            "note_id": [100 + c for c in range(1, 103)],
            "deck_id": [(c % 3) + 10 for c in range(1, 103)],
        }
    ).to_parquet(u1_cards / "data.parquet", index=False)
    pd.DataFrame(
        {
            "deck_id": [10, 11, 12],
            "parent_id": [-1, 10, 10],
            "preset_id": [1, 2, 2],
        }
    ).to_parquet(u1_decks / "data.parquet", index=False)

    # User 2: revlogs and decks exist, but cards/user_id=2 directory is absent
    u2_rev = tmp_path / "revlogs" / "user_id=2"
    u2_decks = tmp_path / "decks" / "user_id=2"
    u2_rev.mkdir(parents=True)
    u2_decks.mkdir(parents=True)
    raw_df.to_parquet(u2_rev / "data.parquet")
    pd.DataFrame({"deck_id": [10], "parent_id": [-1], "preset_id": [1]}).to_parquet(
        u2_decks / "data.parquet", index=False
    )

    # User 3: too few reviews (< 6) -> InsufficientUserDataError
    u3_rev = tmp_path / "revlogs" / "user_id=3"
    u3_rev.mkdir(parents=True)
    raw_df.iloc[:4].to_parquet(u3_rev / "data.parquet")

    for part_mode in ("deck", "preset"):
        cfg_fast = _make_config(
            "--algo",
            "FSRS-7",
            "--short",
            "--secs",
            "--equalize_test_with_non_secs",
            "--partitions",
            part_mode,
            data_path=tmp_path,
        )
        cfg_file = _make_config(
            "--algo",
            "FSRS-7",
            "--short",
            "--secs",
            "--equalize_test_with_non_secs",
            "--partitions",
            part_mode,
            "--file",
            data_path=tmp_path,
        )
        df_fast_u1 = UserDataLoader(cfg_fast).load_user_data(1)
        df_file_u1 = UserDataLoader(cfg_file).load_user_data(1)
        _assert_frames_bit_exact(df_fast_u1, df_file_u1, full_columns=False)
        assert (df_fast_u1.loc[df_fast_u1["card_id"] == 103, "partition"] == -1).all()

        df_fast_u2 = UserDataLoader(cfg_fast).load_user_data(2)
        df_file_u2 = UserDataLoader(cfg_file).load_user_data(2)
        _assert_frames_bit_exact(df_fast_u2, df_file_u2, full_columns=False)
        assert (df_fast_u2["partition"] == -1).all()

    cfg_u3 = _make_config("--algo", "FSRS-7", "--short", "--secs", data_path=tmp_path)
    with pytest.raises(InsufficientUserDataError):
        UserDataLoader(cfg_u3).load_user_data(3)


def test_discover_user_ids_and_pretrain_ipc(tmp_path: Path) -> None:
    raw_df = _build_synthetic_revlogs()
    for uid in (7, 2, 15):
        u_dir = tmp_path / "revlogs" / f"user_id={uid}"
        u_dir.mkdir(parents=True)
        raw_df.to_parquet(u_dir / "data.parquet")

    with patch.object(
        sys, "argv", ["script.py", "--algo", "FSRS-6", "--data", str(tmp_path)]
    ):
        import pretrain
        import script

        discovered = script._discover_user_ids(tmp_path / "revlogs")
        pq_ds = pq.ParquetDataset(tmp_path / "revlogs")
        expected = sorted(u.as_py() for u in pq_ds.partitioning.dictionaries[0])
        assert discovered == expected == [2, 7, 15]

        pretrain.config = _make_config(
            "--algo", "FSRS-6", "--short", "--secs", data_path=tmp_path
        )
        _, df_direct = pretrain.process_user(7)
        uid_ipc, df_part, packed = pretrain._process_user_ipc(7)
        assert uid_ipc == 7
        assert packed is not None
        df_unpacked = pretrain._unpack_user_ipc(df_part, packed)
        assert list(df_unpacked.columns) == list(df_direct.columns)
        _assert_frames_bit_exact(df_unpacked, df_direct, full_columns=True)
