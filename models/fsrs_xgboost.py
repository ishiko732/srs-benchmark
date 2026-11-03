"""
FSRS-XGBoost: an XGBoost log-odds residual on top of any trainable FSRS model.

    python script.py --algo FSRS-7 --xgboost per-user --short --secs --recency
    python script.py --algo FSRS-6 --default --xgboost per-user --short --secs

Each fold starts from that fold's ordinary fit of the base model (`--default` gives the
default-parameter prior instead, as for the base model itself). A gradient-boosted tree
ensemble then predicts a correction delta(x) in logit space, used as base_margin:

    p = sigmoid(logit(p_base) + alpha * delta(x))

alpha in [0, 1] is chosen on a walk-forward validation slice of the fold's train set, and
the booster is dropped (alpha = 0) when it does not improve that slice's log loss.

Training rows never see the fold's own weights: their base predictions come from the
previous fold's model (out-of-fold), and fold 0 uses a fit on its first 75% of rows. The
36 features (FEATURE_NAMES) cover the base model's prediction and memory state, the
current interval, the card's prior review history and the user's recent recall rate and
base-model error, all strictly causal. Features a base model does not have (FSRS-7's
short-term stability and two curve components) are NaN, which XGBoost routes natively.

Scopes:
- per-user: one booster per user and fold, trained on that user's train split.
  With --weights, the last fold's booster is saved per user (_save_user_booster),
  matching the last-fold parameters in the result file.
- global: one booster per fold index, pooled over --xgboost_anchors users picked evenly
  along the user-size order (each user weighs the same, at most 15,000 recent rows per
  fold). They are trained into --xgboost_boosters on the first run and reused after
  that; the meta file records the base model and prior they belong to.
- global-split: as global, but the users are split in two halves (user_id < 5000 and
  >= 5000, as for RWKV) with anchors and boosters per half. Each user is scored with the
  other half's boosters, so no user is scored by a booster trained on its own reviews.

    python script.py --algo FSRS-7 --xgboost global --xgboost_boosters boosters/fsrs7 \
        --short --secs --recency

Like the base model, this is a prediction made at review time from the review history.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.signal
import torch
import xgboost as xgb

from config import Config


PROB_EPS: float = 1e-6
TIME_EPS: float = 1e-5

FEATURE_NAMES: tuple[str, ...] = (
    # Group A: base-model probability & memory state (10; FSRS-7-only columns are NaN otherwise)
    "fsrs_logit_p",
    "fsrs_p",
    "fsrs_s_long_log",
    "fsrs_s_short_log",
    "fsrs_s_ratio_log",
    "fsrs_d",
    "fsrs_r1",
    "fsrs_r2",
    "fsrs_r_gap",
    "fsrs_logit_vs_default",
    # Group B: Current interval & time-interval deviations (8)
    "log_delta_t",
    "is_same_day",
    "log_t_over_s_long",
    "log_t_over_s_short",
    "log_t_over_prev_t",
    "log_t_over_mean_t",
    "log_t_over_max_t",
    "subday_vs_day_gap",
    # Group C: Card repetition, lapse & rating history strictly prior to current review (10)
    "log_rep_count",
    "log_non_sameday_reps",
    "log_lapses",
    "log_total_fails",
    "card_hist_recall_rate",
    "card_mean_rating",
    "last_raw_rating",
    "first_hist_rating",
    "consec_passes_log",
    "log_time_since_last_fail",
    # Group D: Strictly causal user-session & online residual tracking (8)
    "log_nth_today",
    "log_user_day_gap",
    "log_card_age",
    "user_ewma_recall_20",
    "user_ewma_recall_200",
    "user_mode_ewma_recall_50",
    "user_ewma_fsrs_err_50",
    "user_mode_ewma_fsrs_err_100",
)


def clip_prob(p: np.ndarray, eps: float = PROB_EPS) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)


def logit_from_prob(p: np.ndarray, eps: float = PROB_EPS) -> np.ndarray:
    """Transform probability p in (0, 1) to log-odds logit(p) = ln(p / (1 - p))."""
    pc = clip_prob(p, eps=eps)
    return np.log(pc / (1.0 - pc))


def sigmoid(logits: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    x = np.asarray(logits, dtype=np.float64)
    pos = x >= 0.0
    out = np.empty_like(x, dtype=np.float64)
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def bce_with_logits(
    logits: np.ndarray,
    y_true: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Numerically stable binary cross-entropy loss from raw logits (never fails on single-class slices)."""
    x = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.float64)
    per_row = np.maximum(x, 0.0) - x * y + np.log1p(np.exp(-np.abs(x)))
    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=np.float64)
        w_sum = float(np.sum(w))
        if w_sum > 0.0:
            return float(np.sum(per_row * w) / w_sum)
    return float(np.mean(per_row))


@dataclass(frozen=True)
class BaseStateArrays:
    """Pre-review base-model outputs aligned 1-to-1 with a DataFrame's rows.

    s_short, r1 and r2 exist only for FSRS-7; other models fill them with NaN.
    """

    p_fsrs: np.ndarray
    logit_p: np.ndarray
    s_long: np.ndarray
    s_short: np.ndarray
    d: np.ndarray
    r1: np.ndarray
    r2: np.ndarray

    def slice_rows(self, idx: np.ndarray) -> BaseStateArrays:
        return BaseStateArrays(
            p_fsrs=self.p_fsrs[idx],
            logit_p=self.logit_p[idx],
            s_long=self.s_long[idx],
            s_short=self.s_short[idx],
            d=self.d[idx],
            r1=self.r1[idx],
            r2=self.r2[idx],
        )

    @staticmethod
    def concat(parts: list[BaseStateArrays]) -> BaseStateArrays:
        return BaseStateArrays(
            *(
                np.concatenate([getattr(p, f) for p in parts])
                for f in BaseStateArrays.__dataclass_fields__
            )
        )


def _empty_states() -> BaseStateArrays:
    e = np.empty(0, dtype=np.float64)
    return BaseStateArrays(e, e, e, e, e, e, e)


def extract_base_states(
    model: Any, dataset: pd.DataFrame, config: Config
) -> BaseStateArrays:
    """Pre-review base predictions (identical to Collection.batch_predict) plus memory state."""
    from models.fsrs_v7 import FSRS7

    if len(dataset) == 0:
        return _empty_states()
    if isinstance(model, FSRS7):
        return _extract_fsrs7_states(model, dataset, config.device)

    from utils import Collection

    retentions, stabilities, difficulties = Collection(model, config).batch_predict(
        dataset
    )
    n = len(dataset)
    p = np.asarray(retentions, dtype=np.float64)
    nan = np.full(n, np.nan)
    return BaseStateArrays(
        p_fsrs=p,
        logit_p=logit_from_prob(p),
        s_long=np.asarray(stabilities, dtype=np.float64) if stabilities else nan,
        s_short=nan,
        d=np.asarray(difficulties, dtype=np.float64) if difficulties else nan,
        r1=nan,
        r2=nan,
    )


def _extract_fsrs7_states(
    model: Any, dataset: pd.DataFrame, device: torch.device
) -> BaseStateArrays:
    """Run FSRS-7 recurrence over strictly prior history tensors and extract pre-review
    retention probability p_fsrs (bit-exact with Collection.batch_predict) plus
    dual-stability states (s_long, s_short, d) and recall components (r1, r2).
    """
    from fsrs_optimizer import BatchDataset, BatchLoader, DevicePrefetchLoader

    model = model.to(device=device)
    model.eval()

    n_rows = len(dataset)
    if n_rows == 0:
        return _empty_states()

    seq_lens_arr = np.fromiter(
        (int(t.shape[0]) for t in dataset["tensor"]),
        dtype=np.int64,
        count=n_rows,
    )
    sort_idx = np.argsort(seq_lens_arr, kind="stable")
    inv_idx = np.empty(n_rows, dtype=np.int64)
    inv_idx[sort_idx] = np.arange(n_rows, dtype=np.int64)
    sorted_dataset = dataset.iloc[sort_idx]

    batch_dataset = BatchDataset(sorted_dataset, batch_size=2048, sort_by_length=False)
    batch_loader = BatchLoader(batch_dataset, shuffle=False)
    device_loader = DevicePrefetchLoader(batch_loader, target_device=device)

    p_parts: list[np.ndarray] = []
    sl_parts: list[np.ndarray] = []
    ss_parts: list[np.ndarray] = []
    d_parts: list[np.ndarray] = []
    r1_parts: list[np.ndarray] = []
    r2_parts: list[np.ndarray] = []

    with torch.no_grad():
        w = model.w
        decay2 = -w[24].clamp(0.01, 0.95)
        factor2 = w[26].pow(decay2.pow(-1.0)) - 1.0
        w32_shift = w[32] - 0.3

        for sequences, delta_ts, _labels, seq_lens, _weights in device_loader:
            real_batch_size = int(seq_lens.shape[0])
            outputs, _ = model.forward(sequences)
            final = outputs[
                seq_lens - 1,
                torch.arange(real_batch_size, device=device),
            ]
            s_long = final[:, 0]
            s_short = final[:, 1]
            d = final[:, 2]

            t_clamped = delta_ts.clamp(min=0.0)
            r1 = model.short_component_recall(t_clamped, s_short)
            d_timescale = ((d - 5.0) * w32_shift).exp()
            r2 = ((t_clamped / s_long) * factor2 * d_timescale + 1.0).pow(decay2)
            retentions = model.forgetting_curve(delta_ts, s_long, s_short, d)

            p_parts.append(retentions.detach().cpu().numpy().astype(np.float64))
            sl_parts.append(s_long.detach().cpu().numpy().astype(np.float64))
            ss_parts.append(s_short.detach().cpu().numpy().astype(np.float64))
            d_parts.append(d.detach().cpu().numpy().astype(np.float64))
            r1_parts.append(r1.detach().cpu().numpy().astype(np.float64))
            r2_parts.append(r2.detach().cpu().numpy().astype(np.float64))

    p_fsrs = np.concatenate(p_parts, axis=0)[inv_idx]
    return BaseStateArrays(
        p_fsrs=p_fsrs,
        logit_p=logit_from_prob(p_fsrs),
        s_long=np.concatenate(sl_parts, axis=0)[inv_idx],
        s_short=np.concatenate(ss_parts, axis=0)[inv_idx],
        d=np.concatenate(d_parts, axis=0)[inv_idx],
        r1=np.concatenate(r1_parts, axis=0)[inv_idx],
        r2=np.concatenate(r2_parts, axis=0)[inv_idx],
    )


def causal_ewma(series: np.ndarray, alpha: float, init_val: float) -> np.ndarray:
    """Strictly causal EWMA: out[0] = init_val, out[m] = (1 - alpha)*out[m-1] + alpha*series[m-1].
    out[m] NEVER depends on series[m] or any series[k] for k >= m.
    """
    n = len(series)
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    out[0] = init_val
    if n == 1:
        return out
    shifted = np.asarray(series[:-1], dtype=np.float64) - init_val
    filtered = scipy.signal.lfilter([alpha], [1.0, -(1.0 - alpha)], shifted)
    out[1:] = init_val + filtered
    return out


def causal_masked_ewma(
    series: np.ndarray,
    mask: np.ndarray,
    alpha: float,
    init_val: float,
) -> np.ndarray:
    """Strictly causal EWMA updated ONLY on prior rows r < m where mask[r] is True."""
    n = len(series)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    bool_mask = np.asarray(mask, dtype=bool)
    mask_int = bool_mask.astype(np.int64)
    prior_true_count = np.cumsum(mask_int) - mask_int
    vals = np.asarray(series, dtype=np.float64)[bool_mask]
    n_true = len(vals)
    ewma_after_k = np.empty(n_true + 1, dtype=np.float64)
    ewma_after_k[0] = init_val
    if n_true > 0:
        ewma_after_k[1:] = init_val + scipy.signal.lfilter(
            [alpha],
            [1.0, -(1.0 - alpha)],
            vals - init_val,
        )
    return ewma_after_k[prior_true_count]


def causal_mode_matched_ewma(
    series: np.ndarray,
    is_same_day: np.ndarray,
    alpha: float,
    init_val: float,
) -> np.ndarray:
    """At each row m, returns the strictly causal EWMA over prior same-day rows (r < m)
    if row m is same-day, else over prior non-same-day rows (r < m).
    """
    sd_mask = np.asarray(is_same_day, dtype=bool)
    ewma_sd = causal_masked_ewma(series, sd_mask, alpha=alpha, init_val=init_val)
    ewma_lt = causal_masked_ewma(series, ~sd_mask, alpha=alpha, init_val=init_val)
    return np.where(sd_mask, ewma_sd, ewma_lt)


@dataclass(frozen=True)
class StaticUserFeatures:
    """Precomputed fold-invariant strictly causal features aligned to full user `dataset`."""

    review_th: np.ndarray
    y: np.ndarray
    is_same_day_bool: np.ndarray
    # Group B static parts
    log_delta_t: np.ndarray
    is_same_day: np.ndarray
    log_prev_t: np.ndarray
    log_mean_t: np.ndarray
    log_max_t: np.ndarray
    subday_vs_day_gap: np.ndarray
    # Group C card history parts (from pre-review `tensor` ONLY)
    log_rep_count: np.ndarray
    log_non_sameday_reps: np.ndarray
    log_lapses: np.ndarray
    log_total_fails: np.ndarray
    card_hist_recall_rate: np.ndarray
    card_mean_rating: np.ndarray
    last_raw_rating: np.ndarray
    first_hist_rating: np.ndarray
    consec_passes_log: np.ndarray
    log_time_since_last_fail: np.ndarray
    # Group D user-session & causal recall EWMAs
    log_nth_today: np.ndarray
    log_user_day_gap: np.ndarray
    log_card_age: np.ndarray
    user_ewma_recall_20: np.ndarray
    user_ewma_recall_200: np.ndarray
    user_mode_ewma_recall_50: np.ndarray

    def locate_rows(self, sub_df: pd.DataFrame) -> np.ndarray:
        sub_rth = sub_df["review_th"].to_numpy(dtype=np.int64)
        pos = np.searchsorted(self.review_th, sub_rth)
        if np.any(pos >= len(self.review_th)) or not np.array_equal(
            self.review_th[pos], sub_rth
        ):
            raise RuntimeError(
                "Sub-DataFrame review_th does not align with user dataset"
            )
        return pos


def precompute_static_user_features(dataset: pd.DataFrame) -> StaticUserFeatures:
    """Vectorized extraction of all fold-invariant strictly causal features for one user.

    Contract:
    - Card-level history features are derived ONLY from `dataset["tensor"]` (pre-review
      [(t_0, r_0), ..., (t_{j-1}, r_{j-1})]), `delta_t`, `delta_t_int`, and `i`.
    - User-level recall history features use strict causal lag (`r < m` in ascending
      `review_th` order) and NEVER observe current or future `y`, `rating`, `duration`, or `state`.
    """
    review_th = dataset["review_th"].to_numpy(dtype=np.int64)
    if len(review_th) > 1 and np.any(np.diff(review_th) <= 0):
        raise ValueError("dataset must be strictly sorted by ascending review_th")

    n = len(dataset)
    delta_t = np.maximum(dataset["delta_t"].to_numpy(dtype=np.float64), 0.0)
    delta_t_int = np.maximum(dataset["delta_t_int"].to_numpy(dtype=np.float64), 0.0)
    i_col = np.maximum(dataset["i"].to_numpy(dtype=np.float64), 1.0)
    nth_today = np.maximum(dataset["nth_today"].to_numpy(dtype=np.float64), 1.0)
    day_offset = (
        dataset["day_offset"].to_numpy(dtype=np.float64)
        if "day_offset" in dataset.columns
        else np.zeros(n, dtype=np.float64)
    )
    y = dataset["y"].to_numpy(dtype=np.float64)

    is_same_day_bool = delta_t_int == 0.0
    is_same_day = is_same_day_bool.astype(np.float64)
    log_delta_t = np.log(delta_t + TIME_EPS)
    subday_vs_day_gap = log_delta_t - np.log(delta_t_int + TIME_EPS)

    tensors = list(dataset["tensor"])
    lengths = np.fromiter((int(t.shape[0]) for t in tensors), dtype=np.int64, count=n)
    if np.any(lengths < 1):
        raise ValueError(
            "Expected every evaluated review in FSRS dataset to have seq_len >= 1"
        )

    flat_tensor = torch.cat(tensors, dim=0).detach().cpu().numpy()
    del tensors
    flat_t = np.maximum(flat_tensor[:, 0].astype(np.float64), 0.0)
    flat_r = flat_tensor[:, 1].astype(np.float64)
    del flat_tensor

    offsets = np.cumsum(lengths, dtype=np.int64)
    starts = offsets - lengths
    ends = offsets - 1

    not_init = np.ones(len(flat_t), dtype=bool)
    not_init[starts] = False

    first_hist_rating = flat_r[starts]
    last_raw_rating = flat_r[ends]
    prev_delta_t = flat_t[ends]

    is_pass = (flat_r > 1.5).astype(np.int32)
    n_pass = np.add.reduceat(is_pass, starts).astype(np.float64)
    del is_pass
    lengths_f = lengths.astype(np.float64)
    n_fail = lengths_f - n_pass

    is_prior_lapse = (not_init & (flat_t > 0.0) & (flat_r < 1.5)).astype(np.int32)
    del not_init
    n_lapses = np.add.reduceat(is_prior_lapse, starts).astype(np.float64)
    del is_prior_lapse

    sum_ratings = np.add.reduceat(flat_r, starts)
    card_mean_rating = sum_ratings / lengths_f
    card_hist_recall_rate = (n_pass + 1.0) / (lengths_f + 2.0)

    sum_prev_t = np.add.reduceat(flat_t, starts)
    max_prev_t = np.maximum.reduceat(flat_t, starts)
    n_pos_steps = np.maximum(lengths_f - 1.0, 1.0)
    mean_prev_t = sum_prev_t / n_pos_steps

    card_age_now = sum_prev_t + delta_t

    flat_idx = np.arange(len(flat_r), dtype=np.int64)
    fail_pos = np.where(flat_r < 1.5, flat_idx, -1)
    del flat_idx
    last_fail_idx = np.maximum.reduceat(fail_pos, starts)
    del fail_pos
    has_prior_fail = last_fail_idx >= starts
    consec_passes = np.where(has_prior_fail, ends - last_fail_idx, lengths).astype(
        np.float64
    )

    cum_t = np.empty(len(flat_t) + 1, dtype=np.float64)
    cum_t[0] = 0.0
    np.cumsum(flat_t, out=cum_t[1:])
    safe_after_fail_idx = np.where(has_prior_fail, last_fail_idx + 1, starts)
    time_since_last_fail = (cum_t[offsets] - cum_t[safe_after_fail_idx]) + delta_t

    day_gap = np.empty(n, dtype=np.float64)
    if n > 0:
        day_gap[0] = 0.0
        if n > 1:
            day_gap[1:] = np.maximum(np.diff(day_offset), 0.0)

    user_ewma_recall_20 = causal_ewma(y, alpha=0.05, init_val=0.85)
    user_ewma_recall_200 = causal_ewma(y, alpha=0.005, init_val=0.85)
    user_mode_ewma_recall_50 = causal_mode_matched_ewma(
        y,
        is_same_day_bool,
        alpha=0.02,
        init_val=0.85,
    )

    return StaticUserFeatures(
        review_th=review_th,
        y=y,
        is_same_day_bool=is_same_day_bool,
        log_delta_t=log_delta_t,
        is_same_day=is_same_day,
        log_prev_t=np.log(prev_delta_t + TIME_EPS),
        log_mean_t=np.log(mean_prev_t + TIME_EPS),
        log_max_t=np.log(max_prev_t + TIME_EPS),
        subday_vs_day_gap=subday_vs_day_gap,
        log_rep_count=np.log(lengths_f),
        log_non_sameday_reps=np.log(i_col),
        log_lapses=np.log1p(n_lapses),
        log_total_fails=np.log1p(n_fail),
        card_hist_recall_rate=card_hist_recall_rate,
        card_mean_rating=card_mean_rating,
        last_raw_rating=last_raw_rating,
        first_hist_rating=first_hist_rating,
        consec_passes_log=np.log1p(consec_passes),
        log_time_since_last_fail=np.log(time_since_last_fail + TIME_EPS),
        log_nth_today=np.log(nth_today),
        log_user_day_gap=np.log1p(day_gap),
        log_card_age=np.log(card_age_now + TIME_EPS),
        user_ewma_recall_20=user_ewma_recall_20,
        user_ewma_recall_200=user_ewma_recall_200,
        user_mode_ewma_recall_50=user_mode_ewma_recall_50,
    )


def compute_causal_fsrs_error_features(
    y_seq: np.ndarray,
    p_fsrs_seq: np.ndarray,
    is_same_day_seq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute strictly causal EWMAs of prior FSRS prediction residuals (y_{r} - p_{fsrs, r}, r < m)
    over a chronologically ordered sequence (e.g. [train_fold, test_fold]).
    """
    err = np.asarray(y_seq, dtype=np.float64) - np.asarray(p_fsrs_seq, dtype=np.float64)
    ewma_err_50 = causal_ewma(err, alpha=0.02, init_val=0.0)
    mode_ewma_err_100 = causal_mode_matched_ewma(
        err,
        is_same_day_seq,
        alpha=0.01,
        init_val=0.0,
    )
    return ewma_err_50, mode_ewma_err_100


def build_feature_matrix(
    static: StaticUserFeatures,
    row_pos: np.ndarray,
    base_states: BaseStateArrays,
    default_logit_p: np.ndarray,
    ewma_fsrs_err_50: np.ndarray,
    mode_ewma_fsrs_err_100: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the [N, 36] float32 feature matrix X and float32 base_margin = logit(p_fsrs)."""
    log_sl = np.log(np.maximum(base_states.s_long, 0.0) + TIME_EPS)
    log_ss = np.log(np.maximum(base_states.s_short, 0.0) + TIME_EPS)
    log_dt = static.log_delta_t[row_pos]

    cols = (
        # Group A (10)
        base_states.logit_p,
        base_states.p_fsrs,
        log_sl,
        log_ss,
        log_ss - log_sl,
        base_states.d,
        base_states.r1,
        base_states.r2,
        base_states.r1 - base_states.r2,
        base_states.logit_p - default_logit_p,
        # Group B (8)
        log_dt,
        static.is_same_day[row_pos],
        log_dt - log_sl,
        log_dt - log_ss,
        log_dt - static.log_prev_t[row_pos],
        log_dt - static.log_mean_t[row_pos],
        log_dt - static.log_max_t[row_pos],
        static.subday_vs_day_gap[row_pos],
        # Group C (10)
        static.log_rep_count[row_pos],
        static.log_non_sameday_reps[row_pos],
        static.log_lapses[row_pos],
        static.log_total_fails[row_pos],
        static.card_hist_recall_rate[row_pos],
        static.card_mean_rating[row_pos],
        static.last_raw_rating[row_pos],
        static.first_hist_rating[row_pos],
        static.consec_passes_log[row_pos],
        static.log_time_since_last_fail[row_pos],
        # Group D (8)
        static.log_nth_today[row_pos],
        static.log_user_day_gap[row_pos],
        static.log_card_age[row_pos],
        static.user_ewma_recall_20[row_pos],
        static.user_ewma_recall_200[row_pos],
        static.user_mode_ewma_recall_50[row_pos],
        ewma_fsrs_err_50,
        mode_ewma_fsrs_err_100,
    )
    if len(cols) != len(FEATURE_NAMES):
        raise RuntimeError(
            f"Expected {len(FEATURE_NAMES)} feature columns, got {len(cols)}"
        )

    X = np.column_stack(cols).astype(np.float32, copy=False)
    base_margin = base_states.logit_p.astype(np.float32, copy=False)
    return X, base_margin


@dataclass(frozen=True)
class XGBoostConfig:
    """Hyperparameters for the walk-forward XGBoost residual."""

    max_depth: int = 4
    learning_rate: float = 0.03
    num_boost_round: int = 250
    early_stopping_rounds: int = 25
    subsample: float = 0.8
    colsample_bytree: float = 0.75
    reg_lambda: float = 8.0
    reg_alpha: float = 0.5
    max_delta_step: float = 1.5
    min_child_hessian_ratio: float = 0.015
    min_child_weight_floor: float = 2.0
    min_child_weight_ceil: float = 40.0
    recency_floor: float = 0.35
    recency_power: float = 2.0
    in_sample_b0_weight: float = 0.35
    min_subtrain_rows: int = 80
    min_val_rows: int = 25
    min_improvement_eps: float = 1e-6
    min_alpha: float = 0.01
    nthread: int = 1
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_overrides(cls, overrides: dict[str, Any] | None) -> XGBoostConfig:
        if not overrides:
            return cls()
        valid_keys = set(cls.__dataclass_fields__.keys())
        unknown = set(overrides.keys()) - valid_keys
        if unknown:
            raise ValueError(f"Unknown XGBoostConfig keys: {sorted(unknown)}")
        return cls(**overrides)


@dataclass
class FittedBooster:
    """Trained per-fold residual booster + validation-calibrated shrinkage factor alpha."""

    booster: xgb.Booster | None
    best_ntree: int
    alpha: float
    base_val_loss: float
    cal_val_loss: float
    min_child_weight: float
    gain_importance: dict[str, float]


def compute_recency_weights(
    n_rows: int,
    config: XGBoostConfig,
    *,
    b0_sub_end: int = 0,
) -> np.ndarray:
    """Compute normalized sample weights (mean = 1.0) emphasizing more recent reviews,
    with optional downweighting of in-sample Fold-0 subtrain rows.
    """
    if n_rows <= 0:
        return np.empty(0, dtype=np.float64)
    x = np.arange(n_rows, dtype=np.float64) / float(max(n_rows, 1))
    w = config.recency_floor + (1.0 - config.recency_floor) * np.power(
        x, config.recency_power
    )
    if 0 < b0_sub_end <= n_rows:
        w[:b0_sub_end] *= config.in_sample_b0_weight
    mean_w = float(np.mean(w))
    if mean_w > 0.0:
        w = w / mean_w
    return w


def _adaptive_min_child_weight(
    base_margin: np.ndarray,
    sample_weight: np.ndarray,
    config: XGBoostConfig,
) -> float:
    """Scale min_child_weight to the prior logistic Hessian sum sum_i w_i * p_i * (1 - p_i)."""
    p0 = sigmoid(base_margin)
    w_norm = np.asarray(sample_weight, dtype=np.float64)
    w_mean = float(np.mean(w_norm))
    if w_mean > 0.0:
        w_norm = w_norm / w_mean
    hessian_sum = float(np.sum(w_norm * p0 * (1.0 - p0)))
    raw = config.min_child_hessian_ratio * hessian_sum
    return float(
        np.clip(raw, config.min_child_weight_floor, config.min_child_weight_ceil)
    )


def _xgb_params(
    config: XGBoostConfig,
    min_child_weight: float,
    seed: int,
) -> dict[str, Any]:
    return {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "max_depth": config.max_depth,
        "learning_rate": config.learning_rate,
        "subsample": config.subsample,
        "colsample_bytree": config.colsample_bytree,
        "reg_lambda": config.reg_lambda,
        "reg_alpha": config.reg_alpha,
        "max_delta_step": config.max_delta_step,
        "min_child_weight": min_child_weight,
        "nthread": config.nthread,
        "seed": seed,
        "verbosity": 0,
    }


def _make_dmatrix(
    X: np.ndarray,
    base_margin: np.ndarray,
    y: np.ndarray | None = None,
    weight: np.ndarray | None = None,
    *,
    nthread: int = 1,
) -> xgb.DMatrix:
    X_f32 = np.asarray(X, dtype=np.float32)
    m_f32 = np.asarray(base_margin, dtype=np.float32)
    y_f32 = np.asarray(y, dtype=np.float32) if y is not None else None
    w_f32 = np.asarray(weight, dtype=np.float32) if weight is not None else None
    dmat = xgb.DMatrix(
        X_f32,
        label=y_f32,
        weight=w_f32,
        feature_names=list(FEATURE_NAMES),
        nthread=nthread,
    )
    dmat.set_base_margin(m_f32)
    return dmat


def fit_fold_calibrator(
    X_sub: np.ndarray,
    y_sub: np.ndarray,
    m_sub: np.ndarray,
    w_sub: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    m_val: np.ndarray,
    w_val: np.ndarray,
    X_full: np.ndarray,
    y_full: np.ndarray,
    m_full: np.ndarray,
    w_full: np.ndarray,
    config: XGBoostConfig,
    *,
    fold_idx: int = 0,
) -> FittedBooster:
    """Fit walk-forward XGBoost residual booster with OOF early stopping,
    1D convex log-odds validation shrinkage, and full-fold refit.
    """
    base_val_loss = bce_with_logits(m_val, y_val, sample_weight=w_val)

    if (
        len(y_sub) < config.min_subtrain_rows
        or len(y_val) < config.min_val_rows
        or np.all(y_sub == y_sub[0])
    ):
        return FittedBooster(
            booster=None,
            best_ntree=0,
            alpha=0.0,
            base_val_loss=base_val_loss,
            cal_val_loss=base_val_loss,
            min_child_weight=config.min_child_weight_floor,
            gain_importance={},
        )

    w_sub_norm = np.asarray(w_sub, dtype=np.float64)
    if float(np.mean(w_sub_norm)) > 0.0:
        w_sub_norm = w_sub_norm / float(np.mean(w_sub_norm))

    w_val_norm = np.asarray(w_val, dtype=np.float64)
    if float(np.mean(w_val_norm)) > 0.0:
        w_val_norm = w_val_norm / float(np.mean(w_val_norm))

    mcw_sub = _adaptive_min_child_weight(m_sub, w_sub_norm, config)
    fold_seed = int(config.seed + fold_idx * 97)

    dsub = _make_dmatrix(X_sub, m_sub, y=y_sub, weight=w_sub_norm)
    dval = _make_dmatrix(X_val, m_val, y=y_val, weight=w_val_norm)

    val_booster = xgb.train(
        params=_xgb_params(config, min_child_weight=mcw_sub, seed=fold_seed),
        dtrain=dsub,
        num_boost_round=config.num_boost_round,
        evals=[(dval, "val")],
        early_stopping_rounds=config.early_stopping_rounds,
        verbose_eval=False,
    )

    best_ntree = int(getattr(val_booster, "best_iteration", 0)) + 1
    if best_ntree <= 0:
        return FittedBooster(
            booster=None,
            best_ntree=0,
            alpha=0.0,
            base_val_loss=base_val_loss,
            cal_val_loss=base_val_loss,
            min_child_weight=mcw_sub,
            gain_importance={},
        )

    raw_margin_val = val_booster.predict(
        dval,
        iteration_range=(0, best_ntree),
        output_margin=True,
    ).astype(np.float64)
    m_val_f64 = np.asarray(m_val, dtype=np.float64)
    delta_val = raw_margin_val - m_val_f64

    if np.all(np.abs(delta_val) < 1e-9):
        return FittedBooster(
            booster=None,
            best_ntree=0,
            alpha=0.0,
            base_val_loss=base_val_loss,
            cal_val_loss=base_val_loss,
            min_child_weight=mcw_sub,
            gain_importance={},
        )

    def _val_objective(alpha: float) -> float:
        return bce_with_logits(
            m_val_f64 + float(alpha) * delta_val,
            y_val,
            sample_weight=w_val_norm,
        )

    opt = scipy.optimize.minimize_scalar(
        _val_objective,
        bounds=(0.0, 1.0),
        method="bounded",
        options={"xatol": 1e-4},
    )
    alpha_star = float(np.clip(opt.x, 0.0, 1.0))
    cal_val_loss = _val_objective(alpha_star)

    # Also check endpoints explicitly in case optimum sits at alpha=1.0 or alpha=0.0
    loss_at_one = _val_objective(1.0)
    if loss_at_one < cal_val_loss:
        alpha_star = 1.0
        cal_val_loss = loss_at_one

    if (
        cal_val_loss >= base_val_loss - config.min_improvement_eps
        or alpha_star < config.min_alpha
    ):
        return FittedBooster(
            booster=None,
            best_ntree=0,
            alpha=0.0,
            base_val_loss=base_val_loss,
            cal_val_loss=base_val_loss,
            min_child_weight=mcw_sub,
            gain_importance={},
        )

    w_full_norm = np.asarray(w_full, dtype=np.float64)
    if float(np.mean(w_full_norm)) > 0.0:
        w_full_norm = w_full_norm / float(np.mean(w_full_norm))
    mcw_full = _adaptive_min_child_weight(m_full, w_full_norm, config)
    dfull = _make_dmatrix(X_full, m_full, y=y_full, weight=w_full_norm)

    final_booster = xgb.train(
        params=_xgb_params(config, min_child_weight=mcw_full, seed=fold_seed),
        dtrain=dfull,
        num_boost_round=best_ntree,
        verbose_eval=False,
    )

    raw_scores = final_booster.get_score(importance_type="gain")
    gain_importance = {str(k): float(v) for k, v in raw_scores.items()}

    return FittedBooster(
        booster=final_booster,
        best_ntree=best_ntree,
        alpha=alpha_star,
        base_val_loss=base_val_loss,
        cal_val_loss=cal_val_loss,
        min_child_weight=mcw_full,
        gain_importance=gain_importance,
    )


def predict_calibrated_prob(
    fitted: FittedBooster,
    X_test: np.ndarray,
    m_test: np.ndarray,
    *,
    eps: float = PROB_EPS,
) -> np.ndarray:
    """Predict calibrated recall probability p_calibrated = sigmoid(logit(p_fsrs) + alpha * delta_xgb)."""
    if len(m_test) == 0:
        return np.empty(0, dtype=np.float64)

    m_f32 = np.asarray(m_test, dtype=np.float32)
    m_f64 = m_f32.astype(np.float64)

    if fitted.booster is None or fitted.alpha <= 0.0 or fitted.best_ntree <= 0:
        return clip_prob(sigmoid(m_f64), eps=eps)

    dtest = _make_dmatrix(X_test, m_f32)
    raw_margin = fitted.booster.predict(
        dtest,
        iteration_range=(0, fitted.best_ntree),
        output_margin=True,
    ).astype(np.float64)
    delta_margin = raw_margin - m_f64
    calibrated_logits = m_f64 + fitted.alpha * delta_margin
    return clip_prob(sigmoid(calibrated_logits), eps=eps)


# --------------------------------------------------------------------------------------
# Walk-forward orchestration (called from script.py after the base-model fold fits)
# --------------------------------------------------------------------------------------

GLOBAL_META_FILE = "global_boosters_meta.json"
_GLOBAL_CACHE: dict[str, dict[tuple[int | None, int], FittedBooster]] = {}


def read_global_meta(boosters_dir: str | Path) -> dict[str, Any] | None:
    meta_file = Path(boosters_dir) / GLOBAL_META_FILE
    if not meta_file.exists():
        return None
    return json.loads(meta_file.read_text(encoding="utf-8"))


def user_half(user_id: int) -> int:
    """global-split halves: 0 for user_id < 5000, 1 otherwise."""
    return int(user_id >= 5000)


def check_global_meta(meta: dict[str, Any], config: Config) -> None:
    """Refuse boosters trained for another scope, base model or prior."""
    # Boosters trained before the meta recorded these were global FSRS-7 ones.
    scope = str(meta.get("scope", "global"))
    base_model = str(meta.get("baseModel", "FSRS-7"))
    prior = "default" if config.default_params else "trained"
    if (
        scope != config.xgboost
        or base_model != config.model_name
        or meta.get("priorMode", prior) != prior
    ):
        raise ValueError(
            f"boosters were trained for {scope} {base_model} ({meta.get('priorMode')} prior), "
            f"not {config.xgboost} {config.model_name} ({prior} prior)"
        )


def load_global_boosters(
    boosters_dir: str | Path,
) -> dict[tuple[int | None, int], FittedBooster]:
    """Load pretrained boosters keyed by (half, fold); half is None for global."""
    key = str(boosters_dir)
    if key not in _GLOBAL_CACHE:
        bdir = Path(boosters_dir)
        meta = read_global_meta(bdir) or {}
        fold_map: dict[tuple[int | None, int], FittedBooster] = {}
        for f in meta.get("folds", []):
            booster = None
            if f.get("modelFile"):
                booster = xgb.Booster(params={"nthread": 1})
                booster.load_model(str(bdir / str(f["modelFile"])))
            fold_map[(f.get("half"), int(f["fold"]))] = FittedBooster(
                booster=booster,
                best_ntree=int(f.get("bestTrees", 0)),
                alpha=float(f.get("alpha", 0.0)),
                base_val_loss=float(f.get("baseValLoss", float("nan"))),
                cal_val_loss=float(f.get("calValLoss", float("nan"))),
                min_child_weight=float(f.get("minChildWeight", 5.0)),
                gain_importance={},
            )
        _GLOBAL_CACHE[key] = fold_map
    return _GLOBAL_CACHE[key]


class _OOFStates:
    """Base-model states for every row of a user, filled fold by fold out of sample."""

    def __init__(self, n: int):
        self.cols = {
            f: np.full(n, np.nan) for f in BaseStateArrays.__dataclass_fields__
        }
        self.ready = np.zeros(n, dtype=bool)
        self.in_sample_b0 = np.zeros(n, dtype=bool)

    def store(self, pos: np.ndarray, states: BaseStateArrays, overwrite: bool) -> None:
        mask = np.ones(len(pos), dtype=bool) if overwrite else ~self.ready[pos]
        for f, arr in self.cols.items():
            arr[pos[mask]] = getattr(states, f)[mask]
        self.ready[pos[mask]] = True

    def take(self, pos: np.ndarray) -> BaseStateArrays:
        return BaseStateArrays(**{f: arr[pos] for f, arr in self.cols.items()})


@dataclass
class _Fold:
    """One fold/partition of a user, with base states for its train and test rows."""

    split_i: int
    part_test: pd.DataFrame
    X_train: np.ndarray
    m_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    m_test: np.ndarray
    test_states: BaseStateArrays
    b0_sub_end: int
    prev_len: int


def _walk_folds(
    dataset: pd.DataFrame,
    w_list: list[dict[Any, Any]],
    trainsets: list[pd.DataFrame],
    testsets: list[pd.DataFrame],
    config: Config,
    fit_weights: Callable[[pd.DataFrame], Any],
):
    """Yield every fold/partition with out-of-fold base states and feature matrices.

    Training rows never see the fold's own weights: their base predictions come from the
    previous fold's model; fold 0 uses a fit on its first 75% of rows (fit_weights).
    """
    from models.model_factory import create_model

    static = precompute_static_user_features(dataset)
    default_states = extract_base_states(create_model(config), dataset, config)
    oof = _OOFStates(len(dataset))
    prev_model: dict[Any, Any] = {}
    prev_train_len: dict[Any, int] = {}

    for split_i, (w, train_set, test_set) in enumerate(
        zip(w_list, trainsets, testsets)
    ):
        for partition in test_set["partition"].unique():
            part_train = train_set[train_set["partition"] == partition]
            part_test = test_set[test_set["partition"] == partition].copy()
            train_pos = static.locate_rows(part_train)
            test_pos = static.locate_rows(part_test)
            weights = w.get(partition)
            model = create_model(config, weights) if weights else create_model(config)

            b0_sub_end = 0
            if config.default_params:
                train_states = default_states.slice_rows(train_pos)
                test_states = default_states.slice_rows(test_pos)
            else:
                test_states = extract_base_states(model, part_test, config)
                missing = train_pos[~oof.ready[train_pos]]
                if len(missing):
                    if partition in prev_model:
                        fill = extract_base_states(
                            prev_model[partition], dataset.iloc[missing], config
                        )
                    elif len(missing) >= 40:
                        sub_pos = missing[: max(int(round(0.75 * len(missing))), 1)]
                        sub_model = create_model(
                            config, fit_weights(dataset.iloc[sub_pos].copy())
                        )
                        fill = extract_base_states(
                            sub_model, dataset.iloc[missing], config
                        )
                        oof.in_sample_b0[sub_pos] = True
                    else:
                        fill = default_states.slice_rows(missing)
                    oof.store(missing, fill, overwrite=True)
                train_states = oof.take(train_pos)
                b0_sub_end = int(np.sum(oof.in_sample_b0[train_pos]))
                oof.store(test_pos, test_states, overwrite=False)
                prev_model[partition] = model

            n_train = len(train_pos)
            fold_pos = np.concatenate([train_pos, test_pos])
            ewma_50, mode_ewma_100 = compute_causal_fsrs_error_features(
                y_seq=static.y[fold_pos],
                p_fsrs_seq=np.concatenate([train_states.p_fsrs, test_states.p_fsrs]),
                is_same_day_seq=static.is_same_day_bool[fold_pos],
            )
            X_train, m_train = build_feature_matrix(
                static,
                train_pos,
                train_states,
                default_states.logit_p[train_pos],
                ewma_50[:n_train],
                mode_ewma_100[:n_train],
            )
            X_test, m_test = build_feature_matrix(
                static,
                test_pos,
                test_states,
                default_states.logit_p[test_pos],
                ewma_50[n_train:],
                mode_ewma_100[n_train:],
            )
            yield _Fold(
                split_i=split_i,
                part_test=part_test,
                X_train=X_train,
                m_train=m_train,
                y_train=static.y[train_pos],
                X_test=X_test,
                m_test=m_test,
                test_states=test_states,
                b0_sub_end=b0_sub_end,
                prev_len=prev_train_len.get(partition, 0),
            )
            prev_train_len[partition] = n_train


def _training_arrays(
    fold: _Fold, trained_prior: bool, cfg: XGBoostConfig
) -> dict[str, np.ndarray] | None:
    """Split a fold's train rows into sub-train / validation / full arrays with weights.

    Validation is the rows new to this fold (fold 0: the rows after the 75% sub-train
    fit), else the last 20%. None when the fold is too small to fit a booster.
    """
    n_train = len(fold.y_train)
    if n_train < cfg.min_subtrain_rows + cfg.min_val_rows:
        return None
    b0 = fold.b0_sub_end if trained_prior else 0
    start = b0 if fold.split_i == 0 else fold.prev_len
    if start < cfg.min_subtrain_rows or n_train - start < cfg.min_val_rows:
        start = int(round(0.80 * n_train))
    v = int(np.clip(start, cfg.min_subtrain_rows, n_train - cfg.min_val_rows))
    X, m, y = fold.X_train, fold.m_train, fold.y_train
    return {
        "X_sub": X[:v],
        "y_sub": y[:v],
        "m_sub": m[:v],
        "w_sub": compute_recency_weights(v, cfg, b0_sub_end=min(b0, v)),
        "X_val": X[v:],
        "y_val": y[v:],
        "m_val": m[v:],
        "w_val": np.ones(n_train - v, dtype=np.float64),
        "X_full": X,
        "y_full": y,
        "m_full": m,
        "w_full": compute_recency_weights(n_train, cfg, b0_sub_end=b0),
    }


def _no_booster(cfg: XGBoostConfig) -> FittedBooster:
    nan = float("nan")
    return FittedBooster(None, 0, 0.0, nan, nan, cfg.min_child_weight_floor, {})


def xgboost_predict(
    dataset: pd.DataFrame,
    w_list: list[dict[Any, Any]],
    trainsets: list[pd.DataFrame],
    testsets: list[pd.DataFrame],
    config: Config,
    fit_weights: Callable[[pd.DataFrame], Any],
    xgb_config: XGBoostConfig | None = None,
    user_id: int | None = None,
) -> list[pd.DataFrame]:
    """Predict every fold's test set with the base model plus its XGBoost residual.

    w_list, trainsets and testsets are the per-fold outputs of script._fit_folds.
    Returns the fold/partition test sets with columns p, s and d, in fold order.
    """
    cfg = xgb_config or XGBoostConfig()
    global_map = None
    half = None
    if config.xgboost in ("global", "global-split"):
        global_map = load_global_boosters(config.xgboost_boosters)
        if config.xgboost == "global-split":
            assert user_id is not None
            half = 1 - user_half(user_id)  # the other half's boosters
    out: list[pd.DataFrame] = []
    for fold in _walk_folds(dataset, w_list, trainsets, testsets, config, fit_weights):
        if global_map is not None:
            fitted = global_map.get((half, fold.split_i), _no_booster(cfg))
        else:
            arrays = _training_arrays(fold, not config.default_params, cfg)
            fitted = (
                fit_fold_calibrator(**arrays, config=cfg, fold_idx=fold.split_i)
                if arrays is not None
                else _no_booster(cfg)
            )
        if (
            config.save_weights
            and global_map is None
            and fold.split_i == len(w_list) - 1
        ):
            _save_user_booster(fitted, fold, config, user_id)
        part_test = fold.part_test
        part_test["p"] = predict_calibrated_prob(fitted, fold.X_test, fold.m_test)
        part_test["s"] = fold.test_states.s_long
        part_test["d"] = fold.test_states.d
        out.append(part_test)
    return out


def _save_user_booster(
    fitted: FittedBooster, fold: _Fold, config: Config, user_id: int | None
) -> None:
    """--weights: keep the last fold's per-user booster next to the result's parameters
    (also the last fold's), as weights/<result name>/<user>[_<partition>].ubj + .json."""
    out_dir = Path("weights") / config.get_evaluation_file_name()
    out_dir.mkdir(parents=True, exist_ok=True)
    partition = fold.part_test["partition"].iloc[0]
    stem = f"{user_id}" if config.partitions == "none" else f"{user_id}_{partition}"
    model_file = None
    if fitted.booster is not None:
        model_file = f"{stem}.ubj"
        fitted.booster.save_model(str(out_dir / model_file))
    meta = {
        "fold": fold.split_i,
        "alpha": fitted.alpha,
        "bestTrees": fitted.best_ntree,
        "modelFile": model_file,
        "features": list(FEATURE_NAMES),
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(meta) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------------------
# Global boosters: pooled over size-stratified anchor users, one booster per fold index
# --------------------------------------------------------------------------------------

MAX_ROWS_PER_USER_FOLD = 15_000


def select_anchor_users(users_by_size: list[int], n_anchors: int) -> list[int]:
    """Evenly spaced users along a size-ordered list, so all user sizes are represented."""
    if len(users_by_size) <= n_anchors:
        return sorted(users_by_size)
    idx = np.linspace(0, len(users_by_size) - 1, n_anchors, dtype=int)
    return sorted({users_by_size[i] for i in idx})


def xgboost_anchor_arrays(
    dataset: pd.DataFrame,
    w_list: list[dict[Any, Any]],
    trainsets: list[pd.DataFrame],
    testsets: list[pd.DataFrame],
    config: Config,
    fit_weights: Callable[[pd.DataFrame], Any],
    xgb_config: XGBoostConfig | None = None,
) -> dict[int, dict[str, np.ndarray]]:
    """One anchor user's training arrays per fold index, capped to the most recent rows."""
    cfg = xgb_config or XGBoostConfig()
    out: dict[int, dict[str, np.ndarray]] = {}
    for fold in _walk_folds(dataset, w_list, trainsets, testsets, config, fit_weights):
        arrays = _training_arrays(fold, not config.default_params, cfg)
        if arrays is None:
            continue
        for part in ("sub", "val", "full"):
            keep = slice(-MAX_ROWS_PER_USER_FOLD, None)
            w = arrays[f"w_{part}"][keep]
            arrays[f"w_{part}"] = w / max(
                len(w), 1
            )  # every anchor user weighs the same
            for k in ("X", "y", "m"):
                arrays[f"{k}_{part}"] = arrays[f"{k}_{part}"][keep]
        out[fold.split_i] = arrays
    return out


def train_global_boosters(
    per_user: dict[int, dict[int, dict[str, np.ndarray]]],
    config: Config,
    out_dir: str | Path,
    xgb_config: XGBoostConfig | None = None,
) -> None:
    """Fit one pooled booster per fold index (per half for global-split) and write
    [half<h>_]fold_<k>.ubj plus the meta file."""
    cfg = xgb_config or XGBoostConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    halves = (0, 1) if config.xgboost == "global-split" else (None,)
    folds = []
    for half, split_i in ((h, k) for h in halves for k in range(config.n_splits)):
        parts = [
            arrays[split_i]
            for u, arrays in per_user.items()
            if split_i in arrays and (half is None or user_half(u) == half)
        ]
        if not parts:
            continue
        pooled = {
            k: (np.vstack if k.startswith("X") else np.concatenate)(
                [p[k] for p in parts]
            )
            for k in parts[0]
        }
        fitted = fit_fold_calibrator(**pooled, config=cfg, fold_idx=split_i)
        model_file = None
        if fitted.booster is not None:
            model_file = f"{'' if half is None else f'half{half}_'}fold_{split_i}.ubj"
            fitted.booster.save_model(str(out_dir / model_file))
        folds.append(
            {
                "half": half,
                "fold": split_i,
                "pooledTrainRows": int(len(pooled["y_full"])),
                "bestTrees": fitted.best_ntree,
                "alpha": round(fitted.alpha, 6),
                "baseValLoss": round(fitted.base_val_loss, 6),
                "calValLoss": round(fitted.cal_val_loss, 6),
                "minChildWeight": round(fitted.min_child_weight, 4),
                "modelFile": model_file,
            }
        )
    meta = {
        "scope": config.xgboost,
        "baseModel": config.model_name,
        "priorMode": "default" if config.default_params else "trained",
        "xgboostVersion": xgb.__version__,
        "anchorUserIds": sorted(per_user),
        "folds": folds,
    }
    (out_dir / GLOBAL_META_FILE).write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
