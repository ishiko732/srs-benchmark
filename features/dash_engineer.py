from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from shape_extensions import IntVar

from .base import BaseFeatureEngineer, _card_run_bounds, _card_start_indices

_TAU_W = np.array([0.2434, 1.9739, 16.0090, 129.8426], dtype=np.float64)
_TIME_WINDOWS = np.array([1.0, 7.0, 30.0, np.inf], dtype=np.float64)


def _compute_dash_tw_all(df: pd.DataFrame, enable_decay: bool) -> list[torch.Tensor]:
    """Vectorized lag-accumulation across all cards for DASH and DASH[MCM]."""
    n = len(df)
    if n == 0:
        return []

    _, start_idx = _card_start_indices(df["card_id"].to_numpy())
    pos_in_card = np.arange(n).astype(np.int64) - start_idx
    max_lag = int(pos_in_card.max())
    if max_lag == 0:
        return list(torch.zeros((n, 8), dtype=torch.float32).unbind(0))

    t_vals = df["delta_t"].to_numpy(dtype=np.float64)
    r_bin = (df["rating"].to_numpy() > 1).astype(np.float64)

    # Accumulate in float64 to match original np.zeros(8, dtype=float64), then cast to float32
    feats = np.zeros((n, 8), dtype=np.float64)
    active = np.flatnonzero(pos_in_card >= 1)
    cum_t = np.zeros(len(active), dtype=np.float64)

    for lag in range(1, max_lag + 1):
        if lag > 1:
            keep = pos_in_card[active] >= lag
            active = active[keep]
            cum_t = cum_t[keep]
        if len(active) == 0:
            break

        # Reverse cumulative sum order: t[i] + t[i-1] + ... + t[i-lag+1]
        cum_t = cum_t + t_vals[active - (lag - 1)]
        rb = r_bin[active - lag]

        if enable_decay:
            # valid: (M, 4), decay: (M, 4)
            valid = cum_t[:, None] <= _TIME_WINDOWS
            decay = np.where(valid, np.exp(-cum_t[:, None] / _TAU_W), 0.0)
            feats[active, 0::2] += decay
            feats[active, 1::2] += decay * rb[:, None]
        else:
            valid_f = (cum_t[:, None] <= _TIME_WINDOWS).astype(np.float64)
            feats[active, 0::2] += valid_f
            feats[active, 1::2] += valid_f * rb[:, None]

    full = torch.from_numpy(feats.astype(np.float32))
    return list(full.unbind(0))


class DashFeatureEngineer(BaseFeatureEngineer):
    """
    Base feature engineer for DASH model
    DASH uses time window features without decay
    """

    def _model_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create DASH time window features
        """
        cast(Any, df)["tensor"] = _compute_dash_tw_all(df, enable_decay=False)
        return df

    def _dash_tw_features(
        self, r_history: list, t_history: list, enable_decay: bool = False
    ) -> np.ndarray[[8], np.dtype[np.float64]]:
        """
        Create DASH time window features

        Args:
            r_history: Rating history
            t_history: Time history
            enable_decay: Whether to enable exponential decay (for MCM variant)

        Returns:
            Feature vector of length 8
        """
        features = cast(
            "np.ndarray[[8], np.dtype[np.float64]]",
            np.zeros(8),
        )
        r_history_binary = np.array(r_history) > 1  # Convert to binary success/failure
        tau_w = np.array([0.2434, 1.9739, 16.0090, 129.8426])  # Decay constants
        time_windows = np.array([1, 7, 30, np.inf])  # Time windows in days

        # Compute the cumulative sum of t_history in reverse order
        cumulative_times = np.cumsum(t_history[::-1])[::-1]

        for j, time_window in enumerate(time_windows):
            # Calculate decay factors for each time window
            if enable_decay:
                decay_factors = np.exp(-cumulative_times / tau_w[j])
            else:
                decay_factors = np.ones_like(cumulative_times)

            # Identify the indices where cumulative times are within the current time window
            valid_indices = cumulative_times <= time_window

            # Update features using decay factors where valid
            features[j * 2] += np.sum(decay_factors[valid_indices])  # Total count
            features[j * 2 + 1] += np.sum(
                r_history_binary[valid_indices] * decay_factors[valid_indices]
            )  # Success count

        return features


class DashMCMFeatureEngineer(DashFeatureEngineer):
    """
    Feature engineer for DASH[MCM] model
    Same as DASH but with exponential decay enabled
    """

    def _model_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create DASH[MCM] time window features with decay
        """
        cast(Any, df)["tensor"] = _compute_dash_tw_all(df, enable_decay=True)
        return df


class DashACTRFeatureEngineer(BaseFeatureEngineer):
    """
    Feature engineer for DASH[ACT-R] model
    Uses ACT-R style activation features
    """

    def _model_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create DASH[ACT-R] features
        """
        n = len(df)
        if n == 0:
            cast(Any, df)["tensor"] = []
            return df

        r_bin_f32 = torch.from_numpy((df["rating"].to_numpy() > 1).astype(np.float32))
        t_f32 = torch.from_numpy(df["delta_t"].to_numpy(dtype=np.float32, copy=True))
        empty_0_2 = torch.empty((0, 2), dtype=torch.float32)

        starts, ends = _card_run_bounds(df["card_id"].to_numpy())
        tensors: list[torch.Tensor] = []
        for s, e in zip(starts.tolist(), ends.tolist()):
            tensors.append(empty_0_2)
            m = e - s - 1
            if m > 0:
                r_card = r_bin_f32[s : e - 1]
                sp_card = t_f32[s + 1 : e]
                cs_card = torch.cumsum(sp_card, dim=0)
                diff_card = sp_card - cs_card
                time_mat = diff_card.unsqueeze(0) + cs_card.unsqueeze(1)
                r_mat = r_card.unsqueeze(0).expand(m, m)
                card_3d = torch.stack((r_mat, time_mat), dim=2)
                tensors.extend(card_3d[idx, : idx + 1] for idx in range(m))

        cast(Any, df)["tensor"] = tensors
        return df

    def _dash_actr_features[SeqLen: IntVar](
        self, r_history: list, t_history: list
    ) -> torch.Tensor[[SeqLen, 2]]:
        """
        Create ACT-R style features for DASH[ACT-R]

        Args:
            r_history: Rating history
            t_history: Time history

        Returns:
            Feature tensor with shape (sequence_length, 2)
        """
        r_history_tensor = torch.tensor(np.array(r_history) > 1, dtype=torch.float32)
        sp_history = torch.tensor(t_history, dtype=torch.float32)
        cumsum = torch.cumsum(sp_history, dim=0)

        # Features: [success_indicator, time_since_start]
        features = [r_history_tensor, sp_history - cumsum + cumsum[-1:]]
        return torch.stack(features, dim=1)
