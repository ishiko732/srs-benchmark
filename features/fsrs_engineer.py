from typing import Any, cast

import numpy as np
import pandas as pd
import torch

from .base import BaseFeatureEngineer, _card_run_bounds


class FSRSFeatureEngineer(BaseFeatureEngineer):
    """
    Feature engineer for FSRS models (FSRSv1, FSRSv2, FSRSv3, FSRSv4, FSRS-4.5, FSRS-5, FSRS-6)
    Also handles RNN, GRU, Transformer, SM2-trainable, Anki, and 90% models
    """

    def _model_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create tensor features for FSRS-style models
        These models use (time_history, rating_history) tensors
        """
        n = len(df)
        if n == 0:
            cast(Any, df)["tensor"] = []
            return df

        # Build ONE user-wide contiguous [N, 2] float32 tensor from (delta_t, rating),
        # then slice zero-copy prefix views per card.
        arr = np.empty((n, 2), dtype=np.float32)
        arr[:, 0] = df["delta_t"].to_numpy(dtype=np.float32)
        arr[:, 1] = df["rating"].to_numpy(dtype=np.float32)
        full = torch.from_numpy(arr)

        starts, ends = _card_run_bounds(df["card_id"].to_numpy())
        cast(Any, df)["tensor"] = [
            full[s : s + j]
            for s, e in zip(starts.tolist(), ends.tolist())
            for j in range(e - s)
        ]

        return df
