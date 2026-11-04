from typing import Any, cast

import numpy as np
import pandas as pd
import torch

from .base import BaseFeatureEngineer, _card_run_bounds


class FSRSOneStepFeatureEngineer(BaseFeatureEngineer):
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
            cast(Any, df)["inputs"] = []
            return df

        dt_np = df["delta_t"].to_numpy()
        r_np = df["rating"].to_numpy()

        arr = np.empty((n, 2), dtype=np.float32)
        arr[:, 0] = dt_np.astype(np.float32)
        arr[:, 1] = r_np.astype(np.float32)
        full = torch.from_numpy(arr)

        pairs = list(zip(list(dt_np), list(r_np)))
        starts, ends = _card_run_bounds(df["card_id"].to_numpy())
        starts_l = starts.tolist()
        ends_l = ends.tolist()

        tensors: list = []
        inputs: list = []
        for s, e in zip(starts_l, ends_l):
            m = e - s
            card_pairs = tuple(pairs[s : e - 1])
            tensors.extend(full[s : s + j] for j in range(m))
            inputs.extend(card_pairs[:j] for j in range(m))

        cast(Any, df)["tensor"] = tensors
        cast(Any, df)["inputs"] = inputs

        return df
