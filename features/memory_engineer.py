from typing import Any, cast

import pandas as pd

from .base import BaseFeatureEngineer, _card_run_bounds


class SM2FeatureEngineer(BaseFeatureEngineer):
    """
    Feature engineer for SM2 model
    SM2 only needs rating history as a string sequence
    """

    def _model_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create SM2 features: rating history as string sequence
        """
        # SM2 uses the r_history field which is already created in base preprocessing
        df["sequence"] = df["r_history"]
        return df


class EbisuFeatureEngineer(BaseFeatureEngineer):
    """
    Feature engineer for Ebisu models
    Ebisu needs (time_interval, rating) tuples for each review
    """

    def _model_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create Ebisu features: sequence of (time_interval, rating) tuples
        """
        n = len(df)
        if n == 0:
            cast(Any, df)["sequence"] = []
            return df

        pairs = list(zip(list(df["delta_t"].to_numpy()), list(df["rating"].to_numpy())))
        starts, ends = _card_run_bounds(df["card_id"].to_numpy())
        sequences: list = []
        for s, e in zip(starts.tolist(), ends.tolist()):
            card_pairs = tuple(pairs[s : e - 1])
            sequences.extend(card_pairs[:j] for j in range(e - s))

        cast(Any, df)["sequence"] = sequences
        return df
