import numpy as np
import pandas as pd

from config import Config
from features import create_features


class InsufficientUserDataError(ValueError):
    """Raised when a user has too few reviews to run the benchmark."""


class UserDataLoader:
    """A class responsible for loading and preprocessing user data."""

    def __init__(self, config: Config):
        self.config = config
        self.data_path = config.data_path

    def _read_user_partition(
        self, kind: str, user_id: int, columns: list[str] | None = None
    ) -> pd.DataFrame:
        """Read one user's cards/decks from the direct hive-partition path.

        Equivalent to ``read_parquet(kind, filters=[("user_id", "=", uid)])`` followed
        by dropping the ``user_id`` column, but avoids pyarrow re-discovering all ~3000
        partitions on every call (~700 ms -> ~3 ms). The partition path encodes
        ``user_id``, so it is absent from the returned columns either way. Users with no
        partition directory (e.g. revlogs but no cards) fall back to the filtered read,
        which yields the same empty frame.
        """
        path = self.data_path / kind / f"user_id={user_id}"
        if path.exists():
            df = pd.read_parquet(path, columns=columns)
            if "user_id" in df.columns:
                df = df.drop(columns=["user_id"])
            return df
        filter_cols = (
            [*columns, "user_id"]
            if columns is not None and "user_id" not in columns
            else columns
        )
        df = pd.read_parquet(
            self.data_path / kind,
            filters=[("user_id", "=", user_id)],
            columns=filter_cols,
        )
        return df.drop(columns=["user_id"])

    def load_user_data(self, user_id: int) -> pd.DataFrame:
        """
        Load and prepare user data including revlogs, cards, and decks.

        Args:
            user_id: The ID of the user whose data to load

        Returns:
            pd.DataFrame: Processed dataset with features

        Raises:
            InsufficientUserDataError: If there is not enough data for the user
        """
        # Load revlogs (prune unused columns when not exporting evaluation TSVs)
        revlog_path = self.data_path / "revlogs" / f"{user_id=}"
        if self.config.save_evaluation_file:
            df_revlogs = pd.read_parquet(revlog_path)
        else:
            revlog_cols = ["card_id", "rating", "elapsed_days"]
            if self.config.use_secs_intervals:
                revlog_cols.append("elapsed_seconds")
            if self.config.lstm_use_duration and self.config.model_name in (
                "LSTM",
                "GRU",
            ):
                revlog_cols.append("duration")
            df_revlogs = pd.read_parquet(revlog_path, columns=revlog_cols)

        # Create initial features
        dataset = create_features(df_revlogs, config=self.config)

        if dataset.shape[0] < 6:
            raise InsufficientUserDataError(f"{user_id} does not have enough data.")

        # Handle partitions if needed
        if self.config.partitions != "none":
            if self.config.save_evaluation_file:
                # Full merge when exporting evaluation files so card/deck metadata columns exist
                df_cards = self._read_user_partition("cards", user_id)
                df_decks = self._read_user_partition("decks", user_id)
                dataset = dataset.merge(df_cards, on="card_id", how="left").merge(
                    df_decks, on="deck_id", how="left"
                )
                fill_cols = [
                    col
                    for col in dataset.columns
                    if pd.api.types.is_numeric_dtype(dataset[col])
                ]
                dataset[fill_cols] = dataset[fill_cols].fillna(-1)
                if self.config.partitions == "preset":
                    dataset["partition"] = dataset["preset_id"].astype(int)
                elif self.config.partitions == "deck":
                    dataset["partition"] = dataset["deck_id"].astype(int)
            else:
                # Match baseline fillna(-1) on numeric columns (e.g. rmse_bins_lapse
                # on same-day rows under --equalize_test_with_non_secs) without touching
                # string/object/tensor columns.
                for col in dataset.columns:
                    if (
                        pd.api.types.is_numeric_dtype(dataset[col])
                        and dataset[col].isna().any()
                    ):
                        dataset[col] = dataset[col].fillna(-1)

                # Map card_id -> partition directly on small metadata tables without
                # copying the full review/tensor DataFrame twice.
                card_ids_int = dataset["card_id"].astype(np.int64)
                df_cards = self._read_user_partition(
                    "cards", user_id, columns=["card_id", "deck_id"]
                )
                if self.config.partitions == "deck":
                    if df_cards.empty:
                        dataset["partition"] = -1
                    else:
                        deck_by_card = df_cards.set_index("card_id")["deck_id"]
                        dataset["partition"] = (
                            card_ids_int.map(deck_by_card).fillna(-1).astype(int)
                        )
                elif self.config.partitions == "preset":
                    df_decks = self._read_user_partition(
                        "decks", user_id, columns=["deck_id", "preset_id"]
                    )
                    if df_cards.empty or df_decks.empty:
                        dataset["partition"] = -1
                    else:
                        preset_by_deck = df_decks.set_index("deck_id")["preset_id"]
                        preset_by_card = pd.Series(
                            df_cards["deck_id"].map(preset_by_deck).to_numpy(),
                            index=df_cards["card_id"],
                        )
                        dataset["partition"] = (
                            card_ids_int.map(preset_by_card).fillna(-1).astype(int)
                        )
                dataset.reset_index(drop=True, inplace=True)
        else:
            dataset["partition"] = 0

        return dataset
