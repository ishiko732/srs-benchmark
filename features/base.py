from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from config import Config
from utils import cum_concat

# Per-card cumulative history. Nested three deep: outer = cards, middle = reviews within a
# card, inner = the cumulative prefix of (numpy) interval/rating values up to that review.
HistoryLists = list[list[list[float]]]

_FIRST_RATING_STR_LUT = np.array(["", "1", "2", "3", "4"], dtype=object)


def _card_start_indices(card_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(is_start, start_idx)`` for a 1D ``card_ids`` array sorted by card."""
    n = len(card_ids)
    is_start = np.empty(n, dtype=bool)
    is_start[0] = True
    is_start[1:] = card_ids[1:] != card_ids[:-1]
    start_idx = np.maximum.accumulate(
        np.where(is_start, np.arange(n).astype(np.int64), 0)
    )
    return is_start, start_idx


def _card_run_bounds(card_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(starts, ends)`` slice bounds for contiguous ``card_id`` blocks."""
    n = len(card_ids)
    if n == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    changes = np.flatnonzero(card_ids[1:] != card_ids[:-1]) + 1
    m = len(changes) + 1
    starts = np.empty(m, dtype=np.int64)
    ends = np.empty(m, dtype=np.int64)
    starts[0] = 0
    starts[1:] = changes
    ends[:-1] = changes
    ends[-1] = n
    return starts, ends


def _segmented_cumsum(vals: np.ndarray, start_idx: np.ndarray) -> np.ndarray:
    """Segmented integer cumulative sum over contiguous runs defined by ``start_idx``."""
    cs = np.cumsum(vals, dtype=np.int64)
    offset = np.where(start_idx > 0, cs[start_idx - 1], 0)
    return cs - offset


def _prefix_history_strings(card_ids: np.ndarray, values: np.ndarray) -> list[str]:
    """Build comma-separated prefix strings (excluding current row) per contiguous card."""
    n = len(values)
    if n == 0:
        return []
    strs = [str(v) for v in values.tolist()]
    out = [""] * n
    acc = ""
    prev_cid = card_ids[0]
    for idx in range(n):
        cid = card_ids[idx]
        if idx == 0 or cid != prev_cid:
            out[idx] = ""
            acc = strs[idx]
            prev_cid = cid
        else:
            out[idx] = acc
            acc = f"{acc},{strs[idx]}"
    return out


def _compute_last_rating_fast(
    is_start: np.ndarray, t_non_secs: np.ndarray, ratings: np.ndarray
) -> np.ndarray:
    """Vectorized equivalent of ``BaseFeatureEngineer._compute_last_rating``."""
    n = len(ratings)
    if n == 0:
        return np.empty(0, dtype=ratings.dtype)
    idx_arr = np.arange(n).astype(np.int64)
    latest_valid = np.maximum.accumulate(
        np.where(is_start | (t_non_secs > 0), idx_arr, 0)
    )
    src = np.empty(n, dtype=np.int64)
    src[0] = 0
    src[1:] = np.where(is_start[1:], idx_arr[1:], latest_valid[:-1])
    return ratings[src]


def _cumulative_lists_by_card(df: pd.DataFrame, col: str) -> HistoryLists:
    """Per-card list of cumulative-prefix lists for ``col``, in df row order.

    Bit-identical replacement for::

        df.groupby("card_id", group_keys=False)[col].apply(
            lambda x: cum_concat([[i] for i in x]))

    but without the pandas per-group ``apply`` overhead (the dominant create_features
    cost). Requires df sorted by ``card_id`` (guaranteed by _common_preprocessing, which
    is exactly what the groupby-apply assignment also relies on). Values are kept as numpy
    scalars (matching Series iteration), so ``str()`` and ``torch.tensor()`` downstream
    produce identical results.
    """
    card_ids = df["card_id"].values
    vals = list(df[col].values)
    out: HistoryLists = []
    n = len(vals)
    start = 0
    for i in range(1, n + 1):
        if i == n or card_ids[i] != card_ids[start]:  # type: ignore
            block = vals[start:i]
            out.append([block[: j + 1] for j in range(len(block))])
            start = i
    return out


class BaseFeatureEngineer(ABC):
    """
    Base abstract class for feature engineering
    Each specific model feature engineer should inherit from this class and implement the corresponding methods
    """

    def __init__(self, config: Config):
        """
        Initialize the feature engineer

        Args:
            config: Configuration object containing model parameters
        """
        self.config = config
        # Lazily populated if a caller invokes get_history_lists().
        self._cached_t_history_list: HistoryLists | None = None
        self._cached_r_history_list: HistoryLists | None = None

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main feature creation method that includes common preprocessing steps for all models

        Args:
            df: Input raw dataframe

        Returns:
            Processed feature dataframe
        """
        # Execute common preprocessing steps
        df = self._common_preprocessing(df.copy())

        # Compute history records
        df = self._compute_histories(df)

        # Model-specific feature engineering
        df = self._model_specific_features(df)

        # Execute common postprocessing steps
        df = self._common_postprocessing(df)

        # Model-specific postprocessing steps
        df = self._model_specific_postprocessing(df)

        return df

    def create_alignment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Lightweight non-seconds pass for ``--equalize_test_with_non_secs`` alignment.

        Computes exact ``review_th``, ``i``, ``rmse_bins_lapse``, and row filtering
        without building history strings or model-specific feature tensors.
        """
        df = self._common_preprocessing(df.copy())
        return self._common_postprocessing(df)

    def _common_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Common preprocessing steps shared by all models
        """
        # Add review sequence number
        n_raw = df.shape[0]
        df["review_th"] = np.arange(n_raw).astype(np.int64) + 1
        if self.config.save_evaluation_file and "day_offset" in df.columns:
            df["nth_today"] = df.groupby("day_offset").cumcount() + 1
        df.sort_values(by=["card_id", "review_th"], inplace=True)

        # Filter invalid ratings
        rating_vals = df["rating"].to_numpy()
        if pd.api.types.is_integer_dtype(rating_vals):
            valid_rating = (rating_vals >= 1) & (rating_vals <= 4)
        else:
            valid_rating = np.isin(rating_vals, (1, 2, 3, 4))
        if not valid_rating.all():
            df = df[valid_rating].copy()

        # Handle two-button mode
        if self.config.two_buttons and len(df) > 0:
            r_arr = df["rating"].to_numpy()
            df["rating"] = np.where((r_arr == 2) | (r_arr == 4), 3, r_arr)

        # Calculate review count on contiguous card runs
        if len(df) > 0:
            _, start_idx = _card_start_indices(df["card_id"].to_numpy())
            i_vals = np.arange(len(df)).astype(np.int64) - start_idx + 1
            df["i"] = i_vals
            max_i = self.config.max_seq_len * 2
            if (i_vals > max_i).any():
                df = df[i_vals <= max_i].copy()
        else:
            df["i"] = np.empty(0, dtype=np.int64)

        # Process time intervals
        df = self._process_time_intervals(df)

        # Handle short-term reviews
        if not self.config.include_short_term:
            ed_vals = df["elapsed_days"].to_numpy()
            if (ed_vals == 0).any():
                df = df[ed_vals != 0].copy()
            if len(df) > 0:
                _, start_idx = _card_start_indices(df["card_id"].to_numpy())
                df["i"] = np.arange(len(df)).astype(np.int64) - start_idx + 1
            else:
                df["i"] = np.empty(0, dtype=np.int64)

        return df

    def _process_time_intervals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process time interval related fields
        """
        if (
            "delta_t" not in df.columns
            and "elapsed_days" in df.columns
            and ("elapsed_seconds" in df.columns or not self.config.use_secs_intervals)
        ):
            df["delta_t"] = np.maximum(0, df["elapsed_days"].to_numpy())
            if self.config.use_secs_intervals:
                df["delta_t_secs"] = np.maximum(
                    0.0, df["elapsed_seconds"].to_numpy() / 86400
                )
        else:
            df["delta_t"] = np.maximum(0, df["delta_t"].to_numpy())
        df["delta_t_int"] = np.maximum(0, df["elapsed_days"].to_numpy())
        return df

    def _compute_histories(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute time and rating history records on demand
        """
        self._cached_r_history_list = None
        self._cached_t_history_list = None

        need_r = self.config.needs_r_history()
        need_t = self.config.needs_t_history()
        need_eval = self.config.save_evaluation_file

        if need_r or need_t or need_eval:
            card_ids = df["card_id"].to_numpy()
            t_non_secs = df["delta_t"].to_numpy()
            ratings = df["rating"].to_numpy()

            if need_eval:
                if len(df) > 0:
                    is_start, _ = _card_start_indices(card_ids)
                    df["last_rating"] = _compute_last_rating_fast(
                        is_start, t_non_secs, ratings
                    )
                else:
                    df["last_rating"] = np.empty(0, dtype=np.int64)

            if need_r:
                df["r_history"] = _prefix_history_strings(card_ids, ratings)

            if need_t:
                if self.config.use_secs_intervals:
                    t_secs = df["delta_t_secs"].to_numpy()
                    if self.config.equalize_test_with_non_secs:
                        df["t_history"] = _prefix_history_strings(card_ids, t_non_secs)
                        if need_eval:
                            df["t_history_secs"] = _prefix_history_strings(
                                card_ids, t_secs
                            )
                    else:
                        df["t_history"] = _prefix_history_strings(card_ids, t_secs)
                else:
                    df["t_history"] = _prefix_history_strings(card_ids, t_non_secs)

        if self.config.use_secs_intervals:
            df["delta_t"] = df["delta_t_secs"]

        return df

    def _compute_last_rating(
        self, t_history_list: HistoryLists, r_history_list: HistoryLists
    ) -> list[int]:
        """
        Calculate the previous rating for each review
        """
        last_rating = []
        for t_sublist, r_sublist in zip(t_history_list, r_history_list):
            for t_history, r_history in zip(t_sublist, r_sublist):
                flag = True
                for t, r in zip(reversed(t_history[:-1]), reversed(r_history[:-1])):
                    if t > 0:
                        last_rating.append(r)
                        flag = False
                        break
                if flag:
                    last_rating.append(r_history[0])
        return last_rating

    def _set_time_histories(
        self,
        df: pd.DataFrame,
        t_history_non_secs_list: HistoryLists,
        t_history_secs_list: HistoryLists | None = None,
    ) -> pd.DataFrame:
        """
        Set time history string fields
        """
        if t_history_secs_list is not None:
            if self.config.equalize_test_with_non_secs:
                df["t_history"] = [
                    ",".join(map(str, item[:-1]))
                    for sublist in t_history_non_secs_list
                    for item in sublist
                ]
                df["t_history_secs"] = [
                    ",".join(map(str, item[:-1]))
                    for sublist in t_history_secs_list
                    for item in sublist
                ]
            else:
                df["t_history"] = [
                    ",".join(map(str, item[:-1]))
                    for sublist in t_history_secs_list
                    for item in sublist
                ]
            df["delta_t"] = df["delta_t_secs"]
        else:
            df["t_history"] = [
                ",".join(map(str, item[:-1]))
                for sublist in t_history_non_secs_list
                for item in sublist
            ]

        return df

    @abstractmethod
    def _model_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Model-specific feature engineering logic, needs to be implemented in subclasses

        Args:
            df: Dataframe after common preprocessing

        Returns:
            Dataframe with model-specific features added
        """

    def _common_postprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Common postprocessing steps shared by all models
        """
        if len(df) == 0:
            df["first_rating"] = pd.Series(dtype=str)
            df["y"] = np.empty(0, dtype=np.int64)
            df["rmse_bins_lapse"] = np.empty(0, dtype=np.int64)
            if not self.config.use_secs_intervals:
                raise ValueError(
                    "No data after handling outliers and non-continuous rows"
                )
            return df

        card_ids = df["card_id"].to_numpy()
        ratings = df["rating"].to_numpy()
        delta_t_vals = df["delta_t"].to_numpy()
        _, start_idx = _card_start_indices(card_ids)

        # Set first rating and labels
        df["first_rating"] = _FIRST_RATING_STR_LUT[ratings[start_idx]]
        df["y"] = (ratings > 1).astype(np.int64)

        # Find lapses for RMSE (bins). Note: in original code, delta_t.astype(str) != "0"
        # evaluates to True for all float64 values (e.g. 0.0 -> "0.0") and delta_t != 0 for ints.
        if pd.api.types.is_float_dtype(df["delta_t"]):
            is_lapse = (ratings == 1).astype(np.int64)
        else:
            is_lapse = ((ratings == 1) & (delta_t_vals != 0)).astype(np.int64)
        df["rmse_bins_lapse"] = _segmented_cumsum(is_lapse, start_idx) - is_lapse

        # Handle short-term reviews
        if self.config.include_short_term:
            keep_short = (delta_t_vals != 0) | (df["i"].to_numpy() == 1)
            if not keep_short.all():
                df = df[keep_short].copy()
                if len(df) > 0:
                    _, start_idx = _card_start_indices(df["card_id"].to_numpy())

        # Recalculate review sequence number
        if len(df) > 0:
            gt0 = (df["elapsed_days"].to_numpy() > 0).astype(np.int64)
            df["i"] = _segmented_cumsum(gt0, start_idx) + 1

        # Handle outliers and non-continuous rows (only for non-seconds intervals)
        if not self.config.use_secs_intervals:
            df = self._handle_outliers_and_continuity(df)
            if df.empty:
                raise ValueError(
                    "No data after handling outliers and non-continuous rows"
                )

        pos_mask = df["delta_t"].to_numpy() > 0
        return df[pos_mask].sort_values(by=["review_th"])

    def _handle_outliers_and_continuity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle outliers and non-continuous rows (bit-identical to fsrs_optimizer
        ``remove_outliers`` + ``remove_non_continuous_rows``, vectorized over contiguous cards).
        """
        n = len(df)
        if n == 0:
            return pd.DataFrame()

        i_vals = df["i"].to_numpy()
        i2_pos = np.flatnonzero(i_vals == 2)
        if len(i2_pos) == 0:
            return pd.DataFrame()

        first_ratings_i2 = df["first_rating"].to_numpy()[i2_pos]
        delta_t_i2 = df["delta_t"].to_numpy()[i2_pos]
        keep_row = np.ones(n, dtype=bool)
        any_i2_kept = False

        for fr in ("1", "2", "3", "4"):
            fr_mask = first_ratings_i2 == fr
            if not fr_mask.any():
                continue
            dt_sub = delta_t_i2[fr_mask]
            uniq_dt, counts = np.unique(dt_sub, return_counts=True)
            # Sort by (count ASC, delta_t DESC); uniq_dt is strictly unique so order is unique.
            order = np.lexsort((-uniq_dt, counts))
            total = len(dt_sub)
            threshold = max(total * 0.05, 20.0)
            max_dt = 365 if fr == "4" else 100
            removed_dts: list = []
            has_been_removed = 0
            for c, dt_val in zip(counts[order].tolist(), uniq_dt[order].tolist()):
                if has_been_removed + c >= threshold:
                    if c < 6 or dt_val > max_dt:
                        removed_dts.append(dt_val)
                        has_been_removed += c
                else:
                    removed_dts.append(dt_val)
                    has_been_removed += c

            if removed_dts:
                sub_keep = ~np.isin(dt_sub, removed_dts)
                keep_row[i2_pos[fr_mask]] = sub_keep
                if sub_keep.any():
                    any_i2_kept = True
            else:
                any_i2_kept = True

        # Handle any unexpected first_rating values outside {"1","2","3","4"} defensively
        other_mask = ~np.isin(first_ratings_i2, ("1", "2", "3", "4"))
        if other_mask.any():
            for fr in np.unique(first_ratings_i2[other_mask]):
                fr_mask = first_ratings_i2 == fr
                dt_sub = delta_t_i2[fr_mask]
                uniq_dt, counts = np.unique(dt_sub, return_counts=True)
                order = np.lexsort((-uniq_dt, counts))
                total = len(dt_sub)
                threshold = max(total * 0.05, 20.0)
                max_dt = 365 if fr == "4" else 100
                removed_dts = []
                has_been_removed = 0
                for c, dt_val in zip(counts[order].tolist(), uniq_dt[order].tolist()):
                    if has_been_removed + c >= threshold:
                        if c < 6 or dt_val > max_dt:
                            removed_dts.append(dt_val)
                            has_been_removed += c
                    else:
                        removed_dts.append(dt_val)
                        has_been_removed += c
                if removed_dts:
                    sub_keep = ~np.isin(dt_sub, removed_dts)
                    keep_row[i2_pos[fr_mask]] = sub_keep
                    if sub_keep.any():
                        any_i2_kept = True
                else:
                    any_i2_kept = True

        if not any_i2_kept:
            return pd.DataFrame()

        if not keep_row.all():
            df = df[keep_row].copy()
            int_cols = [c for c in df.columns if pd.api.types.is_integer_dtype(df[c])]
            if int_cols:
                df[int_cols] = df[int_cols].astype(np.float64)

        # Vectorized remove_non_continuous_rows over contiguous card runs
        m = len(df)
        if m == 0:
            return pd.DataFrame()
        card_ids = df["card_id"].to_numpy()
        i_rem = df["i"].to_numpy()
        is_start, start_idx = _card_start_indices(card_ids)
        is_discont = np.zeros(m, dtype=bool)
        is_discont[1:] = (~is_start[1:]) & ((i_rem[1:] - i_rem[:-1]) != 1)
        if not is_discont.any():
            return df

        if df.index.is_monotonic_increasing:
            cum_discont = np.cumsum(is_discont, dtype=np.int64)
            return df[cum_discont == cum_discont[start_idx]]

        # Fallback for synthetic DataFrames with non-monotonic index labels within cards
        from fsrs_optimizer import remove_non_continuous_rows

        # pyrefly: ignore [bad-index]
        return df.groupby("card_id", as_index=False, group_keys=False)[
            df.columns
        ].apply(remove_non_continuous_rows)

    def get_time_history_list(self, df: pd.DataFrame) -> pd.Series:
        """
        Get time history list for feature engineering

        Args:
            df: Input dataframe

        Returns:
            Time history list as pandas Series
        """
        if self.config.use_secs_intervals:
            t_history_list = df.groupby("card_id", group_keys=False)[
                (
                    "delta_t_secs"
                    if not self.config.equalize_test_with_non_secs
                    else "delta_t"
                )
            ].apply(lambda x: cum_concat([[i] for i in x]))
        else:
            t_history_list = df.groupby("card_id", group_keys=False)["delta_t"].apply(
                lambda x: cum_concat([[i] for i in x])
            )
        return t_history_list

    def get_rating_history_list(self, df: pd.DataFrame) -> pd.Series:
        """
        Get rating history list for feature engineering

        Args:
            df: Input dataframe

        Returns:
            Rating history list as pandas Series
        """
        r_history_list = df.groupby("card_id", group_keys=False)["rating"].apply(
            lambda x: cum_concat([[i] for i in x])
        )
        return r_history_list

    def get_nth_today_history_list(self, df: pd.DataFrame) -> pd.Series:
        """
        Get nth today history list for feature engineering
        """
        n_history_list = df.groupby("card_id", group_keys=False)["nth_today"].apply(
            lambda x: cum_concat([[i] for i in x])
        )
        return n_history_list

    def get_history_lists(self, df: pd.DataFrame):
        """
        Get history record lists for feature engineering

        Returns:
            Tuple of (time_history_list, rating_history_list)
        """
        if self._cached_t_history_list is None or self._cached_r_history_list is None:
            t_col = (
                "delta_t_secs"
                if self.config.use_secs_intervals and "delta_t_secs" in df.columns
                else "delta_t"
            )
            self._cached_t_history_list = _cumulative_lists_by_card(df, t_col)
            self._cached_r_history_list = _cumulative_lists_by_card(df, "rating")
        return self._cached_t_history_list, self._cached_r_history_list

    def _model_specific_postprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Model-specific post processing

        Args:
            df: Dataframe after common post processing

        Returns:
            Dataframe with model-specific post processing
        """
        return df
