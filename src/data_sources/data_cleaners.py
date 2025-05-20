import pandas as pd
import numpy as np
import logging

class DataCleaner:
    @staticmethod
    def remove_na(df, method='ffill', drop_threshold=0.1):
        """
        Remove or fill NA values in the DataFrame.
        - method: 'ffill', 'bfill', or 'drop'
        - drop_threshold: if more than this fraction of a row is NA, drop the row
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        na_fraction = df.isna().mean(axis=1)
        rows_to_drop = na_fraction > drop_threshold
        if rows_to_drop.any():
            logging.info(f"Dropping {rows_to_drop.sum()} rows with >{drop_threshold*100}% missing values.")
            df = df.loc[~rows_to_drop]

        if method == 'ffill':
            df = df.fillna(method='ffill').fillna(method='bfill')
        elif method == 'bfill':
            df = df.fillna(method='bfill').fillna(method='ffill')
        elif method == 'drop':
            df = df.dropna()
        else:
            raise ValueError("Unknown method for NA handling.")

        return df

    @staticmethod
    def check_gaps(df, max_gap=5):
        """
        Check for large gaps in the index (assumes DateTimeIndex).
        Returns a list of (start, end, gap_size) for gaps larger than max_gap days.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a pandas DatetimeIndex.")

        gaps = []
        diffs = df.index.to_series().diff().dt.days.fillna(0)
        large_gaps = diffs[diffs > max_gap]
        for idx in large_gaps.index:
            prev_idx = df.index[df.index.get_loc(idx) - 1]
            gap_size = (idx - prev_idx).days
            gaps.append((prev_idx, idx, gap_size))
            logging.warning(f"Large gap detected: {prev_idx} to {idx} ({gap_size} days)")
        return gaps

    @staticmethod
    def clean(df, na_method='ffill', drop_threshold=0.1, max_gap=5):
        """
        Full cleaning pipeline: remove/fill NAs and check for large gaps.
        Returns cleaned DataFrame and list of gaps.
        """
        df_clean = DataCleaner.remove_na(df, method=na_method, drop_threshold=drop_threshold)
        gaps = DataCleaner.check_gaps(df_clean, max_gap=max_gap)
        return df_clean, gaps