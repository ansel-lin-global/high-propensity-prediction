"""
Tests for the rolling time-series split logic used in split_data_by_time_series.
The logic is replicated here from components/train.py since it lives inside a KFP
component body and cannot be imported directly.
"""

import pandas as pd


def generate_splits(df: pd.DataFrame, date_col: str, gap: int, prediction_window: int):
    unique_dates = sorted(df[date_col].unique())
    splits = []
    for i in range(len(unique_dates) - gap - prediction_window + 1):
        train_end = unique_dates[i]
        test_range = unique_dates[i + gap : i + gap + prediction_window]
        train_idx = df[df[date_col] <= train_end].index.tolist()
        test_idx = df[df[date_col].isin(test_range)].index.tolist()
        splits.append((train_idx, test_idx))
    return splits


def _make_df(n_dates: int = 10, rows_per_date: int = 3) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="D")
    rows = [{"date": d, "value": i} for i, d in enumerate(dates) for _ in range(rows_per_date)]
    return pd.DataFrame(rows)


def test_split_count_matches_formula():
    df = _make_df(n_dates=10)
    splits = generate_splits(df, "date", gap=3, prediction_window=1)
    # expected: n_dates - gap - prediction_window + 1 = 10 - 3 - 1 + 1 = 7
    assert len(splits) == 7


def test_train_and_test_indices_are_disjoint():
    df = _make_df(n_dates=8)
    for train_idx, test_idx in generate_splits(df, "date", gap=2, prediction_window=1):
        assert set(train_idx).isdisjoint(set(test_idx))


def test_gap_enforces_no_leakage():
    df = _make_df(n_dates=8)
    for train_idx, test_idx in generate_splits(df, "date", gap=2, prediction_window=1):
        train_dates = set(df.loc[train_idx, "date"])
        test_dates = set(df.loc[test_idx, "date"])
        # All test dates must be strictly after all train dates
        assert max(train_dates) < min(test_dates)


def test_empty_dataframe_returns_no_splits():
    df = pd.DataFrame(columns=["date", "value"])
    splits = generate_splits(df, "date", gap=3, prediction_window=1)
    assert splits == []


def test_prediction_window_greater_than_one():
    df = _make_df(n_dates=10)
    splits = generate_splits(df, "date", gap=2, prediction_window=3)
    # expected: 10 - 2 - 3 + 1 = 6
    assert len(splits) == 6
    for _, test_idx in splits:
        test_dates = df.loc[test_idx, "date"].unique()
        assert len(test_dates) == 3
