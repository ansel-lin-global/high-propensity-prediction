"""
Tests for the PSI (Population Stability Index) calculation used in detect_data_drift_psi.
The formula is replicated here from components/drift.py since it lives inside a KFP
component body and cannot be imported directly.
"""

import numpy as np
import pytest


def calculate_psi(expected, actual, buckets: int = 10) -> float:
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    expected_counts = np.histogram(expected, bins=breakpoints)[0] + 1e-6
    actual_counts = np.histogram(actual, bins=breakpoints)[0] + 1e-6
    expected_percents = expected_counts / expected_counts.sum()
    actual_percents = actual_counts / actual_counts.sum()
    return float(
        np.sum((expected_percents - actual_percents) * np.log(expected_percents / actual_percents))
    )


def test_psi_identical_distributions_is_zero():
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, 1000)
    assert calculate_psi(data, data) == pytest.approx(0.0, abs=1e-6)


def test_psi_heavily_shifted_distribution_signals_strong_drift():
    rng = np.random.default_rng(42)
    expected = rng.normal(0, 1, 1000)
    actual = rng.normal(5, 1, 1000)
    assert calculate_psi(expected, actual) > 0.2


def test_psi_is_non_negative():
    rng = np.random.default_rng(99)
    expected = rng.uniform(0, 1, 500)
    actual = rng.uniform(0.5, 1.5, 500)
    assert calculate_psi(expected, actual) >= 0.0


def test_psi_slight_shift_below_threshold():
    rng = np.random.default_rng(7)
    expected = rng.normal(0, 1, 2000)
    actual = rng.normal(0.05, 1, 2000)  # tiny shift — stable
    assert calculate_psi(expected, actual) < 0.1


def test_psi_with_fewer_buckets():
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1, 200)
    assert calculate_psi(data, data, buckets=5) == pytest.approx(0.0, abs=1e-6)
