"""Median Latin hypercube samplers.

The published model (built in Lumina Analytica) used median Latin hypercube
sampling with n = 10,000. Median LHS partitions the unit interval into n
equiprobable strata and evaluates each distribution at the *midpoint* quantile
of every stratum, then shuffles the assignment independently per variable. This
reproduces Analytica's default behaviour and gives stable, low-variance
estimates that match the manuscript tables closely without requiring very large
sample sizes.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def median_lhs_probabilities(n: int, rng: np.random.Generator) -> np.ndarray:
    """Return n shuffled stratum-midpoint probabilities in (0, 1)."""
    p = (np.arange(n) + 0.5) / n
    rng.shuffle(p)
    return p


def sample_truncated_normal(
    n: int,
    mean: float,
    sd: float,
    low: float,
    high: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Median-LHS draws from a normal truncated to [low, high]."""
    a, b = (low - mean) / sd, (high - mean) / sd
    p = median_lhs_probabilities(n, rng)
    return stats.truncnorm.ppf(p, a, b, loc=mean, scale=sd)


def sample_uniform(
    n: int, low: float, high: float, rng: np.random.Generator
) -> np.ndarray:
    """Median-LHS draws from Uniform[low, high]."""
    p = median_lhs_probabilities(n, rng)
    return low + p * (high - low)


def sample_lognormal_median(
    n: int, median: float, gsdev: float, rng: np.random.Generator
) -> np.ndarray:
    """Median-LHS draws from a lognormal specified by its median and gsdev.

    Matches Analytica's ``Lognormal(median:, gsdev:)`` parameterization, where
    the geometric standard deviation maps to the shape parameter s = ln(gsdev)
    and the median maps to the scale parameter.
    """
    p = median_lhs_probabilities(n, rng)
    s = np.log(gsdev)
    return stats.lognorm.ppf(p, s, scale=median)
