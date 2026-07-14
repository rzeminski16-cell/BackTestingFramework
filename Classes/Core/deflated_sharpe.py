"""
Probabilistic and Deflated Sharpe Ratios (Bailey & López de Prado).

When an optimizer examines N parameter combinations and reports the best
one, the winning Sharpe ratio is inflated by selection: even pure noise
produces an impressive "best of N". The Deflated Sharpe Ratio (DSR) asks
the honest question — what is the probability that the chosen
configuration's true Sharpe exceeds the Sharpe you would expect from the
LUCKIEST of N skill-less trials?

    PSR(SR*) = Phi( (SR_hat - SR*) * sqrt(n - 1)
                    / sqrt(1 - skew*SR_hat + ((kurt - 1)/4)*SR_hat^2) )

    DSR = PSR(SR*) with SR* = E[max SR over N trials]
        = sqrt(V[trial SRs]) * ((1-gamma)*z(1 - 1/N) + gamma*z(1 - 1/(N*e)))

where SR_hat is the observed PER-PERIOD Sharpe, n the number of return
observations, skew/kurt the return distribution's moments (normal defaults
0 / 3), z the standard-normal quantile and gamma the Euler-Mascheroni
constant. DSR > 0.95 means the result survives its own search at 95%
confidence.

References:
    Bailey, D.H. & López de Prado, M. (2014), "The Deflated Sharpe Ratio:
    Correcting for Selection Bias, Backtest Overfitting and Non-Normality".
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

import numpy as np
from scipy.stats import norm

EULER_MASCHERONI = 0.5772156649015329


def probabilistic_sharpe_ratio(observed_sr: float,
                               benchmark_sr: float,
                               n_obs: int,
                               skew: float = 0.0,
                               kurtosis: float = 3.0) -> float:
    """
    Probability that the true Sharpe ratio exceeds ``benchmark_sr``.

    All Sharpe ratios are PER-PERIOD (not annualized) — divide an annual
    Sharpe by sqrt(periods_per_year) first.

    Args:
        observed_sr: Estimated per-period Sharpe ratio
        benchmark_sr: Threshold per-period Sharpe to beat (0 = "any skill")
        n_obs: Number of return observations the estimate is based on
        skew: Skewness of the returns (0 for normal)
        kurtosis: Kurtosis of the returns (3 for normal)

    Returns:
        PSR in [0, 1]; 0.0 when n_obs < 2
    """
    if n_obs < 2:
        return 0.0
    denom_sq = 1.0 - skew * observed_sr + ((kurtosis - 1.0) / 4.0) * observed_sr ** 2
    if denom_sq <= 0:
        # Degenerate moment combination; fall back to the normal case.
        denom_sq = 1.0
    z = (observed_sr - benchmark_sr) * math.sqrt(n_obs - 1) / math.sqrt(denom_sq)
    return float(norm.cdf(z))


def expected_max_sharpe(n_trials: int, trial_sr_variance: float) -> float:
    """
    Expected maximum per-period Sharpe among ``n_trials`` independent
    skill-less trials whose SR estimates have variance ``trial_sr_variance``.

    Returns 0.0 when fewer than 2 trials or no variance (no selection
    pressure to correct for).
    """
    if n_trials < 2 or trial_sr_variance <= 0:
        return 0.0
    g = EULER_MASCHERONI
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(math.sqrt(trial_sr_variance) * ((1.0 - g) * z1 + g * z2))


def deflated_sharpe_ratio(observed_sharpe_annual: float,
                          trial_sharpes_annual: Sequence[float],
                          n_obs: int,
                          periods_per_year: int = 252,
                          skew: float = 0.0,
                          kurtosis: float = 3.0) -> Dict[str, Any]:
    """
    Deflated Sharpe Ratio for the winner of a parameter search.

    Args:
        observed_sharpe_annual: The chosen configuration's ANNUALIZED Sharpe
        trial_sharpes_annual: Annualized Sharpe of every configuration the
            search examined (including the winner and failures)
        n_obs: Number of return observations behind the observed Sharpe
            (e.g. training-window bars)
        periods_per_year: Annualization factor used for the Sharpe ratios
        skew / kurtosis: Return-distribution moments (normal defaults)

    Returns:
        Dict with:
            dsr: probability the true SR beats the luckiest-noise benchmark
            psr_vs_zero: probability the true SR is simply > 0 (no
                multiplicity correction) — for comparison
            expected_max_sharpe_annual: the annualized selection benchmark
            n_trials: number of finite trials used
            n_obs: echoed back
    """
    ann = math.sqrt(periods_per_year)
    sr_hat = observed_sharpe_annual / ann

    trials = np.asarray(
        [s for s in trial_sharpes_annual if s is not None and np.isfinite(s)],
        dtype=float,
    ) / ann
    n_trials = int(trials.size)

    sr_star = expected_max_sharpe(n_trials, float(np.var(trials))) if n_trials >= 2 else 0.0

    return {
        "dsr": probabilistic_sharpe_ratio(sr_hat, sr_star, n_obs, skew, kurtosis),
        "psr_vs_zero": probabilistic_sharpe_ratio(sr_hat, 0.0, n_obs, skew, kurtosis),
        "expected_max_sharpe_annual": sr_star * ann,
        "n_trials": n_trials,
        "n_obs": int(n_obs),
    }
