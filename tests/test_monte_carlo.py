"""Unit tests for the vectorized Monte Carlo simulation engine."""

import time
from app.ml.monte_carlo import MonteCarloSimulator


def test_monte_carlo_speed_benchmark():
    """Verify 10,000 iterations execute under 200ms."""
    sim = MonteCarloSimulator(random_seed=42)
    # Warmup call
    sim.simulate_chase(target_runs=150, current_score=50, overs_completed=5.0, wickets_fallen=1, iterations=100)

    start = time.perf_counter()
    res = sim.simulate_chase(
        target_runs=175,
        current_score=80,
        overs_completed=10.0,
        wickets_fallen=2,
        iterations=10000,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert res["simulation_time_ms"] < 200.0, f"Monte Carlo took {res['simulation_time_ms']}ms, which exceeds 200ms threshold!"
    assert elapsed_ms < 300.0, f"Total elapsed took {elapsed_ms:.2f}ms!"
    assert res["iterations"] == 10000
    assert 0.0 <= res["win_probability"] <= 1.0


def test_monte_carlo_monotonic_confidence_intervals():
    """Verify ordering of confidence intervals."""
    sim = MonteCarloSimulator(random_seed=123)
    res = sim.simulate_chase(
        target_runs=180,
        current_score=90,
        overs_completed=10.0,
        wickets_fallen=3,
        iterations=10000,
    )
    ci = res["confidence_intervals"]
    # 95% CI low <= 80% CI low <= 50% CI low <= 50% CI high <= 80% CI high <= 95% CI high
    assert ci["95%"][0] <= ci["80%"][0] <= ci["50%"][0]
    assert ci["50%"][1] <= ci["80%"][1] <= ci["95%"][1]
    assert ci["50%"][0] <= ci["50%"][1]


def test_monte_carlo_deterministic_with_seed():
    """Verify identical results with identical seed."""
    sim1 = MonteCarloSimulator(random_seed=999)
    res1 = sim1.simulate_chase(target_runs=180, current_score=100, overs_completed=12.0, wickets_fallen=2, iterations=5000)

    sim2 = MonteCarloSimulator(random_seed=999)
    res2 = sim2.simulate_chase(target_runs=180, current_score=100, overs_completed=12.0, wickets_fallen=2, iterations=5000)

    assert res1["win_probability"] == res2["win_probability"]
    assert res1["projected_score_mean"] == res2["projected_score_mean"]
