"""
execution_metrics.py

Core execution-quality metrics:
- Implementation Shortfall (IS)
- Square-root market impact model
- VWAP and VWAP participation scheduling

See README.md, Sections 3-5, for the full derivations.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Implementation Shortfall
# ---------------------------------------------------------------------------

def implementation_shortfall(decision_price, execution_prices, execution_quantities):
    """
    IS = actual cost paid - planned cost at the decision price.

    Positive IS on a buy order means you paid more than planned.
    Positive IS on a sell order means you received less than planned.

    Parameters
    ----------
    decision_price : float
        S0, the benchmark price at order inception.
    execution_prices : array-like
        Fill price for each slice.
    execution_quantities : array-like
        Shares executed in each slice.
    """
    execution_prices = np.asarray(execution_prices, dtype=float)
    execution_quantities = np.asarray(execution_quantities, dtype=float)

    total_quantity = np.sum(execution_quantities)
    actual_cost = np.sum(execution_prices * execution_quantities)
    planned_cost = total_quantity * decision_price
    return actual_cost - planned_cost


# ---------------------------------------------------------------------------
# Square-root market impact model
# ---------------------------------------------------------------------------

def estimate_impact_pct(order_size, daily_volume, volatility):
    """Square-root impact model, expressed as a percentage of price."""
    return volatility * np.sqrt(order_size / daily_volume)


def estimate_impact_dollars(order_size, daily_volume, volatility, price):
    """Converts percentage impact into a dollar cost."""
    impact_pct = estimate_impact_pct(order_size, daily_volume, volatility)
    return impact_pct * price * order_size


# ---------------------------------------------------------------------------
# VWAP
# ---------------------------------------------------------------------------

def calculate_vwap(prices, volumes):
    """Volume-weighted average price."""
    prices = np.asarray(prices, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    return np.sum(prices * volumes) / np.sum(volumes)


def vwap_schedule(order_size, expected_volumes):
    """
    Generates a participation schedule proportional to expected volume,
    so the order tracks VWAP rather than trading at a uniform rate.
    """
    expected_volumes = np.asarray(expected_volumes, dtype=float)
    volume_fractions = expected_volumes / np.sum(expected_volumes)
    return order_size * volume_fractions


if __name__ == "__main__":
    # Minimal smoke test
    is_value = implementation_shortfall(75.00, [75.2, 75.5], [1000, 1000])
    print(f"Implementation shortfall: ${is_value:.2f}")

    impact_pct = estimate_impact_pct(50_000, 2_000_000, 0.02)
    print(f"Estimated impact: {impact_pct:.4%}")

    vwap = calculate_vwap([75.0, 75.5, 76.0], [1000, 2000, 1500])
    print(f"VWAP: ${vwap:.4f}")
