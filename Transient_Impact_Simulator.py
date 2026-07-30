import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# TRANSIENT IMPACT SIMULATOR (Obizhaeva-Wang, 2013)
# ============================================================
# FIX vs original version:
#   1) The decaying impact function now actually enters the price
#      path (before, it was computed and thrown away except for a
#      separate decorative plot — changing rho had zero effect on
#      the simulated price).
#   2) Impact is updated recursively in O(N) instead of re-summing
#      the full trade history at every step in O(N^2). The two are
#      mathematically equivalent for the classic O-W exponential
#      kernel: I_t = I_{t-1} * exp(-rho*dt) + eta * size_t
#   3trade size used in the kernel is v*dt (shares actually traded
#      in the interval), not the rate v — units now consistent.


def simulate_transient_impact(rho, Q_total, T, N, sigma_daily, eta_impact=0.05):
    """
    rho          : resilience parameter (higher = faster decay = more resilient book)
    Q_total      : total shares to execute
    T            : horizon (same time units as sigma_daily, e.g. 1.0 = one day)
    N            : number of slices
    sigma_daily  : volatility over the horizon T
    eta_impact   : impact coefficient per unit traded (temporary/transient impact strength)
    """
    dt = T / N
    v = Q_total / N          # constant execution rate (shares per unit time)
    trade_size = v * dt      # shares actually traded in each slice

    decay = np.exp(-rho * dt)  # one-step decay factor

    fundamental = 100.0      # pure random-walk component, unaffected by our trades
    impact = 0.0              # current transient impact level, I_t (decays over time)

    fundamentals = [fundamental]
    impact_history = [impact]
    prices = [fundamental + impact]

    for i in range(N):
        # 1) fundamental price evolves as a pure random walk (no impact here)
        noise = np.random.normal(0, sigma_daily * np.sqrt(dt))
        fundamental = fundamental + noise

        # 2) impact decays from the previous level, then gets a fresh kick
        #    from the trade executed in this slice
        impact = impact * decay + eta_impact * trade_size

        # 3) OBSERVED price = fundamental + impact overlay. This is the line
        #    that was missing before: impact now actually moves the price,
        #    and as rho changes, the impact overlay — and therefore the
        #    simulated price path — changes with it.
        price = fundamental + impact

        fundamentals.append(fundamental)
        impact_history.append(impact)
        prices.append(price)

    return np.array(prices), np.array(impact_history)


# Parameters
T_sim = 1.0
N_sim = 100
Q_sim = 100000
vol = 0.01

np.random.seed(42)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

results = {}
for r_val, label in zip([0.5, 5.0], ['Low Resilience (Slow decay)', 'High Resilience (Fast decay)']):
    p, impact = simulate_transient_impact(r_val, Q_sim, T_sim, N_sim, vol)
    results[label] = (p, impact)
    t_axis = np.linspace(0, T_sim, N_sim + 1)
    axes[0].plot(t_axis, p, label=f'{label} (ρ={r_val})')
    axes[1].plot(t_axis, impact, label=f'Impact level: {label} (ρ={r_val})')

axes[0].set_title('Simulated Price Path', fontweight='bold')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Price')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].set_title('Transient Impact Level I(t) (Obizhaeva-Wang)', fontweight='bold')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Price Displacement')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
