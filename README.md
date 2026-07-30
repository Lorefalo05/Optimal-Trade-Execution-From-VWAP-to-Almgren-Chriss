# Optimal-Trade-Execution-From-VWAP-to-Almgren-Chriss
A quantitative framework for minimizing implementation shortfall in algorithmic order execution
## Table of Contents
1. [Motivation](#1-motivation)
2. [Problem Setup](#2-problem-setup)
3. [Implementation Shortfall](#3-implementation-shortfall)
4. [The Square-Root Market Impact Model](#4-the-square-root-market-impact-model)
5. [VWAP: Benchmark, Not Optimizer](#5-vwap-benchmark-not-optimizer)
6. [Temporary vs. Permanent Impact](#6-temporary-vs-permanent-impact)
7. [The Almgren-Chriss Framework](#7-the-almgren-chriss-framework)
8. [Case Study: A 50,000-Share Momentum Order](#8-case-study-a-50000-share-momentum-order)
9. [Extension: Transient Impact (Obizhaeva-Wang)](#9-extension-transient-impact-obizhaeva-wang)
10. [Limitations & Future Work](#10-limitations--future-work)
11. [References](#11-references)
---
 
## 1. Motivation
 
Every large order faces a dilemma that has no free lunch:
 
- Trade **fast** → you cross the spread and consume liquidity aggressively, paying a steep **market impact** cost.
- Trade **slow** → you reduce impact, but you're now exposed to the market drifting against you before you're done — **timing risk**.
This project builds the mathematical toolkit to reason about that trade-off quantitatively, culminating in the Almgren-Chriss (2000) closed-form optimal execution trajectory, and extends it toward more realistic, decaying impact dynamics.
 
---
 
## 2. Problem Setup
 
| Symbol | Meaning |
|---|---|
| `Q` | Total shares to execute (positive = buy, negative = sell) |
| `T` | Time horizon for the order |
| `x(t)` | Shares remaining at time `t`, with `x(0) = Q`, `x(T) = 0` |
| `v(t) = -ẋ(t)` | Trading rate (shares per unit time) |
| `S₀` | Decision price — the price when the order was initiated |
| `S̃(t)` | Execution price at time `t` |
 
The decision price `S₀` represents the frictionless benchmark: what you *could* have gotten with zero market impact and instant execution. Every cost measure in this project is defined relative to it.
 
---
 
## 3. Implementation Shortfall
 
**Implementation Shortfall (IS)** is the standard measure of execution quality — the gap between the planned cost (at the decision price) and the actual cost paid:
 
```
IS = ∫₀ᵀ v(t)·S̃(t) dt − Q·S₀
```
 
For a buy order (`Q > 0`), positive IS means you paid *more* than planned. For a sell order (`Q < 0`), positive IS means you received *less* than planned. In both cases, **positive IS is bad** — it is the total dollar cost of friction in the execution process.
 
```python
import numpy as np
 
def implementation_shortfall(decision_price, execution_prices, execution_quantities):
    """
    decision_price       : S0, the benchmark price at order inception
    execution_prices     : array of fill prices per slice
    execution_quantities : array of shares executed per slice
    """
    total_quantity = np.sum(execution_quantities)
    actual_cost = np.sum(execution_prices * execution_quantities)
    planned_cost = total_quantity * decision_price
    return actual_cost - planned_cost
```
 
---
 
## 4. The Square-Root Market Impact Model
 
Before building a full trajectory optimizer, we need a way to *estimate* impact. The empirically robust **square-root law** relates impact to order size relative to liquidity:
 
```
Impact (%) = σ · √(Q / V)
```
 
| Symbol | Meaning |
|---|---|
| `σ` | Daily volatility of the stock |
| `Q` | Order size (shares) |
| `V` | Average daily volume (shares) |
 
Converting to dollar terms:
 
```
Impact Cost = Impact (%) × Price × Shares
```
 
```python
import numpy as np
 
def estimate_impact_pct(order_size, daily_volume, volatility):
    """Square-root impact model, in percent of price."""
    return volatility * np.sqrt(order_size / daily_volume)
 
def estimate_impact_dollars(order_size, daily_volume, volatility, price):
    """Converts percentage impact into a dollar cost."""
    impact_pct = estimate_impact_pct(order_size, daily_volume, volatility)
    return impact_pct * price * order_size
```
 
**Rules of thumb from the literature:**
- Orders above ~5% of daily volume face significant impact.
- Institutional order impact typically runs 10–100 basis points.
- The square-root law is empirically robust across markets and time periods.
---
 
## 5. VWAP: Benchmark, Not Optimizer
 
**Volume-Weighted Average Price (VWAP)** is the average price paid by the "typical" market participant over a period:
 
```
VWAP = Σ(Pᵢ · Vᵢ) / Σ(Vᵢ)
```
 
Beating VWAP on a buy order means your average execution price came in *below* it — better-than-average execution.
 
### VWAP Participation Scheduling
 
To track VWAP, you trade in proportion to *expected* volume in each period:
 
```
Shares in period i = Q × (Vᵢ_expected / Σⱼ Vⱼ_expected)
```
 
This requires forecasting the intraday volume curve — fortunately, volume patterns are highly consistent (high at the open, low at midday, high at the close), so historical averages work reasonably well.
 
```python
import numpy as np
 
def calculate_vwap(prices, volumes):
    return np.sum(prices * volumes) / np.sum(volumes)
 
def vwap_schedule(order_size, expected_volumes):
    """Generates a participation schedule proportional to expected volume."""
    expected_volumes = np.asarray(expected_volumes)
    volume_fractions = expected_volumes / np.sum(expected_volumes)
    return order_size * volume_fractions
```
 
**Key insight:** VWAP algorithms don't try to *beat* the market — they try to *match* the average. That's the ceiling of what VWAP can do, and it's exactly what motivates a genuine optimizer.
 
---
 
## 6. Temporary vs. Permanent Impact
 
Price impact from trading splits into two components:
 
- **Temporary impact** — displacement from your immediate demand for liquidity. You "walk the book" and pay up; the effect reverts once you stop trading.
```
  ΔS_temp = η · v
```
  where `η` is the temporary impact coefficient (\$ per share per unit time) and `v` is the trading rate.
 
- **Permanent impact** — the lasting price shift from information leakage: the market infers a buyer (or seller) is present, and re-prices accordingly.
```
  ΔS_perm = γ · n
```
  where `γ` is the permanent impact coefficient (\$ per share) and `n` is cumulative shares traded so far.
 
### The Linear Impact Model
 
Combining both, the execution price for the `n`-th share traded at rate `v`, given `n_prior` shares already executed, is:
 
```
S̃ = S₀ + γ·n_prior + η·v
```
 
and the impact cost (excess over the unaffected price) is:
 
```
Impact Cost = n × (γ·n_prior + η·v)
```
 
```python
import numpy as np
 
def temporary_impact(trading_rate, eta):
    return eta * trading_rate
 
def permanent_impact(shares_traded, gamma):
    return gamma * shares_traded
 
def execution_price(base_price, shares_prior, trading_rate, gamma, eta):
    return base_price + eta * trading_rate + gamma * shares_prior
```
 
### Sliced Execution Cost
 
Executing an order in `N` discrete slices, each slice faces permanent impact compounded from *all prior* slices:
 
```
Total Cost = Σᵢ nᵢ × (S₀ + γ·Σⱼ<ᵢ nⱼ + η·vᵢ)
```
 
```python
import numpy as np
 
def sliced_execution_cost(base_price, slice_sizes, slice_rates, gamma, eta):
    total_cost = 0.0
    cumulative_shares = 0.0
    for n_i, v_i in zip(slice_sizes, slice_rates):
        exec_price = base_price + gamma * cumulative_shares + eta * v_i
        total_cost += n_i * exec_price
        cumulative_shares += n_i
    return total_cost
 
def impact_cost(base_price, slice_sizes, slice_rates, gamma, eta):
    total_cost = sliced_execution_cost(base_price, slice_sizes, slice_rates, gamma, eta)
    baseline_cost = np.sum(slice_sizes) * base_price
    return total_cost - baseline_cost
```
 
Trading slowly lowers temporary impact but extends exposure to adverse price moves (timing risk). Trading quickly does the reverse. **Finding the balance is the core problem of execution optimization** — which is exactly what Almgren-Chriss solves.
 
---
 
## 7. The Almgren-Chriss Framework
 
VWAP ignores three realities: (1) you move the price, (2) you're exposed to volatility while your order is unfilled, and (3) different traders have different risk tolerances. Almgren and Chriss (2000) address all three by framing execution as **mean-variance optimization**:
 
```
min_trajectory  E[Cost] + λ·Var[Cost]
```
 
- `E[Cost]` — expected impact cost.
- `Var[Cost]` — variance from price volatility during execution.
- `λ` — risk-aversion parameter: higher λ trades certainty for a lower variance, at the cost of higher expected impact.
### Timing Risk
 
While shares remain unexecuted, their value is exposed to a random walk `dS = σ·dW`. The variance of total cost depends on the entire holding trajectory:
 
```
Var[Cost] = σ² ∫₀ᵀ x(t)² dt
```
 
The longer — and the more — you hold, the greater your exposure to adverse price drift.
 
### Closed-Form Optimal Trajectory
 
For the linear impact model above, Almgren-Chriss admits an **analytical solution**:
 
```
x*(t) = Q · sinh(κ(T − t)) / sinh(κT)
```
 
where the **urgency parameter** is:
 
```
κ = √(λσ² / η)
```
 
`κ` fuses risk aversion, volatility, and temporary impact into a single number governing trajectory shape:
 
- **High κ** (high risk aversion / volatility, or low temporary impact) → front-loaded, urgent execution.
- **Low κ** (low risk aversion / volatility, or high temporary impact) → slow, even execution.
### The Risk-Neutral Limit (κ → 0)
 
As `λ → 0`, `sinh(κx) ≈ κx`, and the trajectory collapses to a straight line:
 
```
x*(t) = Q · (T − t) / T
```
 
— a uniform trading rate `Q/T`, i.e. equal shares per slice. This is the textbook "TWAP" schedule, recovered as the zero-risk-aversion special case of Almgren-Chriss.
 
### Implementation Recipe
 
1. Compute urgency: `κ = √(λσ²/η)`
2. Generate time points `tₖ = k·τ`, `τ = T/N`, for `k = 0, ..., N`
3. Compute the trajectory:
   - if `κ > 0`: `xₖ = Q · sinh(κ(T − tₖ)) / sinh(κT)`
   - if `κ = 0`: `xₖ = Q · (N − k) / N`
4. Compute the per-slice schedule: `nₖ = xₖ₋₁ − xₖ`
```python
import numpy as np
 
def almgren_chriss_trajectory(Q, T, N, lam, sigma, eta):
    """Returns (time_points, shares_remaining, per_slice_schedule)."""
    tau = T / N
    t_points = np.arange(N + 1) * tau
    kappa = np.sqrt(lam * sigma**2 / eta)
 
    if kappa > 0:
        trajectory = Q * np.sinh(kappa * (T - t_points)) / np.sinh(kappa * T)
    else:
        trajectory = Q * (T - t_points) / T
 
    schedule = -np.diff(trajectory)
    return t_points, trajectory, schedule
```
 
---
 
## 8. Case Study: A 50,000-Share Momentum Order
 
**Scenario:** a momentum signal predicts a +2% move with a 2-hour alpha half-life. The desk must buy 50,000 shares of a mid-cap stock within a 4-hour horizon.
 
| Parameter | Value |
|---|---|
| Shares (`Q`) | 50,000 |
| Horizon (`T`) | 4 hours |
| Slices (`N`) | 8 (30-min bars) |
| Volatility (`σ`) | 1.5%/hour |
| Permanent impact (`γ`) | 5e-6 |
| Temporary impact (`η`) | 8e-5 |
| Decision price | \$75.00 |
 
**Pre-trade: choosing λ from alpha decay.** Since the signal's edge decays with a 2-hour half-life, urgency should roughly match that decay rate: `κ_target = 1 / alpha_halflife`. Solving `κ = √(λσ²/η)` for λ gives:
 
```
λ_suggested = κ_target² · η / σ²
```
 
This ties the risk-aversion parameter directly to how fast the alpha is expected to disappear — a fast-decaying signal justifies a more urgent (front-loaded) trajectory even before considering pure risk aversion.
 
```python
import numpy as np
 
Q, T, N = 50_000, 4.0, 8
sigma, gamma, eta = 0.015, 0.000005, 0.00008
decision_price = 75.00
 
alpha_halflife = 2.0
target_kappa = 1.0 / alpha_halflife
lambda_risk = (target_kappa ** 2) * eta / (sigma ** 2)
 
t_points, trajectory, schedule = almgren_chriss_trajectory(
    Q, T, N, lambda_risk, sigma, eta
)
```
 
**Cost decomposition** (permanent + temporary impact, plus timing-risk standard deviation):
 
```python
tau = T / N
cumulative = 0.0
perm_cost = 0.0
for n_i in schedule:
    perm_cost += gamma * n_i * cumulative
    cumulative += n_i
 
temp_cost = eta * np.sum(schedule**2 / tau)
expected_cost = perm_cost + temp_cost
std_dev = np.sqrt(sigma**2 * tau * np.sum(trajectory[1:]**2))
```
 
**Post-trade evaluation.** After simulating fills (impact + noise), the realized Implementation Shortfall is compared against the pre-trade expected cost:
 
```
ratio = IS_actual / E[Cost]
```
 
- `|ratio − 1| < 0.25` → execution performed as expected.
- `ratio < 0.75` → execution beat expectations.
- otherwise → cost exceeded the pre-trade estimate; worth a post-mortem.
> The full simulation — price-path generation, fill logging, and a 4-panel diagnostic plot (trajectory, schedule, price evolution, cost breakdown) — is in [`case_study.py`](./case_study.py) in this repo.
 
### Simulated Output
 
![Case study output: execution trajectory, schedule, price evolution, and cost breakdown](./assets/case_study.png)
 
*Top left: optimal shares-remaining trajectory (λ = 0.089, κ = 0.500), visibly front-loaded. Top right: the corresponding per-slice execution schedule against the uniform (TWAP) benchmark. Bottom left: the simulated price path (black), individual fill prices (red), the decision price (blue, $75.00), and realized VWAP (green, $76.34). Bottom right: cost decomposition into permanent impact, temporary impact, total expected cost, and actual implementation shortfall.*
 
Four observations follow directly from the plots:
 
- **The trajectory is front-loaded, as κ = 0.5 > 0 predicts.** The execution schedule starts at roughly 11,500 shares in the first slice and tapers to under 3,500 by the last — well above and below the uniform `Q/N = 6,250` benchmark respectively. This reflects the pre-trade choice of λ tied to the signal's 2-hour alpha half-life: the model is deliberately urgent because the edge it's trying to capture is decaying.
- **Fill prices fall even though the underlying price path rises.** The black price path drifts up slowly from \$75.00 to about \$75.30 — mostly permanent impact plus noise. Yet the red fill prices *start higher* (near \$76.90) and *decline* toward the price path over the course of execution. This isn't a contradiction: the fill price includes the temporary-impact term `η·v`, and `v` is largest in the earliest, most urgent slices. As the trading rate tapers off, the temporary-impact premium shrinks even as the underlying price keeps creeping upward — the two effects pull fill prices in opposite directions.
- **Temporary impact dominates the cost, not permanent impact.** The cost breakdown attributes only \$5,324 to permanent impact but \$59,239 — roughly 92% of the \$64,564 total expected cost — to temporary impact. That's the direct consequence of front-loading: an aggressive early trading rate is exactly what the temporary-impact term penalizes most.
- **Realized IS tracks the pre-trade estimate closely.** The actual implementation shortfall (\$67,050) sits within about 4% of the \$64,564 expected cost — comfortably inside the "good execution" band (within 25% of expectation) defined in the assessment logic above.
This is the practical payoff of the closed-form solution: the pre-trade cost decomposition — which cost dominates, and why — is available *before* a single share trades, from just `Q`, `T`, `σ`, `γ`, `η`, and the chosen `λ`.
 
---
 
## 9. Extension: Transient Impact (Obizhaeva-Wang)
 
**Limitation of Almgren-Chriss:** impact is modeled as *either* purely temporary (reverts instantly) *or* purely permanent (never decays). Real order books sit between these extremes — impact decays gradually as the book refills.
 
**Obizhaeva & Wang (2013)** model this with a block-shaped limit order book whose impact decays exponentially at resilience rate `ρ`:
 
```
ΔS(t) = ∫₀ᵗ e^(−ρ(t−s)) dQ(s)
```
 
**Gatheral & Schied (2011)** generalize this further to an arbitrary decay kernel `G`:
 
```
S(t) = S₀ + ∫₀ᵗ G(t−s) · ẋ(s) ds
```
 
These models capture a realistic effect: large trades move prices for minutes to hours before the book fully refills, rather than snapping back instantly or never reverting at all.
 
For the exponential (Obizhaeva-Wang) kernel, the impact level updates recursively in `O(N)` — mathematically equivalent to summing the full decayed trade history, but far cheaper to compute:
 
```
I_t = I_{t-1} · e^(−ρ·dt) + η · size_t
```
 
```python
import numpy as np
 
def simulate_transient_impact(rho, Q_total, T, N, sigma_daily, eta_impact=0.05):
    """
    rho         : resilience parameter (higher = faster decay = more resilient book)
    Q_total     : total shares to execute
    T           : horizon (same units as sigma_daily)
    N           : number of slices
    sigma_daily : volatility over horizon T
    eta_impact  : impact coefficient per unit traded
    """
    dt = T / N
    trade_size = (Q_total / N) * dt      # shares traded per slice (constant rate)
    decay = np.exp(-rho * dt)             # one-step decay factor
 
    fundamental = 100.0
    impact = 0.0
    prices = [fundamental + impact]
 
    for _ in range(N):
        # fundamental price: pure random walk, unaffected by our trading
        fundamental += np.random.normal(0, sigma_daily * np.sqrt(dt))
 
        # impact decays from its previous level, then gets a fresh kick
        impact = impact * decay + eta_impact * trade_size
 
        # observed price = fundamental + impact overlay
        prices.append(fundamental + impact)
 
    return np.array(prices)
```
 
Comparing a **low-resilience** book (`ρ = 0.5`, slow decay — impact lingers) against a **high-resilience** book (`ρ = 5.0`, fast decay — the book refills quickly) shows how much of an Almgren-Chriss-style permanent-impact assumption is really an approximation to a fast-decaying transient effect.
 
> Full plotting code (price path and impact-level overlay for both regimes) is in [`transient_impact.py`](./transient_impact.py).
 
### Simulated Output: Low vs. High Resilience
 
![Transient impact simulation: price path and impact level for low vs. high resilience](./assets/transient_impact.png)
 
*Left: simulated price path under low (ρ = 0.5) and high (ρ = 5.0) resilience. Right: the corresponding transient impact level `I(t)` driving the overlay on top of the (unaffected) fundamental price. Constant trading rate, `T = 1`, `N = 100` slices, 100,000 shares, `σ_daily = 0.01`.*
 
Two distinct regimes emerge, and they correspond to the two limiting cases already covered in Section 6:
 
- **Low resilience (ρ = 0.5, blue curve).** Impact decays slowly relative to the rate of new trades arriving, so each fresh kick (`η · size_t`) lands on top of an impact level that has barely decayed since the previous slice. The impact level accumulates almost linearly over the full horizon, reaching roughly 40 in displacement by `t = 1`. This is visually indistinguishable from the **permanent impact** assumption in Almgren-Chriss (`ΔS_perm = γ·n`): a slowly-resilient book behaves, over the execution horizon, as if every share traded leaves a permanent mark on the price.
- **High resilience (ρ = 5.0, orange curve).** Impact decays fast enough that within a few slices the loss from decay roughly balances the gain from each new trade, so the impact level converges to a steady-state plateau (around 10 in this simulation) instead of growing without bound. This is the **temporary impact** limit (`ΔS_temp = η·v`): the displacement saturates at a level proportional to the trading rate itself, exactly as the constant-rate temporary-impact term in the linear model would predict.
**This is the central insight the transient model adds beyond Almgren-Chriss:** permanent and temporary impact aren't two separate physical phenomena that get added together — they're the two limiting behaviors of a single decaying-impact process, as the resilience parameter `ρ → 0` and `ρ → ∞` respectively. Real order books sit somewhere in between, and `ρ` is exactly the parameter that interpolates between them.
 
**Key paper:** Gatheral, Schied & Slynko (2012), *"Transient Linear Price Impact and Fredholm Integral Equations."*
 
---
 
## 10. Limitations & Future Work
 
- The linear impact model (η, γ constant) is a simplification — real impact coefficients vary with volatility regime, liquidity conditions, and order-book depth.
- Almgren-Chriss assumes continuous, deterministic trajectories; real execution is discrete and subject to fills, queue position, and adverse selection.
- The transient impact model here uses a single exponential kernel; Gatheral-Schied's general kernel `G(t)` opens the door to power-law decay, which better matches some empirical microstructure studies.
- Natural next step: recast the transient-impact case as its own optimal-control problem (rather than just simulating a fixed-rate schedule) and compare the resulting trajectory to the classical Almgren-Chriss solution.
---
 
## 11. References
 
- Almgren, R., & Chriss, N. (2000). *Optimal Execution of Portfolio Transactions.* Journal of Risk, 3, 5-40.
- Obizhaeva, A., & Wang, J. (2013). *Optimal Trading Strategy and Supply/Demand Dynamics.* Journal of Financial Markets, 16(1), 1-32.
- Gatheral, J., & Schied, A. (2011). *Optimal Trade Execution under Geometric Brownian Motion in the Almgren and Chriss Framework.* International Journal of Theoretical and Applied Finance, 14(3), 353-368.
- Gatheral, J., Schied, A., & Slynko, A. (2012). *Transient Linear Price Impact and Fredholm Integral Equations.* Mathematical Finance, 22(3), 445-474.
---
 
## Repository Structure
 
```
.
├── README.md                  # this document
├── execution_metrics.py       # IS, square-root impact, VWAP
├── impact_models.py           # linear temporary/permanent impact, sliced cost
├── almgren_chriss.py          # closed-form optimal trajectory + urgency parameter
├── case_study.py              # full pre-trade / execution / post-trade walkthrough
└── transient_impact.py        # Obizhaeva-Wang exponential-decay extension
```
 
---
