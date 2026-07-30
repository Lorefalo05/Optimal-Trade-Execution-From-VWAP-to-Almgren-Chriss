import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# SCENARIO: Momentum signal on a mid-cap stock
# ============================================================
print("=" * 60)
print("EXECUTION SCENARIO")
print("=" * 60)
print("Signal: Momentum model predicts +2% move")
print("Alpha half-life: 2 hours")
print("Order: Buy 50,000 shares")
print("Time horizon: 4 hours")
print("Current price: 75.00 dollars")
print()

# Market parameters
Q = 50000           # Shares to buy
T = 4.0             # Hours
N = 8               # 30-minute slices
sigma = 0.015       # 1.5% per hour volatility
gamma = 0.000005    # Permanent impact
eta = 0.00008       # Temporary impact
decision_price = 75.00

# ============================================================
# STEP 1: PRE-TRADE - Choose lambda based on alpha decay
# ============================================================
print("STEP 1: PRE-TRADE ANALYSIS")
print("-" * 40)

alpha_halflife = 2.0  # hours
target_kappa = 1.0 / alpha_halflife
suggested_lambda = (target_kappa ** 2) * eta / (sigma ** 2)

print(f"Alpha half-life: {alpha_halflife} hours")
print(f"Suggested lambda: {suggested_lambda:.3f}")
print(f"Target kappa: {target_kappa:.3f}")

# Use suggested lambda
lambda_risk = suggested_lambda
tau = T / N

# Calculate kappa and trajectory
kappa = np.sqrt(lambda_risk * sigma ** 2 / eta)
t_points = np.arange(N + 1) * tau
trajectory = Q * np.sinh(kappa * (T - t_points)) / np.sinh(kappa * T)
schedule = -np.diff(trajectory)

# Estimate costs
perm_cost = 0.0
cumulative = 0.0
for i in range(len(schedule)):
  perm_cost += gamma * schedule[i] * cumulative
  cumulative += schedule[i]
temp_cost = eta * np.sum(schedule ** 2 / tau)
expected_cost = perm_cost + temp_cost
variance = sigma ** 2 * tau * np.sum(trajectory[1:] ** 2)
std_dev = np.sqrt(variance)

print(f"Expected impact cost: {expected_cost:.2f} dollars")
print(f"Cost std dev: {std_dev:.2f} dollars")
print(f"Expected cost per share: {expected_cost/Q:.4f} dollars")
print()

# ============================================================
# STEP 2: EXECUTION - Simulate with realistic fills
# ============================================================
print("STEP 2: EXECUTION")
print("-" * 40)

# Simulate execution with some randomness
np.random.seed(42)
price_path = [decision_price]
execution_prices = []
execution_shares = []
actual_trajectory = [Q]

current_price = decision_price
for i in range(N):
  # Price evolves with drift (our impact) and noise
  shares_this_slice = schedule[i]

  # Permanent impact from our trading
  current_price += gamma * shares_this_slice

  # Random noise
  current_price += sigma * np.sqrt(tau) * np.random.randn() * current_price / 100

  # Temporary impact for this fill
  fill_price = current_price + eta * (shares_this_slice / tau)

  execution_prices.append(fill_price)
  execution_shares.append(shares_this_slice)
  actual_trajectory.append(actual_trajectory[-1] - shares_this_slice)
  price_path.append(current_price)

execution_prices = np.array(execution_prices)
execution_shares = np.array(execution_shares)

# Print execution log
print(f"{'Slice':<8} {'Shares':<12} {'Price':<12} {'Cumulative':<12}")
print("-" * 44)
cum_shares = 0
for i in range(N):
  cum_shares += execution_shares[i]
  print(f"{i+1:<8} {execution_shares[i]:<12.0f} {execution_prices[i]:<12.4f} {cum_shares:<12.0f}")
print()

# ============================================================
# STEP 3: POST-TRADE ANALYSIS
# ============================================================
print("STEP 3: POST-TRADE ANALYSIS")
print("-" * 40)

# Calculate implementation shortfall
total_shares = np.sum(execution_shares)
vwap = np.sum(execution_prices * execution_shares) / total_shares
is_per_share = vwap - decision_price
is_total = is_per_share * total_shares

print(f"Decision price: {decision_price:.4f} dollars")
print(f"VWAP: {vwap:.4f} dollars")
print(f"Implementation shortfall: {is_per_share:.4f} dollars/share")
print(f"Total IS: {is_total:.2f} dollars")
print(f"IS vs expected: {is_total:.2f} vs {expected_cost:.2f} (expected)")
print()

# Store results for plotting
result = {
    'lambda': lambda_risk,
    'price_path': price_path,
    'execution_prices': execution_prices,
    'pre_trade': {
        'trajectory': trajectory,
        'schedule': schedule,
        'perm_cost': perm_cost,
        'temp_cost': temp_cost,
        'expected_cost': expected_cost,
        'kappa': kappa
    },
    'is_result': {
        'vwap': vwap,
        'is_total': is_total
    }
}

order = {
    'horizon': T,
    'slices': N,
    'shares': Q,
    'decision_price': decision_price
}

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

pre_trade = result['pre_trade']
t_points = np.linspace(0, order['horizon'], order['slices'] + 1)
tau = order['horizon'] / order['slices']
print("\n--- ASSESSMENT ---")
ratio = result['is_result']['is_total'] / pre_trade['expected_cost'] if pre_trade['expected_cost'] > 0 else 0
if abs(ratio - 1) < 0.25:
    assessment = "GOOD: Actual cost within 25% of expected"
elif ratio < 0.75:
    assessment = "EXCELLENT: Beat expectations significantly"
else:
    assessment = "REVIEW: Cost exceeded expectations"
print(f"IS / E[Cost]:     {ratio:.2f}x")
print(f"Assessment:       {assessment}")
print("=" * 70)
# Trajectory
axes[0, 0].plot(t_points, pre_trade['trajectory'], 'b-', linewidth=2.5, label='Optimal Trajectory')
axes[0, 0].fill_between(t_points, 0, pre_trade['trajectory'], alpha=0.2)
axes[0, 0].set_xlabel('Time (hours)')
axes[0, 0].set_ylabel('Shares Remaining')
axes[0, 0].set_title(f'Execution Trajectory (λ={result["lambda"]:.3f}, κ={pre_trade["kappa"]:.3f})', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Schedule
slice_times = t_points[:-1] + tau/2
axes[0, 1].bar(slice_times, pre_trade['schedule'], width=tau*0.8, color='#2ecc71', alpha=0.8, edgecolor='#27ae60')
axes[0, 1].axhline(y=order['shares']/order['slices'], color='gray', linestyle='--', label='Uniform')
axes[0, 1].set_xlabel('Time (hours)')
axes[0, 1].set_ylabel('Shares per Slice')
axes[0, 1].set_title('Execution Schedule', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Price path
axes[1, 0].plot(t_points, result['price_path'], 'k-', linewidth=2, label='Price Path')
axes[1, 0].scatter(slice_times, result['execution_prices'], c='red', s=50, zorder=5, label='Fill Prices')
axes[1, 0].axhline(y=order['decision_price'], color='blue', linestyle='--', label='Decision')
axes[1, 0].axhline(y=result['is_result']['vwap'], color='green', linestyle='--', label='VWAP')
axes[1, 0].set_xlabel('Time (hours)')
axes[1, 0].set_ylabel('Price (dollars)')
axes[1, 0].set_title('Price Evolution During Execution', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Cost breakdown
labels = ['Permanent\nImpact', 'Temporary\nImpact', 'Total\nExpected', 'Actual\nIS']
values = [pre_trade['perm_cost'], pre_trade['temp_cost'], pre_trade['expected_cost'], result['is_result']['is_total']]
colors = ['#e74c3c', '#3498db', '#9b59b6', '#2ecc71']
axes[1, 1].bar(labels, values, color=colors, edgecolor='black', linewidth=1.5)
axes[1, 1].set_ylabel('Cost (dollars)')
axes[1, 1].set_title('Cost Breakdown', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Add value labels
for i, v in enumerate(values):
    axes[1, 1].text(i, v + 50, f'${v:.0f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()
