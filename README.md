# Multi-Echelon Inventory Optimization with Policy Gradient Methods

A reinforcement learning approach to multi-echelon inventory control, comparing REINFORCE and Actor-Critic algorithms against classical base-stock policies.

**Course**: ISyE 8803 - Learning and Optimization in Operations (Georgia Tech)   
**Date**: January - April 2025

## Problem

Multi-echelon inventory systems involve multiple stocking locations in a supply chain (e.g., supplier - warehouse - retailer). The goal is to find optimal replenishment policies that minimize total costs including holding, shortage, and transportation costs, under stochastic demand.

### System Structure

```
External Supply --[L3]--> E3 (Supplier) --[L2]--> E2 (Warehouse) --[L1]--> E1 (Retailer) --> Demand (Poisson)
```

- **State**: On-hand inventory + in-transit pipeline at each echelon
- **Action**: Order quantities at each echelon per period
- **Reward**: Negative total cost (holding + shortage + transportation)
- **Horizon**: T = 50 periods, discount factor gamma = 0.97

## Experiments and Results

### Experiment 1: Base-Stock Convergence (2-Echelon, REINFORCE)
**File**: `MEIM_nn_policy.py`

Validates that REINFORCE can learn optimal base-stock levels (theta1, theta2) in a simple 2-echelon system with K=0 and linear costs.

| Policy | Avg Reward |
|--------|-----------|
| Base-stock (REINFORCE) | -1,112 |
| NN (REINFORCE) | -1,278 |
| Optimal base-stock (grid search) | -945 |

**Finding**: Both converge, but REINFORCE-trained theta values are suboptimal compared to grid search.

### Experiment 2: Three-Echelon + Lead Time (REINFORCE vs AC)
**File**: `MEIM_three_echelon.py` (REINFORCE), `MEIM_ac_three_echelon.py` (Actor-Critic)

Tests whether NN policies can scale to 8-dimensional state space with lead times (2, 2, 1).

| Policy | Avg Reward |
|--------|-----------|
| Base-stock (REINFORCE) | -2,368 |
| NN (REINFORCE) | -14,508 (failed) |
| NN (Actor-Critic) | -1,662 |
| Optimal base-stock (grid search) | -1,495 |

**Finding**: REINFORCE NN completely fails in the 8-dim state space. Actor-Critic successfully converges, approaching the optimal base-stock level. The credit assignment problem across lead times makes REINFORCE's high-variance gradients insufficient.

### Experiment 3: Fixed Ordering Cost K > 0 (AC discovers (s,S) policy)
**File**: `MEIM_ac_fixed_cost.py`

With fixed ordering cost K=50, base-stock (order every period) is no longer optimal. The (s,S) policy - order up to S only when inventory drops below s - avoids paying K every period.

| Policy | Avg Reward |
|--------|-----------|
| Base-stock (every period pays K) | -6,091 |
| AC (NN, no gate) | **-3,306** |

**Finding**: AC's NN policy autonomously discovers (s,S)-like behavior: it orders 0 in 50% of periods (when inventory is sufficient) and places large orders in the remaining periods. This reduces cost by **46%** compared to base-stock.

Ordering pattern analysis:
- 25/50 periods: no order (order = 0)
- 25/50 periods: large orders (avg = 23.1 units)

### Experiment 4: REINFORCE vs Actor-Critic (2-Echelon)
**File**: `MEIM_actor_critic.py`

Direct comparison of REINFORCE and Actor-Critic on the same 2-echelon environment.

| Policy | Avg Reward |
|--------|-----------|
| Optimal base-stock | -945 |
| REINFORCE (NN) | Unstable (diverged in some runs) |
| Actor-Critic (NN) | -1,036 |

**Finding**: REINFORCE is unstable across runs; Actor-Critic provides more reliable convergence due to the learned value baseline reducing gradient variance.

### Experiment 5: Nonlinear Cost Functions
**File**: `MEIM_nonlinear_cost.py`

Quadratic holding cost (h * x^2) and piecewise shortage cost to test whether NN can exploit nonlinear structure.

| Policy | Avg Reward |
|--------|-----------|
| Best base-stock (grid search) | -2,059 |
| Actor-Critic (NN) | -2,118 |

**Finding**: AC matches but does not significantly surpass base-stock. Quadratic holding costs alone don't sufficiently break base-stock's near-optimality, as base-stock can still control the mean inventory level effectively.

## Key Takeaways

1. **Base-stock is hard to beat when it's optimal** (K=0, linear costs, backlog): NN policies converge to similar performance but cannot surpass it (Clark-Scarf theorem).

2. **Actor-Critic >> REINFORCE for complex environments**: In 8-dim state space with lead times, REINFORCE fails entirely while AC converges. The value baseline dramatically reduces gradient variance.

3. **NN shines when base-stock is suboptimal**: With fixed ordering costs (K>0), the NN policy discovers (s,S)-like behavior autonomously, reducing costs by 46%.

4. **Credit assignment across lead times**: Upstream parameters (supplier) are hardest to learn because their actions take multiple periods to affect downstream costs.

## Algorithm Details

### REINFORCE (Monte Carlo Policy Gradient)
```
gradient = E[sum_t(grad(log pi(a_t|s_t)) * G_t)]
baseline: reward normalization (mean/std of returns)
```

### Actor-Critic
```
Actor:  policy network, updated with advantage A_t = G_t^GAE - V(s_t)
Critic: value network V(s), trained to predict discounted returns
GAE:    lambda = 0.95, blends TD and MC for advantage estimation
Stabilization: reward normalization, gradient clipping, Smooth L1 loss for critic
```

### Vectorized Environment
All environments support batch execution (n_envs parallel simulations) using numpy array operations, providing 10-15x speedup over sequential Python loops.

## File Structure

```
.
|-- MEIM_nn_policy.py          # Exp 1: 2-echelon, REINFORCE, NN vs base-stock
|-- MEIM_three_echelon.py      # Exp 2: 3-echelon + lead time, REINFORCE (vectorized)
|-- MEIM_ac_three_echelon.py   # Exp 2: 3-echelon + lead time, Actor-Critic
|-- MEIM_actor_critic.py       # Exp 4: REINFORCE vs AC comparison (2-echelon)
|-- MEIM_fixed_cost.py         # Exp 3: Fixed ordering cost K>0, REINFORCE
|-- MEIM_ac_fixed_cost.py      # Exp 3: Fixed ordering cost K>0, Actor-Critic
|-- MEIM_nonlinear_cost.py     # Exp 5: Nonlinear costs, Actor-Critic
|-- experiments/               # Original/early experiment files
|-- results/                   # Plots and figures
`-- README.md
```

## Requirements

```
numpy
torch
matplotlib
```

## Usage

Each experiment file can be run independently:

```bash
python MEIM_nn_policy.py           # ~10 min
python MEIM_ac_three_echelon.py    # ~10 min
python MEIM_ac_fixed_cost.py       # ~10 min
python MEIM_actor_critic.py        # ~10 min
python MEIM_nonlinear_cost.py      # ~12 min
```

Results (plots and metrics) are printed to stdout and saved as PNG files in the working directory.
