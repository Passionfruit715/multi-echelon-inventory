"""
Filename: MEIM_three_echelon.py
Description: Three-echelon serial inventory system with lead times,
             trained via REINFORCE. Compares base-stock vs NN policy.
             Vectorized environment for fast batch training.
Course: ISyE 8803 - Learning and Optimization in Operations
Contributors: Jinghua Weng, Yitong Wu
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Tuple
import matplotlib.pyplot as plt
import time


T = 50
gamma = 0.97


# ---------------------------------------------------------------------------
# Vectorized environment: runs N envs in parallel using numpy arrays
# ---------------------------------------------------------------------------
class ThreeEchelonInvVec:
    """
    Vectorized three-echelon serial inventory system.

    Instead of running 1 environment at a time, this runs `n_envs`
    environments simultaneously using numpy array operations.

    All state variables are arrays of shape (n_envs,).
    Pipelines are arrays of shape (n_envs, L_i).

    Structure: Retailer (E1) <-- Warehouse (E2) <-- Supplier (E3) <-- External
    """

    def __init__(self, n_envs,
                 h1=2.0, p1=5.0, h2=1.0, h3=0.5,
                 demand_lambda=5,
                 lead_times=(2, 2, 1),
                 init_x=(10.0, 15.0, 20.0),
                 c1=1.0, c2=1.0, c3=1.0):
        self.n_envs = n_envs
        self.h1 = h1
        self.p1 = p1
        self.h2 = h2
        self.h3 = h3
        self.demand_lambda = demand_lambda
        self.lead_times = lead_times
        self.init_x = init_x
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.L1, self.L2, self.L3 = lead_times
        self.state_dim = 3 + sum(lead_times)
        self.reset()

    def reset(self) -> np.ndarray:
        n = self.n_envs
        self.x1 = np.full(n, self.init_x[0], dtype=np.float32)
        self.x2 = np.full(n, self.init_x[1], dtype=np.float32)
        self.x3 = np.full(n, self.init_x[2], dtype=np.float32)
        self.pipeline1 = np.zeros((n, self.L1), dtype=np.float32)
        self.pipeline2 = np.zeros((n, self.L2), dtype=np.float32)
        self.pipeline3 = np.zeros((n, self.L3), dtype=np.float32)
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Return state array of shape (n_envs, state_dim)."""
        return np.column_stack([
            self.x1, self.pipeline1,
            self.x2, self.pipeline2,
            self.x3, self.pipeline3,
        ])

    def next_step(self, a1: np.ndarray, a2: np.ndarray, a3: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Step all n_envs environments simultaneously.

        Args:
            a1, a2, a3: arrays of shape (n_envs,)
        Returns:
            states: (n_envs, state_dim)
            costs:  (n_envs,)
        """
        # --- Demand: sample n_envs Poisson values at once ---
        demand = np.random.poisson(self.demand_lambda, size=self.n_envs).astype(np.float32)

        # --- Receive arriving shipments (first column of each pipeline) ---
        arriving1 = self.pipeline1[:, 0] if self.L1 > 0 else np.zeros(self.n_envs, dtype=np.float32)
        arriving2 = self.pipeline2[:, 0] if self.L2 > 0 else np.zeros(self.n_envs, dtype=np.float32)
        arriving3 = self.pipeline3[:, 0] if self.L3 > 0 else np.zeros(self.n_envs, dtype=np.float32)

        self.x1 += arriving1
        self.x2 += arriving2
        self.x3 += arriving3

        # --- Fulfill demand at retailer ---
        self.x1 -= demand
        cost_x1 = np.where(self.x1 >= 0, self.h1 * self.x1, self.p1 * (-self.x1))

        # --- Ship from E2 to E1 ---
        a1 = np.maximum(a1, 0.0)
        ship_to_1 = np.minimum(a1, np.maximum(self.x2, 0.0))
        self.x2 -= ship_to_1
        cost_ship1 = self.c1 * ship_to_1

        # --- Ship from E3 to E2 ---
        a2 = np.maximum(a2, 0.0)
        ship_to_2 = np.minimum(a2, np.maximum(self.x3, 0.0))
        self.x3 -= ship_to_2
        cost_ship2 = self.c2 * ship_to_2

        # --- Ship from external to E3 (unlimited) ---
        a3 = np.maximum(a3, 0.0)
        ship_to_3 = a3
        cost_ship3 = self.c3 * ship_to_3

        # --- Holding costs (after shipping) ---
        cost_x2 = self.h2 * np.maximum(self.x2, 0.0)
        cost_x3 = self.h3 * np.maximum(self.x3, 0.0)

        # --- Update pipelines: shift left, append new shipment ---
        if self.L1 > 1:
            self.pipeline1[:, :-1] = self.pipeline1[:, 1:]
        if self.L1 > 0:
            self.pipeline1[:, -1] = ship_to_1

        if self.L2 > 1:
            self.pipeline2[:, :-1] = self.pipeline2[:, 1:]
        if self.L2 > 0:
            self.pipeline2[:, -1] = ship_to_2

        if self.L3 > 1:
            self.pipeline3[:, :-1] = self.pipeline3[:, 1:]
        if self.L3 > 0:
            self.pipeline3[:, -1] = ship_to_3

        total_cost = cost_x1 + cost_x2 + cost_x3 + cost_ship1 + cost_ship2 + cost_ship3
        return self._get_state(), total_cost


# ---------------------------------------------------------------------------
# Neural network policy
# ---------------------------------------------------------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim=3, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softplus(),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state: torch.Tensor):
        mu = self.net(state)
        std = self.log_std.exp().clamp(min=1e-3)
        return mu, std


# ---------------------------------------------------------------------------
# REINFORCE training - NN policy (vectorized)
# ---------------------------------------------------------------------------
def train_nn_policy(env_kwargs, num_episodes=5000, batch_size=16,
                    lr=3e-4, print_every=200):
    state_dim = 3 + sum(env_kwargs.get('lead_times', (2, 2, 1)))
    policy = PolicyNetwork(state_dim=state_dim, action_dim=3)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    cost_history = []

    for epi in range(1, num_episodes + 1):
        # all batch envs run in parallel
        env = ThreeEchelonInvVec(n_envs=batch_size, **env_kwargs)
        states = env.reset()  # (batch_size, state_dim)

        all_log_probs = []   # list of T tensors, each (batch_size,)
        all_rewards = []     # list of T arrays, each (batch_size,)

        for t in range(T):
            s = torch.tensor(states, dtype=torch.float32)       # (batch_size, state_dim)
            mu, std = policy(s)                                  # (batch_size, 3)
            dist = Normal(mu, std)
            actions = dist.sample()                              # (batch_size, 3)
            log_prob = dist.log_prob(actions).sum(dim=-1)        # (batch_size,)

            a_np = actions.detach().numpy()
            states, costs = env.next_step(a_np[:, 0], a_np[:, 1], a_np[:, 2])

            all_log_probs.append(log_prob)
            all_rewards.append(-costs)

        # compute returns for all envs at once
        # all_rewards: list of T arrays, each (batch_size,)
        returns_batch = np.zeros((T, batch_size), dtype=np.float32)
        G = np.zeros(batch_size, dtype=np.float32)
        for t in reversed(range(T)):
            G = all_rewards[t] + gamma * G
            returns_batch[t] = G

        returns_t = torch.tensor(returns_batch, dtype=torch.float32)    # (T, batch_size)
        returns_t = (returns_t - returns_t.mean(dim=0)) / (returns_t.std(dim=0) + 1e-8)

        # policy loss
        log_probs_t = torch.stack(all_log_probs)    # (T, batch_size)
        loss = (-log_probs_t * returns_t).sum(dim=0).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_reward = sum(all_rewards).mean()
        cost_history.append(avg_reward)

        if epi % print_every == 0:
            std_vals = policy.log_std.exp().detach().numpy()
            print(f"[NN  Episode {epi:5d}] avg_reward={avg_reward:.1f}  "
                  f"std={np.round(std_vals, 3)}")

    return policy, cost_history


# ---------------------------------------------------------------------------
# REINFORCE training - base-stock policy (vectorized)
# ---------------------------------------------------------------------------
def train_basestock(env_kwargs, num_episodes=5000, batch_size=16,
                    lr=3e-3, print_every=200):
    lead_times = env_kwargs.get('lead_times', (2, 2, 1))
    L1, L2, L3 = lead_times

    theta1 = torch.tensor(15.0, requires_grad=True)
    theta2 = torch.tensor(25.0, requires_grad=True)
    theta3 = torch.tensor(35.0, requires_grad=True)
    optimizer = optim.Adam([theta1, theta2, theta3], lr=lr)
    sigma = torch.tensor(0.6)

    cost_history = []
    theta_history = {'θ1': [], 'θ2': [], 'θ3': []}

    for epi in range(1, num_episodes + 1):
        env = ThreeEchelonInvVec(n_envs=batch_size, **env_kwargs)
        states = env.reset()   # (batch_size, state_dim)

        all_log_probs = []
        all_rewards = []

        for t in range(T):
            s = torch.tensor(states, dtype=torch.float32)  # (batch_size, state_dim)

            # parse state columns
            idx = 0
            x1 = s[:, idx]; idx += 1
            p1 = s[:, idx:idx+L1]; idx += L1
            x2 = s[:, idx]; idx += 1
            p2 = s[:, idx:idx+L2]; idx += L2
            x3 = s[:, idx]; idx += 1
            p3 = s[:, idx:idx+L3]; idx += L3

            inv_pos1 = x1 + p1.sum(dim=1)
            inv_pos2 = x2 + p2.sum(dim=1)
            inv_pos3 = x3 + p3.sum(dim=1)

            mu1 = torch.relu(theta1 - inv_pos1)   # (batch_size,)
            mu2 = torch.relu(theta2 - inv_pos2)
            mu3 = torch.relu(theta3 - inv_pos3)

            dist1 = Normal(mu1, sigma)
            dist2 = Normal(mu2, sigma)
            dist3 = Normal(mu3, sigma)
            a1 = dist1.sample()
            a2 = dist2.sample()
            a3 = dist3.sample()

            log_prob = dist1.log_prob(a1) + dist2.log_prob(a2) + dist3.log_prob(a3)  # (batch_size,)
            all_log_probs.append(log_prob)

            states, costs = env.next_step(
                a1.detach().numpy(), a2.detach().numpy(), a3.detach().numpy())
            all_rewards.append(-costs)

        # compute returns
        returns_batch = np.zeros((T, batch_size), dtype=np.float32)
        G = np.zeros(batch_size, dtype=np.float32)
        for t in reversed(range(T)):
            G = all_rewards[t] + gamma * G
            returns_batch[t] = G

        returns_t = torch.tensor(returns_batch, dtype=torch.float32)
        returns_t = (returns_t - returns_t.mean(dim=0)) / (returns_t.std(dim=0) + 1e-8)

        log_probs_t = torch.stack(all_log_probs)
        loss = (-log_probs_t * returns_t).sum(dim=0).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_reward = sum(all_rewards).mean()
        cost_history.append(avg_reward)
        theta_history['θ1'].append(theta1.item())
        theta_history['θ2'].append(theta2.item())
        theta_history['θ3'].append(theta3.item())

        if epi % print_every == 0:
            print(f"[BS  Episode {epi:5d}] avg_reward={avg_reward:.1f}  "
                  f"θ1={theta1.item():.2f}  θ2={theta2.item():.2f}  θ3={theta3.item():.2f}")

    print(f"Base-stock converged: θ1={theta1.item():.2f}, θ2={theta2.item():.2f}, θ3={theta3.item():.2f}")
    return [theta1, theta2, theta3], cost_history, theta_history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_nn(policy, env_kwargs, n_eval=500):
    env = ThreeEchelonInvVec(n_envs=n_eval, **env_kwargs)
    states = env.reset()
    total_rewards = np.zeros(n_eval, dtype=np.float32)
    for t in range(T):
        with torch.no_grad():
            s = torch.tensor(states, dtype=torch.float32)
            mu, _ = policy(s)
            a_np = mu.numpy().clip(min=0.0)
        states, costs = env.next_step(a_np[:, 0], a_np[:, 1], a_np[:, 2])
        total_rewards += -costs
    return total_rewards.mean(), total_rewards.std()


def evaluate_basestock(thetas, env_kwargs, n_eval=500):
    lead_times = env_kwargs.get('lead_times', (2, 2, 1))
    L1, L2, L3 = lead_times
    t1, t2, t3 = [th.item() for th in thetas]

    env = ThreeEchelonInvVec(n_envs=n_eval, **env_kwargs)
    states = env.reset()
    total_rewards = np.zeros(n_eval, dtype=np.float32)
    for t in range(T):
        idx = 0
        x1 = states[:, idx]; idx += 1
        p1 = states[:, idx:idx+L1]; idx += L1
        x2 = states[:, idx]; idx += 1
        p2 = states[:, idx:idx+L2]; idx += L2
        x3 = states[:, idx]; idx += 1
        p3 = states[:, idx:idx+L3]; idx += L3

        inv_pos1 = x1 + p1.sum(axis=1)
        inv_pos2 = x2 + p2.sum(axis=1)
        inv_pos3 = x3 + p3.sum(axis=1)

        a1 = np.maximum(t1 - inv_pos1, 0.0)
        a2 = np.maximum(t2 - inv_pos2, 0.0)
        a3 = np.maximum(t3 - inv_pos3, 0.0)

        states, costs = env.next_step(a1, a2, a3)
        total_rewards += -costs
    return total_rewards.mean(), total_rewards.std()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    env_kwargs = dict(
        h1=2.0, p1=5.0, h2=1.0, h3=0.5,
        demand_lambda=5,
        lead_times=(2, 2, 1),
        init_x=(10.0, 15.0, 20.0),
        c1=1.0, c2=1.0, c3=1.0,
    )

    print("=" * 60)
    print("Three-echelon system (vectorized): lead_times=(2, 2, 1)")
    print(f"State dimension: {3 + sum(env_kwargs['lead_times'])}")
    print("=" * 60)

    t0 = time.time()

    print()
    print("Training base-stock policy ...")
    print("-" * 60)
    thetas, bs_cost_hist, theta_hist = train_basestock(
        env_kwargs, num_episodes=5000, batch_size=16)

    t1 = time.time()
    print(f"Base-stock training time: {t1 - t0:.1f}s")

    print()
    print("Training neural network policy ...")
    print("-" * 60)
    nn_policy, nn_cost_hist = train_nn_policy(
        env_kwargs, num_episodes=5000, batch_size=16)

    t2 = time.time()
    print(f"NN training time: {t2 - t1:.1f}s")

    print()
    print("=" * 60)
    print("Evaluation (500 episodes)")
    print("=" * 60)
    bs_mean, bs_std = evaluate_basestock(thetas, env_kwargs)
    nn_mean, nn_std = evaluate_nn(nn_policy, env_kwargs)
    print(f"Base-stock:  avg_reward = {bs_mean:.1f} +/- {bs_std:.1f}")
    print(f"NN policy:   avg_reward = {nn_mean:.1f} +/- {nn_std:.1f}")

    # Plots
    def smooth(arr, window=100):
        kernel = np.ones(window) / window
        return np.convolve(arr, kernel, mode='valid')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(smooth(bs_cost_hist), label='Base-stock', alpha=0.8)
    axes[0].plot(smooth(nn_cost_hist), label='NN policy', alpha=0.8)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Episode Reward (smoothed)')
    axes[0].set_title('3-Echelon: Base-Stock vs NN Policy')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(theta_hist['θ1'], label='θ₁ (retailer)', alpha=0.8)
    axes[1].plot(theta_hist['θ2'], label='θ₂ (warehouse)', alpha=0.8)
    axes[1].plot(theta_hist['θ3'], label='θ₃ (supplier)', alpha=0.8)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Base-Stock Level')
    axes[1].set_title('Base-Stock Parameter Convergence')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('learning_curves_3echelon.png', dpi=150)
    plt.show()
    print(f"\nTotal time: {time.time() - t0:.1f}s")
    print("Plot saved to learning_curves_3echelon.png")
