"""
Filename: MEIM_nn_policy.py
Description: Two-echelon inventory management with neural network policy trained via REINFORCE.
             Compares NN policy against learned base-stock policy.
Course: ISyE 8803 - Learning and Optimization in Operations
Contributors: Jinghua Weng, Yitong Wu
Created: 2025-4-14, Jinghua Weng
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Tuple
import matplotlib.pyplot as plt


T = 50
gamma = 0.97


class TwoEchelonInv:
    """Two-echelon inventory environment with stochastic Poisson demand."""

    def __init__(self,
                 h1=2.0, p1=5.0, h2=1.0,
                 demand_lambda=5,
                 init_x1=10.0, init_w1=0.0, init_x2=20.0, init_w2=0.0,
                 K=0.0, c=1.0, c1=1.0):
        self.h1 = h1
        self.p1 = p1
        self.h2 = h2
        self.demand_lambda = demand_lambda
        self.init_x1 = init_x1
        self.init_w1 = init_w1
        self.init_x2 = init_x2
        self.init_w2 = init_w2
        self.K = K
        self.c = c
        self.c1 = c1
        self.reset()

    def reset(self) -> np.ndarray:
        self.x1 = self.init_x1
        self.w1 = self.init_w1
        self.x2 = self.init_x2
        self.w2 = self.init_w2
        self.t = 0
        return np.array([self.x1, self.w1, self.x2, self.w2], dtype=np.float32)

    def shortage_storage_cost(self, x: float) -> float:
        return self.h1 * x if x >= 0 else self.p1 * (-x)

    def next_step(self, a1: float, a2: float) -> Tuple[np.ndarray, float]:
        demand = np.random.poisson(self.demand_lambda)

        x2_local = self.x2 - self.w1 - self.x1 + self.w2

        self.x1 = self.x1 + self.w1 - demand
        self.w1 = min(a1, x2_local)
        cost_w1 = self.c1 * self.w1
        cost_x1 = self.shortage_storage_cost(self.x1)

        x2_local = self.x2 - self.w1 - self.x1
        self.x2 = self.x2 + self.w2 - demand
        x2_local = x2_local - self.w1
        self.w2 = a2
        cost_w2 = self.c1 * self.w2
        cost_x2 = max(self.h2 * x2_local, 0)

        total_period_cost = cost_x1 + cost_x2 + cost_w1 + cost_w2
        next_state = np.array([self.x1, self.w1, self.x2, self.w2], dtype=np.float32)
        return next_state, total_period_cost


# ---------------------------------------------------------------------------
# Neural network policy: state -> action mean
# ---------------------------------------------------------------------------
class PolicyNetwork(nn.Module):
    """MLP that maps inventory state to replenishment action means."""

    def __init__(self, state_dim=4, action_dim=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softplus(),  # ensure non-negative order quantities
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state: torch.Tensor):
        """Return action means and std."""
        mu = self.net(state)
        std = self.log_std.exp().clamp(min=1e-3)
        return mu, std


# ---------------------------------------------------------------------------
# REINFORCE training
# ---------------------------------------------------------------------------
def train_nn_policy(num_episodes=5000, batch_size=16, lr=3e-4, print_every=200):
    env_kwargs = dict(h1=2.0, p1=5.0, h2=1.0, demand_lambda=5,
                      init_x1=10.0, init_w1=0.0, init_x2=20.0, init_w2=0.0,
                      K=0.0, c=1.0, c1=1.0)

    policy = PolicyNetwork()
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    cost_history = []

    for epi in range(1, num_episodes + 1):
        batch_loss = []
        batch_cost = []

        for _ in range(batch_size):
            env = TwoEchelonInv(**env_kwargs)
            state = env.reset()
            log_probs = []
            rewards = []

            for t in range(T):
                s = torch.tensor(state, dtype=torch.float32)
                mu, std = policy(s)
                dist = Normal(mu, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum()

                a1 = max(action[0].item(), 0.0)
                a2 = max(action[1].item(), 0.0)
                state, cost = env.next_step(a1, a2)

                log_probs.append(log_prob)
                rewards.append(-cost)

            # compute discounted returns
            G = 0.0
            returns = []
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)

            returns = torch.tensor(returns, dtype=torch.float32)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            loss = torch.stack([(-lp * Gt) for lp, Gt in zip(log_probs, returns)]).sum()
            batch_loss.append(loss)
            batch_cost.append(sum(rewards))

        policy_loss = torch.stack(batch_loss).mean()
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        avg_cost = np.mean(batch_cost)
        cost_history.append(avg_cost)

        if epi % print_every == 0:
            std_vals = policy.log_std.exp().detach().numpy()
            print(f"[Episode {epi:5d}] avg_reward={avg_cost:.1f}  "
                  f"std=[{std_vals[0]:.3f}, {std_vals[1]:.3f}]")

    return policy, cost_history


# ---------------------------------------------------------------------------
# Base-stock baseline for comparison
# ---------------------------------------------------------------------------
def train_basestock(num_episodes=5000, batch_size=16, lr=3e-3, print_every=200):
    env_kwargs = dict(h1=2.0, p1=5.0, h2=1.0, demand_lambda=5,
                      init_x1=10.0, init_w1=0.0, init_x2=20.0, init_w2=0.0,
                      K=0.0, c=1.0, c1=1.0)

    theta1 = torch.tensor(15.0, requires_grad=True)
    theta2 = torch.tensor(25.0, requires_grad=True)
    optimizer = optim.Adam([theta1, theta2], lr=lr)
    sigma = torch.tensor(0.6)

    cost_history = []
    theta_history = {'θ1': [], 'θ2': []}

    for epi in range(1, num_episodes + 1):
        batch_loss = []
        batch_cost = []

        for _ in range(batch_size):
            env = TwoEchelonInv(**env_kwargs)
            state = env.reset()
            log_probs = []
            rewards = []

            for t in range(T):
                x1, w1, x2, w2 = [torch.tensor(s, dtype=torch.float32) for s in state]

                mu1 = torch.relu(theta1 - x1 - w1)
                mu2 = torch.relu(theta2 - x2 - w2)

                dist1 = Normal(mu1, sigma)
                dist2 = Normal(mu2, sigma)
                a1 = dist1.sample()
                a2 = dist2.sample()

                log_probs.append(dist1.log_prob(a1) + dist2.log_prob(a2))

                state, cost = env.next_step(a1.item(), a2.item())
                rewards.append(-cost)

            G = 0.0
            returns = []
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns, dtype=torch.float32)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            loss = torch.stack([(-lp * Gt) for lp, Gt in zip(log_probs, returns)]).sum()
            batch_loss.append(loss)
            batch_cost.append(sum(rewards))

        policy_loss = torch.stack(batch_loss).mean()
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        avg_cost = np.mean(batch_cost)
        cost_history.append(avg_cost)
        theta_history['θ1'].append(theta1.item())
        theta_history['θ2'].append(theta2.item())

        if epi % print_every == 0:
            print(f"[Episode {epi:5d}] avg_reward={avg_cost:.1f}  "
                  f"θ1={theta1.item():.2f}  θ2={theta2.item():.2f}")

    print(f"Base-stock converged: θ1={theta1.item():.2f}, θ2={theta2.item():.2f}")
    return theta1, theta2, cost_history, theta_history


# ---------------------------------------------------------------------------
# Evaluation: run a trained policy for N episodes, return average total cost
# ---------------------------------------------------------------------------
def evaluate(policy, n_eval=500):
    env_kwargs = dict(h1=2.0, p1=5.0, h2=1.0, demand_lambda=5,
                      init_x1=10.0, init_w1=0.0, init_x2=20.0, init_w2=0.0,
                      K=0.0, c=1.0, c1=1.0)
    total_rewards = []
    for _ in range(n_eval):
        env = TwoEchelonInv(**env_kwargs)
        state = env.reset()
        ep_reward = 0.0
        for t in range(T):
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32)
                mu, _ = policy(s)
                a1 = max(mu[0].item(), 0.0)
                a2 = max(mu[1].item(), 0.0)
            state, cost = env.next_step(a1, a2)
            ep_reward += -cost
        total_rewards.append(ep_reward)
    return np.mean(total_rewards), np.std(total_rewards)


def evaluate_basestock(theta1_val, theta2_val, n_eval=500):
    env_kwargs = dict(h1=2.0, p1=5.0, h2=1.0, demand_lambda=5,
                      init_x1=10.0, init_w1=0.0, init_x2=20.0, init_w2=0.0,
                      K=0.0, c=1.0, c1=1.0)
    total_rewards = []
    for _ in range(n_eval):
        env = TwoEchelonInv(**env_kwargs)
        state = env.reset()
        ep_reward = 0.0
        for t in range(T):
            x1, w1, x2, w2 = state
            a1 = max(theta1_val - x1 - w1, 0.0)
            a2 = max(theta2_val - x2 - w2, 0.0)
            state, cost = env.next_step(a1, a2)
            ep_reward += -cost
        total_rewards.append(ep_reward)
    return np.mean(total_rewards), np.std(total_rewards)


# ---------------------------------------------------------------------------
# Main: train both, compare, and plot
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Training base-stock policy (scalar θ1, θ2) ...")
    print("=" * 60)
    theta1, theta2, bs_cost_hist, theta_hist = train_basestock(
        num_episodes=5000, batch_size=16)

    print()
    print("=" * 60)
    print("Training neural network policy ...")
    print("=" * 60)
    nn_policy, nn_cost_hist = train_nn_policy(num_episodes=5000, batch_size=16)

    # Evaluate both
    print()
    print("=" * 60)
    print("Evaluation (500 episodes, deterministic actions)")
    print("=" * 60)
    bs_mean, bs_std = evaluate_basestock(theta1.item(), theta2.item())
    nn_mean, nn_std = evaluate(nn_policy)
    print(f"Base-stock:  avg_reward = {bs_mean:.1f} ± {bs_std:.1f}")
    print(f"NN policy:   avg_reward = {nn_mean:.1f} ± {nn_std:.1f}")

    # Smoothed learning curves
    def smooth(arr, window=100):
        kernel = np.ones(window) / window
        return np.convolve(arr, kernel, mode='valid')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: learning curves comparison
    axes[0].plot(smooth(bs_cost_hist), label='Base-stock policy', alpha=0.8)
    axes[0].plot(smooth(nn_cost_hist), label='NN policy', alpha=0.8)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Episode Reward (smoothed)')
    axes[0].set_title('Learning Curves: Base-Stock vs NN Policy')
    axes[0].legend()
    axes[0].grid(True)

    # Plot 2: base-stock theta convergence
    axes[1].plot(theta_hist['θ1'], label='θ₁', alpha=0.8)
    axes[1].plot(theta_hist['θ2'], label='θ₂', alpha=0.8)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Base-Stock Level')
    axes[1].set_title('Base-Stock Parameter Convergence')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=150)
    plt.show()
    print("Plot saved to learning_curves.png")
