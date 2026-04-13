"""
Filename: MEIM_ac_fixed_cost.py
Description: Two-echelon inventory with fixed ordering cost K > 0.
             Actor-Critic with gated NN policy to learn (s, S)-like behavior.
             Compares against base-stock baseline.
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
# Vectorized two-echelon environment with fixed ordering cost
# ---------------------------------------------------------------------------
class TwoEchelonInvVec:
    def __init__(self, n_envs,
                 h1=2.0, p1=5.0, h2=1.0,
                 demand_lambda=5,
                 init_x1=10.0, init_w1=0.0, init_x2=20.0, init_w2=0.0,
                 K1=0.0, K2=0.0, c1=1.0):
        self.n_envs = n_envs
        self.h1 = h1
        self.p1 = p1
        self.h2 = h2
        self.demand_lambda = demand_lambda
        self.init_x1 = init_x1
        self.init_w1 = init_w1
        self.init_x2 = init_x2
        self.init_w2 = init_w2
        self.K1 = K1
        self.K2 = K2
        self.c1 = c1
        self.state_dim = 4
        self.reset()

    def reset(self):
        n = self.n_envs
        self.x1 = np.full(n, self.init_x1, dtype=np.float32)
        self.w1 = np.full(n, self.init_w1, dtype=np.float32)
        self.x2 = np.full(n, self.init_x2, dtype=np.float32)
        self.w2 = np.full(n, self.init_w2, dtype=np.float32)
        return self._get_state()

    def _get_state(self):
        return np.column_stack([self.x1, self.w1, self.x2, self.w2])

    def next_step(self, a1, a2):
        demand = np.random.poisson(self.demand_lambda, size=self.n_envs).astype(np.float32)

        x2_local = self.x2 - self.w1 - self.x1 + self.w2

        self.x1 = self.x1 + self.w1 - demand
        self.w1 = np.minimum(np.maximum(a1, 0.0), np.maximum(x2_local, 0.0))
        cost_x1 = np.where(self.x1 >= 0, self.h1 * self.x1, self.p1 * (-self.x1))

        # fixed cost K1 when ordering > 0
        cost_w1 = np.where(self.w1 > 0, self.K1 + self.c1 * self.w1, 0.0)

        x2_local = self.x2 - self.w1 - self.x1
        self.x2 = self.x2 + self.w2 - demand
        x2_local = x2_local - self.w1
        self.w2 = np.maximum(a2, 0.0)
        cost_w2 = np.where(self.w2 > 0, self.K2 + self.c1 * self.w2, 0.0)
        cost_x2 = np.maximum(self.h2 * x2_local, 0.0)

        total_cost = cost_x1 + cost_x2 + cost_w1 + cost_w2
        return self._get_state(), total_cost


# ---------------------------------------------------------------------------
# Gated policy network (actor)
# ---------------------------------------------------------------------------
class PolicyNetwork(nn.Module):
    """
    Simple policy: outputs unconstrained action mean.
    Negative mean -> high probability of sampling action <= 0 -> no order -> no K.
    Positive mean -> order that amount.
    The environment clips negative actions to 0.
    """
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            # NO activation: output can be negative (= don't order)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        mu = self.net(state)
        std = self.log_std.exp().clamp(min=1e-3)
        return mu, std


# ---------------------------------------------------------------------------
# Value network (critic)
# ---------------------------------------------------------------------------
class ValueNetwork(nn.Module):
    def __init__(self, state_dim=4, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------
def compute_gae(rewards, values, gamma=0.97, lam=0.95):
    T_len, batch_size = rewards.shape
    values_np = values.numpy()
    advantages = np.zeros((T_len, batch_size), dtype=np.float32)
    gae = np.zeros(batch_size, dtype=np.float32)

    for t in reversed(range(T_len)):
        next_value = 0.0 if t == T_len - 1 else values_np[t + 1]
        delta = rewards[t] + gamma * next_value - values_np[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae

    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# Actor-Critic with gated policy
# ---------------------------------------------------------------------------
def train_ac(env_kwargs, num_episodes=10000, batch_size=16,
             lr_actor=1e-4, lr_critic=3e-4, critic_updates=3,
             print_every=500):
    state_dim = 4
    actor = PolicyNetwork(state_dim=state_dim)
    critic = ValueNetwork(state_dim=state_dim)
    optimizer_actor = optim.Adam(actor.parameters(), lr=lr_actor)
    optimizer_critic = optim.Adam(critic.parameters(), lr=lr_critic)
    cost_history = []

    reward_mean = 0.0
    reward_std = 1.0

    for epi in range(1, num_episodes + 1):
        env = TwoEchelonInvVec(n_envs=batch_size, **env_kwargs)
        states = env.reset()

        all_log_probs = []
        all_rewards = []
        all_states = []

        for t in range(T):
            s = torch.tensor(states, dtype=torch.float32)
            all_states.append(s)

            mu, std = actor(s)
            dist = Normal(mu, std)
            actions = dist.sample()
            log_prob = dist.log_prob(actions).sum(dim=-1)

            # clip to 0: negative action = no order = no K cost
            a_np = actions.detach().numpy().clip(min=0.0)
            states, costs = env.next_step(a_np[:, 0], a_np[:, 1])

            all_log_probs.append(log_prob)
            all_rewards.append(-costs)

        states_t = torch.stack(all_states)
        rewards_np = np.array(all_rewards)
        log_probs_t = torch.stack(all_log_probs)

        # normalize rewards
        batch_r_mean = rewards_np.mean()
        batch_r_std = rewards_np.std() + 1e-8
        reward_mean = 0.99 * reward_mean + 0.01 * batch_r_mean
        reward_std = 0.99 * reward_std + 0.01 * batch_r_std
        rewards_normed = (rewards_np - reward_mean) / reward_std

        with torch.no_grad():
            values_detached = critic(states_t).detach()

        advantage, returns_target = compute_gae(rewards_normed, values_detached)
        advantage = (advantage - advantage.mean(dim=0)) / (advantage.std(dim=0) + 1e-8)

        # update actor
        actor_loss = (-log_probs_t * advantage).sum(dim=0).mean()
        optimizer_actor.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
        optimizer_actor.step()

        # update critic
        for _ in range(critic_updates):
            values_pred = critic(states_t)
            critic_loss = nn.functional.smooth_l1_loss(values_pred, returns_target.detach())
            optimizer_critic.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
            optimizer_critic.step()

        avg_reward = rewards_np.sum(axis=0).mean()
        cost_history.append(avg_reward)

        if epi % print_every == 0:
            std_vals = actor.log_std.exp().detach().numpy()
            print(f"[AC Episode {epi:5d}] avg_reward={avg_reward:.1f}  "
                  f"std={np.round(std_vals, 3)}  critic_loss={critic_loss.item():.4f}")

    return actor, cost_history


# ---------------------------------------------------------------------------
# Base-stock training (REINFORCE, for comparison)
# ---------------------------------------------------------------------------
def train_basestock(env_kwargs, num_episodes=5000, batch_size=16,
                    lr=3e-3, print_every=500):
    theta1 = torch.tensor(15.0, requires_grad=True)
    theta2 = torch.tensor(25.0, requires_grad=True)
    optimizer = optim.Adam([theta1, theta2], lr=lr)
    sigma = torch.tensor(0.6)
    cost_history = []

    for epi in range(1, num_episodes + 1):
        env = TwoEchelonInvVec(n_envs=batch_size, **env_kwargs)
        states = env.reset()
        all_log_probs = []
        all_rewards = []

        for t in range(T):
            s = torch.tensor(states, dtype=torch.float32)
            x1, w1, x2, w2 = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
            mu1 = torch.relu(theta1 - x1 - w1)
            mu2 = torch.relu(theta2 - x2 - w2)
            dist1 = Normal(mu1, sigma)
            dist2 = Normal(mu2, sigma)
            a1 = dist1.sample()
            a2 = dist2.sample()
            log_prob = dist1.log_prob(a1) + dist2.log_prob(a2)
            all_log_probs.append(log_prob)
            states, costs = env.next_step(a1.detach().numpy(), a2.detach().numpy())
            all_rewards.append(-costs)

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
        cost_history.append(sum(all_rewards).mean())

        if epi % print_every == 0:
            print(f"[BS Episode {epi:5d}] avg_reward={cost_history[-1]:.1f}  "
                  f"θ1={theta1.item():.2f}  θ2={theta2.item():.2f}")

    return theta1, theta2, cost_history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_basestock(t1, t2, env_kwargs, n_eval=1000):
    env = TwoEchelonInvVec(n_envs=n_eval, **env_kwargs)
    states = env.reset()
    total_rewards = np.zeros(n_eval, dtype=np.float32)
    for t in range(T):
        x1, w1, x2, w2 = states[:, 0], states[:, 1], states[:, 2], states[:, 3]
        a1 = np.maximum(t1 - x1 - w1, 0.0)
        a2 = np.maximum(t2 - x2 - w2, 0.0)
        states, costs = env.next_step(a1, a2)
        total_rewards += -costs
    return total_rewards.mean(), total_rewards.std()


def evaluate_ac(actor, env_kwargs, n_eval=1000):
    env = TwoEchelonInvVec(n_envs=n_eval, **env_kwargs)
    states = env.reset()
    total_rewards = np.zeros(n_eval, dtype=np.float32)
    for t in range(T):
        with torch.no_grad():
            s = torch.tensor(states, dtype=torch.float32)
            mu, _ = actor(s)
            a_np = mu.numpy().clip(min=0.0)
        states, costs = env.next_step(a_np[:, 0], a_np[:, 1])
        total_rewards += -costs
    return total_rewards.mean(), total_rewards.std()


# ---------------------------------------------------------------------------
# Analyze policy behavior
# ---------------------------------------------------------------------------
def analyze_policy(actor, env_kwargs):
    env = TwoEchelonInvVec(n_envs=1, **env_kwargs)
    states = env.reset()
    inv_positions = []
    order_quantities = []

    for t in range(T):
        with torch.no_grad():
            s = torch.tensor(states, dtype=torch.float32)
            mu, _ = actor(s)
            a_np = mu.numpy().clip(min=0.0)

        x1, w1 = states[0, 0], states[0, 1]
        inv_positions.append(x1 + w1)
        order_quantities.append(a_np[0, 0])

        states, costs = env.next_step(a_np[:, 0], a_np[:, 1])

    return inv_positions, order_quantities


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    K_val = 50.0
    env_kwargs = dict(
        h1=2.0, p1=5.0, h2=1.0,
        demand_lambda=5,
        init_x1=10.0, init_w1=0.0, init_x2=20.0, init_w2=0.0,
        K1=K_val, K2=K_val, c1=1.0,
    )

    print("=" * 60)
    print(f"Two-echelon with K={K_val}: AC + Gated NN vs Base-stock")
    print("=" * 60)
    t0 = time.time()

    # Base-stock
    print("\n--- Base-stock ---")
    theta1, theta2, bs_hist = train_basestock(env_kwargs, num_episodes=5000, batch_size=16)
    t1 = time.time()
    print(f"Time: {t1-t0:.1f}s")

    # AC
    print("\n--- Actor-Critic ---")
    ac_actor, ac_hist = train_ac(env_kwargs, num_episodes=10000, batch_size=16)
    t2 = time.time()
    print(f"Time: {t2-t1:.1f}s")

    # Evaluate
    print("\n" + "=" * 60)
    print("Evaluation (1000 episodes)")
    print("=" * 60)
    bs_mean, bs_std = evaluate_basestock(theta1.item(), theta2.item(), env_kwargs)
    ac_mean, ac_std = evaluate_ac(ac_actor, env_kwargs)
    print(f"Base-stock:   avg_reward = {bs_mean:.1f} +/- {bs_std:.1f}")
    print(f"AC (NN):      avg_reward = {ac_mean:.1f} +/- {ac_std:.1f}")

    # Analyze ordering pattern
    inv_pos, orders = analyze_policy(ac_actor, env_kwargs)
    n_orders = sum(1 for o in orders if o > 0.1)
    print(f"\nAC policy: ordered {n_orders}/{T} periods ({100*n_orders/T:.0f}%)")
    print(f"Base-stock: ordered {T}/{T} periods (100%)")

    # Plots
    def smooth(arr, window=200):
        kernel = np.ones(window) / window
        return np.convolve(arr, kernel, mode='valid')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Learning curves
    axes[0].plot(smooth(bs_hist), label='Base-stock', alpha=0.8)
    axes[0].plot(smooth(ac_hist), label='AC + Gated NN', alpha=0.8)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Episode Reward (smoothed)')
    axes[0].set_title(f'K={K_val}: Base-Stock vs Actor-Critic')
    axes[0].legend()
    axes[0].grid(True)

    # Policy behavior
    axes[1].bar(range(T), orders, alpha=0.6, label='Order qty', color='steelblue')
    axes[1].plot(range(T), inv_pos, 'r-o', markersize=3, label='Inv position (E1)', alpha=0.8)
    axes[1].axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    axes[1].set_xlabel('Time step')
    axes[1].set_ylabel('Quantity')
    axes[1].set_title('AC Policy Behavior (E1)')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('ac_fixed_cost.png', dpi=150)
    plt.show()
    print(f"\nTotal time: {time.time()-t0:.1f}s")
    print("Plot saved to ac_fixed_cost.png")
