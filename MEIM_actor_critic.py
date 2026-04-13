"""
Filename: MEIM_actor_critic.py
Description: Two-echelon inventory management comparing REINFORCE vs Actor-Critic.
             Both use NN policy. Actor-Critic adds a value network as baseline
             to reduce gradient variance.
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
# Vectorized two-echelon environment
# ---------------------------------------------------------------------------
class TwoEchelonInvVec:
    """Vectorized two-echelon inventory with Poisson demand."""

    def __init__(self, n_envs,
                 h1=2.0, p1=5.0, h2=1.0,
                 demand_lambda=5,
                 init_x1=10.0, init_w1=0.0, init_x2=20.0, init_w2=0.0,
                 c1=1.0):
        self.n_envs = n_envs
        self.h1 = h1
        self.p1 = p1
        self.h2 = h2
        self.demand_lambda = demand_lambda
        self.init_x1 = init_x1
        self.init_w1 = init_w1
        self.init_x2 = init_x2
        self.init_w2 = init_w2
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
        cost_w1 = self.c1 * self.w1
        cost_x1 = np.where(self.x1 >= 0, self.h1 * self.x1, self.p1 * (-self.x1))

        x2_local = self.x2 - self.w1 - self.x1
        self.x2 = self.x2 + self.w2 - demand
        x2_local = x2_local - self.w1
        self.w2 = np.maximum(a2, 0.0)
        cost_w2 = self.c1 * self.w2
        cost_x2 = np.maximum(self.h2 * x2_local, 0.0)

        total_cost = cost_x1 + cost_x2 + cost_w1 + cost_w2
        return self._get_state(), total_cost


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------
class PolicyNetwork(nn.Module):
    """Actor: state -> action distribution."""

    def __init__(self, state_dim=4, action_dim=2, hidden_dim=64):
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

    def forward(self, state):
        mu = self.net(state)
        std = self.log_std.exp().clamp(min=1e-3)
        return mu, std


class ValueNetwork(nn.Module):
    """Critic: state -> V(s) scalar."""

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
        return self.net(state).squeeze(-1)  # (batch,)


# ---------------------------------------------------------------------------
# REINFORCE (baseline = reward normalization, same as before)
# ---------------------------------------------------------------------------
def train_reinforce(env_kwargs, num_episodes=5000, batch_size=16,
                    lr=3e-4, print_every=200):
    policy = PolicyNetwork()
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    cost_history = []

    for epi in range(1, num_episodes + 1):
        env = TwoEchelonInvVec(n_envs=batch_size, **env_kwargs)
        states = env.reset()
        all_log_probs = []
        all_rewards = []

        for t in range(T):
            s = torch.tensor(states, dtype=torch.float32)
            mu, std = policy(s)
            dist = Normal(mu, std)
            actions = dist.sample()
            log_prob = dist.log_prob(actions).sum(dim=-1)

            a_np = actions.detach().numpy()
            states, costs = env.next_step(a_np[:, 0], a_np[:, 1])

            all_log_probs.append(log_prob)
            all_rewards.append(-costs)

        # discounted returns
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

        if epi % print_every == 0:
            print(f"[REINFORCE Episode {epi:5d}] avg_reward={avg_reward:.1f}")

    return policy, cost_history


# ---------------------------------------------------------------------------
# Actor-Critic (learned value baseline)
# ---------------------------------------------------------------------------
def compute_gae(rewards, values, gamma=0.97, lam=0.95):
    """
    Generalized Advantage Estimation (GAE).

    Instead of advantage = G_t - V(s_t) (high variance),
    GAE blends TD errors across multiple steps:
      delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
      A_t = delta_t + (gamma*lam)*delta_{t+1} + (gamma*lam)^2*delta_{t+2} + ...

    lam=0: pure TD (low variance, high bias)
    lam=1: pure MC (high variance, low bias)
    lam=0.95: good balance

    Args:
        rewards: (T, batch) numpy array
        values:  (T, batch) tensor (detached)
    Returns:
        advantages: (T, batch) tensor
        returns:    (T, batch) tensor (for critic target)
    """
    T_len, batch_size = rewards.shape
    values_np = values.numpy()

    advantages = np.zeros((T_len, batch_size), dtype=np.float32)
    gae = np.zeros(batch_size, dtype=np.float32)

    for t in reversed(range(T_len)):
        if t == T_len - 1:
            next_value = 0.0  # terminal
        else:
            next_value = values_np[t + 1]
        delta = rewards[t] + gamma * next_value - values_np[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae

    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = advantages + values  # target for critic
    return advantages, returns


def train_actor_critic(env_kwargs, num_episodes=5000, batch_size=16,
                       lr_actor=1e-4, lr_critic=3e-4, critic_updates=3,
                       print_every=200):
    actor = PolicyNetwork()
    critic = ValueNetwork()
    optimizer_actor = optim.Adam(actor.parameters(), lr=lr_actor)
    optimizer_critic = optim.Adam(critic.parameters(), lr=lr_critic)
    cost_history = []

    # reward normalization stats (running mean/std)
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

            # actor
            mu, std = actor(s)
            dist = Normal(mu, std)
            actions = dist.sample()
            log_prob = dist.log_prob(actions).sum(dim=-1)

            a_np = actions.detach().numpy()
            states, costs = env.next_step(a_np[:, 0], a_np[:, 1])

            all_log_probs.append(log_prob)
            all_rewards.append(-costs)

        # stack
        states_t = torch.stack(all_states)           # (T, batch, 4)
        rewards_np = np.array(all_rewards)            # (T, batch)
        log_probs_t = torch.stack(all_log_probs)      # (T, batch)

        # normalize rewards to stabilize critic training
        batch_r_mean = rewards_np.mean()
        batch_r_std = rewards_np.std() + 1e-8
        reward_mean = 0.99 * reward_mean + 0.01 * batch_r_mean
        reward_std = 0.99 * reward_std + 0.01 * batch_r_std
        rewards_normed = (rewards_np - reward_mean) / reward_std

        with torch.no_grad():
            values_detached = critic(states_t).detach()

        # GAE on normalized rewards
        advantage, returns_target = compute_gae(rewards_normed, values_detached)
        advantage = (advantage - advantage.mean(dim=0)) / (advantage.std(dim=0) + 1e-8)

        # --- update actor (1 step) ---
        actor_loss = (-log_probs_t * advantage).sum(dim=0).mean()
        optimizer_actor.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
        optimizer_actor.step()

        # --- update critic ---
        for _ in range(critic_updates):
            values_pred = critic(states_t)
            critic_loss = nn.functional.smooth_l1_loss(values_pred, returns_target.detach())
            optimizer_critic.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
            optimizer_critic.step()

        avg_reward = rewards_np.sum(axis=0).mean()  # unnormalized for logging
        cost_history.append(avg_reward)

        if epi % print_every == 0:
            avg_v = values_detached.mean().item()
            print(f"[A-C Episode {epi:5d}] avg_reward={avg_reward:.1f}  "
                  f"critic_loss={critic_loss.item():.4f}  avg_V={avg_v:.2f}")

    return actor, cost_history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(policy, env_kwargs, n_eval=500):
    env = TwoEchelonInvVec(n_envs=n_eval, **env_kwargs)
    states = env.reset()
    total_rewards = np.zeros(n_eval, dtype=np.float32)
    for t in range(T):
        with torch.no_grad():
            s = torch.tensor(states, dtype=torch.float32)
            mu, _ = policy(s)
            a_np = mu.numpy().clip(min=0.0)
        states, costs = env.next_step(a_np[:, 0], a_np[:, 1])
        total_rewards += -costs
    return total_rewards.mean(), total_rewards.std()


def evaluate_basestock(theta1_val, theta2_val, env_kwargs, n_eval=500):
    env = TwoEchelonInvVec(n_envs=n_eval, **env_kwargs)
    states = env.reset()
    total_rewards = np.zeros(n_eval, dtype=np.float32)
    for t in range(T):
        x1, w1, x2, w2 = states[:, 0], states[:, 1], states[:, 2], states[:, 3]
        a1 = np.maximum(theta1_val - x1 - w1, 0.0)
        a2 = np.maximum(theta2_val - x2 - w2, 0.0)
        states, costs = env.next_step(a1, a2)
        total_rewards += -costs
    return total_rewards.mean(), total_rewards.std()


# ---------------------------------------------------------------------------
# Base-stock training (for reference)
# ---------------------------------------------------------------------------
def train_basestock(env_kwargs, num_episodes=5000, batch_size=16,
                    lr=3e-3, print_every=200):
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

    print(f"Base-stock: θ1={theta1.item():.2f}, θ2={theta2.item():.2f}")
    return theta1, theta2, cost_history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    env_kwargs = dict(h1=2.0, p1=5.0, h2=1.0, demand_lambda=5,
                      init_x1=10.0, init_w1=0.0, init_x2=20.0, init_w2=0.0,
                      c1=1.0)

    N_EPI = 5000
    BATCH = 16

    print("=" * 60)
    print("Two-echelon, K=0: REINFORCE vs Actor-Critic")
    print("=" * 60)
    t0 = time.time()

    # 1. Base-stock
    print("\n--- Base-stock ---")
    theta1, theta2, bs_hist = train_basestock(env_kwargs, num_episodes=N_EPI, batch_size=BATCH)
    t1 = time.time()
    print(f"Time: {t1-t0:.1f}s")

    # 2. REINFORCE
    print("\n--- REINFORCE (NN) ---")
    rf_policy, rf_hist = train_reinforce(env_kwargs, num_episodes=N_EPI, batch_size=BATCH)
    t2 = time.time()
    print(f"Time: {t2-t1:.1f}s")

    # 3. Actor-Critic
    print("\n--- Actor-Critic (NN) ---")
    ac_policy, ac_hist = train_actor_critic(env_kwargs, num_episodes=N_EPI, batch_size=BATCH)
    t3 = time.time()
    print(f"Time: {t3-t2:.1f}s")

    # Grid search for true optimal base-stock
    print("\n--- Grid search optimal base-stock ---")
    best_r, best_t1, best_t2 = -1e9, 0, 0
    for t1 in np.arange(8, 20, 0.5):
        for t2 in np.arange(16, 30, 0.5):
            r, _ = evaluate_basestock(t1, t2, env_kwargs, n_eval=1000)
            if r > best_r:
                best_r, best_t1, best_t2 = r, t1, t2
    print(f"Optimal: θ1={best_t1:.1f}, θ2={best_t2:.1f}, reward={best_r:.1f}")

    # Evaluate all
    print("\n" + "=" * 60)
    print("Evaluation (1000 episodes)")
    print("=" * 60)
    opt_mean, opt_std = evaluate_basestock(best_t1, best_t2, env_kwargs, n_eval=1000)
    bs_mean, bs_std = evaluate_basestock(theta1.item(), theta2.item(), env_kwargs, n_eval=1000)
    rf_mean, rf_std = evaluate(rf_policy, env_kwargs, n_eval=1000)
    ac_mean, ac_std = evaluate(ac_policy, env_kwargs, n_eval=1000)
    print(f"Optimal BS:    avg_reward = {opt_mean:.1f} +/- {opt_std:.1f}  (θ1={best_t1:.1f}, θ2={best_t2:.1f})")
    print(f"Learned BS:    avg_reward = {bs_mean:.1f} +/- {bs_std:.1f}  (θ1={theta1.item():.2f}, θ2={theta2.item():.2f})")
    print(f"REINFORCE:     avg_reward = {rf_mean:.1f} +/- {rf_std:.1f}")
    print(f"Actor-Critic:  avg_reward = {ac_mean:.1f} +/- {ac_std:.1f}")

    # Plot
    def smooth(arr, window=100):
        kernel = np.ones(window) / window
        return np.convolve(arr, kernel, mode='valid')

    plt.figure(figsize=(10, 5))
    plt.plot(smooth(bs_hist), label='Base-stock', alpha=0.8)
    plt.plot(smooth(rf_hist), label='REINFORCE (NN)', alpha=0.8)
    plt.plot(smooth(ac_hist), label='Actor-Critic (NN)', alpha=0.8)
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward (smoothed)')
    plt.title('Two-Echelon K=0: REINFORCE vs Actor-Critic')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('reinforce_vs_ac.png', dpi=150)
    plt.show()
    print(f"\nTotal time: {time.time()-t0:.1f}s")
    print("Plot saved to reinforce_vs_ac.png")
