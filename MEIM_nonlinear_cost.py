"""
Filename: MEIM_nonlinear_cost.py
Description: Three-echelon inventory with lead times and nonlinear costs.
             - Holding cost: quadratic h * x^2 (convex, penalizes excess)
             - Shortage cost: piecewise (p1 for 0-5 units, p2 for 5+ units)
             AC NN vs grid-search-optimal base-stock.
Course: ISyE 8803 - Learning and Optimization in Operations
Contributors: Jinghua Weng, Yitong Wu
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
import time


T = 50
gamma = 0.97


# ---------------------------------------------------------------------------
# Vectorized three-echelon environment
# ---------------------------------------------------------------------------
class ThreeEchelonInvVec:
    def __init__(self, n_envs,
                 h1=0.1, p1_low=5.0, p1_high=20.0, p1_threshold=5.0,
                 h2=0.05, h3=0.02,
                 demand_lambda=5,
                 lead_times=(2, 2, 1),
                 init_x=(10.0, 15.0, 20.0),
                 c1=1.0, c2=1.0, c3=1.0):
        self.n_envs = n_envs
        self.h1 = h1          # quadratic holding cost coefficient at E1
        self.p1_low = p1_low   # shortage penalty for first p1_threshold units
        self.p1_high = p1_high # shortage penalty beyond p1_threshold units
        self.p1_threshold = p1_threshold
        self.h2 = h2          # quadratic holding cost coefficient at E2
        self.h3 = h3          # quadratic holding cost coefficient at E3
        self.demand_lambda = demand_lambda
        self.lead_times = lead_times
        self.init_x = init_x
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.L1, self.L2, self.L3 = lead_times
        self.state_dim = 3 + sum(lead_times)
        self.reset()

    def reset(self):
        n = self.n_envs
        self.x1 = np.full(n, self.init_x[0], dtype=np.float32)
        self.x2 = np.full(n, self.init_x[1], dtype=np.float32)
        self.x3 = np.full(n, self.init_x[2], dtype=np.float32)
        self.pipeline1 = np.zeros((n, self.L1), dtype=np.float32)
        self.pipeline2 = np.zeros((n, self.L2), dtype=np.float32)
        self.pipeline3 = np.zeros((n, self.L3), dtype=np.float32)
        return self._get_state()

    def _get_state(self):
        return np.column_stack([
            self.x1, self.pipeline1,
            self.x2, self.pipeline2,
            self.x3, self.pipeline3,
        ])

    def next_step(self, a1, a2, a3):
        demand = np.random.poisson(self.demand_lambda, size=self.n_envs).astype(np.float32)

        arriving1 = self.pipeline1[:, 0] if self.L1 > 0 else np.zeros(self.n_envs, dtype=np.float32)
        arriving2 = self.pipeline2[:, 0] if self.L2 > 0 else np.zeros(self.n_envs, dtype=np.float32)
        arriving3 = self.pipeline3[:, 0] if self.L3 > 0 else np.zeros(self.n_envs, dtype=np.float32)

        self.x1 += arriving1
        self.x2 += arriving2
        self.x3 += arriving3

        self.x1 -= demand
        # nonlinear holding cost: h * x^2 (convex)
        # nonlinear shortage cost: piecewise linear
        #   shortage <= threshold: p1_low * shortage
        #   shortage > threshold:  p1_low * threshold + p1_high * (shortage - threshold)
        shortage = np.maximum(-self.x1, 0.0)
        cost_holding1 = self.h1 * np.maximum(self.x1, 0.0) ** 2
        cost_shortage1 = np.where(
            shortage <= self.p1_threshold,
            self.p1_low * shortage,
            self.p1_low * self.p1_threshold + self.p1_high * (shortage - self.p1_threshold)
        )
        cost_x1 = cost_holding1 + cost_shortage1

        a1 = np.maximum(a1, 0.0)
        ship_to_1 = np.minimum(a1, np.maximum(self.x2, 0.0))
        self.x2 -= ship_to_1
        cost_ship1 = self.c1 * ship_to_1

        a2 = np.maximum(a2, 0.0)
        ship_to_2 = np.minimum(a2, np.maximum(self.x3, 0.0))
        self.x3 -= ship_to_2
        cost_ship2 = self.c2 * ship_to_2

        a3 = np.maximum(a3, 0.0)
        ship_to_3 = a3
        cost_ship3 = self.c3 * ship_to_3

        # quadratic holding costs at E2, E3
        cost_x2 = self.h2 * np.maximum(self.x2, 0.0) ** 2
        cost_x3 = self.h3 * np.maximum(self.x3, 0.0) ** 2

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
# Networks
# ---------------------------------------------------------------------------
class PolicyNetwork(nn.Module):
    MAX_ORDER = 25.0  # clamp output to prevent inventory explosion

    def __init__(self, state_dim, action_dim=3, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        raw = self.net(state)
        # sigmoid * MAX_ORDER: bounded in [0, MAX_ORDER]
        mu = torch.sigmoid(raw) * self.MAX_ORDER
        std = self.log_std.exp().clamp(min=1e-3, max=2.0)
        return mu, std


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
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
# Actor-Critic training
# ---------------------------------------------------------------------------
def train_ac(env_kwargs, num_episodes=8000, batch_size=16,
             lr_actor=1e-4, lr_critic=3e-4, critic_updates=3,
             print_every=400):
    state_dim = 3 + sum(env_kwargs.get('lead_times', (2, 2, 1)))
    actor = PolicyNetwork(state_dim=state_dim, action_dim=3)
    critic = ValueNetwork(state_dim=state_dim)
    optimizer_actor = optim.Adam(actor.parameters(), lr=lr_actor)
    optimizer_critic = optim.Adam(critic.parameters(), lr=lr_critic)
    cost_history = []

    reward_mean = 0.0
    reward_std = 1.0

    for epi in range(1, num_episodes + 1):
        env = ThreeEchelonInvVec(n_envs=batch_size, **env_kwargs)
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

            a_np = actions.detach().numpy().clip(min=0.0, max=PolicyNetwork.MAX_ORDER)
            states, costs = env.next_step(a_np[:, 0], a_np[:, 1], a_np[:, 2])

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
# Base-stock training (REINFORCE, vectorized)
# ---------------------------------------------------------------------------
def train_basestock(env_kwargs, num_episodes=5000, batch_size=16,
                    lr=3e-3, print_every=500):
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
        states = env.reset()
        all_log_probs = []
        all_rewards = []

        for t in range(T):
            s = torch.tensor(states, dtype=torch.float32)
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

            mu1 = torch.relu(theta1 - inv_pos1)
            mu2 = torch.relu(theta2 - inv_pos2)
            mu3 = torch.relu(theta3 - inv_pos3)

            dist1 = Normal(mu1, sigma)
            dist2 = Normal(mu2, sigma)
            dist3 = Normal(mu3, sigma)
            a1, a2, a3 = dist1.sample(), dist2.sample(), dist3.sample()

            log_prob = dist1.log_prob(a1) + dist2.log_prob(a2) + dist3.log_prob(a3)
            all_log_probs.append(log_prob)

            states, costs = env.next_step(
                a1.detach().numpy(), a2.detach().numpy(), a3.detach().numpy())
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
        theta_history['θ1'].append(theta1.item())
        theta_history['θ2'].append(theta2.item())
        theta_history['θ3'].append(theta3.item())

        if epi % print_every == 0:
            print(f"[BS Episode {epi:5d}] avg_reward={cost_history[-1]:.1f}  "
                  f"θ1={theta1.item():.2f}  θ2={theta2.item():.2f}  θ3={theta3.item():.2f}")

    return [theta1, theta2, theta3], cost_history, theta_history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_ac(actor, env_kwargs, n_eval=1000):
    env = ThreeEchelonInvVec(n_envs=n_eval, **env_kwargs)
    states = env.reset()
    total_rewards = np.zeros(n_eval, dtype=np.float32)
    for t in range(T):
        with torch.no_grad():
            s = torch.tensor(states, dtype=torch.float32)
            mu, _ = actor(s)
            a_np = mu.numpy().clip(min=0.0)
        states, costs = env.next_step(a_np[:, 0], a_np[:, 1], a_np[:, 2])
        total_rewards += -costs
    return total_rewards.mean(), total_rewards.std()


def evaluate_basestock(thetas, env_kwargs, n_eval=1000):
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
        h1=0.5, p1_low=5.0, p1_high=30.0, p1_threshold=3.0,
        h2=0.2, h3=0.1,
        demand_lambda=5,
        lead_times=(2, 2, 1),
        init_x=(10.0, 15.0, 20.0),
        c1=1.0, c2=1.0, c3=1.0,
    )

    state_dim = 3 + sum(env_kwargs['lead_times'])
    print("=" * 60)
    print(f"Three-echelon, lead_times=(2,2,1), state_dim={state_dim}")
    print("Nonlinear costs: quadratic holding + piecewise shortage")
    print(f"  Holding: h*x^2  (h1={env_kwargs['h1']}, h2={env_kwargs['h2']}, h3={env_kwargs['h3']})")
    print(f"  Shortage E1: {env_kwargs['p1_low']}/unit (<=5), {env_kwargs['p1_high']}/unit (>5)")
    print("=" * 60)
    t0 = time.time()

    # Actor-Critic
    print("\n--- Actor-Critic (NN) ---")
    ac_actor, ac_hist = train_ac(
        env_kwargs, num_episodes=15000, batch_size=16)
    t1 = time.time()
    print(f"Time: {t1-t0:.1f}s")

    # Grid search for best base-stock
    print("\n--- Grid search optimal base-stock ---")
    L1, L2, L3 = env_kwargs['lead_times']
    best_r, best_t1, best_t2, best_t3 = -1e9, 0, 0, 0
    for t1v in np.arange(6, 22, 2.0):
        for t2v in np.arange(8, 28, 2.0):
            for t3v in np.arange(8, 28, 2.0):
                r, _ = evaluate_basestock([
                    torch.tensor(t1v), torch.tensor(t2v), torch.tensor(t3v)
                ], env_kwargs, n_eval=500)
                if r > best_r:
                    best_r, best_t1, best_t2, best_t3 = r, t1v, t2v, t3v

    # fine grid
    for t1v in np.arange(best_t1-2, best_t1+3, 0.5):
        for t2v in np.arange(best_t2-3, best_t2+4, 1.0):
            for t3v in np.arange(best_t3-3, best_t3+4, 1.0):
                r, _ = evaluate_basestock([
                    torch.tensor(t1v), torch.tensor(t2v), torch.tensor(t3v)
                ], env_kwargs, n_eval=1000)
                if r > best_r:
                    best_r, best_t1, best_t2, best_t3 = r, t1v, t2v, t3v

    print(f"Optimal BS: θ1={best_t1:.1f}, θ2={best_t2:.1f}, θ3={best_t3:.1f}, reward={best_r:.1f}")

    # Evaluate
    print("\n" + "=" * 60)
    print("Evaluation (1000 episodes)")
    print("=" * 60)
    bs_mean, bs_std = evaluate_basestock([
        torch.tensor(best_t1), torch.tensor(best_t2), torch.tensor(best_t3)
    ], env_kwargs)
    ac_mean, ac_std = evaluate_ac(ac_actor, env_kwargs)
    print(f"Best BS:       avg_reward = {bs_mean:.1f} +/- {bs_std:.1f}  (θ1={best_t1:.1f}, θ2={best_t2:.1f}, θ3={best_t3:.1f})")
    print(f"Actor-Critic:  avg_reward = {ac_mean:.1f} +/- {ac_std:.1f}")

    # Plots
    def smooth(arr, window=100):
        kernel = np.ones(window) / window
        return np.convolve(arr, kernel, mode='valid')

    plt.figure(figsize=(10, 5))
    plt.plot(smooth(ac_hist), label='Actor-Critic (NN)', alpha=0.8)
    plt.axhline(y=best_r, color='blue', linestyle='--', label=f'Best base-stock ({best_r:.0f})', alpha=0.8)
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward (smoothed)')
    plt.title('3-Echelon Nonlinear Cost: AC vs Best Base-Stock')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('nonlinear_cost.png', dpi=150)
    plt.show()
    print(f"\nTotal time: {time.time()-t0:.1f}s")
    print("Plot saved to nonlinear_cost.png")
