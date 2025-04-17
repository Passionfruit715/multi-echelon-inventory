"""
Filename: main.py
Description: test the convergence on multi-echelon inventory model's convergence on base stock policy.
Course: ISyE 8803 - Learning and Optimization in Operations
Contributors: Jinghua Weng, Yitong Wu
Created: 2025-4-14, Jinghua Weng
""" 
import numpy as np 
import math 
import torch
import torch.optim as optim
from torch.distributions import Normal
from torch.nn.functional import relu
from typing import Tuple, Dict, Any

T = 50          # assuming there are T period left.
gamma = 0.95    # discount factor.

class TwoEchelonInv:
    """
    the setup of two-echelon inventory model.
    """
    def __init__(self, 
                 h1 = 1.0,     # marginal holding cost for installation 1.
                 p1 = 10.0,    # marginal shortage cost for installation 1.
                 h2 = 1.0,     # marginal holding cost for installation 2.
                 demand_lambda = 5,     # suppose the demand follows possion distribution, \lambda = 5.
                 
                 # initial state variables.    
                 init_x1 = 20.0,     # initial stock on hand for installation 1.
                 init_w1 = 0.0,      # initial stock in transit.
                 init_x2 = 40.0,     # initial system stock. 
                 K = 0.0,            # purchase setup one time cost.
                 c = 1.0,            # purchase per unit cost.
                 c1 = 1.0,           # transit cost per unit within installations.
                 seed = None         # add stochasticity.  
                 ):

        self.h1 = h1 
        self.p1 = p1
        self.h2 = h2 
        self.demand_lambda = demand_lambda
        self.init_x1 = init_x1
        self.init_w1 = init_w1
        self.init_x2 = init_x2
        self.K = K 
        self.c = c
        self.c1 = c1
        
        if seed is not None:
            np.random.seed(seed)
        self.reset()

        # current state variables.
        self.x1 = init_x1 
        self.w1 = init_w1
        self.x2 = init_x2
        self.t = 0     # tracking the current time step. 

    def reset(self) -> np.ndarray:
        """
        reset the model to initial state, to be used in monte carlo simulation.

        Return: a 3 dimension array containing state variables(np.ndarray)  
        """
        self.x1 = self.init_x1
        self.w1 = self.init_x2
        self.x2 = self.init_x2
        self.t = 0

        return np.array([self.x1, self.w1, self.x2], dtype=np.float32)

    
    def purchase_cost(self, z: float) -> float:
        """
        compute the purchase cost

        Args:
        z (float): purchase quantity.

        Return:
        float: if z > 0, return setup cost + per unit cost, else no cost.
        """
        return self.K + self.c * z if z > 0 else 0.0
    
    def shortage_storage_cost(self, x: float) -> float:
        """
        compute the shortage and storage cost.

        Args:
        x (float): current order level at installation 1 after the demand is taken (can be positive and negitive).

        Return:
        float: the shortage and storage cost.
        """
        return self.h1 * x if x >= 0 else self.p1 * (-x)

    def next_step(self, a1: float, a2: float) -> Tuple[np.ndarray, float, bool]:
        """
        calculate the total cost of current time step, and update the new state variables.

        Args:
        a1: order request of installation 1 to be delivered at the beginning of next period.
        a2: order request of installation 2 to be delivered at the beginning of next period.

        Return:
        new states variables and the current step cost.
        """
        # 
        x2_local = self.x2 - self.w1 - self.x1
        
        cost_x1 = self.shortage_storage_cost(self.x1)
        cost_x2 = self.h2 * self.x2
        
        demand = np.random.poisson(demand_lambda)
        self.x1 = self.x1 + self.w1 - demand
        
        self.w1 = min(a1, x2_local)
        cost_w1 = self.c1 * self.w1
        total_period_cost = cost_x1 + cost_x2 + cost_w1

        self.x2 = self.x2  - demand + a2
        next_state = np.array([self.x1, self.w1, self.x2], dtype=np.float32)         

        done = (self.t >= T)
        return (next_state, total_period_cost, done)


if __name__ == "__main__":
    # initialization of variables.
    h1 = 1.0
    p1 = 10.0 
    h2 = 1.0
    demand_lambda = 5
    init_x1 = 20.0 
    init_w1 = 0.0 
    init_x2 = 40.0
    K = 0.0 
    c = 1.0 
    c1 = 1.0

    # assuming base stock policy.
    theta1 = torch.tensor(20.0, requires_grad=True)
    theta2 = torch.tensor(45.0, requires_grad=True)
    params = [theta1, theta2]
    optimizer = optim.Adam(params, lr=0.03)     # TODO: test Adam here, try other optimizers later

    # gaussion noise params
    sigma1 = torch.tensor(1.0)
    sigma2 = torch.tensor(1.0)
    
    num_batches = 400
    batech_sizes = 20
    # apply the policy gradient theorem, monte carlo sampling times = 1000.  
    for batch_idx in range(1, num_batches+1):
        batch_terms = []
        for k in range(batech_sizes):
            inv = TwoEchelonInv(h1=h1, p1=p1, h2=h2, demand_lambda=demand_lambda, 
                                init_x1=init_x1, init_w1=init_w1, init_x2=init_x2,
                                K=K, c=c, c1=c1, seed = batch_idx * batech_sizes + k)
            state = inv.reset()
            rewards = []
            log_probs = []

            for t in range(T):
                x1, w1, x2 = [torch.tensor(s, dtype=torch.float32, requires_grad=False) for s in state]

                mu1 = relu(theta1 - x1)
                dist1 = Normal(mu1, sigma1)
                a1 = dist1.sample()
                logp1 = dist1.log_prob(a1)
            
            
                mu2 = relu(theta2 - x2)
                dist2 = Normal(mu2, sigma2)
                a2 = dist2.sample()
                logp2 = dist2.log_prob(a2)

                new_state, cost, done = inv.next_step(a1.item(), a2.item())
                
                log_probs.append(logp1 + logp2)
                rewards.append(-cost)
                state = new_state
                if done:
                    break

            G = 0
            returns = []
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)

            returns = torch.tensor(returns, dtype=torch.float32)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        
            batch_terms += [(- logp * G_t) for logp, G_t in zip(log_probs, returns)]
        
        # find the mean of policy loss within its batch.
        policy_loss = torch.stack(batch_terms).mean()
        
        # clear the previous gradient data.
        optimizer.zero_grad()
        # calculate the gradient w.r.t. theta. 
        policy_loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"[Batch {batch_idx:4d}/{num_batches}] "
                  f"loss={policy_loss.item():.3f}  "
                  f"θ1={theta1.item():.2f}  θ2={theta2.item():.2f}")












