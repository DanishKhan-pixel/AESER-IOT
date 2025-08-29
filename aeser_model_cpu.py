# AESER: Adaptive, Energy- & Security-aware Efficient Routing (CPU-only)
# -------------------------------------------------------------------

import os
import random
from dataclasses import dataclass
from typing import List, Tuple
from collections import namedtuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from gym import spaces
from tqdm import trange
import matplotlib.pyplot as plt

# CPU-only
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_num_threads(max(1, os.cpu_count() // 2))

# PyG
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GCNConv, global_mean_pool

# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# -----------------------------
# Environment: Energy & Security-aware Graph Routing
# -----------------------------
class AESERGraphEnv(gym.Env):
    """
    Connected random graph with simple energy dynamics and an anomaly flag.

    - Nodes: sensors with battery in [0,1].
    - Each step: the packet is at current node c, agent selects a neighbor j.
    - Transition: current <- j; batteries: battery[j] -= tx_cost(j), battery[c] -= rx_cost(c) (small).
    - Failure if battery <= 0 (node dies) -> episode ends with penalty.
    - Reward: +arrival_reward on reaching target; per-step: -step_cost;
              -energy_weight * (energy_spent_tx + energy_spent_rx);
              -security_weight if chosen next-hop is anomalous.
    - Observation: PyG Data with x=[is_current, is_target, degree_norm, battery, anomaly_score].
    - Action masking: only neighbors of current can be chosen, and (optionally) nodes with battery>min_batt.
    """

    metadata = {"render.modes": []}

    def __init__(self,
                 N: int = 18,
                 p_edge: float = 0.22,
                 max_steps: int = 40,
                 seed: int = 123,
                 arrival_reward: float = 1.0,
                 step_cost: float = 0.02,
                 energy_weight: float = 0.6,
                 security_weight: float = 0.2,
                 min_viable_batt: float = 0.02):
        super().__init__()
        self.N = N
        self.p_edge = p_edge
        self.max_steps = max_steps
        self.arrival_reward = arrival_reward
        self.step_cost = step_cost
        self.energy_weight = energy_weight
        self.security_weight = security_weight
        self.min_viable_batt = min_viable_batt
        self._seed = seed
        self.seed(seed)
        self.action_space = spaces.Discrete(self.N)
        # dummy observation_box for Gym compliance; actual obs is PyG Data
        self.observation_space = spaces.Box(low=-10, high=10, shape=(1,), dtype=np.float32)
        self._build_graph()

    # --- graph generation ---
    def seed(self, s=None):
        self._seed = s
        self.rng = np.random.RandomState(s)
        random.seed(s)

    def _build_graph(self):
        import networkx as nx
        while True:
            G = nx.erdos_renyi_graph(self.N, self.p_edge, seed=None)
            if nx.is_connected(G):
                break
        self.G = G
        self.adj = nx.to_numpy_array(G, dtype=int)
        self.neighbors = [list(G.neighbors(i)) for i in range(self.N)]

    # --- episode lifecycle ---
    def reset(self):
        self._build_graph()
        self.source = random.randrange(self.N)
        self.target = random.randrange(self.N)
        while self.target == self.source:
            self.target = random.randrange(self.N)
        self.current = self.source
        self.steps = 0
        # initialize battery and anomaly scores
        self.battery = self.rng.uniform(0.5, 1.0, size=self.N).astype(np.float32)
        # a few anomalous nodes (e.g., compromised): higher anomaly score
        self.anomaly = np.zeros(self.N, dtype=np.float32)
        for _ in range(self.rng.randint(0, max(1, self.N // 8))):
            self.anomaly[self.rng.randint(0, self.N)] = self.rng.uniform(0.6, 1.0)
        return self._obs()

    def _degree_norm(self):
        deg = self.adj.sum(axis=1)
        return (deg / (self.N - 1 + 1e-9)).astype(np.float32)

    def _obs(self):
        deg_norm = self._degree_norm()
        x = []
        for i in range(self.N):
            x.append([
                1.0 if i == self.current else 0.0,
                1.0 if i == self.target else 0.0,
                deg_norm[i],
                float(self.battery[i]),
                float(self.anomaly[i]),
            ])
        x = torch.tensor(x, dtype=torch.float32)
        edge_index = torch.tensor(np.vstack(np.where(self.adj)).astype(np.int64), dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)
        data.num_nodes = self.N
        data.adj = torch.tensor(self.adj, dtype=torch.float32)
        data.current = self.current
        data.target = self.target
        # mask for viable nodes (battery above threshold)
        data.viable = torch.tensor((self.battery > self.min_viable_batt).astype(np.float32))
        return data

    # simple energy model
    def _tx_cost(self, j: int) -> float:
        # more cost for high-degree nodes (busy relays)
        deg = max(1, int(self.adj[j].sum()))
        return 0.01 + 0.004 * deg

    def _rx_cost(self, i: int) -> float:
        return 0.005

    def step(self, action: int):
        self.steps += 1
        done = False
        reward = -self.step_cost
        # invalid move penalty (should not happen with mask)
        if action not in self.neighbors[self.current]:
            reward -= 0.5
        else:
            # energy spending
            tx = self._tx_cost(action)
            rx = self._rx_cost(self.current)
            self.battery[action] -= tx
            self.battery[self.current] -= rx
            # security penalty if next hop is anomalous
            reward -= self.security_weight * float(self.anomaly[action])
            # energy penalty
            reward -= self.energy_weight * (tx + rx)
            # move packet
            self.current = int(action)

        # terminal conditions
        if self.current == self.target:
            reward += self.arrival_reward
            done = True
        if self.steps >= self.max_steps:
            done = True
        if (self.battery <= 0).any():
            reward -= 1.0
            done = True
        return self._obs(), float(reward), done, {}

# -----------------------------
# Security Autoencoder (optional offline pretrain)
# -----------------------------
class TrafficAE(nn.Module):
    def __init__(self, in_dim=5, hidden=32, latent=8):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, latent), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, in_dim))
    def forward(self, x):
        z = self.enc(x)
        r = self.dec(z)
        return r

@torch.no_grad()
def infer_anomaly_scores(model: TrafficAE, x: torch.Tensor) -> np.ndarray:
    # x: [N, in_dim]
    recon = model(x)
    err = ((recon - x) ** 2).mean(dim=1).cpu().numpy()
    # normalize to [0,1]
    if err.max() > 0:
        err = (err - err.min()) / (err.max() - err.min())
    return err.astype(np.float32)

# -----------------------------
# AESER Model
# -----------------------------
class GNNEncoder(nn.Module):
    def __init__(self, in_feats=5, hidden=64, n_layers=2, node_emb=64, conv="SAGE"):
        super().__init__()
        conv_cls = SAGEConv if conv == "SAGE" else GCNConv
        self.layers = nn.ModuleList()
        self.layers.append(conv_cls(in_feats, hidden))
        for _ in range(n_layers - 1):
            self.layers.append(conv_cls(hidden, hidden))
        self.lin = nn.Linear(hidden, node_emb)

    def forward(self, data: Data):
        h = data.x
        for conv in self.layers:
            h = torch.relu(conv(h, data.edge_index))
        node_h = torch.relu(self.lin(h))
        batch = getattr(data, 'batch', None)
        if batch is None:
            batch = torch.zeros(node_h.size(0), dtype=torch.long)
        g = global_mean_pool(node_h, batch)
        return node_h, g

class ActorCritic(nn.Module):
    def __init__(self, node_emb=64, graph_emb=64, hidden=128, N=18):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(node_emb + graph_emb, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.critic = nn.Sequential(
            nn.Linear(graph_emb, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.N = N

    def forward(self, node_h, g):
        G = g.repeat(node_h.size(0), 1)
        logits = self.actor(torch.cat([node_h, G], dim=-1)).squeeze(-1)
        value = self.critic(g).squeeze(-1)
        return logits, value

Transition = namedtuple('Transition', ['obs', 'action', 'logp', 'value', 'reward', 'done'])

class AESER_PPO:
    def __init__(self, env: AESERGraphEnv, enc: GNNEncoder, ac: ActorCritic,
                 lr=3e-4, gamma=0.99, clip=0.2, epochs=4, minibatch=64):
        self.env = env
        self.enc = enc
        self.ac = ac
        self.opt = optim.Adam(list(enc.parameters()) + list(ac.parameters()), lr=lr)
        self.gamma = gamma
        self.clip = clip
        self.epochs = epochs
        self.minibatch = minibatch

    @staticmethod
    def _masked_dist(logits: torch.Tensor, valid: torch.Tensor):
        neg_inf = -1e9
        masked = logits + (1.0 - valid) * neg_inf
        probs = torch.softmax(masked, dim=0)
        return torch.distributions.Categorical(probs), probs

    def _valid_mask(self, data: Data):
        cur = int(data.current)
        neigh_mask = (data.adj[cur] > 0).to(torch.float32)  # neighbors
        viable = getattr(data, 'viable', torch.ones_like(neigh_mask))
        valid = neigh_mask * viable  # require battery above threshold
        if valid.sum() == 0:
            valid = neigh_mask  # fallback
        if valid.sum() == 0:
            valid = torch.ones_like(neigh_mask)  # extreme fallback
        return valid

    def collect(self, steps=1024):
        traj: List[Transition] = []
        obs = self.env.reset()
        step = 0
        rs = []
        while step < steps:
            node_h, g = self.enc(obs)
            logits, v = self.ac(node_h, g)
            valid = self._valid_mask(obs)
            dist, _ = self._masked_dist(logits, valid)
            a = int(dist.sample().item())
            logp = float(dist.log_prob(torch.tensor(a)))
            new_obs, r, d, _ = self.env.step(a)
            traj.append(Transition(obs, a, logp, float(v.item()), r, d))
            rs.append(r)
            obs = new_obs
            step += 1
            if d:
                obs = self.env.reset()
        return traj, float(np.mean(rs))

    def _gae(self, rewards, values, dones, last_v, lam=0.95):
        adv, g = [], 0.0
        values = values + [last_v]
        for t in reversed(range(len(rewards))):
            mask = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * values[t+1] * mask - values[t]
            g = delta + self.gamma * lam * mask * g
            adv.insert(0, g)
        return adv

    def train(self, total_epochs=200, steps_per_epoch=512, verbose_every=10):
        curve = []
        for ep in range(1, total_epochs + 1):
            traj, mean_roll = self.collect(steps_per_epoch)
            curve.append(mean_roll)
            rewards = [t.reward for t in traj]
            dones = [t.done for t in traj]
            values = [t.value for t in traj]
            with torch.no_grad():
                last_obs = traj[-1].obs
                _, g_last = self.enc(last_obs)
                last_v = float(self.ac.critic(g_last).item())
            adv = self._gae(rewards, values, dones, last_v)
            rets = [a + v for a, v in zip(adv, values)]

            actions = torch.tensor([t.action for t in traj], dtype=torch.long)
            old_logp = torch.tensor([t.logp for t in traj], dtype=torch.float32)
            adv_t = torch.tensor(adv, dtype=torch.float32)
            ret_t = torch.tensor(rets, dtype=torch.float32)
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

            idx = np.arange(len(traj))
            for _ in range(self.epochs):
                np.random.shuffle(idx)
                for s in range(0, len(traj), self.minibatch):
                    mb = idx[s:s + self.minibatch]
                    mb_logp, mb_val = [], []
                    for k in mb:
                        data = traj[k].obs
                        node_h, g = self.enc(data)
                        logits, v = self.ac(node_h, g)
                        valid = self._valid_mask(data)
                        dist, _ = self._masked_dist(logits, valid)
                        a_k = actions[k]
                        mb_logp.append(dist.log_prob(a_k))
                        mb_val.append(v.squeeze())
                    mb_logp = torch.stack(mb_logp)
                    mb_val = torch.stack(mb_val)
                    ratio = torch.exp(mb_logp - old_logp[mb])
                    surr1 = ratio * adv_t[mb]
                    surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * adv_t[mb]
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = nn.MSELoss()(mb_val, ret_t[mb])
                    loss = policy_loss + 0.5 * value_loss
                    self.opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(self.enc.parameters()) + list(self.ac.parameters()), 0.5)
                    self.opt.step()

            if ep % verbose_every == 0:
                print(f"[AESER] Epoch {ep}/{total_epochs}  mean_rollout_reward={mean_roll:.4f}")
        return curve

# -----------------------------
# Deterministic Evaluation
# -----------------------------
@torch.no_grad()
def eval_deterministic(env: AESERGraphEnv, enc: GNNEncoder, ac: ActorCritic) -> Tuple[float, bool, int]:
    obs = env.reset()
    total_r, steps = 0.0, 0
    for _ in range(env.max_steps):
        node_h, g = enc(obs)
        logits, _ = ac(node_h, g)
        valid = (obs.adj[obs.current] > 0).to(torch.float32)
        viable = getattr(obs, 'viable', torch.ones_like(valid))
        valid = valid * viable
        neg_inf = -1e9
        masked = logits + (1.0 - valid) * neg_inf
        a = int(torch.argmax(masked).item())
        obs, r, d, _ = env.step(a)
        total_r += r
        steps += 1
        if d:
            return total_r, True, steps
    return total_r, False, steps

# -----------------------------
# Main
# -----------------------------

def main():
    set_seed(42)

    # --- Optional: pretrain anomaly detector on benign samples (synthetic) ---
    # We will not integrate it directly to mutate env.anomaly (kept for extension),
    # but you can use infer_anomaly_scores(ae, data.x) to compute anomaly features.
    ae = TrafficAE(in_dim=5, hidden=32, latent=8)
    ae_opt = optim.Adam(ae.parameters(), lr=1e-3)
    # quick unsupervised warmup on random benign-like vectors
    for _ in range(300):
        x = torch.rand(64, 5)
        recon = ae(x)
        loss = ((recon - x) ** 2).mean()
        ae_opt.zero_grad(); loss.backward(); ae_opt.step()

    # --- Env & Model ---
    env = AESERGraphEnv(N=18, p_edge=0.22, max_steps=40,
                        arrival_reward=1.0, step_cost=0.02,
                        energy_weight=0.6, security_weight=0.2,
                        seed=7)

    encoder = GNNEncoder(in_feats=5, hidden=64, n_layers=2, node_emb=64, conv="SAGE")
    ac = ActorCritic(node_emb=64, graph_emb=64, hidden=128, N=env.N)
    agent = AESER_PPO(env, encoder, ac, lr=3e-4, gamma=0.99, clip=0.2, epochs=4, minibatch=64)

    # --- Train ---
    EPOCHS = 160
    STEPS_PER_EPOCH = 512
    curve = agent.train(total_epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, verbose_every=10)

    # --- Save training curve ---
    plt.figure()
    plt.plot(curve)
    plt.xlabel('Epoch')
    plt.ylabel('Mean rollout reward')
    plt.title('AESER Training Curve')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('aeser_train_curve.png')
    plt.close()

    # --- Evaluate ---
    TEST_EPISODES = 300
    rewards = []
    successes = 0
    steps_sum = 0
    for _ in range(TEST_EPISODES):
        env_eval = AESERGraphEnv(N=18, p_edge=0.22, max_steps=40,
                                 arrival_reward=1.0, step_cost=0.02,
                                 energy_weight=0.6, security_weight=0.2)
        r, ok, steps = eval_deterministic(env_eval, encoder, ac)
        rewards.append(r)
        successes += int(ok)
        steps_sum += steps

    avg_reward = float(np.mean(rewards))
    success_rate = successes / TEST_EPISODES
    avg_steps = steps_sum / TEST_EPISODES

    df = pd.DataFrame({
        'metric': ['avg_reward', 'success_rate', 'avg_steps'],
        'value': [avg_reward, success_rate, avg_steps]
    })
    df.to_csv('aeser_results.csv', index=False)

    with open('aeser_eval_summary.txt', 'w') as f:
        f.write(f"AESER Evaluation (deterministic)\n")
        f.write(f"Episodes: {TEST_EPISODES}\n")
        f.write(f"Average reward: {avg_reward:.4f}\n")
        f.write(f"Success rate:  {success_rate*100:.2f}%\n")
        f.write(f"Average steps: {avg_steps:.2f}\n")

    print("\nSaved: aeser_train_curve.png, aeser_results.csv, aeser_eval_summary.txt")


if __name__ == "__main__":
    main()
