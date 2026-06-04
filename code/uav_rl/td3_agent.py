"""Minimal TD3 implementation for online ArduPilot training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done) -> None:
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[idx], device=device),
            "actions": torch.as_tensor(self.actions[idx], device=device),
            "rewards": torch.as_tensor(self.rewards[idx], device=device),
            "next_obs": torch.as_tensor(self.next_obs[idx], device=device),
            "dones": torch.as_tensor(self.dones[idx], device=device),
        }


def mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, out_dim),
    )


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, act_limit: float):
        super().__init__()
        self.net = mlp(obs_dim, act_dim)
        self.act_limit = float(act_limit)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs)) * self.act_limit


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = mlp(obs_dim + act_dim, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))


@dataclass
class TD3Config:
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    exploration_noise: float = 0.1
    batch_size: int = 256
    replay_size: int = 200000


class TD3Agent:
    def __init__(self, obs_dim: int, act_dim: int, act_limit: float, config: TD3Config, device: torch.device):
        self.cfg = config
        self.device = device
        self.act_limit = float(act_limit)
        self.actor = Actor(obs_dim, act_dim, act_limit).to(device)
        self.actor_target = Actor(obs_dim, act_dim, act_limit).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1 = Critic(obs_dim, act_dim).to(device)
        self.critic2 = Critic(obs_dim, act_dim).to(device)
        self.critic1_target = Critic(obs_dim, act_dim).to(device)
        self.critic2_target = Critic(obs_dim, act_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_opt = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=config.critic_lr,
        )
        self.total_updates = 0

    def select_action(self, obs: np.ndarray, noise_scale: float = 0.0) -> np.ndarray:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs_tensor).cpu().numpy()[0]
        if noise_scale > 0.0:
            action = action + np.random.normal(0.0, noise_scale, size=action.shape)
        return np.clip(action, -self.act_limit, self.act_limit).astype(np.float32)

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        self.total_updates += 1
        with torch.no_grad():
            noise = (
                torch.randn_like(batch["actions"]) * self.cfg.policy_noise
            ).clamp(-self.cfg.noise_clip, self.cfg.noise_clip)
            next_action = (self.actor_target(batch["next_obs"]) + noise).clamp(-self.act_limit, self.act_limit)
            target_q1 = self.critic1_target(batch["next_obs"], next_action)
            target_q2 = self.critic2_target(batch["next_obs"], next_action)
            target_q = batch["rewards"] + (1.0 - batch["dones"]) * self.cfg.gamma * torch.min(target_q1, target_q2)

        current_q1 = self.critic1(batch["obs"], batch["actions"])
        current_q2 = self.critic2(batch["obs"], batch["actions"])
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss_value = 0.0
        if self.total_updates % self.cfg.policy_delay == 0:
            actor_loss = -self.critic1(batch["obs"], self.actor(batch["obs"])).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            actor_loss_value = float(actor_loss.item())
            self._soft_update()

        return {
            "critic_loss": float(critic_loss.item()),
            "q1_mean": float(current_q1.mean().item()),
            "q2_mean": float(current_q2.mean().item()),
            "actor_loss": actor_loss_value,
        }

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "total_updates": self.total_updates,
            "config": vars(self.cfg),
        }

    def load_state_dict(self, payload: dict) -> None:
        self.actor.load_state_dict(payload["actor"])
        self.actor_target.load_state_dict(payload["actor_target"])
        self.critic1.load_state_dict(payload["critic1"])
        self.critic2.load_state_dict(payload["critic2"])
        self.critic1_target.load_state_dict(payload["critic1_target"])
        self.critic2_target.load_state_dict(payload["critic2_target"])
        self.actor_opt.load_state_dict(payload["actor_opt"])
        self.critic_opt.load_state_dict(payload["critic_opt"])
        self.total_updates = int(payload.get("total_updates", 0))

    def _soft_update(self) -> None:
        for src, dst in zip(self.actor.parameters(), self.actor_target.parameters()):
            dst.data.copy_(self.cfg.tau * src.data + (1.0 - self.cfg.tau) * dst.data)
        for src, dst in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            dst.data.copy_(self.cfg.tau * src.data + (1.0 - self.cfg.tau) * dst.data)
        for src, dst in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            dst.data.copy_(self.cfg.tau * src.data + (1.0 - self.cfg.tau) * dst.data)
