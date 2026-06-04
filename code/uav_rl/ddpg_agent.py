"""Minimal DDPG implementation for online ArduPilot training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .td3_agent import Actor, Critic, ReplayBuffer


@dataclass
class DDPGConfig:
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    exploration_noise: float = 0.1
    batch_size: int = 256
    replay_size: int = 200000


class DDPGAgent:
    """Deterministic actor-critic with one critic and target networks.

    This keeps the same actor / critic network shape as the TD3 agent so the
    comparison changes the algorithm, not the model capacity.
    """

    def __init__(self, obs_dim: int, act_dim: int, act_limit: float, config: DDPGConfig, device: torch.device):
        self.cfg = config
        self.device = device
        self.act_limit = float(act_limit)
        self.actor = Actor(obs_dim, act_dim, act_limit).to(device)
        self.actor_target = Actor(obs_dim, act_dim, act_limit).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = Critic(obs_dim, act_dim).to(device)
        self.critic_target = Critic(obs_dim, act_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
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
            next_action = self.actor_target(batch["next_obs"])
            target_q = self.critic_target(batch["next_obs"], next_action)
            target_q = batch["rewards"] + (1.0 - batch["dones"]) * self.cfg.gamma * target_q

        current_q = self.critic(batch["obs"], batch["actions"])
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss = -self.critic(batch["obs"], self.actor(batch["obs"])).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self._soft_update()

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "q_mean": float(current_q.mean().item()),
        }

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "total_updates": self.total_updates,
            "config": vars(self.cfg),
        }

    def load_state_dict(self, payload: dict) -> None:
        self.actor.load_state_dict(payload["actor"])
        self.actor_target.load_state_dict(payload["actor_target"])
        self.critic.load_state_dict(payload["critic"])
        self.critic_target.load_state_dict(payload["critic_target"])
        self.actor_opt.load_state_dict(payload["actor_opt"])
        self.critic_opt.load_state_dict(payload["critic_opt"])
        self.total_updates = int(payload.get("total_updates", 0))

    def _soft_update(self) -> None:
        for src, dst in zip(self.actor.parameters(), self.actor_target.parameters()):
            dst.data.copy_(self.cfg.tau * src.data + (1.0 - self.cfg.tau) * dst.data)
        for src, dst in zip(self.critic.parameters(), self.critic_target.parameters()):
            dst.data.copy_(self.cfg.tau * src.data + (1.0 - self.cfg.tau) * dst.data)
