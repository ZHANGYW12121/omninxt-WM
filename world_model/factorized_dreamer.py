"""DreamerV3 world-model core backed by :class:`FactorizedRSSM`.

This module is intentionally separate from the original ``dreamer.Dreamer``.
It establishes the factorized posterior, shared task heads, and world-model
loss API first; online replay/optimizer orchestration is added without changing
the vanilla agent.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn
import networks

from factorized_rssm import FactorizedRSSM, FactorizedState
from modules.goal_conditioning import goal_features_torch
from modules.reward_components import RewardComponentHead


class FactorizedDreamer(nn.Module):
    """Factorized world model plus one joint Actor/Critic/Reward/Continue set."""

    def __init__(
        self,
        encoder: nn.Module,
        dynamics: FactorizedRSSM,
        actor: nn.Module,
        value: nn.Module,
        reward: nn.Module,
        cont: nn.Module,
        prediction_heads: nn.Module | None = None,
        *,
        kl_free: float = 1.0,
        horizon: int = 333,
        lamb: float = 0.95,
        act_entropy: float = 3.0e-4,
        slow_target_fraction: float = 0.02,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.rssm = dynamics
        self.actor = actor
        self.value = value
        self.reward = reward
        self.reward_components = RewardComponentHead(dynamics.joint_feat_size)
        self.cont = cont
        self.prediction_heads = prediction_heads
        self.kl_free = float(kl_free)
        self.horizon = int(horizon)
        self.lamb = float(lamb)
        self.act_entropy = float(act_entropy)
        self.slow_target_fraction = float(slow_target_fraction)
        self.slow_value = copy.deepcopy(value)
        self.return_ema = networks.ReturnEMA(device=dynamics.ego_rssm._device)
        for parameter in self.slow_value.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _embeddings(encoded: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # New encoders should expose per-person tokens. A pooled human_embed is
        # deliberately rejected because it cannot drive per-slot Human RSSM.
        human = encoded.get("human_tokens")
        if human is None:
            human = encoded.get("human_embed_slots")
        if human is None:
            raise KeyError("factorized encoder must return human_tokens [B,T,N,E]")
        return {
            "ego": encoded["ego_embed"],
            "human": human,
        }

    @staticmethod
    def _mean_nll(head: nn.Module, feat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dist = head(feat)
        if not hasattr(dist, "log_prob"):
            raise TypeError("Reward/Continue heads must return distribution-like objects")
        loss = -dist.log_prob(target.float())
        return loss.mean()

    def posterior(
        self,
        batch: dict[str, torch.Tensor],
        initial: FactorizedState | None = None,
    ) -> tuple[FactorizedState, dict[str, Any], dict[str, torch.Tensor]]:
        encoded = self.encoder(batch)
        embeds = self._embeddings(encoded)
        human_mask = batch.get("human_mask", encoded.get("human_token_mask"))
        if human_mask is None:
            raise KeyError("batch/encoder must provide human_mask or human_token_mask")
        human_is_first = batch.get("human_is_first")
        if human_is_first is None:
            raise KeyError("batch must provide human_is_first for slot-safe posterior updates")
        batch_size, _, max_people = human_mask.shape
        if "goal" not in batch:
            raise KeyError("batch must provide body-relative goal features")
        if initial is None:
            initial = self.rssm.initial(batch_size, max_people)
        state, aux = self.rssm.observe(
            embeds,
            batch["action"],
            initial,
            batch["is_first"],
            human_mask,
            human_is_first,
            goal=batch["goal"],
        )
        return state, aux, encoded

    def world_model_loss(
        self,
        batch: dict[str, torch.Tensor],
        initial: FactorizedState | None = None,
    ) -> tuple[dict[str, torch.Tensor], FactorizedState, dict[str, Any]]:
        states, aux, encoded = self.posterior(batch, initial)
        losses = self.rssm.kl_loss(
            aux["post_logits"], aux["prior_logits"], aux["human_mask"], self.kl_free,
        )
        joint = aux["joint_feat"]
        losses["rew"] = self._mean_nll(self.reward, joint, batch["reward"])
        if "reward_components" in batch:
            losses["rew_components"] = self.reward_components.loss(
                joint, batch["reward_components"]
            )
        continuation = 1.0 - batch["is_terminal"].float()
        losses["con"] = self._mean_nll(self.cont, joint, continuation)
        return losses, states, {**aux, "encoded": encoded}

    def prediction_loss(self, states: FactorizedState, batch: dict[str, torch.Tensor]):
        if self.prediction_heads is None:
            return {}, {}
        branch_feats = self.rssm.get_branch_feats(states)
        predictions, losses = self.prediction_heads.forward_loss(branch_feats, batch)
        return losses, predictions

    @staticmethod
    def last_state(states: FactorizedState) -> FactorizedState:
        return {
            branch: {key: value[:, -1] for key, value in values.items()}
            for branch, values in states.items()
        }

    def imagine(
        self,
        initial: FactorizedState,
        horizon: int,
        human_mask: torch.Tensor,
        *,
        goal_position: torch.Tensor,
        sample: bool = True,
    ):
        return self._imagine_goal_conditioned(
            initial, horizon, human_mask, goal_position, sample=sample,
        )

    def task_predictions(self, joint_feat: torch.Tensor) -> dict[str, Any]:
        """Shared task heads all read the same joint latent feature."""
        return {
            "actor": self.actor(joint_feat),
            "value": self.value(joint_feat),
            "reward": self.reward(joint_feat),
            "continue": self.cont(joint_feat),
        }

    def initial_agent_state(self, batch_size: int, max_people: int, action_dim: int) -> dict[str, Any]:
        return {
            "latent": self.rssm.initial(batch_size, max_people),
            "prev_action": torch.zeros(batch_size, int(action_dim), device=next(self.parameters()).device),
        }

    @staticmethod
    def replay_state_fields(state: FactorizedState) -> dict[str, torch.Tensor]:
        """Flatten a structured latent for TensorDict/ReplayBuffer storage."""
        return {
            "ego_stoch": state["ego"]["stoch"], "ego_deter": state["ego"]["deter"],
            "human_stoch": state["human"]["stoch"], "human_deter": state["human"]["deter"],
        }

    @staticmethod
    def state_from_replay_fields(fields: dict[str, torch.Tensor]) -> FactorizedState:
        return {
            "ego": {"stoch": fields["ego_stoch"], "deter": fields["ego_deter"]},
            "human": {"stoch": fields["human_stoch"], "deter": fields["human_deter"]},
        }

    @torch.no_grad()
    def act_step(
        self,
        observation: dict[str, torch.Tensor],
        agent_state: dict[str, Any],
        *,
        evaluation: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """One online posterior update followed by one joint policy action.

        Online observations may be ``[B,...]`` or ``[B,1,...]``. Encoders are
        expected to preserve a singleton time dimension when one is provided.
        """
        encoded = self.encoder(observation)
        embeds = self._embeddings(encoded)
        if embeds["ego"].ndim == 3:
            embeds = {key: value[:, 0] for key, value in embeds.items()}
        human_mask = observation.get("human_mask", encoded.get("human_token_mask"))
        human_first = observation.get("human_is_first")
        is_first = observation["is_first"]
        for name, value in (("human_mask", human_mask), ("human_is_first", human_first)):
            if value is None:
                raise KeyError(f"online observation must provide {name}")
        if human_mask.ndim == 3:
            human_mask = human_mask[:, 0]
        if human_first.ndim == 3:
            human_first = human_first[:, 0]
        if is_first.ndim > 1:
            is_first = is_first.reshape(is_first.shape[0], -1).any(dim=1)
        goal = observation.get("goal")
        if goal is None:
            raise KeyError("online observation must provide goal")
        if goal.ndim == 3:
            goal = goal[:, 0]
        latent, aux = self.rssm.obs_step(
            agent_state["latent"], agent_state["prev_action"], embeds, is_first,
            human_mask, human_first, goal=goal,
        )
        dist = self.actor(aux["joint_feat"])
        if torch.is_tensor(dist):
            action = dist
        elif evaluation:
            action = self._dist_mode(dist)
        elif hasattr(dist, "rsample"):
            action = dist.rsample()
        else:
            action = dist.sample()
        return action, {"latent": latent, "prev_action": action}

    def checkpoint_state(self, optimizer=None, scheduler=None, scaler=None) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.state_dict()}
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            payload["scheduler"] = scheduler.state_dict()
        if scaler is not None:
            payload["scaler"] = scaler.state_dict()
        return payload

    def load_checkpoint_state(self, payload, optimizer=None, scheduler=None, scaler=None, strict=True) -> None:
        self.load_state_dict(payload["model"], strict=strict)
        for name, obj in (("optimizer", optimizer), ("scheduler", scheduler), ("scaler", scaler)):
            if obj is not None and name in payload:
                obj.load_state_dict(payload[name])

    @torch.no_grad()
    def update_slow_value(self, fraction: float | None = None) -> None:
        mix = self.slow_target_fraction if fraction is None else float(fraction)
        for source, target in zip(self.value.parameters(), self.slow_value.parameters()):
            target.data.copy_(mix * source.data + (1.0 - mix) * target.data)

    @staticmethod
    def _flatten_state_time(states: FactorizedState) -> FactorizedState:
        return {
            branch: {
                key: value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])
                for key, value in values.items()
            }
            for branch, values in states.items()
        }

    @staticmethod
    def _dist_mode(dist):
        mode = getattr(dist, "mode", None)
        if mode is not None:
            return mode() if callable(mode) else mode
        return dist.mean

    @staticmethod
    def lambda_return(last, terminal, reward, value, bootstrap, discount, lamb):
        if not (last.shape == terminal.shape == reward.shape == value.shape == bootstrap.shape):
            raise ValueError("lambda-return inputs must have identical shapes")
        live = (1.0 - terminal.float())[:, 1:] * discount
        cont = (1.0 - last.float())[:, 1:] * lamb
        interm = reward[:, 1:] + (1.0 - cont) * live * bootstrap[:, 1:]
        outputs = [bootstrap[:, -1]]
        for index in reversed(range(live.shape[1])):
            outputs.append(interm[:, index] + live[:, index] * cont[:, index] * outputs[-1])
        return torch.stack(list(reversed(outputs))[:-1], dim=1)

    def _imagine_goal_conditioned(
        self,
        initial: FactorizedState,
        horizon: int,
        human_mask: torch.Tensor,
        goal_position: torch.Tensor,
        *,
        sample: bool = True,
    ):
        """Roll out coupled priors and recompute Goal token at every step.

        The fixed target remains in the episode-start local frame.  Ego14 is
        decoded from each imagined Ego latent, then converted to an updated
        body-relative Goal feature.  Goal therefore conditions the policy but
        never enters either branch's physical transition.
        """
        if self.prediction_heads is None or not hasattr(self.prediction_heads, "decode_ego_state"):
            raise RuntimeError("goal-conditioned imagination requires the Ego state decoder")
        if int(horizon) <= 0:
            raise ValueError("horizon must be positive")
        if goal_position.ndim != 2 or goal_position.shape[-1] != 3:
            raise ValueError("goal_position must be [B,3]")
        state = initial
        states, joints, actions, goals = [], [], [], []
        for _ in range(int(horizon)):
            ego_feat = self.rssm.get_branch_feats(state)["ego"]
            ego_state = self.prediction_heads.decode_ego_state(ego_feat)
            goal = goal_features_torch(ego_state, goal_position)
            joint, _ = self.rssm.get_joint_feat(state, human_mask, goal)
            action = self.rssm._sample_actor(self.actor, joint, sample)
            states.append(state)
            joints.append(joint)
            actions.append(action)
            goals.append(goal)
            state, _ = self.rssm.img_step(state, action, human_mask)
        return self._stack_imagined_states(states), {
            "joint_feat": torch.stack(joints, 1),
            "action": torch.stack(actions, 1),
            "goal": torch.stack(goals, 1),
            "human_mask": human_mask,
            "final_state": state,
        }

    @staticmethod
    def _stack_imagined_states(states: list[FactorizedState]) -> FactorizedState:
        return {
            branch: {
                key: torch.stack([state[branch][key] for state in states], dim=1)
                for key in ("stoch", "deter")
            }
            for branch in ("ego", "human")
        }

    def actor_critic_loss(
        self,
        posterior_states: FactorizedState,
        posterior_aux: dict[str, Any],
        batch: dict[str, torch.Tensor],
        imag_horizon: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """DreamerV3 imagination and replay-value objectives on joint features."""

        b, t = batch["action"].shape[:2]
        start = self._flatten_state_time(posterior_states)
        mask = posterior_aux["human_mask"].reshape(b * t, -1)
        if "goal_position" not in batch:
            raise KeyError("actor-critic imagination requires goal_position")
        goal_position = batch["goal_position"]
        if goal_position.ndim == 2:
            goal_position = goal_position[:, None].expand(-1, t, -1)
        goal_position = goal_position.reshape(b * t, 3)

        # World-model rollout and sampled actions are targets for policy/value
        # optimization, matching the detached imagination path in DreamerV3.
        with torch.no_grad():
            _, imagined = self._imagine_goal_conditioned(
                start, int(imag_horizon) + 1, mask, goal_position, sample=True,
            )
            imag_feat = imagined["joint_feat"].detach()
            imag_action = imagined["action"].detach()
            imag_reward = self._dist_mode(self.reward(imag_feat))
            imag_cont = self.cont(imag_feat).mean
            imag_value = self._dist_mode(self.value(imag_feat))
            imag_slow_value = self._dist_mode(self.slow_value(imag_feat))
            discount = 1.0 - 1.0 / self.horizon
            weight = torch.cumprod(imag_cont * discount, dim=1)
            ret = self.lambda_return(
                torch.zeros_like(imag_cont), 1.0 - imag_cont, imag_reward,
                imag_value, imag_value, discount, self.lamb,
            )
            ret_offset, ret_scale = self.return_ema(ret)
            advantage = (ret - imag_value[:, :-1]) / ret_scale

        policy_dist = self.actor(imag_feat)
        if not hasattr(policy_dist, "log_prob"):
            raise TypeError("Actor must return a distribution for Dreamer policy loss")
        log_prob = policy_dist.log_prob(imag_action)[:, :-1]
        entropy = policy_dist.entropy()[:, :-1]
        if log_prob.ndim == 2:
            log_prob = log_prob[..., None]
        if entropy.ndim == 2:
            entropy = entropy[..., None]
        policy_loss = (
            weight[:, :-1].detach()
            * -(log_prob * advantage.detach() + self.act_entropy * entropy)
        ).mean()

        value_dist = self.value(imag_feat)
        padded_return = torch.cat((ret, torch.zeros_like(ret[:, -1:])), dim=1)
        value_nll = -value_dist.log_prob(padded_return.detach())
        slow_nll = -value_dist.log_prob(imag_slow_value.detach())
        if value_nll.ndim == 2:
            value_nll = value_nll[..., None]
            slow_nll = slow_nll[..., None]
        value_loss = (weight[:, :-1].detach() * (value_nll + slow_nll)[:, :-1]).mean()

        # Replay value keeps gradients through posterior joint features and thus
        # through both world-model branches and their policy interaction.
        replay_feat = posterior_aux["joint_feat"]
        replay_value = self._dist_mode(self.value(replay_feat))
        replay_slow = self._dist_mode(self.slow_value(replay_feat))
        boot = ret[:, 0].reshape(b, t, *ret.shape[2:])
        replay_return = self.lambda_return(
            batch["is_last"].float(), batch["is_terminal"].float(), batch["reward"].float(),
            replay_value.detach(), boot.detach(), 1.0 - 1.0 / self.horizon, self.lamb,
        )
        replay_padded = torch.cat((replay_return, torch.zeros_like(replay_return[:, -1:])), dim=1)
        replay_dist = self.value(replay_feat)
        replay_nll = -replay_dist.log_prob(replay_padded.detach())
        replay_slow_nll = -replay_dist.log_prob(replay_slow.detach())
        if replay_nll.ndim == 2:
            replay_nll = replay_nll[..., None]
            replay_slow_nll = replay_slow_nll[..., None]
        replay_weight = 1.0 - batch["is_last"].float()
        repval_loss = (replay_weight[:, :-1] * (replay_nll + replay_slow_nll)[:, :-1]).mean()
        return {"policy": policy_loss, "value": value_loss, "repval": repval_loss}, {
            "imag_reward": imag_reward.mean(), "imag_continue": imag_cont.mean(),
            "imag_value": imag_value.mean(), "action_entropy": entropy.mean(),
            "imag_return": ret.mean(), "return_normalized": ((ret - ret_offset) / ret_scale).mean(),
            "return_005": self.return_ema.ema_vals[0], "return_095": self.return_ema.ema_vals[1],
            "advantage": advantage.mean(), "advantage_std": advantage.std(),
        }

    def train(self, mode: bool = True):
        super().train(mode)
        self.slow_value.train(False)
        return self
