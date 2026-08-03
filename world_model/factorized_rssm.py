"""Conditionally coupled two-branch Ego/Human RSSM."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

import rssm
from modules.latent_policy_attention import ActionTokenLatentAttention, LatentPolicyAttentionConfig
from modules.sparse_ego_human_attention import (
    SparseEgoHumanAttention,
    SparseEgoHumanAttentionConfig,
)


BranchState = dict[str, torch.Tensor]
FactorizedState = dict[str, BranchState]


def _stack_states(states: list[FactorizedState]) -> FactorizedState:
    return {
        branch: {
            key: torch.stack([state[branch][key] for state in states], 1)
            for key in ("stoch", "deter")
        }
        for branch in ("ego", "human")
    }


class FactorizedRSSM(nn.Module):
    """Separate Ego/Human states with causal, asymmetric prior coupling.

    The recurrent parameters remain factorized.  Before each transition the
    current latent states are fused as follows: Ego reads all valid Humans;
    Human n reads only Ego and itself.  The resulting context is concatenated
    with the drone action and passed to that branch's transition.  This keeps
    imagination observation-free while avoiding an information mismatch with
    the observation-contextualized posteriors.
    """

    def __init__(self, rssm_config: Any, embed_sizes: dict[str, int], action_dim: int, *,
                 goal_dim: int = 8,
                 latent_attention_config: LatentPolicyAttentionConfig | None = None,
                 coupling_attention_config: SparseEgoHumanAttentionConfig | None = None,
                 joint_feat_dim: int | None = None, **_legacy_kwargs) -> None:
        super().__init__()
        if set(embed_sizes) != {"ego", "human"}:
            raise ValueError("embed_sizes must contain exactly ego/human")
        self.action_dim = int(action_dim)
        self.goal_dim = int(goal_dim)
        policy_cfg = latent_attention_config or LatentPolicyAttentionConfig(
            model_dim=int(joint_feat_dim or 128)
        )
        coupling_cfg = coupling_attention_config or SparseEgoHumanAttentionConfig(
            model_dim=policy_cfg.model_dim,
            num_heads=policy_cfg.num_heads,
            ff_mult=policy_cfg.ff_mult,
            dropout=policy_cfg.dropout,
        )
        self.coupling_dim = int(coupling_cfg.model_dim)
        transition_dim = self.action_dim + self.coupling_dim
        self.ego_rssm = rssm.RSSM(rssm_config, int(embed_sizes["ego"]), transition_dim)
        self.human_rssm = rssm.RSSM(rssm_config, int(embed_sizes["human"]), transition_dim)
        self.ego_coupling_projector = nn.Sequential(
            nn.LayerNorm(self.ego_rssm.feat_size),
            nn.Linear(self.ego_rssm.feat_size, self.coupling_dim),
        )
        self.human_coupling_projector = nn.Sequential(
            nn.LayerNorm(self.human_rssm.feat_size),
            nn.Linear(self.human_rssm.feat_size, self.coupling_dim),
        )
        self.latent_coupling_attention = SparseEgoHumanAttention(coupling_cfg)
        self.latent_policy_attention = ActionTokenLatentAttention(
            self.ego_rssm.feat_size, self.human_rssm.feat_size, self.goal_dim,
            config=policy_cfg,
        )
        self.joint_feat_size = int(policy_cfg.model_dim)
        self.branch_feat_size = self.ego_rssm.feat_size

    def initial(self, batch_size: int, max_people: int, **_ignored) -> FactorizedState:
        ego = self.ego_rssm.initial(batch_size)
        human = self.human_rssm.initial(batch_size * max_people)
        return {
            "ego": {"stoch": ego[0], "deter": ego[1]},
            "human": {
                "stoch": human[0].reshape(batch_size, max_people, *human[0].shape[1:]),
                "deter": human[1].reshape(batch_size, max_people, -1),
            },
        }

    def get_branch_feats(self, state: FactorizedState) -> dict[str, torch.Tensor]:
        return {
            "ego": self.ego_rssm.get_feat(**state["ego"]),
            "human": self.human_rssm.get_feat(**state["human"]),
        }

    def get_transition_context(self, state: FactorizedState,
                               human_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.get_branch_feats(state)
        coupled = self.latent_coupling_attention(
            self.ego_coupling_projector(feats["ego"]),
            self.human_coupling_projector(feats["human"]),
            human_mask.bool(),
        )
        return {
            "ego": coupled["ego"][:, 0],
            "human": coupled["human"],
            "attention": coupled,
        }

    def get_joint_feat(self, state: FactorizedState, human_mask: torch.Tensor,
                       goal: torch.Tensor, **_ignored):
        feats = self.get_branch_feats(state)
        result = self.latent_policy_attention(
            goal, feats["ego"], feats["human"], human_mask.bool(),
        )
        return result["joint_feat"], result

    @staticmethod
    def _reset_mask(is_first: torch.Tensor) -> torch.Tensor:
        reset = is_first.bool()
        return reset.reshape(reset.shape[0], -1).any(1) if reset.ndim > 1 else reset

    def _transition_inputs(self, state: FactorizedState, action: torch.Tensor,
                           human_mask: torch.Tensor,
                           human_reset: torch.Tensor | None = None):
        hmask = human_mask.bool()
        batch, people = hmask.shape
        context_state = state
        if human_reset is not None:
            # A reassigned/reappearing slot must not leak its previous person's
            # latent into the Ego prior before the Human RSSM reset is applied.
            reset_slots = human_reset.bool()
            context_state = {
                "ego": state["ego"],
                "human": {
                    "stoch": state["human"]["stoch"].masked_fill(
                        reset_slots[..., None, None], 0.0),
                    "deter": state["human"]["deter"].masked_fill(
                        reset_slots[..., None], 0.0),
                },
            }
        context = self.get_transition_context(context_state, hmask)
        ego_input = torch.cat((action, context["ego"]), dim=-1)
        human_action = action[:, None].expand(-1, people, -1)
        human_input = torch.cat((human_action, context["human"]), dim=-1)
        return ego_input, human_input.reshape(batch * people, -1), context

    def obs_step(self, prev_state: FactorizedState, prev_action: torch.Tensor,
                 embeddings: dict[str, torch.Tensor], is_first: torch.Tensor,
                 human_mask: torch.Tensor, human_is_first: torch.Tensor, *,
                 goal: torch.Tensor, **_ignored):
        hmask = human_mask.bool()
        batch, people = hmask.shape
        reset = self._reset_mask(is_first).to(prev_action.device)
        human_reset_matrix = reset[:, None] | human_is_first.bool() | ~hmask
        ego_input, human_input, coupling = self._transition_inputs(
            prev_state, prev_action, hmask, human_reset=human_reset_matrix,
        )
        ego_stoch, ego_deter, ego_post = self.ego_rssm.obs_step(
            prev_state["ego"]["stoch"], prev_state["ego"]["deter"], ego_input,
            embeddings["ego"], reset,
        )
        human_reset = human_reset_matrix.reshape(-1)
        human_stoch, human_deter, human_post = self.human_rssm.obs_step(
            prev_state["human"]["stoch"].reshape(
                batch * people, *prev_state["human"]["stoch"].shape[2:]),
            prev_state["human"]["deter"].reshape(batch * people, -1),
            human_input, embeddings["human"].reshape(batch * people, -1), human_reset,
        )
        human_stoch = human_stoch.reshape(batch, people, *human_stoch.shape[1:])
        human_deter = human_deter.reshape(batch, people, -1)
        human_post = human_post.reshape(batch, people, *human_post.shape[1:])
        human_stoch = human_stoch.masked_fill(~hmask[..., None, None], 0.0)
        human_deter = human_deter.masked_fill(~hmask[..., None], 0.0)
        human_post = human_post.masked_fill(~hmask[..., None, None], 0.0)
        state = {
            "ego": {"stoch": ego_stoch, "deter": ego_deter},
            "human": {"stoch": human_stoch, "deter": human_deter},
        }
        joint, latent = self.get_joint_feat(state, hmask, goal)
        _, ego_prior = self.ego_rssm.prior(ego_deter)
        _, human_prior = self.human_rssm.prior(human_deter.reshape(batch * people, -1))
        human_prior = human_prior.reshape(batch, people, *human_prior.shape[1:])
        human_prior = human_prior.masked_fill(~hmask[..., None, None], 0.0)
        return state, {
            "post_logits": {"ego": ego_post, "human": human_post},
            "prior_logits": {"ego": ego_prior, "human": human_prior},
            "joint_feat": joint,
            "latent_attention": latent,
            "coupling_attention": coupling["attention"],
            "human_mask": hmask,
            "goal": goal,
        }

    def observe(self, embeddings, actions, initial, is_first, human_mask,
                human_is_first, *, goal, **_ignored):
        states, joints, latent, coupling = [], [], [], []
        post = {key: [] for key in ("ego", "human")}
        prior = {key: [] for key in post}
        state = initial
        for t in range(actions.shape[1]):
            state, aux = self.obs_step(
                state, actions[:, t], {key: value[:, t] for key, value in embeddings.items()},
                is_first[:, t], human_mask[:, t], human_is_first[:, t], goal=goal[:, t],
            )
            states.append(state)
            joints.append(aux["joint_feat"])
            latent.append(aux["latent_attention"])
            coupling.append(aux["coupling_attention"])
            for key in post:
                post[key].append(aux["post_logits"][key])
                prior[key].append(aux["prior_logits"][key])
        return _stack_states(states), {
            "post_logits": {key: torch.stack(value, 1) for key, value in post.items()},
            "prior_logits": {key: torch.stack(value, 1) for key, value in prior.items()},
            "joint_feat": torch.stack(joints, 1),
            "latent_attention": latent,
            "coupling_attention": coupling,
            "human_mask": human_mask.bool(),
            "goal": goal,
        }

    @staticmethod
    def _masked_kl(model, post, prior, mask, free):
        dyn, rep = model.kl_loss(post, prior, free)
        weight = mask.to(dyn)
        return ((dyn * weight).sum() / weight.sum().clamp_min(1),
                (rep * weight).sum() / weight.sum().clamp_min(1))

    def kl_loss(self, post_logits, prior_logits, human_mask, free, **_ignored):
        dyn, rep = self.ego_rssm.kl_loss(
            post_logits["ego"], prior_logits["ego"], free,
        )
        dyn_human, rep_human = self._masked_kl(
            self.human_rssm, post_logits["human"], prior_logits["human"],
            human_mask, free,
        )
        return {
            "dyn_ego": dyn.mean(), "rep_ego": rep.mean(),
            "dyn_human": dyn_human, "rep_human": rep_human,
        }

    def img_step(self, state, action, human_mask, **_ignored):
        hmask = human_mask.bool()
        batch, people = hmask.shape
        ego_input, human_input, coupling = self._transition_inputs(state, action, hmask)
        ego_stoch, ego_deter = self.ego_rssm.img_step(
            state["ego"]["stoch"], state["ego"]["deter"], ego_input,
        )
        human_stoch, human_deter = self.human_rssm.img_step(
            state["human"]["stoch"].reshape(
                batch * people, *state["human"]["stoch"].shape[2:]),
            state["human"]["deter"].reshape(batch * people, -1), human_input,
        )
        human_stoch = human_stoch.reshape(batch, people, *human_stoch.shape[1:])
        human_deter = human_deter.reshape(batch, people, -1)
        human_stoch = human_stoch.masked_fill(~hmask[..., None, None], 0.0)
        human_deter = human_deter.masked_fill(~hmask[..., None], 0.0)
        return {
            "ego": {"stoch": ego_stoch, "deter": ego_deter},
            "human": {"stoch": human_stoch, "deter": human_deter},
        }, {"human_mask": hmask, "coupling_attention": coupling["attention"]}

    @staticmethod
    def _sample_actor(actor, feat, sample):
        out = actor(feat)
        if torch.is_tensor(out):
            return out
        if sample and hasattr(out, "rsample"):
            return out.rsample()
        mode = getattr(out, "mode", None)
        if mode is not None:
            return mode() if callable(mode) else mode
        return out.sample()

    def imagine(self, initial, actor, horizon, human_mask, *, goal, sample=True,
                **_ignored):
        """Compatibility rollout for a supplied goal-feature sequence/constant.

        Full Dreamer imagination recomputes relative goal features from the
        decoded imagined Ego state in :class:`FactorizedDreamer`.
        """
        if int(horizon) <= 0:
            raise ValueError("horizon must be positive")
        state, states, joints, actions = initial, [], [], []
        for step in range(int(horizon)):
            step_goal = goal[:, step] if goal.ndim == 3 else goal
            joint, _ = self.get_joint_feat(state, human_mask, step_goal)
            action = self._sample_actor(actor, joint, sample)
            states.append(state)
            joints.append(joint)
            actions.append(action)
            state, _ = self.img_step(state, action, human_mask)
        return _stack_states(states), {
            "joint_feat": torch.stack(joints, 1),
            "action": torch.stack(actions, 1),
            "human_mask": human_mask,
            "final_state": state,
        }
