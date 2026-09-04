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
    source latent states are fused as follows: Ego reads all Humans that were
    valid in the source observation; Human n reads only Ego and itself.  A
    destination-frame appearance mask must never enter the Ego prior because it
    is not known when the source action is selected.  Newly assigned Human
    slots use a reset Human context, independently of the causal Ego context.
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

    def initial(
        self, batch_size: int, max_people: int, *, sample_state: bool = True,
        **_ignored,
    ) -> FactorizedState:
        ego = self.ego_rssm.initial(batch_size, sample=sample_state)
        human = self.human_rssm.initial(
            batch_size * max_people, sample=sample_state)
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
        ego_token = self.ego_coupling_projector(feats["ego"])
        human_tokens = self.human_coupling_projector(feats["human"])
        coupled = self.latent_coupling_attention(
            ego_token,
            human_tokens,
            human_mask.bool(),
        )
        # The RSSM core already receives its own previous stochastic and
        # deterministic state directly.  Feeding the attention residual token
        # back as additional "action" duplicates that identity recurrence and
        # creates a second, un-gated Jacobian path through every observation
        # step.  The coupling input must be the learned interaction increment,
        # while the direct branch recurrence remains owned by Deter.
        ego_interaction = coupled["ego"][:, 0] - ego_token
        human_interaction = coupled["human"] - human_tokens
        return {
            "ego": ego_interaction,
            "human": human_interaction,
            "attention": coupled,
        }

    def get_joint_feat(self, state: FactorizedState, human_mask: torch.Tensor,
                       goal: torch.Tensor, **_ignored):
        feats = self.get_branch_feats(state)
        result = self.latent_policy_attention(
            goal, feats["ego"], feats["human"], human_mask.bool(),
        )
        weights = human_mask.to(result["human_tokens"].dtype)[..., None]
        human_pool = (
            (result["human_tokens"] * weights).sum(dim=-2)
            / weights.sum(dim=-2).clamp_min(1.0)
        )
        result["human_pool"] = human_pool
        result["actor_feat"] = torch.cat((
            result["private_goal_token"], result["private_ego_token"],
            result["joint_feat"], human_pool,
        ), dim=-1)
        return result["joint_feat"], result

    @staticmethod
    def _reset_mask(is_first: torch.Tensor) -> torch.Tensor:
        reset = is_first.bool()
        return reset.reshape(reset.shape[0], -1).any(1) if reset.ndim > 1 else reset

    def _transition_inputs(self, state: FactorizedState, action: torch.Tensor,
                           human_mask: torch.Tensor,
                           human_reset: torch.Tensor | None = None, *,
                           context_human_mask: torch.Tensor | None = None):
        hmask = human_mask.bool()
        batch, people = hmask.shape
        source_mask = (
            hmask if context_human_mask is None
            else context_human_mask.bool())
        if source_mask.shape != hmask.shape:
            raise ValueError(
                "context_human_mask must match destination human_mask [B,N]")
        human_context_state = state
        if human_reset is not None:
            # A reassigned/reappearing slot must not leak its previous person's
            # latent into the new Human branch.  Do not apply this reset to the
            # Ego context: that previous person belongs to the source state and
            # is causal for the transition that has already begun.
            reset_slots = human_reset.bool()
            human_context_state = {
                "ego": state["ego"],
                "human": {
                    "stoch": state["human"]["stoch"].masked_fill(
                        reset_slots[..., None, None], 0.0),
                    "deter": state["human"]["deter"].masked_fill(
                        reset_slots[..., None], 0.0),
                },
            }
        ego_context = self.get_transition_context(state, source_mask)
        if human_reset is None and torch.equal(source_mask, hmask):
            human_context = ego_context
        else:
            human_context = self.get_transition_context(
                human_context_state, hmask)
        ego_input = torch.cat((action, ego_context["ego"]), dim=-1)
        # Human observations and recurrent states live in the moving UAV body
        # frame. The applied UAV action is therefore a genuine cause of their
        # next representation even though pedestrian intent is exogenous. A
        # stop-gradient would leave the forward rollout action-dependent while
        # giving the Actor the derivative of a different model. The structured
        # Human head separately constrains physical root/joint motion to the
        # exact frame transform plus bounded Human-velocity corrections.
        human_action = action[:, None].expand(-1, people, -1)
        human_input = torch.cat(
            (human_action, human_context["human"]), dim=-1)
        context = {
            "ego": ego_context["ego"],
            "human": human_context["human"],
            # Preserve the historical diagnostic key for the causal source
            # attention and expose the reset-aware Human attention separately.
            "attention": ego_context["attention"],
            "human_attention": human_context["attention"],
        }
        return ego_input, human_input.reshape(batch * people, -1), context

    def obs_step(self, prev_state: FactorizedState, prev_action: torch.Tensor,
                 embeddings: dict[str, torch.Tensor], is_first: torch.Tensor,
                 human_mask: torch.Tensor, human_is_first: torch.Tensor, *,
                 goal: torch.Tensor,
                 previous_human_mask: torch.Tensor | None = None,
                 sample_state: bool = True,
                 **_ignored):
        hmask = human_mask.bool()
        batch, people = hmask.shape
        reset = self._reset_mask(is_first).to(prev_action.device)
        human_reset_matrix = reset[:, None] | human_is_first.bool() | ~hmask
        if previous_human_mask is None:
            # Backwards-compatible single-step callers cannot recover a
            # disappeared/recycled source identity.  They can still avoid
            # destination-frame look-ahead by admitting only continuing slots.
            source_hmask = hmask & ~human_is_first.bool()
        else:
            source_hmask = previous_human_mask.bool()
            if source_hmask.shape != hmask.shape:
                raise ValueError(
                    "previous_human_mask must match human_mask [B,N]")
        source_hmask = source_hmask & ~reset[:, None]
        ego_input, human_input, coupling = self._transition_inputs(
            prev_state, prev_action, hmask,
            human_reset=human_reset_matrix,
            context_human_mask=source_hmask,
        )
        ego_stoch, ego_deter, ego_post = self.ego_rssm.obs_step(
            prev_state["ego"]["stoch"], prev_state["ego"]["deter"], ego_input,
            embeddings["ego"], reset, sample=sample_state,
        )
        human_reset = human_reset_matrix.reshape(-1)
        human_stoch, human_deter, human_post = self.human_rssm.obs_step(
            prev_state["human"]["stoch"].reshape(
                batch * people, *prev_state["human"]["stoch"].shape[2:]),
            prev_state["human"]["deter"].reshape(batch * people, -1),
            human_input, embeddings["human"].reshape(batch * people, -1),
            human_reset, sample=sample_state,
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
        # Only logits are consumed here. Use the categorical mode so this
        # diagnostic likelihood does not advance the RNG with discarded
        # samples or perturb a later Actor/world draw.
        _, ego_prior = self.ego_rssm.prior(ego_deter, sample=False)
        _, human_prior = self.human_rssm.prior(
            human_deter.reshape(batch * people, -1), sample=False)
        human_prior = human_prior.reshape(batch, people, *human_prior.shape[1:])
        human_prior = human_prior.masked_fill(~hmask[..., None, None], 0.0)
        return state, {
            "post_logits": {"ego": ego_post, "human": human_post},
            "prior_logits": {"ego": ego_prior, "human": human_prior},
            "joint_feat": joint,
            "actor_feat": latent["actor_feat"],
            "latent_attention": latent,
            "coupling_attention": coupling["attention"],
            "human_mask": hmask,
            "goal": goal,
        }

    def observe(self, embeddings, actions, initial, is_first, human_mask,
                human_is_first, *, goal, sample_state: bool = True,
                **_ignored):
        states, joints, actor_feats, latent, coupling = [], [], [], [], []
        post = {key: [] for key in ("ego", "human")}
        prior = {key: [] for key in post}
        state = initial
        previous_human_mask = torch.zeros_like(human_mask[:, 0], dtype=torch.bool)
        for t in range(actions.shape[1]):
            state, aux = self.obs_step(
                state, actions[:, t], {key: value[:, t] for key, value in embeddings.items()},
                is_first[:, t], human_mask[:, t], human_is_first[:, t],
                goal=goal[:, t], previous_human_mask=previous_human_mask,
                sample_state=sample_state,
            )
            states.append(state)
            joints.append(aux["joint_feat"])
            actor_feats.append(aux["actor_feat"])
            latent.append(aux["latent_attention"])
            coupling.append(aux["coupling_attention"])
            for key in post:
                post[key].append(aux["post_logits"][key])
                prior[key].append(aux["prior_logits"][key])
            previous_human_mask = human_mask[:, t].bool()
        return _stack_states(states), {
            "post_logits": {key: torch.stack(value, 1) for key, value in post.items()},
            "prior_logits": {key: torch.stack(value, 1) for key, value in prior.items()},
            "joint_feat": torch.stack(joints, 1),
            "actor_feat": torch.stack(actor_feats, 1),
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

    def kl_loss(
        self, post_logits, prior_logits, human_mask, free,
        sequence_valid=None, **_ignored,
    ):
        dyn, rep = self.ego_rssm.kl_loss(
            post_logits["ego"], prior_logits["ego"], free,
        )
        if sequence_valid is None:
            ego_weight = torch.ones_like(dyn)
        else:
            ego_weight = sequence_valid.bool().reshape(
                *sequence_valid.shape[:2], -1).all(-1).to(dyn)
            while ego_weight.ndim < dyn.ndim:
                ego_weight = ego_weight.unsqueeze(-1)
            ego_weight = ego_weight.expand_as(dyn)
        valid_human_mask = human_mask
        if sequence_valid is not None:
            row_valid = sequence_valid.bool()
            while row_valid.ndim < human_mask.ndim:
                row_valid = row_valid.unsqueeze(-1)
            valid_human_mask = human_mask.bool() & row_valid
        dyn_human, rep_human = self._masked_kl(
            self.human_rssm, post_logits["human"], prior_logits["human"],
            valid_human_mask, free,
        )
        return {
            "dyn_ego": (dyn * ego_weight).sum() / ego_weight.sum().clamp_min(1),
            "rep_ego": (rep * ego_weight).sum() / ego_weight.sum().clamp_min(1),
            "dyn_human": dyn_human, "rep_human": rep_human,
        }

    def img_step(
        self, state, action, human_mask, *, sample_state: bool = True,
        **_ignored,
    ):
        hmask = human_mask.bool()
        batch, people = hmask.shape
        ego_input, human_input, coupling = self._transition_inputs(
            state, action, hmask)
        ego_stoch, ego_deter = self.ego_rssm.img_step(
            state["ego"]["stoch"], state["ego"]["deter"], ego_input,
            sample=sample_state,
        )
        human_stoch, human_deter = self.human_rssm.img_step(
            state["human"]["stoch"].reshape(
                batch * people, *state["human"]["stoch"].shape[2:]),
            state["human"]["deter"].reshape(batch * people, -1), human_input,
            sample=sample_state,
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
        state, states, joints, actor_feats, actions = initial, [], [], [], []
        for step in range(int(horizon)):
            step_goal = goal[:, step] if goal.ndim == 3 else goal
            joint, latent = self.get_joint_feat(state, human_mask, step_goal)
            actor_feat = latent["actor_feat"]
            actor_input = (
                actor_feat
                if getattr(actor, "input_dim", None) == actor_feat.shape[-1]
                else joint)
            action = self._sample_actor(actor, actor_input, sample)
            states.append(state)
            joints.append(joint)
            actor_feats.append(actor_feat)
            actions.append(action)
            state, _ = self.img_step(state, action, human_mask)
        return _stack_states(states), {
            "joint_feat": torch.stack(joints, 1),
            "actor_feat": torch.stack(actor_feats, 1),
            "action": torch.stack(actions, 1),
            "human_mask": human_mask,
            "final_state": state,
        }
