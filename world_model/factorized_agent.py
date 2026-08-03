"""OnlineTrainer-compatible construction and lifecycle for FactorizedDreamer."""

from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

import torch
from tensordict import TensorDict
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

import networks
from factorized_dreamer import FactorizedDreamer
from factorized_rssm import FactorizedRSSM
from factorized_trainer import FactorizedTrainStep
from modules.factorized_encoders import FactorizedEncoderConfig, FactorizedObservationEncoder
from modules.factorized_prediction_heads import FactorizedPredictionConfig, FactorizedPredictionHeads
from modules.latent_policy_attention import LatentPolicyAttentionConfig
from modules.sparse_ego_human_attention import SparseEgoHumanAttentionConfig
from optim import LaProp


def _shape(space):
    return tuple(int(x) for x in space.shape)


class FactorizedDreamerAgent(nn.Module):
    """Adapter exposing the same act/update/state surface used by OnlineTrainer."""

    is_factorized = True

    def __init__(self, config, obs_space, act_space) -> None:
        super().__init__()
        self.device = torch.device(config.device)
        self.act_dim = act_space.n if hasattr(act_space, "n") else sum(act_space.shape)
        fcfg = config.factorized
        private_dim = int(fcfg.private_dim)
        if "goal" not in obs_space.spaces:
            raise ValueError("two-branch factorized observations require goal [G]")
        goal_dim = _shape(obs_space.spaces["goal"])[-1]
        encoder = FactorizedObservationEncoder(FactorizedEncoderConfig(
            model_dim=private_dim,
            ego_state_mean=tuple(fcfg.ego_state_mean) if fcfg.get("ego_state_mean") is not None else None,
            ego_state_std=tuple(fcfg.ego_state_std) if fcfg.get("ego_state_std") is not None else None,
            human_root_dim=int(fcfg.human_root_dim),
            human_feat_dim=int(fcfg.get("human_feature_dim", 7)),
            use_stgcn=True, pose_history=int(fcfg.pose_history),
            observation_heads=int(fcfg.observation_attention.num_heads),
            observation_ff_mult=int(fcfg.observation_attention.ff_mult),
        ))
        latent_cfg = LatentPolicyAttentionConfig(
            model_dim=int(fcfg.latent_policy_attention.model_dim),
            num_heads=int(fcfg.latent_policy_attention.num_heads),
            num_layers=int(fcfg.latent_policy_attention.num_layers),
            ff_mult=int(fcfg.latent_policy_attention.ff_mult),
        )
        dynamics = FactorizedRSSM(
            config.rssm, {"ego": private_dim, "human": private_dim}, self.act_dim,
            goal_dim=goal_dim, latent_attention_config=latent_cfg,
            coupling_attention_config=SparseEgoHumanAttentionConfig(
                model_dim=int(fcfg.prior_coupling.model_dim),
                num_heads=int(fcfg.prior_coupling.num_heads),
                ff_mult=int(fcfg.prior_coupling.ff_mult),
            ),
        )
        joint_dim = dynamics.joint_feat_size
        config.actor.shape = (act_space.n,) if hasattr(act_space, "n") else _shape(act_space)
        if hasattr(act_space, "multi_discrete"):
            config.actor.dist = config.actor.dist.multi_disc
        elif hasattr(act_space, "discrete"):
            config.actor.dist = config.actor.dist.disc
        else:
            config.actor.dist = config.actor.dist.cont
        actor = networks.MLPHead(config.actor, joint_dim)
        value = networks.MLPHead(config.critic, joint_dim)
        reward = networks.MLPHead(config.reward, joint_dim)
        cont = networks.MLPHead(config.cont, joint_dim)

        prediction_cfg = FactorizedPredictionConfig(
            hidden_dim=int(config.get("units", 256)), num_joints=int(fcfg.num_joints),
        )
        prediction = FactorizedPredictionHeads(
            dynamics.ego_rssm.feat_size, dynamics.human_rssm.feat_size, prediction_cfg,
        )
        self.model = FactorizedDreamer(
            encoder, dynamics, actor, value, reward, cont, prediction,
            kl_free=float(config.kl_free), horizon=int(config.horizon), lamb=float(config.lamb),
            act_entropy=float(config.act_entropy), slow_target_fraction=float(config.slow_target_fraction),
        )
        self._named_params = OrderedDict(self.model.named_parameters())
        self._optimizer = LaProp(
            self._named_params.values(), lr=config.lr,
            betas=(config.beta1, config.beta2), eps=config.eps,
        )
        self._scheduler = LambdaLR(
            self._optimizer,
            lr_lambda=lambda step: min(1.0, (step + 1) / config.warmup) if config.warmup else 1.0,
        )
        self._train_step = FactorizedTrainStep(
            self.model, self._optimizer, imag_horizon=int(config.imag_horizon),
            loss_scales=dict(config.loss_scales), agc=float(config.agc), pmin=float(config.pmin),
            amp_device=str(config.device),
            slow_target_update=int(config.slow_target_update),
        )
        self.max_people = int(fcfg.max_people)

    def _state_dict(self, state):
        return self.model.state_from_replay_fields({key: state[key] for key in (
            "ego_stoch", "ego_deter", "human_stoch", "human_deter")})

    @torch.no_grad()
    def get_initial_state(self, batch_size):
        latent = self.model.rssm.initial(batch_size, self.max_people)
        fields = self.model.replay_state_fields(latent)
        fields["prev_action"] = torch.zeros(batch_size, self.act_dim, device=self.device)
        return TensorDict(fields, batch_size=(batch_size,))

    @torch.no_grad()
    def act(self, obs, state, eval=False):
        latent = self._state_dict(state)
        action, next_state = self.model.act_step(
            obs, {"latent": latent, "prev_action": state["prev_action"]}, evaluation=eval,
        )
        fields = self.model.replay_state_fields(next_state["latent"])
        fields["prev_action"] = action
        return action, TensorDict(fields, batch_size=state.batch_size)

    def update(self, replay_buffer):
        data, index, initial_fields = replay_buffer.sample()
        initial = self.model.state_from_replay_fields(initial_fields)
        metrics = self._train_step(data, initial)
        self._scheduler.step()
        metrics["opt/lr"] = float(self._scheduler.get_last_lr()[0])
        if hasattr(replay_buffer, "update_factorized"):
            replay_buffer.update_factorized(
                index, self.model.replay_state_fields(self._train_step.last_states)
            )
        return metrics

    def training_state_dict(self):
        return {
            "optimizer": self._optimizer.state_dict(),
            "scheduler": self._scheduler.state_dict(),
            "scaler": self._train_step.scaler.state_dict(),
            "update_count": self._train_step.update_count,
        }

    def load_training_state_dict(self, payload):
        if "optimizer" in payload: self._optimizer.load_state_dict(payload["optimizer"])
        if "scheduler" in payload: self._scheduler.load_state_dict(payload["scheduler"])
        if "scaler" in payload: self._train_step.scaler.load_state_dict(payload["scaler"])
        self._train_step.update_count = int(payload.get("update_count", 0))

    @torch.no_grad()
    def video_pred(self, data, initial):
        raise NotImplementedError("Factorized model logs BEV/skeleton open-loop metrics instead of RGB video")
