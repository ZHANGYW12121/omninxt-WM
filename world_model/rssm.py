from collections.abc import Mapping

import torch
from torch import distributions as torchd
from torch import nn

import distributions as dists
from modules.structured_posterior import StructuredPosteriorAdapter, flatten_structured_embed, structured_embed_size
from networks import BlockLinear, LambdaLayer
from tools import rpad, weight_init_


class Deter(nn.Module):
    def __init__(self, deter, stoch, act_dim, hidden, blocks, dynlayers, act="SiLU"):
        super().__init__()
        self.blocks = int(blocks)
        self.dynlayers = int(dynlayers)
        act = getattr(torch.nn, act)
        self._dyn_in0 = nn.Sequential(
            nn.Linear(deter, hidden, bias=True), nn.RMSNorm(hidden, eps=1e-04, dtype=torch.float32), act()
        )
        self._dyn_in1 = nn.Sequential(
            nn.Linear(stoch, hidden, bias=True), nn.RMSNorm(hidden, eps=1e-04, dtype=torch.float32), act()
        )
        self._dyn_in2 = nn.Sequential(
            nn.Linear(act_dim, hidden, bias=True), nn.RMSNorm(hidden, eps=1e-04, dtype=torch.float32), act()
        )
        self._dyn_hid = nn.Sequential()
        in_ch = (3 * hidden + deter // self.blocks) * self.blocks
        for i in range(self.dynlayers):
            self._dyn_hid.add_module(f"dyn_hid_{i}", BlockLinear(in_ch, deter, self.blocks))
            self._dyn_hid.add_module(f"norm_{i}", nn.RMSNorm(deter, eps=1e-04, dtype=torch.float32))
            self._dyn_hid.add_module(f"act_{i}", act())
            in_ch = deter
        self._dyn_gru = BlockLinear(in_ch, 3 * deter, self.blocks)
        self.flat2group = lambda x: x.reshape(*x.shape[:-1], self.blocks, -1)
        self.group2flat = lambda x: x.reshape(*x.shape[:-2], -1)

    def forward(self, stoch, deter, action):
        """Deterministic state transition (block-GRU style)."""
        # (B, S, K), (B, D), (B, A)
        B = action.shape[0]

        # Flatten stochastic state and normalize action magnitude.
        # (B, S*K)
        stoch = stoch.reshape(B, -1)
        action = action / torch.clip(torch.abs(action), min=1.0).detach()
        # (B, U)
        x0 = self._dyn_in0(deter)
        x1 = self._dyn_in1(stoch)
        x2 = self._dyn_in2(action)

        # Concatenate projected inputs and broadcast over blocks.
        # (B, 3*U)
        x = torch.cat([x0, x1, x2], -1)
        # (B, G, 3*U)
        x = x.unsqueeze(-2).expand(-1, self.blocks, -1)

        # Combine per-block deterministic state with per-block inputs.
        # (B, G, D/G + 3*U) -> (B, D + 3*U*G)
        x = self.group2flat(torch.cat([self.flat2group(deter), x], -1))

        # (B, D)
        x = self._dyn_hid(x)
        # (B, 3*D)
        x = self._dyn_gru(x)

        # Split GRU-style gates block-wise.
        # (B, G, 3*D/G)
        gates = torch.chunk(self.flat2group(x), 3, dim=-1)

        # (B, D)
        reset, cand, update = (self.group2flat(x) for x in gates)
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1)
        # (B, D)
        return update * cand + (1 - update) * deter


def _cfg_get(config, name, default):
    return getattr(config, name, default)


class RSSM(nn.Module):
    def __init__(self, config, embed_size, act_dim):
        super().__init__()
        self._stoch = int(config.stoch)
        self._deter = int(config.deter)
        self._hidden = int(config.hidden)
        self._discrete = int(config.discrete)
        act = getattr(torch.nn, config.act)
        self._unimix_ratio = float(config.unimix_ratio)
        self._initial = str(config.initial)
        if self._initial not in {"zeros", "learned"}:
            raise ValueError(
                "RSSM initial state must be either 'zeros' or 'learned'")
        self._device = torch.device(config.device)
        self._act_dim = act_dim
        self._obs_layers = int(config.obs_layers)
        self._img_layers = int(config.img_layers)
        self._dyn_layers = int(config.dyn_layers)
        self._blocks = int(config.blocks)
        self._structured_embed = isinstance(embed_size, Mapping)
        self._structured_embed_keys: tuple[str, ...] = ()
        self._flat_embed_size = structured_embed_size(embed_size) if self._structured_embed else int(embed_size)
        self.flat_stoch = self._stoch * self._discrete
        self.feat_size = self.flat_stoch + self._deter
        self._deter_net = Deter(
            self._deter,
            self.flat_stoch,
            act_dim,
            self._hidden,
            blocks=self._blocks,
            dynlayers=self._dyn_layers,
            act=config.act,
        )

        self._structured_obs_adapter = None
        if self._structured_embed:
            # Structured posterior support is retained for backwards-compatible
            # experiments.  The factorized world model is implemented in a
            # separate wrapper and does not turn this path on for vanilla RSSM.
            stream_dims = {str(k): int(v) for k, v in embed_size.items()}
            self._structured_embed_keys = tuple(stream_dims.keys())
            structured_cfg = _cfg_get(config, "structured_posterior", None)
            structured_hidden = int(_cfg_get(structured_cfg, "hidden", self._hidden))
            structured_layers = int(_cfg_get(structured_cfg, "layers", max(1, self._obs_layers)))
            structured_dropout = float(_cfg_get(structured_cfg, "dropout", 0.0))
            self._structured_obs_adapter = StructuredPosteriorAdapter(
                deter_dim=self._deter,
                stream_dims=stream_dims,
                hidden_dim=structured_hidden,
                layers=structured_layers,
                act=config.act,
                dropout=structured_dropout,
            )

        self._obs_net = nn.Sequential()
        inp_dim = (
            self._structured_obs_adapter.out_dim
            if self._structured_obs_adapter is not None
            else self._deter + int(embed_size)
        )
        for i in range(self._obs_layers):
            self._obs_net.add_module(f"obs_net_{i}", nn.Linear(inp_dim, self._hidden, bias=True))
            self._obs_net.add_module(f"obs_net_n_{i}", nn.RMSNorm(self._hidden, eps=1e-04, dtype=torch.float32))
            self._obs_net.add_module(f"obs_net_a_{i}", act())
            inp_dim = self._hidden
        self._obs_net.add_module("obs_net_logit", nn.Linear(inp_dim, self._stoch * self._discrete, bias=True))
        self._obs_net.add_module(
            "obs_net_lambda",
            LambdaLayer(lambda x: x.reshape(*x.shape[:-1], self._stoch, self._discrete)),
        )

        self._img_net = nn.Sequential()
        inp_dim = self._deter
        for i in range(self._img_layers):
            self._img_net.add_module(f"img_net_{i}", nn.Linear(inp_dim, self._hidden, bias=True))
            self._img_net.add_module(f"img_net_n_{i}", nn.RMSNorm(self._hidden, eps=1e-04, dtype=torch.float32))
            self._img_net.add_module(f"img_net_a_{i}", act())
            inp_dim = self._hidden
        self._img_net.add_module("img_net_logit", nn.Linear(inp_dim, self._stoch * self._discrete))
        self._img_net.add_module(
            "img_net_lambda",
            LambdaLayer(lambda x: x.reshape(*x.shape[:-1], self._stoch, self._discrete)),
        )
        # ``initial: learned`` is part of the configured Dreamer state
        # contract. The previous implementation stored the option but always
        # returned an all-zero deterministic and stochastic state. A zero
        # stochastic tensor is outside the one-hot support used everywhere
        # else by this RSSM, and repeated Human-slot resets drove that invalid
        # state through zero-initialized Linear -> RMSNorm stacks. Learn the
        # deterministic prior state and draw its categorical state from the
        # same prior network used by imagination.
        self._initial_deter = (
            nn.Parameter(torch.zeros(self._deter, dtype=torch.float32))
            if self._initial == "learned" else None
        )
        self.apply(weight_init_)

    def initial(self, batch_size, *, sample: bool = True):
        """Return an initial latent state."""
        # (B, D), (B, S, K)
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("RSSM initial batch size must be positive")
        if self._initial_deter is None:
            deter = torch.zeros(
                batch_size, self._deter, dtype=torch.float32,
                device=self._device)
            stoch = torch.zeros(
                batch_size, self._stoch, self._discrete,
                dtype=torch.float32, device=self._device)
        else:
            deter = torch.tanh(self._initial_deter).to(
                device=self._device)[None].expand(batch_size, -1)
            stoch, _ = self.prior(deter, sample=sample)
        return stoch, deter

    def observe(self, embed, action, initial, reset):
        """Posterior rollout using observations."""
        # (B, T, E) or mapping of (B, T, E_i), (B, T, A), ((B, S, K), (B, D)) (B, T)
        L = action.shape[1]
        stoch, deter = initial
        stochs, deters, logits = [], [], []
        for i in range(L):
            # (B, S, K), (B, D), (B, S, K)
            stoch, deter, logit = self.obs_step(stoch, deter, action[:, i], self._embed_step(embed, i), reset[:, i])
            stochs.append(stoch)
            deters.append(deter)
            logits.append(logit)
        # (B, T, S, K), (B, T, D), (B, T, S, K)
        stochs = torch.stack(stochs, dim=1)
        deters = torch.stack(deters, dim=1)
        logits = torch.stack(logits, dim=1)
        return stochs, deters, logits

    def _embed_step(self, embed, index: int):
        if isinstance(embed, Mapping):
            keys = self._structured_embed_keys or tuple(embed.keys())
            return {key: embed[key][:, index] for key in keys}
        return embed[:, index]

    def obs_step(
        self, stoch, deter, prev_action, embed, reset, *, sample: bool = True,
    ):
        """Single posterior step."""
        # (B, S, K), (B, D), (B, A), (B, E) or mapping of (B, E_i), (B,)
        reset = reset.bool()
        if bool(reset.any()):
            initial_stoch, initial_deter = self.initial(
                stoch.shape[0], sample=sample)
            stoch = torch.where(
                rpad(reset, stoch.dim() - int(reset.dim())),
                initial_stoch.to(stoch), stoch)
            deter = torch.where(
                rpad(reset, deter.dim() - int(reset.dim())),
                initial_deter.to(deter), deter)
        prev_action = torch.where(
            rpad(reset, prev_action.dim() - int(reset.dim())), torch.zeros_like(prev_action), prev_action
        )

        # Deterministic transition then posterior logits conditioned on embed.
        # (B, D)
        deter = self._deter_net(stoch, deter, prev_action)
        if self._structured_obs_adapter is not None:
            if not isinstance(embed, Mapping):
                raise TypeError("This RSSM was configured for structured posterior input but got a tensor embed.")
            # (B, H)
            x = self._structured_obs_adapter(deter, embed)
        else:
            if isinstance(embed, Mapping):
                embed = flatten_structured_embed(embed)
            # (B, D + E)
            x = torch.cat([deter, embed], dim=-1)
        # (B, S, K)
        logit = self._obs_net(x)

        # Training samples the categorical posterior through straight-through
        # Gumbel-Softmax. Deterministic deployment must instead use its mode;
        # selecting only the Actor mean while still sampling this state made
        # supposedly fixed closed-loop evaluations depend on process RNG.
        # (B, S, K)
        distribution = self.get_dist(logit)
        stoch = distribution.rsample() if sample else distribution.mode
        return stoch, deter, logit

    def img_step(self, stoch, deter, prev_action, *, sample: bool = True):
        """Single prior step (no observation)."""

        # (B, D)
        deter = self._deter_net(stoch, deter, prev_action)
        # (B, S, K)
        stoch, _ = self.prior(deter, sample=sample)
        return stoch, deter

    def prior(self, deter, *, sample: bool = True):
        """Compute prior distribution parameters and sample stoch."""

        # (B, S, K)
        logit = self._img_net(deter)
        distribution = self.get_dist(logit)
        stoch = distribution.rsample() if sample else distribution.mode
        return stoch, logit

    def imagine_with_action(self, stoch, deter, actions):
        """Roll out prior dynamics given a sequence of actions."""
        # (B, S, K), (B, D), (B, T, A)
        L = actions.shape[1]
        stochs, deters = [], []
        for i in range(L):
            stoch, deter = self.img_step(stoch, deter, actions[:, i])
            stochs.append(stoch)
            deters.append(deter)
        # (B, T, S, K), (B, T, D)
        stochs = torch.stack(stochs, dim=1)
        deters = torch.stack(deters, dim=1)
        return stochs, deters

    def get_feat(self, stoch, deter):
        """Flatten stoch and concatenate with deter."""
        # (B, S, K), (B, D)
        # (B, S*K)
        stoch = stoch.reshape(*stoch.shape[:-2], self._stoch * self._discrete)
        # (B, S*K + D)
        return torch.cat([stoch, deter], -1)

    def get_dist(self, logit):
        return torchd.independent.Independent(dists.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1)

    def kl_loss(self, post_logit, prior_logit, free):
        kld = dists.kl
        rep_loss = kld(
            post_logit, prior_logit.detach(), self._unimix_ratio,
        ).sum(-1)
        dyn_loss = kld(
            post_logit.detach(), prior_logit, self._unimix_ratio,
        ).sum(-1)
        # Clipped gradients are not backpropagated using torch.clip.
        rep_loss = torch.clip(rep_loss, min=free)
        dyn_loss = torch.clip(dyn_loss, min=free)

        return dyn_loss, rep_loss
