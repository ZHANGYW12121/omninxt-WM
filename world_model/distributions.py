import torch
from torch import distributions as torchd
from torch.nn import functional as F

from tools import to_f32, to_i32


def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x):
    return torch.sign(x) * torch.expm1(torch.abs(x))


class OneHotDist(torchd.one_hot_categorical.OneHotCategorical):
    def __init__(self, logits, unimix_ratio=0.0):
        # (..., K)
        probs = F.softmax(to_f32(logits), dim=-1)
        uniform = unimix_ratio / probs.shape[-1]
        probs = probs * (1.0 - unimix_ratio) + torch.ones_like(probs, dtype=torch.float32) * uniform
        logits = torch.log(probs)
        super().__init__(logits=logits)

    @property
    def mode(self):
        # (..., K)
        _mode = F.one_hot(torch.argmax(self.logits, axis=-1), self.logits.shape[-1])
        return _mode.detach() + self.logits - self.logits.detach()

    def rsample(self, sample_shape=(), temperature=1.0):
        # (..., K)
        return F.gumbel_softmax(self.logits, tau=temperature, hard=True, dim=-1)

    def sample(self, **kwargs):
        raise NotImplementedError


class MultiOneHotDist:
    def __init__(self, logits, shape, unimix_ratio=0.0):
        self.shape = shape
        splits = torch.split(logits, shape, dim=-1)
        self.onehots = [OneHotDist(s, unimix_ratio=unimix_ratio) for s in splits]

    @property
    def mode(self):
        _modes = [dist.mode for dist in self.onehots]
        return torch.cat(_modes, dim=-1)

    def rsample(self, sample_shape=()):
        _rsamples = [dist.rsample() for dist in self.onehots]
        return torch.cat(_rsamples, dim=-1)

    def sample(self, **kwargs):
        raise NotImplementedError

    def log_prob(self, value):
        splits = torch.split(value, self.shape, dim=-1)
        _log_probs = [dist.log_prob(s) for dist, s in zip(self.onehots, splits)]
        return sum(_log_probs)

    def entropy(self):
        _entropies = [dist.entropy() for dist in self.onehots]
        return sum(_entropies)


class TwoHot:
    def __init__(self, logits, bins, squash=None, unsquash=None):
        # (..., N_bins), (N_bins,)
        self.logits = to_f32(logits)
        assert self.logits.shape[-1] == len(bins), (self.logits.shape, len(bins))

        self.bins = bins
        self.probs = F.softmax(self.logits, dim=-1)  # (..., N_bins)
        self.squash = squash if squash is not None else (lambda x: x)
        self.unsquash = unsquash if unsquash is not None else (lambda x: x)

    def mode(self):
        # (..., N_bins), (N_bins,) -> (..., 1)
        expected = (self.probs * self.bins).sum(dim=-1, keepdim=True)
        return self.unsquash(expected)

    def log_prob(self, target):
        # (..., 1)
        assert target.dtype == self.probs.dtype
        target = target.squeeze(-1)  # (...,)
        target_squashed = self.squash(target).detach()  # (...,)
        # below/above: (...,)
        below = to_i32(self.bins <= target_squashed.unsqueeze(-1)).sum(dim=-1) - 1
        above = len(self.bins) - to_i32(self.bins > target_squashed.unsqueeze(-1)).sum(dim=-1)
        below = torch.clamp(below, 0, len(self.bins) - 1)
        above = torch.clamp(above, 0, len(self.bins) - 1)
        equal = below == above
        dist_to_below = torch.where(
            equal,
            torch.tensor(1.0, device=target.device, dtype=torch.float32),
            (self.bins[below] - target_squashed).abs(),
        )
        dist_to_above = torch.where(
            equal,
            torch.tensor(1.0, device=target.device, dtype=torch.float32),
            (self.bins[above] - target_squashed).abs(),
        )
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        oh_below = to_f32(F.one_hot(below, num_classes=len(self.bins)))
        oh_above = to_f32(F.one_hot(above, num_classes=len(self.bins)))
        # (..., N_bins)
        mixed_target = oh_below * weight_below.unsqueeze(-1) + oh_above * weight_above.unsqueeze(-1)
        log_pred = self.logits - torch.logsumexp(self.logits, dim=-1, keepdim=True)  # (..., N_bins)
        return (mixed_target * log_pred).sum(dim=-1)  # (...)


class MSEDist:
    def __init__(self, mode, agg="sum"):
        # (..., D)
        self._mode = to_f32(mode)
        self._agg = agg

    def mode(self):
        return self._mode

    def mean(self):
        return self._mode

    def log_prob(self, value):
        # (..., D)
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        assert self._mode.dtype == value.dtype, (self._mode.dtype, value.dtype)
        distance = (self._mode - value) ** 2
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)
        return -loss  # (...)


class SymlogDist:
    def __init__(self, mode, dist="mse", agg="sum", tol=1e-8):
        # (..., D)
        self._mode = to_f32(mode)
        self._dist = dist
        self._agg = agg
        self._tol = tol

    def mode(self):
        return symexp(self._mode)

    def mean(self):
        return symexp(self._mode)

    def log_prob(self, value):
        # (..., D)
        assert self._mode.shape == value.shape
        assert self._mode.dtype == value.dtype
        if self._dist == "mse":
            distance = (self._mode - symlog(value)) ** 2.0
            distance = torch.where(distance < self._tol, 0, distance)
        elif self._dist == "abs":
            distance = torch.abs(self._mode - symlog(value))
            distance = torch.where(distance < self._tol, 0, distance)
        else:
            raise NotImplementedError(self._dist)
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)
        return -loss  # (...)


class Bound:
    def __init__(self, dist):
        super().__init__()
        self._dist = dist

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    @property
    def mode(self):
        out = self._dist.mean
        return out / torch.clip(torch.abs(out), min=1.0).detach()

    def sample(self, sample_shape=()):
        out = self._dist.rsample(sample_shape)
        return out / torch.clip(torch.abs(out), min=1.0).detach()

    def log_prob(self, x):
        return self._dist.log_prob(x)


class TanhNormal:
    """Reparameterized Normal followed by one bijective ``tanh`` transform.

    The historical ``bounded_normal`` squashed only the Normal mean and then
    clamped samples at the controller boundary.  Its reported log probability
    therefore described a different random variable from the action executed
    by the environment.  This small wrapper keeps sampling, mode and
    ``log_prob`` on the same transformed distribution while preserving the
    lightweight distribution interface used by the Dreamer code.

    ``entropy()`` is the standard one-sample Monte-Carlo estimate of the
    transformed entropy.  There is no closed form after the tanh transform.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor,
                 *, epsilon: float = 1.0e-6) -> None:
        self._mean = to_f32(mean)
        self._std = to_f32(std)
        self._base = torchd.Normal(self._mean, self._std)
        self._epsilon = float(epsilon)

    @property
    def mode(self) -> torch.Tensor:
        return torch.tanh(self._mean)

    @property
    def mean(self) -> torch.Tensor:
        # The exact transformed mean has no elementary closed form.  Dreamer
        # uses this field as the deterministic policy action, i.e. the mode.
        return self.mode

    @property
    def stddev(self) -> torch.Tensor:
        return self._std

    def rsample_with_pre_tanh(
        self, sample_shape=torch.Size(),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pre_tanh = self._base.rsample(sample_shape)
        return torch.tanh(pre_tanh), pre_tanh

    def rsample_with_pre_tanh_antithetic_pairs(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample adjacent batch rows with opposite standard-Normal noise.

        Callers must duplicate every source state into adjacent rows before
        constructing this distribution. Each row retains the exact TanhNormal
        marginal; pairing only reduces Monte-Carlo variance when their losses
        are averaged.
        """
        if self._mean.ndim < 1 or self._mean.shape[0] % 2:
            raise ValueError(
                "antithetic TanhNormal sampling requires adjacent row pairs")
        epsilon = torch.randn_like(self._mean[0::2])
        paired_epsilon = torch.stack(
            (epsilon, -epsilon), dim=1).reshape_as(self._mean)
        pre_tanh = self._mean + self._std * paired_epsilon
        return torch.tanh(pre_tanh), pre_tanh

    def rsample(self, sample_shape=torch.Size()) -> torch.Tensor:
        action, _ = self.rsample_with_pre_tanh(sample_shape)
        return action

    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:
        return torch.tanh(self._base.sample(sample_shape))

    @staticmethod
    def _log_abs_det_jacobian(pre_tanh: torch.Tensor) -> torch.Tensor:
        # Algebraically equal to log(1 - tanh(u)^2), but stable for large |u|.
        return 2.0 * (
            torch.log(pre_tanh.new_tensor(2.0))
            - pre_tanh
            - F.softplus(-2.0 * pre_tanh)
        )

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        value = to_f32(value)
        bounded = value.clamp(
            -1.0 + self._epsilon, 1.0 - self._epsilon)
        pre_tanh = torch.atanh(bounded)
        return self.log_prob_from_pre_tanh(pre_tanh)

    def log_prob_from_pre_tanh(
        self, pre_tanh: torch.Tensor,
    ) -> torch.Tensor:
        """Stable transformed log-probability for a stored latent sample.

        ``tanh`` maps sufficiently large float32 inputs to exactly +/-1, so
        recovering a latent with ``atanh(clamp(action))`` is not invertible at
        the boundary.  Dreamer imagination can retain the original latent and
        use this method for an exact score-function objective.
        """
        pre_tanh = to_f32(pre_tanh)
        elementwise = (
            self._base.log_prob(pre_tanh)
            - self._log_abs_det_jacobian(pre_tanh)
        )
        return elementwise.sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        # Keep the original pre-tanh sample for the Monte-Carlo change of
        # variables estimate.  Going through ``tanh`` and then ``atanh`` is
        # numerically wrong once float32 tanh saturates to exactly +/-1: the
        # clamped inverse no longer equals the sampled latent and can report a
        # huge *positive* entropy for a nearly deterministic boundary action.
        pre_tanh = self._base.rsample()
        elementwise_log_prob = (
            self._base.log_prob(pre_tanh)
            - self._log_abs_det_jacobian(pre_tanh)
        )
        return -elementwise_log_prob.sum(dim=-1)


def bounded_normal(x, min_std, max_std, **kwargs):
    mean, std = torch.chunk(x, 2, dim=-1)
    std = (max_std - min_std) * torch.sigmoid(std + 2.0) + min_std
    return TanhNormal(to_f32(mean), to_f32(std))


def normal_std_fixed(mean, std, **kwargs):
    dist = torchd.normal.Normal(to_f32(mean), to_f32(std))
    return Bound(torchd.independent.Independent(dist, 1))


def onehot(mean, unimix_ratio, **kwargs):
    return OneHotDist(to_f32(mean), unimix_ratio=unimix_ratio)


def multi_onehot(mean, unimix_ratio, shape, **kwargs):
    return MultiOneHotDist(to_f32(mean), shape, unimix_ratio=unimix_ratio)


def binary(logits, **kwargs):
    return torchd.independent.Independent(torchd.bernoulli.Bernoulli(logits=to_f32(logits)), 1)


def symexp_twohot(logits, bin_num, **kwargs):
    """Dreamer symlog two-hot distribution.

    The categorical support lives in *symlog space*.  Targets are transformed
    before their two-hot interpolation and the probability-weighted support is
    transformed back only after taking its expectation.  Building bins in raw
    ``symexp`` space instead makes tiny, otherwise harmless probabilities on
    the +/-4.8e8 endpoint bins dominate the decoded reward or value.
    """
    if int(bin_num) < 2:
        raise ValueError("symexp_twohot requires at least two bins")
    bins = torch.linspace(
        -20.0,
        20.0,
        int(bin_num),
        dtype=torch.float32,
        device=logits.device,
    )
    return TwoHot(
        to_f32(logits), bins, squash=symlog, unsquash=symexp)


def linear_twohot(logits, bin_num, low, high, **kwargs):
    """Two-hot distribution whose decoded mode is an expected raw return.

    Symlog support is useful for unbounded generic rewards, but
    ``symexp(E[symlog(return)])`` is not ``E[return]``.  In a safety task a
    policy-conditioned mixture containing a small probability of a -120
    terminal can therefore bootstrap near zero even when its expected raw
    return is strongly negative.  Pure Dreamer has an explicit finite reward
    guard and uses this bounded raw support for its Critic so lambda-return
    bootstrapping retains the task's actual expected-reward semantics.
    """
    count = int(bin_num)
    lower, upper = float(low), float(high)
    if count < 2:
        raise ValueError("linear_twohot requires at least two bins")
    if not torch.isfinite(torch.tensor((lower, upper))).all() or lower >= upper:
        raise ValueError("linear_twohot requires finite low < high")
    bins = torch.linspace(
        lower, upper, count, dtype=torch.float32, device=logits.device)
    return TwoHot(to_f32(logits), bins)


def symlog_mse(logits, **kwargs):
    return SymlogDist(to_f32(logits))


def mse(logits, **kwargs):
    return MSEDist(to_f32(logits))


def identity(logits, **kwargs):
    return logits


def kl(logits_left, logits_right, unimix_ratio=0.0):
    """Categorical KL under the same unimix law used for RSSM sampling.

    DreamerV3's categorical latent mixes every softmax distribution with a
    small uniform component.  Applying that mixture only while sampling but
    not in the KL objective gives the forward state and the optimized
    distribution different semantics, and leaves the KL derivative
    effectively unbounded as logits polarize.
    """
    # (..., K), (..., K)
    ratio = float(unimix_ratio)
    if not 0.0 <= ratio < 1.0:
        raise ValueError("unimix_ratio must lie in [0,1)")
    if ratio == 0.0:
        logprob_left = torch.log_softmax(logits_left, -1)
        logprob_right = torch.log_softmax(logits_right, -1)
        prob_left = torch.softmax(logits_left, -1)
    else:
        categories = logits_left.shape[-1]
        if logits_right.shape[-1] != categories or categories <= 0:
            raise ValueError("categorical KL logits must share a final axis")
        uniform = ratio / categories
        prob_left = (
            torch.softmax(logits_left.float(), -1) * (1.0 - ratio)
            + uniform
        )
        prob_right = (
            torch.softmax(logits_right.float(), -1) * (1.0 - ratio)
            + uniform
        )
        logprob_left = torch.log(prob_left)
        logprob_right = torch.log(prob_right)
    return (
        prob_left * (logprob_left - logprob_right)
    ).sum(-1)  # (...)
