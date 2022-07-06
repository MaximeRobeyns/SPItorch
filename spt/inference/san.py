# SPItorch: Inference of Stellar Population Properties in PyTorch
#
# Copyright (C) 2022 Maxime Robeyns <dev@maximerobeyns.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Implements a "sequential autoregressive network"; a simple procedure for
generating autoregressive samples.
"""

import typing
import logging
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F

from abc import abstractmethod
from typing import Any, Callable, Optional, Type
from torch.utils.data import DataLoader
from torch.distributions import Beta, Categorical, Normal, MixtureSameFamily

import spt.config as cfg

from spt.inference import Model, ModelParams, InferenceParams, HMC_optimiser
from spt.types import Tensor, tensor_like
from spt.inference.utils import squareplus_f, TruncatedNormal


# Likelihoods -----------------------------------------------------------------


class SAN_Likelihood:
    """Class defining abstract methods to be implemented by all likelihoods
    used in a SAN.
    """

    def __init__(*args, **kwargs):
        pass

    supports_lim: bool = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the name of this distribution, as a string."""
        raise NotImplementedError

    def n_params(self) -> int:
        """Returns the number of parameters required to parametrise a single
        dimension of this distribution.
        e.g. for a Gaussian, this is 2 (loc, scale). For a mixture of K
        distributions each taking N parameters, this is K * N.
        """
        return 2

    @abstractmethod
    def log_prob(self, value: Tensor, params: Tensor) -> Tensor:
        """Evaluate the log probability of `value` under a distribution
        parametrised by `params`"""
        raise NotImplementedError

    @abstractmethod
    def sample(self, params: Tensor, d: Optional[int] = None) -> Tensor:
        """Draw a single sample from a distribution parametrised by `params`"""
        raise NotImplementedError

    @abstractmethod
    def rsample(self, params: Tensor, d: Optional[int] = None) -> Tensor:
        """Draw a single reparametrised sample from a distribution parametrised
        by `params`"""
        raise NotImplementedError

    def to(self, device: t.device = None, dtype: t.dtype = None):
        pass


class TruncatedLikelihood:
    """Disallows sampling outside a given range (useful when prior doesn't have
    support over the entire real line.)"""

    supports_lim: bool = True

    @property
    @abstractmethod
    def lower(self) -> Tensor:
        # Return the lower bound
        raise NotImplementedError

    @property
    @abstractmethod
    def upper(self) -> Tensor:
        # Return the upper bound
        raise NotImplementedError

class Gaussian(SAN_Likelihood):
    """Univariate Gaussian likelihood for each dimension"""

    name: str = "Gaussian"

    def _extract_params(self, params: Tensor) -> tuple[Tensor, Tensor]:
        loc, scale = params.split(1, -1)
        return loc.squeeze(), squareplus_f(scale.squeeze())

    def log_prob(self, value: Tensor, params: Tensor) -> Tensor:
        loc, scale = self._extract_params(params)
        return t.distributions.Normal(
                loc.squeeze(), scale.squeeze()).log_prob(value)

    def sample(self, params: Tensor, _: Optional[int] = None) -> Tensor:
        return t.distributions.Normal(
                *self._extract_params(params)).sample()

    def rsample(self, params: Tensor, _: Optional[int] = None) -> Tensor:
        return t.distributions.Normal(
                *self._extract_params(params)).rsample()


class Laplace(SAN_Likelihood):
    """Models each dimension using a univariate Laplace distribution"""

    name: str = "Laplace"

    def _extract_params(self, params: Tensor) -> tuple[Tensor, Tensor]:
        loc, scale = params.split(1, -1)
        return loc.squeeze(), squareplus_f(scale.squeeze())

    def log_prob(self, value: Tensor, params: Tensor) -> Tensor:
        return t.distributions.Laplace(
                *self._extract_params(params)).log_prob(value)

    def sample(self, params: Tensor, _: Optional[int] = None) -> Tensor:
        return t.distributions.Laplace(
                *self._extract_params(params)).sample()

    def rsample(self, params: Tensor, _: Optional[int] = None) -> Tensor:
        return t.distributions.Laplace(
                *self._extract_params(params)).rsample()


class MoB(SAN_Likelihood, TruncatedLikelihood):
    """Models every dimension with a K-component mixture of Beta distributions"""

    def __init__(self, K: int, lims: Tensor, mult_eps: float=1e-4,
                 abs_eps: float=1e-4):
        """Mixture of Beta distributions.

        Args:
            K: number of mixture components.
            lims: [0, 1] limits on each of the dimensions; size [D, 2] (min, max)
            mult_eps: multiplicative stabilisation term
            abs_eps: additive stabilisation term
        """
        self.K = K
        self.mult_eps = mult_eps
        self.abs_eps = abs_eps
        self._lower, self._upper = lims[..., 0].detach(), lims[..., 1].detach()

    supports_lim = True
    name: str = "MoB"

    @property
    def lower(self) -> Tensor:
        return self._lower

    @property
    def upper(self) -> Tensor:
        return self._upper

    def to(self, device: t.device = None, dtype: t.dtype = None):
        self._lower = self._lower.to(device, dtype).detach()
        self._upper = self._upper.to(device, dtype).detach()

    def n_params(self) -> int:
        return 3 * self.K  # concentration 1 & 2, mixture weight.

    def _extract_params(self, params: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Extracts the mixture model parameters from the tensor returned from
        the network

        Returns: concentration 1 [B, K], concentration 2 [B, K], weights [B, K]
        """
        f = lambda x: x.squeeze(-1)
        B = params.shape[:-2]  # -1
        c1, c2, k = params.reshape(*B, -1, self.K, 3).tensor_split(3, -1)
        # return f(F.softplus(c1)), f(F.softplus(c2)), f(k)
        return f(squareplus_f(c1)), f(squareplus_f(c2)), f(k)

    def _stabilise(self, S: Tensor) -> Tensor:
        while not S.gt(0.).all():
            S = t.where(S.le(0.), S.abs()*(1.+self.mult_eps)+self.abs_eps, S)
            # S = S.abs()*(1.+self.mult_eps) + self.abs_eps
        return S

    def _stable_betas(self, c1: Tensor, c2: Tensor) -> Beta:
        try:
            return Beta(c1, c2)
        except ValueError:
            return Beta(*map(self._stabilise, (c1, c2)))

    def _bmm_from_params(self, params: Tensor) -> MixtureSameFamily:
        c1, c2, k = self._extract_params(params)
        cat = Categorical(logits=k)
        norms = self._stable_betas(c1, c2)
        return MixtureSameFamily(cat, norms)

    def log_prob(self, value: Tensor, params: Tensor) -> Tensor:
        return self._bmm_from_params(params).log_prob(value)

    def sample(self, params: Tensor, _: Optional[int] = None) -> Tensor:
        return self._bmm_from_params(params).sample()

    def rsample(self, params: Tensor, _: Optional[int] = None) -> Tensor:
        """Warning: this is not true reparametrised sampling: for that we
        would need to average over each of the mixture components in proportion
        to the parameters of the categorical distribution. Consequently, don't
        use this to learn the logits (k) parameters. It merely allows us to do
        HMC with this distribution.
        """
        sample_shape = t.Size()
        c1, c2, k = self._extract_params(params)
        gather_dim = len(c1.shape) - 1
        es = t.Size([])

        mix_sample = Categorical(logits=k).sample()
        mix_shape = mix_sample.shape

        comp_samples = self._stable_betas(c1, c2).rsample(sample_shape)

        # gather along the k dimension
        mix_sample_r = mix_sample.reshape(
            mix_shape + t.Size([1] * (len(es) + 1)))
        mix_sample_r = mix_sample_r.repeat(
            t.Size([1] * len(mix_shape)) + t.Size([1]) + es)

        samples = t.gather(comp_samples, gather_dim, mix_sample_r)
        return samples.squeeze(gather_dim)


class MoG(SAN_Likelihood):
    """Models every dimension with a K-component mixture of Gaussians"""

    def __init__(self, K: int, mult_eps: float=1e-4, abs_eps: float=1e-4):
        """Mixture of Gaussians.

        Args:
            K: number of mixture components.
            mult_eps: multiplicative stabilisation term
            abs_eps: additive stabilisation term
        """
        self.K = K
        self.mult_eps = mult_eps
        self.abs_eps = abs_eps

    name: str = "MoG"

    def n_params(self) -> int:
        return 3 * self.K  # loc, scale, mixture weight.

    def _extract_params(self, params: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Extracts the mixture model parameters from the tensor returned from
        the network

        Returns: locations [B, K], scale [B, K], weights [B, K]
        """
        f = lambda x: x.squeeze(-1)
        B = params.shape[:-2]  # -1
        loc, scale, k = params.reshape(*B, -1, self.K, 3).tensor_split(3, -1)
        return f(loc), f(squareplus_f(scale)), f(k)

    def _stabilise(self, S: Tensor) -> Tensor:
        while not S.gt(0.).all():
            S = t.where(S.le(0.), S.abs()*(1.+self.mult_eps)+self.abs_eps, S)
            # S = S.abs()*(1.+self.mult_eps) + self.abs_eps
        return S

    def _stable_norms(self, loc: Tensor, scale: Tensor) -> Normal:
        try:
            return Normal(loc, scale)
        except ValueError:
            return Normal(loc, self._stabilise(scale))

    def _gmm_from_params(self, params: Tensor) -> MixtureSameFamily:
        loc, scale, k = self._extract_params(params)
        cat = Categorical(logits=k)
        norms = self._stable_norms(loc, scale)
        return MixtureSameFamily(cat, norms)

    def log_prob(self, value: Tensor, params: Tensor) -> Tensor:
        return self._gmm_from_params(params).log_prob(value)

    def sample(self, params: Tensor, _: Optional[int] = None) -> Tensor:
        return self._gmm_from_params(params).sample()

    def rsample(self, params: Tensor, _: Optional[int] = None) -> Tensor:
        """Warning: this is not true reparametrised sampling: for that we
        would need to average over each of the mixture components in proportion
        to the parameters of the categorical distribution. Consequently, don't
        use this to learn the logits (k) parameters. It merely allows us to do
        HMC with this distribution.
        """
        sample_shape = t.Size()
        loc, scale, k = self._extract_params(params)
        gather_dim = len(loc.shape) - 1
        es = t.Size([])

        mix_sample = Categorical(logits=k).sample()
        mix_shape = mix_sample.shape

        comp_samples = self._stable_norms(loc, scale).rsample(sample_shape)

        # gather along the k dimension
        mix_sample_r = mix_sample.reshape(
            mix_shape + t.Size([1] * (len(es) + 1)))
        mix_sample_r = mix_sample_r.repeat(
            t.Size([1] * len(mix_shape)) + t.Size([1]) + es)

        samples = t.gather(comp_samples, gather_dim, mix_sample_r)
        return samples.squeeze(gather_dim)


class TruncatedMoG(SAN_Likelihood, TruncatedLikelihood):
    """
    A truncated mixture of Gaussians, which allows us to repsect prior
    parameter limits.
    """

    def __init__(self, lims: Tensor, K: int, mult_eps: float = 1e-4,
                 abs_eps: float = 1e-4, trunc_eps: float = 1e-3,
                 validate_args: bool = True):
        """Truncated mixture of Gaussians. We work with tensors of size
        [B, D, K], for B a (possibly multi-dimensional) batch shape, D the
        number of dimensions in the mixture, and K the number of mixture
        components..

        Args:
            lims: limits on each of the dimensions; size [D, 2] (min, max)
            K: number of mixture components per dimension
        """
        self.K = K
        self.mult_eps, self.abs_eps = mult_eps, abs_eps
        self.trunc_eps = trunc_eps
        assert lims.shape[-1] == 2, "Expected a tensor of [min, max] as lims"
        self._lower, self._upper = lims[..., 0].detach(), lims[..., 1].detach()
        self._val_args = validate_args

    supports_lim = True
    name: str = "TMoG"  # used for file names

    @property
    def lower(self) -> Tensor:
        return self._lower

    @property
    def upper(self) -> Tensor:
        return self._upper

    def n_params(self) -> int:
        return 3 * self.K  # loc, scale, mixture weight

    def to(self, device: t.device = None, dtype: t.dtype = None):
        self._lower = self._lower.to(device, dtype).detach()
        self._upper = self._upper.to(device, dtype).detach()

    def _extract_params(self, params: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        f = lambda x: x.squeeze(-1)
        B = params.shape[:-2]
        loc, scale, k = params.reshape(*B, -1, self.K, 3).tensor_split(3, -1)
        return f(loc), f(squareplus_f(scale)), f(k)

    def _stabilise(self, S: Tensor) -> Tensor:
        while not S.gt(0.).all():
            S = t.where(S.le(0.), S.abs()*(1.+self.mult_eps)+self.abs_eps, S)
            # S = S.abs()*(1.+self.mult_eps) + self.abs_eps
        return S

    def _stable_norms(self, loc: Tensor, scale: Tensor, A: Tensor, B: Tensor
                      ) -> TruncatedNormal:
        # for numerical stability, don't get too close to the edge!
        loc = loc.clamp(A + self.trunc_eps, B - self.trunc_eps)
        try:
            return TruncatedNormal(loc, scale, A, B, self._val_args)
        except ValueError:  # constraint violation
            return TruncatedNormal(loc, self._stabilise(scale), A, B,
                                   self._val_args)

    def _get_bounds(self, loc: Tensor, d: Optional[int] = None
                    ) -> tuple[Tensor, Tensor]:
        if d is None:  # [B, D, n_params]
            A = self._lower[(None,) * (loc.dim() - 2) + (..., None)]
            B = self._upper[(None,) * (loc.dim() - 2) + (..., None)]
        else:  # [B, 1, K]
            A = self._lower[d][(None,) * (loc.dim() - 2) + (..., None)]
            B = self._upper[d][(None,) * (loc.dim() - 2) + (..., None)]
        A = A.expand(loc.shape).to(loc.device, loc.dtype)
        B = B.expand(loc.shape).to(loc.device, loc.dtype)
        return A, B

    def _gmm_from_params(self, params: Tensor, d: Optional[int] = None) -> MixtureSameFamily:
        loc, scale, k = self._extract_params(params)
        cat = t.distributions.Categorical(logits=k)
        A, B = self._get_bounds(loc, d)
        norms = self._stable_norms(loc, scale, A, B)
        return t.distributions.MixtureSameFamily(cat, norms)

    def log_prob(self, value: Tensor, params: Tensor) -> Tensor:
        # We compute the log prob ourselves here due to a bug in
        # MixtureSameFamily which prevents us from identifying the correct
        # support.

        loc, scale, k = self._extract_params(params)
        A, B = self._get_bounds(loc)
        value = value[..., None].expand(loc.shape)
        norms = self._stable_norms(loc, scale, A, B)
        lps = norms.log_prob(value)
        ws = t.log_softmax(k, dim=-1)
        return t.logsumexp(lps + ws, dim=-1)

    def sample(self, params: Tensor, d: Optional[int] = None) -> Tensor:
        return self._gmm_from_params(params, d).sample().nan_to_num()

    def rsample(self, params: Tensor, d: Optional[int] = None) -> Tensor:
        """rsample, implicitly holding the mixture weights frozen for 1 MC
        estimate. See MoG.rsample warning."""

        loc, scale, k = self._extract_params(params)
        gather_dim = len(loc.shape) - 1
        es = t.Size()

        mix_sample = t.distributions.Categorical(logits=k).sample()
        mix_shape = mix_sample.shape

        A, B = self._get_bounds(loc, d)
        comp_samples = self._stable_norms(loc, scale, A, B).rsample().nan_to_num()

        # gather along the kth dimension
        mix_sample_r = mix_sample.reshape(
            mix_shape + t.Size([1] * (len(es) + 1)))
        mix_sample_r = mix_sample_r.repeat(
            t.Size([1] * len(mix_shape)) + t.Size([1]) + es)
        samples = t.gather(comp_samples, gather_dim, mix_sample_r)
        return samples.squeeze(gather_dim)


class MoST(SAN_Likelihood):

    # Don't use this one...

    def __init__(self, K: int) -> None:
        """Mixture of StudentT distributions.

        Args:
            K: number of mixture components.
        """
        self.K = K

    name: str =  "MoST"

    def n_params(self) -> int:
        return 3 * self.K  # loc, scale, mixture weight.

    def _extract_params(self, params: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        B = params.size(0)
        loc, scale, k = params.reshape(B, -1, self.K, 3).tensor_split(3, 3)
        return loc.squeeze(-1), squareplus_f(scale).squeeze(-1), F.softmax(k, -1).squeeze(-1)

    def log_prob(self, value: Tensor, params: Tensor) -> Tensor:
        loc, scale, k = self._extract_params(params)
        cat = t.distributions.Categorical(k)
        sts = t.distributions.StudentT(1., loc, scale)
        return t.distributions.MixtureSameFamily(cat, sts).log_prob(value)

    def sample(self, params: Tensor, _: Optional[int] = None) -> Tensor:
        B = params.size(0)
        loc, scale, k = params.reshape(B, self.K, 3).tensor_split(3, 2)
        loc, scale = loc.squeeze(-1), squareplus_f(scale).squeeze(-1)
        cat = t.distributions.Categorical(F.softmax(k, -1).squeeze(-1))
        sts = t.distributions.StudentT(1., loc, scale)
        return t.distributions.MixtureSameFamily(cat, sts).sample()

    def rsample(self, params: Tensor, _: Optional[int] = None) -> Tensor:
        raise NotImplementedError()


class ACFLikelihood(SAN_Likelihood):
    """A likelihood which is based on a normalising flow (affine coupling layer(s)
    used instead of mixture distribution in SAN).

    WARNING: Not Implemented.
    """

    def name(self) -> str:
        return "Affine Coupling Flow"

    def n_params(self) -> int:
        """number of parameters required to parametrise the flow layer

        This is essentially the nunber of output features for the d'th layer
        head.
        """
        raise NotImplementedError

    def log_prob(self, value: Tensor, params: Tensor) -> Tensor:
        raise NotImplementedError


# SAN Description -------------------------------------------------------------


class SANParams(ModelParams):
    """Configuration class for SAN.

    This defines some required properties, and additionally performs validation
    of user-supplied values.
    """

    def __init__(self):
        super().__init__()
        # perform any required validation here...

    @property
    def first_module_shape(self) -> list[int]:
        """The size of the first module of the network.
        This is used to build useful initial sequence features, and allows the
        subsequent blocks to be smaller, reducing memory requirements.

        Default: the usual module shape
        """
        return self.module_shape

    @property
    @abstractmethod
    def module_shape(self) -> list[int]:
        """Size of each individual 'module block'"""
        pass

    @property
    @abstractmethod
    def sequence_features(self) -> int:
        """Number of features to carry through for each network block"""
        pass

    @property
    @abstractmethod
    def likelihood(self) -> Type[SAN_Likelihood]:
        """Likelihood to use for each p(y_d | y_<d, x)"""
        pass

    @property
    def likelihood_kwargs(self) -> Optional[dict[str, Any]]:
        """Any keyword arguments accepted by likelihood"""
        return None

    @property
    @abstractmethod
    def layer_norm(self) -> bool:
        """Whether to use layer norm (batch normalisation) or not"""
        pass

    @property
    def train_rsample(self) -> bool:
        """Whether to stop gradients at the autoregressive step"""
        return False

    @property
    def opt_lr(self) -> float:
        """Optimiser learning rate"""
        return 3e-3

    @property
    def opt_decay(self) -> float:
        """Optimiser weight decay"""
        return 1e-4

    @property
    def limits(self) -> Optional[Tensor]:
        """Allows the output samples to be constrained to lie within the
        specified range.

        This method returns a (self.data_dim x 2)-dimensional tensor, with the
        (normalised) min and max values of each dimension of the output.
        """
        return None


class SANv2Params(SANParams):
    """Configuration class for SAN v2 (with shared latent variable)"""

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def latent_features(self) -> int:
        """The size of the shared latent features."""
        pass

    @property
    @abstractmethod
    def encoder_layers(self) -> list[int]:
        """The layer sizes for the encoder"""
        pass



# Main SAN Class ==============================================================


class SAN(Model):

    def __init__(self, mp: SANParams,
                 logging_callbacks: list[Callable] = [],
                 init_modules: bool = True):
        """"Sequential autoregressive network" implementation.

        Args:
            mp: the SAN model parameters, set in `config.py`.
            logging_callbacks: list of callables accepting this model instance;
                often used for visualisations and debugging.
        """
        super().__init__(mp, logging_callbacks)

        kwargs = {} if mp.likelihood_kwargs is None else mp.likelihood_kwargs
        self.likelihood: SAN_Likelihood = mp.likelihood(**kwargs)

        self.layer_norm = mp.layer_norm
        self.train_rsample = mp.train_rsample

        if mp.limits is not None:
            if not isinstance(mp.limits, t.Tensor):
                self.limits = t.tensor(mp.limits, dtype=mp.dtype, device=mp.device)
            else:
                self.limits = mp.limits.to(mp.device, mp.dtype)
            if not self.likelihood.supports_lim:
                logging.warning((
                    'SAN limits have been provided by the selected SAN likelihood'
                    f'({self.likelihood.name}) does not support limits.'))
        else:
            self.limits = None

        if init_modules:
            self._do_init(mp)

        # A tensor storing the parameters of the distributions from which
        # samples were drawn for the last forward pass.
        # Useful for evaluating NLL of data under model.
        # Size: [mini-batch, likelihood_params]
        self.last_params: Optional[Tensor] = None

        if mp.device == t.device('cuda'):
            self.to(mp.device, mp.dtype)
            self.likelihood.to(mp.device, mp.dtype)
        else:
            self.likelihood.to(t.device('cpu'), mp.dtype)

        # Strange mypy error requires this to be put here although it is
        # perfectly well defined and typed in the super class ¯\_(ツ)_/¯
        self.savepath_cached: str = ""

    name: str = 'SAN'

    def __repr__(self) -> str:
        return (f'{self.name} with {self.likelihood.name} likelihood, '
                f'module blocks of shape {self.module_shape} '
                f'and {self.sequence_features} features between blocks trained '
                f'for {self.epochs} epochs with batches of size {self.batch_size}')

    def fpath(self, ident: str='') -> str:
        """Returns a file path to save the model to, based on its parameters."""
        base = './results/sanmodels/'
        s = self.module_shape + [self.sequence_features]
        ms = '_'.join([str(l) for l in s])
        name = (f'l{self.likelihood.name}_cd{self.cond_dim}'
                f'_dd{self.data_dim}_ms{ms}_'
                f'lp{self.likelihood.n_params()}_ln{self.layer_norm}_'
                f'lr{self.lr}_ld{self.decay}_e{self.epochs}_'
                f'bs{self.batch_size}_trsample{self.train_rsample}_')
        name += 'lim_' if self.limits is not None else ''
        self.savepath_cached = f'{base}{name}{ident}.pt'

        return self.savepath_cached

    def _do_init(self, mp: ModelParams):
        """Initialises the network modules based on the configuration"""

        assert isinstance(mp, SANParams)

        self.first_module_shape = mp.first_module_shape
        self.module_shape = mp.module_shape
        self.sequence_features = mp.sequence_features
        self.lr = mp.opt_lr
        self.decay = mp.opt_decay

        self.network_blocks = nn.ModuleList()
        self.block_heads = nn.ModuleList()

        for (i, d) in enumerate(range(self.data_dim)):
            b, h = self._sequential_block(self.cond_dim, d,
                    self.first_module_shape if i == 0 else self.module_shape,
                    out_shapes=[self.sequence_features, self.likelihood.n_params()],
                    out_activations=[nn.ReLU, None])
            self.network_blocks.append(b)
            self.block_heads.append(h)

        self.opt = t.optim.Adam(self.parameters(), lr=self.lr,
                                weight_decay=self.decay)

    def _sequential_block(self, cond_dim: int, d: int, module_shape: list[int],
                          out_shapes: list[int], out_activations: list[Any]
                          ) -> tuple[nn.Module, nn.ModuleList]:
        """Initialises a single 'sequential block' of the network.

        Args:
            cond_dim: dimension of conditioning data (e.g. photometric observations)
            d: current dimension in the autoregressive sequence p(y_d | y_<d, x)
            module_shape: sizes of the 'sequential block' layers
            out_shapes: size of sequential block output and parameters, respectively
            out_activations: activation functions to apply to each respective head

        Returns:
            tuple[nn.Module, nn.ModuleList]: sequential block and heads
        """

        block = nn.Sequential()
        heads = nn.ModuleList()

        if d == 0:
            hs = [cond_dim] + module_shape
        else:
            # [x + sequence_features + autoregressive samples, module layers...]
            hs = [cond_dim + out_shapes[0] + d] + module_shape

        for i, (j, k) in enumerate(zip(hs[:-1], hs[1:])):
            block.add_module(name=f'B{d}L{i}', module=nn.Linear(j, k))
            if self.layer_norm:
                block.add_module(name=f'B{d}LN{i}', module=nn.LayerNorm(k))
            block.add_module(name=f'B{d}A{i}', module=nn.ReLU())
        block.to(self.device, self.dtype)

        hn: int = module_shape[-1]
        for i, h in enumerate(out_shapes):
            this_head = nn.Sequential()
            this_head.add_module(name=f'H{d}:{i}H{i}', module=nn.Linear(hn, h))

            a = out_activations[i]
            if a is not None:
                this_head.add_module(name=f'H{d}:{i}A{i}', module=a())
            heads.append(this_head)
        heads.to(self.device, self.dtype)

        return block, heads

    def forward(self, x: Tensor, lp: bool = False,
                rsample: bool = False) -> Tensor:
        """Runs the autoregressive model.

        Args:
            x: some conditioning information [B, cond_dim]
            lp: whether to save the last parameters
            rsample: whether to stop gradients (False) or not (True) in the
                autoregressive sampling step.

        Returns:
            Tensor: a sample from the distribution; y_hat ~ p(y | x)

        Implicit Returns:
            self.last_params: a tensor containing the parameters of each
                dimension's (univariate) distribution of size
                [mini-batch, lparams]; giving p(y | x). Only if lp=True.
        """

        B = x.shape[:-1]
        ys = t.empty(B + (0,), dtype=self.dtype, device=self.device)
        if lp:
            self.last_params = t.empty(B + (0, self.likelihood.n_params()),
                                       dtype=self.dtype, device=self.device)
        else:
            self.last_params = None

        seq_features = t.empty(B + (0,), dtype=self.dtype, device=self.device)

        for d in range(self.data_dim):

            d_input = t.cat((x, seq_features, ys), -1)

            H = self.network_blocks[d](d_input)

            # for passing to next sequential block
            seq_features = self.block_heads[d][0](H)

            # draw single sample from p(y_d | y_<d, x)
            params = self.block_heads[d][1](H)

            if rsample:
                y_d = self.likelihood.rsample(params, d)
            else:
                y_d = self.likelihood.sample(params, d)

            assert not y_d.isnan().all(), "NaN values returned from likelihood."

            y_d = y_d[(...,) + (None, ) * (ys.dim() - y_d.dim())]

            ys = t.cat((ys, y_d), -1)
            if lp:
                self.last_params = t.cat((self.last_params, params.unsqueeze(-2)), x.dim()-1)

        # check we did the sampling right
        assert ys.shape == B + (self.data_dim,)
        return ys

    def offline_train(self, train_loader: DataLoader, ip: InferenceParams,
                      errs: Optional[Tensor] = None,
                      *args, **kwargs) -> None:
        """Train the SAN model offline.

        Args:
            train_loader: DataLoader to load the training data.
            ip: The parameters to use for training, defined in
                `config.py:InferenceParams`.
            errs: observation noise variance (optional).
        kwargs:
            rsample (Bool): whether to stop gradients (False) or not (True) in the
                autoregressive sampling step.
        """
        t.random.seed()
        self.train()

        start_e = self.attempt_checkpoint_recovery(ip)
        for e in range(start_e, self.epochs):
            for i, (x, y) in enumerate(train_loader):
                x, y = self.preprocess(x, y)

                if errs is not None:
                    x = self._obs_noise_augmentation(x, errs)

                # if the likelihood has limits, then filter the y values here:
                if isinstance(self.likelihood, TruncatedLikelihood):
                    mask = t.logical_and(y > self.likelihood.lower,
                                         y < self.likelihood.upper).all(-1)
                    x, y = x[mask], y[mask]

                # The following implicitly updates self.last_params, and
                # returns y_hat (a sample from p(y | x))
                _ = self.forward(x, True, self.train_rsample)
                assert (self.last_params is not None)

                # Minimise the NLL of true ys using training parameters
                LP = self.likelihood.log_prob(y, self.last_params)
                loss = -LP.sum(1).mean(0)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                if i % ip.logging_frequency == 0 or i == len(train_loader)-1:
                    # Run through all logging functions
                    [cb(self) for cb in self.logging_callbacks]
                    logging.info(
                        "Epoch: {:02d}/{:02d}, Batch: {:05d}/{:d}, Loss {:9.4f}"
                        .format(e+1, self.epochs, i, len(train_loader)-1,
                                loss.item()))
            self.checkpoint(ip.ident)

        # Pre-emptively put model in evaluation mode.
        self.eval()


    @typing.no_type_check
    def hmc_retrain_procedure(self, train_loader: DataLoader, ip: InferenceParams,
                              P: Model, epochs: int, K: int, lr: float = 3e-4,
                              decay: float = 1e-4, logging_frequency: int = 1000,
                              simplified: bool = True, errs: Optional[Tensor] = None,
                              ) -> None:
        """Perform the retraining procedure, using intermediate HMC updates to
        generate training paris on-the-fly.

        Note: simplified is for backward compatability: leave as True.
        """
        self.train()
        t.random.seed()

        opt = t.optim.Adam(self.parameters(), lr=lr, weight_decay=decay)
        bounds = t.tensor(cfg.ForwardModelParams().free_param_lims(normalised=True),
                          device=self.device, dtype=self.dtype)

        def get_logpdf(xs: Tensor) -> Callable[[Tensor], Tensor]:
            def logpdf(theta: Tensor) -> Tensor:
                _ = P(theta, True)
                return P.likelihood.log_prob(xs, P.last_params).sum(-1)
            return logpdf

        start_e = self.attempt_checkpoint_recovery(ip)
        for e in range(start_e, epochs):
            for i, (x, y) in enumerate(train_loader):
                x, _ = self.preprocess(x, y)

                # expand to chains
                xs = x.repeat_interleave(K, 0)

                if errs is not None:
                    xs = self._obs_noise_augmentation(xs, errs)

                xs = xs.unsqueeze(-2).expand(-1, ip.hmc_update_C, ip.hmc_update_D)


                logpdf = get_logpdf(xs)
                initial_pos = self(xs)

                theta_hat = HMC_optimiser(
                    logpdf, N=ip.hmc_update_N, initial_pos=initial_pos,
                    rho=ip.hmc_update_rho, L=ip.hmc_update_L,
                    alpha=ip.hmc_update_alpha,
                    bounds=bounds, device=self.device, dtype=self.dtype,
                    quiet=True)
                xs = None  # no longer needed on GPU memory

                _ = self.forward(x, True, rsample=False)
                post_prob = self.likelihood.log_prob(theta_hat, self.last_params)

                loss = -post_prob.sum(-1).mean(0)

                opt.zero_grad()
                loss.backward()
                opt.step()

                if i % logging_frequency == 0:
                    logging.info((f'Objective at epoch: {e:02d}/{epochs:02d}'
                                  f' iter: {i:04d}/{len(train_loader):04d} is '
                                  f'{loss.detach().cpu().item()}'))
            self.checkpoint(ip.ident)

        self.eval()

    def _obs_noise_augmentation(self, x: Tensor, errs: Tensor) -> Tensor:
        """Adds Gaussian noise to the inputs x, with variance errs"""
        errs = t.atleast_2d(errs).to(x.device, x.dtype)
        return t.distributions.Normal(x, errs).sample()

    def _preprocess_sample_input(self, x: tensor_like, n_samples: int = 1000,
                                 errs: Optional[tensor_like] = None) -> Tensor:

        if isinstance(x, np.ndarray):
            x = t.from_numpy(x)

        if not isinstance(x, t.Tensor):
            raise ValueError((
                f'Please provide a PyTorch Tensor (or numpy array) as input '
                f'(got {type(x)})'))

        x = x.unsqueeze(0) if x.dim() == 1 else x
        # TODO remove this
        assert x.dim() == 2, "Please 'flatten' your batch of points to be 2 dimensional"
        x, _ = self.preprocess(x, t.empty(x.shape))
        n, d = x.shape

        if errs is not None:
            if isinstance(errs, np.ndarray):
                errs = t.from_numpy(errs)
            errs = errs.unsqueeze(0) if errs.dim() == 1 else errs
            x = t.distributions.Normal(x, errs).sample((n_samples,)).reshape(-1, d)
        else:
            x = x.repeat_interleave(n_samples, 0)

        assert x.shape == (n * n_samples, d)
        return x

    @typing.no_type_check
    def sample(self, x_in: tensor_like, n_samples: int = 1000,
               rsample: bool = False, errs: Optional[tensor_like] = None
               ) -> Tensor:
        """A convenience method for drawing (conditional) samples from p(y | x)
        for a single conditioning point.

        Args:
            x: the conditioning data; x
            n_samples: the number of samples to draw
            rsample: whether to use reparametrised sampling (default False)
            errs: observation uncertainty in x (optional; only for 'real' observations)

        Returns:
            Tensor: a tensor of shape [n_samples, data_dim]
        """
        x = self._preprocess_sample_input(x_in, n_samples, errs)
        return self(x, rsample)

    def mode(self, x_in: tensor_like, n_samples: int = 1000,
             rsample: bool = False, errs: Optional[tensor_like] = None
             ) -> Tensor:
        """A convenience method which returns the highest posterior mode for a
        given batch of photometric observations, x_in

        Args:
            x_in: the conditioning data
            n_samples: the number of samples to draw when searching for the mode
            rsample: whether to use reparametrised sampling (default False)
            errs: observation uncertainty in x (optional; only use for 'real' observations)

        Returns:
            Tensor: a tensor of modes [data_dim]
        """
        N = n_samples
        x = self._preprocess_sample_input(x_in, n_samples, errs)
        B = int(x.size(0) / N)

        samples = self(x, True, rsample)  # [B*N, data_dim]
        rsamples = samples.reshape(B, n_samples, self.data_dim)  # [B, N, data_dim]

        assert self.last_params is not None
        lps = self.likelihood.log_prob(samples, self.last_params).sum(-1)  #[B*N]
        self.last_params = None  # remove references to GPU memory
        rlps = lps.reshape(B, N)  # [B, N]

        # [B, 1, data_dim]:
        idxs = t.argmax(rlps, dim=1)[:, None, None].expand(B, 1, self.data_dim)
        modes = rsamples.gather(1, idxs).squeeze(1)  # [B, data_dim]

        return modes


class PModel(SAN):
    """A SAN which is slightly adapted to act as a likelihood / forward model
    emulator by switching the xs and thetas in the preprocessing step."""
    def preprocess(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        return y.to(self.device, self.dtype), x.to(self.device, self.dtype)


# V2 SAN (with shared latents) ================================================


class SANv2(SAN):

    def __init__(self, mp: SANv2Params,
                 logging_callbacks: list[Callable] = []):
        """Sequential autoregressive network with shared latent variables
        output by an initial 'encoder' block. Marginals of the target
        distribution are then sequentially generated in an autoregressive
        manner.

        Args:
            mp: The SAN (v2) model parameters, set in `config.py`.
            logging_callbacks: list of callables accepting this model instance;
                often used for visualisations and debugging.
        """
        super().__init__(mp, logging_callbacks, init_modules=False)

        self._do_init(mp)

        if mp.device == t.device('cuda'):
            self.to(mp.device, mp.dtype)
            self.likelihood.to(mp.device, mp.dtype)
        else:
            self.likelihood.to(t.device('cpu'), mp.dtype)

    name: str = 'SANv2'

    def __repr__(self) -> str:
        return (f'{self.name} with {self.likelihood.name} likelihood, '
                f'encoder of shape {self.encoder_layers} and '
                f'{self.latent_features} latent features, '
                f'module blocks of shape {self.module_shape} '
                f'and {self.sequence_features} features between blocks trained '
                f'for {self.epochs} epochs with batches of size {self.batch_size}')

    def fpath(self, ident: str='') -> str:
        """Returns a file path to save the model to, based on its parameters."""
        base = './results/sanv2models/'
        s = self.module_shape + [self.sequence_features]
        ms = '_'.join([str(l) for l in s])
        name = (f'l{self.likelihood.name}_cd{self.cond_dim}_'
                f'ed{self.latent_features}_'
                f'dd{self.data_dim}_ms{ms}_'
                f'lp{self.likelihood.n_params()}_ln{self.layer_norm}_'
                f'lr{self.lr}_ld{self.decay}_e{self.epochs}_'
                f'bs{self.batch_size}_trsample{self.train_rsample}_')
        name += 'lim_' if self.limits is not None else ''
        self.savepath_cached = f'{base}{name}{ident}.pt'

        return self.savepath_cached

    def _do_init(self, mp: ModelParams):
        """Initialises the network modules based on the configuration"""

        assert isinstance(mp, SANv2Params)

        # SANv1 properties
        self.first_module_shape = mp.first_module_shape
        self.module_shape = mp.module_shape
        self.sequence_features = mp.sequence_features
        self.lr = mp.opt_lr
        self.decay = mp.opt_decay
        # SANv2 properties
        self.latent_features = mp.latent_features
        self.encoder_layers = mp.encoder_layers

        # initialise the encoder block:

        self.encoder = nn.Sequential()
        hs = [self.cond_dim] + self.encoder_layers + [self.latent_features]
        for i, (j, k) in enumerate(zip(hs[:-1], hs[1:])):
            self.encoder.add_module(name=f'E{i}', module=nn.Linear(j, k))
            if self.layer_norm:
                self.encoder.add_module(name=f'E_LN{i}', module=nn.LayerNorm(k))
            self.encoder.add_module(name=f'E_A{i}', module=nn.ReLU())
        self.encoder.to(self.device, self.dtype)

        # initialise the sequential blocks:

        self.sequential_blocks = nn.ModuleList()
        self.sequential_block_heads = nn.ModuleList()

        for (i, d) in enumerate(range(self.data_dim)):
            b, h = self._sequential_block(self.latent_features, d,
                    self.module_shape,
                    out_shapes=[self.sequence_features, self.likelihood.n_params()],
                    out_activations=[nn.ReLU, None])
            self.sequential_blocks.append(b)
            self.sequential_block_heads.append(h)

        self.opt = t.optim.Adam(self.parameters(), lr=self.lr,
                                weight_decay=self.decay)

        # We need to initialise an encoder block (taking input_features to latent_features)
        #
        # We then need to initialise the data_dim sequential blocks. This could
        # be done by the superclass for us, so long as we set the .

        self.latent_features = mp.latent_features
        self.module_shape = mp.module_shape
        self.sequence_features = mp.sequence_features
        self.lr = mp.opt_lr
        self.decay = mp.opt_decay

    def forward(self, x: Tensor, lp: bool = False,
                rsample: bool = False) -> Tensor:
        """Forward pass through the model.

        Args:
            x: a (batch of) conditioning information [batch_size, cond_dim]
            lp: whether to save the last parameters (used for evaluating likelihoods)
            rsample: whether to stop gradients (False) or not (True) in the
                autoregressive sampling step.

        Returns:
            Tensor: a sample from the distribution; y_hat ~  p(y | x)

        Implicit Returns:
            self.lats_params: if lp=True, a tensor containing the parameters of
            each marginal distribution of size [batch_size, lparams] is saved.
            This allows us to evaluate p(y | x) later.
        """

        B = x.shape[:-1]

        # marginals
        ys = t.empty(B + (0,), dtype=self.dtype, device=self.device)

        if lp:
            self.last_params = t.empty(B + (0, self.likelihood.n_params()),
                                       dtype=self.dtype, device=self.device)
        else:
            # invalidate any previously saved parameters to avoid confusion
            self.last_params = None

        seq_features = t.empty(B + (0,), dtype=self.dtype, device=self.device)

        z = self.encoder(x)

        for d in range(self.data_dim):

            d_input = t.cat((z, seq_features, ys), -1)

            H = self.sequential_blocks[d](d_input)

            # for passing into next sequential block
            seq_features = self.sequential_block_heads[d][0](H)

            # draw single sample from p(y_d | y_<d, z)
            params = self.sequential_block_heads[d][1](H)
            if rsample:
                y_d = self.likelihood.rsample(params, d)
            else:
                y_d = self.likelihood.sample(params, d)
            assert not y_d.isnan().all(), "NaN values returned from likelihood."

            y_d = y_d[(..., ) + (None, ) * (ys.dim() - y_d.dim())]

            ys = t.cat((ys, y_d), -1)
            if lp:
                self.last_params = t.cat((self.last_params, params.unsqueeze(-2)), x.dim()-1)

        # check we did the sampling right
        assert ys.shape == B + (self.data_dim,)
        return ys


class PModelv2(SANv2):
    """A SANv2 which is slightly adapted to act as a likelihood / forward model
    emulator by switching the xs and thetas in the preprocessing step."""
    def preprocess(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        return y.to(self.device, self.dtype), x.to(self.device, self.dtype)


if __name__ == '__main__':

    from spt import config as cfg
    from spt.load_photometry import load_simulated_data, get_norm_theta

    logging.info(f'Beginning SAN training')
    sp = cfg.SANParams()
    s = SAN(sp)
    logging.info(s)

    fp = cfg.ForwardModelParams()
    ip = cfg.InferenceParams()

    train_loader, test_loader = load_simulated_data(
        path=ip.dataset_loc,
        split_ratio=ip.split_ratio,
        batch_size=sp.batch_size,
        phot_transforms=[t.from_numpy, t.log],
        theta_transforms=[get_norm_theta(fp)],
    )

    s.offline_train(train_loader, ip)

    logging.info(f'Exiting')
