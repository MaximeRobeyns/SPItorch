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
Implements a "sequential autoregressive network"; a simple, sequential
procedure for generating autoregressive samples.
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
from torch.distributions import Categorical, Normal, MixtureSameFamily

import spt.config as cfg

from spt.inference import Model, ModelParams, InferenceParams
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
    # must have lower abd upper properties

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
    """Simple, univariate Gaussian likelihood for each dimension"""

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
        B = params.size(0)
        loc, scale, k = params.reshape(B, -1, self.K, 3).tensor_split(3, 3)
        return loc.squeeze(-1), squareplus_f(scale).squeeze(-1), k.squeeze(-1)
        # F.softmax(k.squeeze(), -1).squeeze(-1)

    def _stabilise(self, S: Tensor) -> Tensor:
        while not S.gt(0.).all():
            S = S.abs()*(1.+self.mult_eps) + self.abs_eps
        return S

    def _stable_norms(self, loc: Tensor, scale: Tensor) -> Normal:
        try:
            return t.distributions.Normal(loc, scale)
        except ValueError:
            return t.distributions.Normal(loc, self._stabilise(scale))

    def _gmm_from_params(self, params: Tensor) -> MixtureSameFamily:
        loc, scale, k = self._extract_params(params)
        cat = t.distributions.Categorical(logits=k)
        norms = self._stable_norms(loc, scale)
        return t.distributions.MixtureSameFamily(cat, norms)

    def log_prob(self, value: Tensor, params: Tensor) -> Tensor:
        return self._gmm_from_params(params).log_prob(value)

    def sample(self, params: Tensor, _: Optional[int] = None) -> Tensor:
        return self._gmm_from_params(params).sample()

    def rsample(self, params: Tensor, _: Optional[int] = None) -> Tensor:
        raise NotImplementedError
        # sample_shape = t.Size()
        # loc, scale, k = self._extract_params(params)
        # gather_dim = len(loc.shape) - 1
        # es = t.Size([])

        # # mix_sample = t.distributions.Categorical(logits=k).sample()
        # # Rather than re-sampling the mixture components, we freeze them.
        # mix_sample = kwargs['mix_sample']
        # mix_shape = mix_sample.shape

        # comp_samples = self._stable_norms(loc, scale).rsample(sample_shape)

        # # gather along the k dimension
        # mix_sample_r = mix_sample.reshape(
        #     mix_shape + t.Size([1] * (len(es) + 1)))
        # mix_sample_r = mix_sample_r.repeat(
        #     t.Size([1] * len(mix_shape)) + t.Size([1]) + es)

        # samples = t.gather(comp_samples, gather_dim, mix_sample_r)
        # return samples.squeeze(gather_dim)

    def rsample_all(self, params: Tensor) -> Tensor:
        """Returns output K times the size of the input"""
        raise NotImplementedError
        # TODO: finish this.
        # loc, scale, _ = self._extract_params(params)
        # comp_samples = self._stable_norms(loc, scale).rsample(t.Size())
        # # concatenate along the k dimension
        # print(comp_samples.shape)
        # return comp_samples


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
        self._lower, self._upper = lims[:, 0].detach(), lims[:, 1].detach()
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

        f = lambda x: x.squeeze(-1)# .nan_to_num()

        B = params.size(0)
        loc, scale, k = params.reshape(B, -1, self.K, 3).tensor_split(3, 3)
        return f(loc), f(squareplus_f(scale)), f(k)
        # return loc.squeeze(-1), squareplus_f(scale).squeeze(-1), k.squeeze(-1)

    def _stabilise(self, S: Tensor) -> Tensor:
        while not S.gt(0.).all():
            S = S.abs() * (1. + self.mult_eps) + self.abs_eps
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

    def _gmm_from_params(self, params: Tensor, d: Optional[int] = None) -> MixtureSameFamily:
        loc, scale, k = self._extract_params(params)
        cat = t.distributions.Categorical(logits=k)

        if d is None:  # [B, D, n_params]
            A, B = self._lower[None, ..., None], self._upper[None, ..., None]
        else:  # [B, 1, K]
            A, B = self._lower[d][None, ..., None], self._upper[d][None, ..., None]
        A = A.expand(loc.shape).to(loc.device, loc.dtype)
        B = B.expand(loc.shape).to(loc.device, loc.dtype)

        norms = self._stable_norms(loc, scale, A, B)
        return t.distributions.MixtureSameFamily(cat, norms)

    def log_prob(self, value: Tensor, params: Tensor) -> Tensor:
        # We compute the log prob ourselves here due to a bug in
        # MixtureSameFamily which prevents us from finding the correct support.

        loc, scale, k = self._extract_params(params)
        A = self._lower[None, ..., None].expand(loc.shape).to(loc.device, loc.dtype)
        B = self._upper[None, ..., None].expand(loc.shape).to(loc.device, loc.dtype)
        value = value[..., None].expand(loc.shape)
        norms = self._stable_norms(loc, scale, A, B)
        LP = norms.log_prob(value)
        wts = t.log_softmax(k, dim=-1)
        return t.logsumexp(LP + wts, dim=-1)

    def sample(self, params: Tensor, d: Optional[int] = None) -> Tensor:
        return self._gmm_from_params(params, d).sample().nan_to_num()

    def rsample(self, params: Tensor, d: Optional[int] = None) -> Tensor:
        raise NotImplementedError

    def rsample_all(self, params: Tensor):
        raise NotImplementedError


#     A _frozen_-weight mixture of Gaussians; that is, we freeze the mixture
#     weights, do some training, select some new ones, do some more training etc.
#     """
#
#     def n_params(self) -> int:
#         return 2 * self.K  # loc, scale
#
#     def _extract_params(self, params: Tensor) -> tuple[Tensor, Tensor]:
#         B = params.size(0)  # batch size
#         loc, scale = params.reshape(B, -1, self.K, 2).tensor_split(2, 2)
#         return loc.squeeze(-1), squareplus_f(scale).squeeze(-1)
#
#     def _dist_from_params(self, params: Tensor) -> t.distributions.Normal:
#         loc, scale = self._extract_params(params)
#         return self._stable_norms(loc, scale)
#
#     def log_prob(self, value: Tensor, params: Tensor) -> Tensor:
#         return self._dist_from_params(params).log_prob(value).mean(-1)
#
#     def sample(self, params: Tensor) -> Tensor:
#         # normal GMM sampling, with even logits
#
#     def rsample(self, params: Tensor) -> Tensor:
#         """Draw reparametrised samples.
#
#         This should only ever be for 1 dimension at a time. To pass in the full
#         set of parameters and draw samples that way is incorrect; since later
#         parameters must depend on previous ones.
#
#         A true sample from the SAN is a forward pass. There is no other way to
#         draw a sample.
#         """


class MoST(SAN_Likelihood):

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
    """A likelihood which is based on a normalising flow"""

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
        return 1e-4

    @property
    def limits(self) -> Optional[Tensor]:
        """Allows the output samples to be constrained to lie within the
        specified range.

        This method returns a (self.data_dim x 2)-dimensional tensor, with the
        (normalised) min and max values of each dimension of the output.
        """
        return None


# Main SAN Class ==============================================================


class SAN(Model):

    def __init__(self, mp: SANParams,
                 logging_callbacks: list[Callable] = []):
        """"Sequential autoregressive network" implementation.

        Args:
            mp: the SAN model parameters, set in `config.py`.
            logging_callbacks: list of callables accepting this model instance;
                often used for visualisations and debugging.
        """
        super().__init__(mp, logging_callbacks)

        self.module_shape = mp.module_shape
        self.sequence_features = mp.sequence_features
        self.lr = mp.opt_lr

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

        # Initialise the network
        self.network_blocks = nn.ModuleList()
        self.block_heads = nn.ModuleList()

        for d in range(self.data_dim):
            b, h = self._sequential_block(self.cond_dim, d, self.module_shape,
                      out_shapes=[self.sequence_features,
                      self.likelihood.n_params()],
                      out_activations=[nn.ReLU, None])
            self.network_blocks.append(b)
            self.block_heads.append(h)

        # A tensor storing the parameters of the distributions from which
        # samples were drawn for the last forward pass.
        # Useful for evaluating NLL of data under model.
        # Size: [mini-batch, likelihood_params]
        self.last_params: Optional[Tensor] = None

        self.opt = t.optim.Adam(self.parameters(), lr=self.lr)

        if mp.device == t.device('cuda'):
            self.to(mp.device, mp.dtype)
            self.likelihood.to(mp.device, mp.dtype)

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
        if self.savepath_cached == "":
            base = './results/sanmodels/'
            s = self.module_shape + [self.sequence_features]
            ms = '_'.join([str(l) for l in s])
            name = (f'l{self.likelihood.name}_cd{self.cond_dim}'
                    f'_dd{self.data_dim}_ms{ms}_'
                    f'lp{self.likelihood.n_params()}_ln{self.layer_norm}_'
                    f'lr{self.lr}_e{self.epochs}_bs{self.batch_size}_'
                    f'trsample{self.train_rsample}_')
            name += 'lim_' if self.limits is not None else ''
            self.savepath_cached = f'{base}{name}{ident}.pt'

        return self.savepath_cached

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

    def forward(self, x: Tensor, rsample: bool = False) -> Tensor:
        """Runs the autoregressive model.

        Args:
            x: some conditioning information
            rsample: whether to stop gradients (False) or not (True) in the
                autoregressive sampling step.

        Returns:
            Tensor: a sample from the distribution; y_hat ~ p(y | x)

        Implicit Returns:
            self.last_params: a tensor containing the parameters of each
                dimension's (univariate) distribution of size
                [mini-batch, lparams]; giving p(y | x)
        """
        assert x.dim() == 2, "Can only run SAN on two dimensional inputs"

        # batch size
        B = x.size(0)
        ys = t.empty((B, 0), dtype=self.dtype, device=self.device)
        self.last_params = t.empty((B, self.data_dim,
                                    self.likelihood.n_params()),
                                   dtype=self.dtype, device=self.device)

        seq_features = t.empty((B, 0), dtype=self.dtype, device=self.device)

        for d in range(self.data_dim):

            d_input = t.cat((x, seq_features, ys), 1)

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

            # TODO: remove
            # verifies that the samples are within the limits
            # assert t.logical_and(y_d >= 0., y_d <= 1.).all(), "some sampled values are out of range!"

            ys = t.cat((ys, y_d), -1)
            self.last_params[:, d] = params  # type: ignore

        # check we did the sampling right
        assert ys.shape == (x.size(0), self.data_dim)
        return ys

    def offline_train(self, train_loader: DataLoader, ip: InferenceParams,
                      *args, **kwargs) -> None:
        """Train the SAN model offline.

        Args:
            train_loader: DataLoader to load the training data.
            ip: The parameters to use for training, defined in
                `config.py:InferenceParams`.
        kwargs:
            rsample (Bool): whether to stop gradients (False) or not (True) in the
                autoregressive sampling step.
        """
        t.random.seed()
        self.train()

        start_e = 0
        # if not ip.retrain_model:
        start_e = self.attempt_checkpoint_recovery(ip)
        for e in range(start_e, self.epochs):
            for i, (x, y) in enumerate(train_loader):
                # print(f'iter e: {e}/{range(self.epochs)}, i: {i}/{len(train_loader)}')
                x, y = self.preprocess(x, y)

                # if the SAN has limits, then filter the y values here:
                #
                # TODO: Perhaps we need to filter out vlaues which are merely
                # close to the edges for better numerical stability?
                # We could define some eps term; thus y > self.likelihood.lower + eps...
                #
                if self.likelihood.supports_lim:
                    mask = t.logical_and(y > self.likelihood.lower,
                                         y < self.likelihood.upper).all(-1)
                    x, y = x[mask], y[mask]

                # The following implicitly updates self.last_params, and
                # returns y_hat (a sample from p(y | x))
                _ = self.forward(x, self.train_rsample)
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
        # This is very important to ensure that batch norm (among other model
        # features) work as expected.
        self.eval()

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
            kwargs: any additional model-specific arguments

        Returns:
            Tensor: a tensor of shape [n_samples, data_dim]
        """

        # if self.training:
        #     logging.warning('Model is still in training mode during sampling! '
        #                     'This is likely to give you unexpected results.')

        if isinstance(x_in, np.ndarray):
            x_in = t.from_numpy(x_in)

        if not isinstance(x_in, t.Tensor):
            raise ValueError((
                f'Please provide a PyTorch Tensor (or numpy array) to sample '
                f'(got {type(x_in)})'))

        x, _ = self.preprocess(x_in, t.empty(x_in.shape))

        x = x.unsqueeze(0) if x.dim() == 1 else x
        assert x.dim() == 2, ""
        n, d = x.shape

        if errs is not None:
            if isinstance(errs, np.ndarray):
                errs = t.from_numpy(errs)
            errs = errs.unsqueeze(0) if errs.dim() == 1 else errs
            x = t.distributions.Normal(x, errs).sample((n_samples,)).reshape(-1, d)
        else:
            x = x.repeat_interleave(n_samples, 0)

        assert x.shape == (n * n_samples, d)

        samples = self.forward(x, rsample)

        return samples


if __name__ == '__main__':

    from spt import config as cfg

    logging.info(f'Beginning SAN')
    sp = cfg.SANParams()
    s = SAN(sp)
    logging.info(s)
