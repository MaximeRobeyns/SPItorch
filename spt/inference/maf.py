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
Implements a masked autoregressive flow.
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
from torch.distributions import Beta, Categorical, Distribution, Normal,\
                                MixtureSameFamily, Uniform
from torch.distributions import TransformedDistribution
from torch.distributions.transforms import SigmoidTransform

import spt.config as cfg

from spt.inference import Model, ModelParams, InferenceParams
from spt.types import Tensor, tensor_like
from spt.inference.utils import squareplus_f, TruncatedNormal


# MADE ------------------------------------------------------------------------

class MaskedLinear(nn.Linear):
    """Linear layer with a configurable mask on the weights"""
    def __init__(self, in_features: Tensor, out_features: Tensor,
                 maskset: Tensor, bias: bool = True,
                 dtype: t.dtype = None, device: t.device = None) -> Tensor:
        super().__init__(in_features, out_features, bias)
        self.maskset = maskset
        self.num_masks = maskset.size(0)
        self.device, self.dtype = device, dtype
        self.register_buffer('mask', maskset[0].T)

    def update_mask(self, idx: int):
        assert idx < self.num_masks, "Index must be less than number of masks"
        self.register_buffer('mask', self.maskset[idx].T)

    def forward(self, x: Tensor):
        return F.linear(x, self.mask * self.weight, self.bias)


class CMADE(nn.Module):

    def __init__(self, in_features: int, cond_features: int, hidden_sizes: int,
                 out_features: int, num_masks: int = 1,
                 natural_ordering: bool = True, dtype: t.dtype = None,
                 device: t.device = None):
        super().__init__()
        self.in_features = in_features
        self.cond_features = cond_features
        self.hidden_sizes = hidden_sizes
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        assert self.out_features % self.in_features == 0

        # Setup GPU mask cache ================================================

        self.mask_idx, self.num_masks = 0, num_masks
        # TODO: extend model to support order-agnostic training.
        # self.natural_ordering = natural_ordering
        self.natural_ordering = True

        # Initialise a list of masks for each layer stored on GPU.
        self.masksets, self.orderings = self.initialise_masks()

        # Setup model ========================================================

        self.net = []
        hs = [cond_features + in_features] + hidden_sizes + [out_features]
        for i, (h0, h1) in enumerate(zip(hs, hs[1:])):
            self.net.extend([
                MaskedLinear(h0, h1, self.masksets[i],
                             device=device, dtype=dtype),
                # WARNING: does this have an impact on the Jacobian?
                # It is perhaps safer to leave it out.
                nn.LayerNorm(h1),
                nn.ReLU(),
            ])
        self.net.pop()  # pop last activation
        self.net.pop()  # pop LayerNorm too
        self.net = nn.Sequential(*self.net).to(self.device, self.dtype)


    def initialise_masks(self) -> tuple[list[Tensor], Tensor]:
        hs: list[int] = [self.cond_features + self.in_features] + \
                        self.hidden_sizes + [self.out_features]
        L = len(self.hidden_sizes)
        assert len(hs) == L+2

        Ms: dict[[int], Tensor] = {}
        if self.natural_ordering:
            Ms[-1] = t.cat((
                t.zeros(self.cond_features, device=self.device, dtype=self.dtype, requires_grad=False),
                t.arange(self.in_features, device=self.device, dtype=self.dtype, requires_grad=False),
                )).repeat((self.num_masks, 1))
        else:
            tmp: list[Tensor] = []
            for _ in range(self.num_masks):
                tmp_m = t.cat((
                    t.zeros(self.cond_features, device=self.device, dtype=self.dtype, requires_grad=False),
                    t.randperm(self.in_features, device=self.device, dtype=self.dtype, requires_grad=False),
                ), -1)[None, :]
                tmp.append(tmp_m)
            Ms[-1] = t.cat(tmp, 0)

        assert Ms[-1].shape == (self.num_masks, self.cond_features + self.in_features)


        for l in range(L):
            tmp: list[Tensor] = []
            for i in range(self.num_masks):
                if self.in_features-1 == 0:
                    tmp_m = t.zeros(
                        size=(self.hidden_sizes[l],),
                        device=self.device, dtype=self.dtype)
                else:
                    tmp_m = t.randint(
                        int(Ms[l-1][i].min().item()),
                        self.in_features-1,
                        size=(self.hidden_sizes[l],),
                        device=self.device, dtype=self.dtype)
                tmp.append(tmp_m[None, :])
            Ms[l] = t.cat(tmp, 0)

        assert len(Ms.values()) == L + 1

        # Now create the corresponding masks ----------------------------------

        masks = [Ms[l-1].unsqueeze(-1) <= Ms[l].unsqueeze(-2) for l in range(L)]
        # masks.append(Ms[L-1].unsqueeze(-1) < Ms[-1].unsqueeze(-2))
        masks.append(Ms[L-1].unsqueeze(-1) < Ms[-1][..., -self.in_features:].unsqueeze(-2))

        if self.out_features > self.in_features:
            k = int(self.out_features / self.in_features)
            # repeat the last mask k times
            masks[-1] = t.cat([masks[-1]]*k, -1)

        masks = [m.to(self.device, t.uint8) for m in masks]

        assert masks[0].shape == (self.num_masks, self.cond_features + self.in_features, self.hidden_sizes[0])
        for l in range(1, L):
            assert masks[l].shape == (self.num_masks, self.hidden_sizes[l-1], self.hidden_sizes[l])
        assert masks[-1].shape == (self.num_masks, self.hidden_sizes[-1], self.out_features)
        assert len(masks) == len(self.hidden_sizes) + 1

        return masks, (Ms[0] - 1)[:, -self.in_features:]

    def rotate_masks(self):
        if self.num_masks == 1: return

        self.mask_idx = (self.mask_idx + 1) % self.num_masks
        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
        for l in layers:
            l.update_mask(self.mask_idx)

    def forward(self, c: Tensor, x: Tensor):
        assert c.size(-1) == self.cond_features
        assert x.size(-1) == self.in_features
        self.rotate_masks()
        return self.net(t.cat((c, x), -1))


# MAF prior -------------------------------------------------------------------


class MAFPrior(Distribution):
    """A prior / base distribution to use with a normalising flow"""

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class LogisticPrior(TransformedDistribution, MAFPrior):

    def __init__(self, dim: int, device: t.device = None,
                 dtype: t.dtype = None):
        TransformedDistribution.__init__(
            self,
            Uniform(
                t.zeros(dim, device=device, dtype=dtype),
                t.ones(dim, device=device, dtype=dtype)),
            SigmoidTransform().inv)

    @property
    def name(self) -> str:
        return "logistic_prior"


class GaussianPrior(Normal, MAFPrior):
    """Standard Gaussian prior"""

    def __init__(self, dim: int, device: t.device = None,
                 dtype: t.dtype = None):
        Normal.__init__(self, t.zeros(dim, device=device, dtype=dtype),
                        t.ones(dim, device=device, dtype=dtype))

    @property
    def name(self) -> str:
        return "Gaussian"


# MAF 'block' -----------------------------------------------------------------
# A single MAF network, multiple of which are composed to give the main MAF
# model (flows of flows are flows), but without all the accompanying wiring that
# comes with SPItorch models.


class MAFBlock(nn.Module):
    """Masked autoregressive flow that uses a conditional MADE-style network for
    fast forward propagation"""

    def __init__(self, cond_dim: int, data_dim: int, hidden_width: int = 24,
                 depth: int = 4, num_masks: int = 1,
                 natural_ordering: bool = True, parity: int = 0,
                 device: t.device = None, dtype: t.dtype = None):
        super().__init__()
        self.dim = data_dim
        self.net = CMADE(in_features=data_dim,
                         cond_features=cond_dim,
                         hidden_sizes=[hidden_width] * depth,
                         out_features=data_dim*2,
                         num_masks=num_masks,
                         natural_ordering=natural_ordering,
                         device=device, dtype=dtype)
        self.parity = parity

    def forward(self, c: Tensor, x: Tensor) -> tuple[Tensor, Tensor]:
        # Evaluate all xs in parallel: fast density estimation
        st = self.net(c, x)
        s, T = st.split(self.dim, dim=1)
        z = x * t.exp(s) + T
        z = z.flip(dims=(1,)) if self.parity == 1 else z
        log_det = t.sum(s, dim=1)
        return z, log_det

    def backward(self, c: Tensor, z: Tensor) -> tuple[Tensor, Tensor]:
        # decode the x one at a time.
        x = t.zeros_like(z)
        log_det = t.zeros(z.size(0), device=z.device, dtype=z.dtype)
        z = z.flip(dims=(1,)) if self.parity == 1 else z
        for i in range(self.dim):
            st = self.net(c, x.clone())  # TODO: is clone necessary?
            s, T = st.split(self.dim, dim=1)
            x[:, i] = (z[..., i] - T[..., i]) * t.exp(-s[..., i])
            log_det += -s[..., i]
        return x, log_det


# Masked autoregressive flow --------------------------------------------------


class MAFParams(ModelParams):
    """Configuration class for MAF"""

    def __init__(self):
        super().__init__()
        # perform any required validation here...

    @property
    @abstractmethod
    def prior(self) -> MAFPrior:
        raise NotImplementedError

    @property
    def depth(self) -> int:
        """Number of MAF stacked on top of each other."""
        return 1

    @property
    def maf_depth(self) -> int:
        """Depth of each of the MAFs"""
        return 1

    @property
    def maf_hidden_width(self) -> int:
        """Width of each of the MAF hidden layers"""
        return 64

    @property
    def maf_num_masks(self) -> int:
        """Number of masks to use in each MAF for connectivity-agnostic training
        in each MADE block.
        (This is basically a regularisation parameter)
        """
        return 1

    @property
    def natural_ordering(self) -> bool:
        """Whether to use natural ordering for order-agnostic training in each
        MADE block."""
        return True

    @property
    def layer_norm(self) -> bool:
        """Whether to use layer norm (batch normalisation) or not"""
        return True

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


class MAF(Model):

    def __init__(self, mp: MAFParams,
                 logging_callbacks: list[Callable] = []):
        """Masked Autoregressive Flow implementation

        Args:
            mp: the MAF parameters, set in `config.py`.
            logging_callbacks: list of callables accepting this model instance;
                useful for visualisations and debugging.
        """
        super().__init__(mp, logging_callbacks)

        self.dim = mp.data_dim
        self.prior = mp.prior(self.dim, mp.device, mp.dtype)

        # TODO find out how to incorporate limits...
        if mp.limits is not None:
            if not isinstance(mp.limits, t.Tensor):
                self.limits = t.tensor(mp.limits, dtype=mp.dtype, device=mp.device)
            else:
                self.limits = mp.limits.to(mp.device, mp.dtype)
        else:
            self.limits = None

        flow_blocks = [
            MAFBlock(cond_dim=mp.cond_dim, data_dim=mp.data_dim,
                     hidden_width=mp.maf_hidden_width,
                     depth=mp.maf_depth, num_masks=mp.maf_num_masks,
                     natural_ordering=mp.natural_ordering, parity=i%2,
                     device=mp.device, dtype=mp.dtype)
            for i in range(mp.depth)
        ]
        self.flows = nn.ModuleList(flow_blocks)

        self.lr = mp.opt_lr
        self.decay = mp.opt_decay
        self.opt = t.optim.Adam(self.parameters(), lr=self.lr,
                                weight_decay=self.decay)

        self.to(mp.device, mp.dtype)

        self.savepath_cached: str = ""

    name: str = 'MAF'

    def __repr__(self) -> str:
        return (f'{self.name} with {self.prior.name} base distribution '
                f'consisting of {self.mp.depth} stacked MAFs, each with '
                f'{self.mp.maf_depth} layers of width {self.mp.maf_hidden_width}'
                f' with {"no" if self.mp.natural_ordering else ""} natural '
                f'ordering, {"no" if self.mp.layer_norm else ""} layer norm'
                f'trained for {self.mp.epochs} epochs with batches of size '
                f'{self.mp.batch_size}, an Adam learning rate of {self.mp.opt_lr}'
                f' and decay of {self.mp.opt_decay}.')

    def fpath(self, ident: str='') -> str:
        """Returns a file path to save the model to, based on its parameters"""
        base = './results/mafmodels/'
        name = (f'p{self.prior.name}_cd{self.cond_dim}_dd{self.data_dim}_'
                f'd{self.mp.depth}_md{self.mp.maf_depth}_mhw_{self.mp.maf_hidden_width}'
                f'_nm{self.mp.maf_num_masks}_no{self.mp.natural_ordering}_'
                f'ln{self.mp.layer_norm}_lr{self.mp.opt_lr}'
                f'_od{self.mp.opt_decay}_')
        name += 'lim_' if self.mp.limits is not None else ''
        self.savepath_cached = f'{base}{name}{ident}.pt'
        return self.savepath_cached


    def flow_forward(self, c: Tensor, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """'forward' in the conditional flow setting.

        Given conditioning information c and data x, this returns the latents
        zs, their prior density and the log determinant term.
        """
        m, _ = x.shape
        log_det = t.zeros(m, dtype=x.dtype, device=x.device)
        zs = [x]
        for flow in self.flows:
            x, ld = flow.forward(c, x)
            log_det += ld
            zs.append(x)
        # prior_logprob = self.prior.log_prob(zs[-1]).view(x.size(0), -1).sum(1)
        prior_logprob = self.prior.log_prob(zs[-1]).view(x.size(0), -1).sum(-1)
        return zs, prior_logprob, log_det

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Map from conditioning information x to data y.

        Note: 'forward' is a bit of a misnomer in this setting because we are
        sampling some latents from the base distribution, and then calling
        'backwards' on the conditional flow to get the data.

        Args:
            x: some conditioning information [B, cond_dim]

        Returns:
            Tensor: a sample from the distribution y_hat ~ p(y | x)
        """
        if x.dim() == 1:
            x = x[None, :]
        zs = self.prior.sample(x.shape[:-1])
        xs, _ = self.flow_backward(x, zs)  # TODO avoid computing log_det
        return xs[-1]

    def flow_backward(self, c: Tensor, z: Tensor) -> tuple[Tensor, Tensor]:
        """'backward' in the conditional flow setting

        Given conditioning information c and some latents z, this returns the
        data x as well as the log determinant term.
        """
        m, _ = z.shape
        log_det = t.zeros(m, dtype=z.dtype, device=z.device)
        xs = [z]
        for flow in self.flows[::-1]:  # iterate through flows in reverse order
            z, ld = flow.backward(c, z)
            log_det += ld
            xs.append(z)
        return xs, log_det

    def offline_train(self, train_loader: DataLoader, ip: InferenceParams,
                      *args, **kwargs) -> None:
        """Train the MAF model.

        Args:
            train_loader: DataLoader to load the training data.
            ip: The parameters to use for training, defined
                in `config.ip:InferenceParams`
        """
        t.random.seed()
        self.train()

        start_e = self.attempt_checkpoint_recovery(ip)
        for e in range(start_e, self.epochs):
            for i, (x, y) in enumerate(train_loader):
                x, y = self.preprocess(x, y)

                _, prior_logprob, log_det = self.flow_forward(x, y)
                LP = prior_logprob + log_det
                # loss = -LP.sum(-1).mean()
                loss = -t.sum(LP) # .mean()

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
        self.eval()

    def _preprocess_sample_input(self, x: tensor_like, n_samples: int = 1000,
                                 errs: Optional[tensor_like] = None) -> Tensor:

        if isinstance(x, np.ndarray):
            x = t.from_numpy(x)

        if not isinstance(x, t.Tensor):
            raise ValueError((
                f'Please provide a PyTorch Tensor (or numpy array) as input '
                f'(got {type(x)})'))

        x = x.unsqueeze(0) if x.dim() == 1 else x

        x, _ = self.preprocess(x, t.empty(x.shape))
        n, d = x.shape

        if errs is not None:
            if isinstance(errs, np.ndarray):
                errs = t.from_numpy(errs)
            errs = errs.unsqueeze(0) if errs.dim() == 1 else errs
            x = Normal(x, errs).sample((n_samples,)).reshape(-1, d)
        else:
            x = x.repeat_interleave(n_samples, 0)

        assert x.shape == (n * n_samples, d)
        return x

    def sample(self, x: tensor_like, n_samples: int = 1000,
               errs: Optional[tensor_like] = None) -> Tensor:
        """Draw conditional samples from p(y | x)

        Args:
            x: the conditioning data; x, [B, cond_dim]
            n_samples: the number of samples to draw.

        Returns:
            Tensor: a tensor of shape [n_samples, data_dim]
        """
        x_pre = self._preprocess_sample_input(x, n_samples, errs)
        z = self.prior.sample(x_pre.shape[:-1])
        xs, _ = self.flow_backward(x_pre, z)
        return xs[-1]

    def mode(self, x: tensor_like, n_samples: int = 1000,
             errs: Optional[tensor_like] = None) -> Tensor:
        """A convenience method which returns the highest posterior mode for a
        given batch of photometric observations, x.

        Args:
            x: the conditioning data.
            n_samples: the number of saples to draw when searching for the mode.

        Returns:
            Tensor: a tensor of modes [data_dim]
        """
        N = n_samples
        x_pre = self._preprocess_sample_input(x, n_samples, errs)
        B = int(x_pre.size(0) / N)

        # get y
        z = self.prior.sample(x_pre.shape[:-1])
        y, _ = self.flow_backward(x_pre, z)

        _, lps, log_det = self.flow_forward(x, y)
        rlps = lps.reshape(B, N)

        # [B, 1, data_dim]
        idxs = t.argmax(rlps, dim=1)[:, None, None].expand(B, 1, self.data_dim)
        modes = y.gather(1, idxs).squeeze(1) # [B, data_dim]

        return modes


class PModel(MAF):
    """A MAF which acts as a likelihood / forward module emulator by switching
    the xs and thetas in the preprocessing step.
    """
    def preprocess(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        return y.to(self.device, self.dtype), x.to(self.device, self.dtype)


if __name__ == '__main__':
    pass

    # from spt import config as cfg
    # from spt.load_photometry import load_simulated_data, get_norm_theta

    # logging.info(f'Beginning SAN training')
    # sp = cfg.SANParams()
    # s = SAN(sp)
    # logging.info(s)

    # fp = cfg.ForwardModelParams()
    # ip = cfg.InferenceParams()

    # train_loader, test_loader = load_simulated_data(
    #     path=ip.dataset_loc,
    #     split_ratio=ip.split_ratio,
    #     batch_size=sp.batch_size,
    #     phot_transforms=[t.from_numpy],
    #     theta_transforms=[get_norm_theta(fp)],
    # )

    # s.offline_train(train_loader, ip)

    # logging.info(f'Exiting')
