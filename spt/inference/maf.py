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
import torch as t
import torch.nn as nn
import torch.nn.functional as F

from abc import abstractmethod
from typing import Any, Callable, Optional, Type
from torch.utils.data import DataLoader
from torch.distributions import Beta, Categorical, Distribution, Normal, MixtureSameFamily, Uniform
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
                 bias: bool = True, dtype: t.dtype = None,
                 device: t.device = None) -> Tensor:
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', t.ones(out_features, in_features,
                                            dtype=dtype, device=device))

    def set_mask(self, mask: Tensor):
        self.mask = mask.to(dtype=t.int8).T

    def forward(self, x: Tensor):
        return F.linear(x, self.mask * self.weight, self.bias)


class MADE(nn.Module):
    def __init__(self, in_features: int, hidden_sizes: int, out_features: int,
                 num_masks: int = 1, natural_ordering: bool = False,
                 dtype: t.dtype = None, device: t.device = None):
        super().__init__()
        self.in_features = in_features
        self.hidden_sizes = hidden_sizes
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        assert self.out_features % self.in_features == 0

        self.net = []
        hs = [in_features] + hidden_sizes + [out_features]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                MaskedLinear(h0, h1, device=device, dtype=dtype),
                nn.LayerNorm(h1),
                nn.ReLU(),
            ])
        self.net.pop() # pop last activation
        self.net.pop() # pop LayerNorm too
        self.net = nn.Sequential(*self.net)

        self.natural_ordering = natural_ordering
        self.num_masks = num_masks
        self.seed = 0

        self.m = {}
        self.update_masks() # builds initial self.m connectivity.

    def update_masks(self):
        if self.m and self.num_masks == 1: return
        L = len(self.hidden_sizes)

        self.seed = (self.seed + 1) % self.num_masks

        if self.natural_ordering:
            self.m[-1] = t.arange(self.in_features, device=self.device,
                                  dtype=self.dtype)
        else:
            self.m[-1] = t.randperm(self.in_features, device=self.device,
                                    dtype=self.dtype)

        for l in range(L):
            self.m[l] = t.randint(int(self.m[l-1].min().item()),
                                  self.in_features-1,
                                  size=(self.hidden_sizes[l],),
                                  device=self.device, dtype=self.dtype)

        # construct mask matrices
        masks = [self.m[l-1][:, None] <= self.m[l][None, :] for l in range(L)]
        masks.append(self.m[L-1][:, None] < self.m[-1][None, :])

        # when out_features = in_features * k, for k > 1
        if self.out_features > self.in_features:
            k = int(self.out_features / self.in_features)
            # replicate the mask across the other outputs
            # masks[-1] = np.concatenate([masks[-1]]*k, axis=1)
            masks[-1] = t.cat([masks[-1]]*k, 1)

        # set the masks in all MaskedLinear layers
        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
        for layer, m in zip(layers, masks):
            layer.set_mask(m)

    def forward(self, x: Tensor):
        return self.net(x)


# Autoregressive multi-layer perceptron ---------------------------------------

class ARMLP(nn.Module):
    """An n-layer autoregressive MLP."""

    def __init__(self, in_features: int, hidden_sizes: list[int],
                 out_features: int, num_masks: int = 1,
                 natural_ordering: bool = True, device: t.device = None,
                 dtype: t.dtype = None):
        super().__init__()
        self.net = MADE(in_features, hidden_sizes, out_features,
                        num_masks=num_masks, natural_ordering=natural_ordering,
                        device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# Masked autoregressive flow --------------------------------------------------


class MAF(nn.Module):
    """Masked autoregressive flow that uses a MADE-style network for fast
    forward propagation."""

    def __init__(self, dim: int, hidden_width: int = 24, depth: int = 4,
                 num_masks: int = 1, natural_ordering: bool = True,
                 parity: bool = True, device: t.device = None,
                 dtype: t.dtype = None):
        super().__init__()
        self.dim = dim
        self.net = ARMLP(in_features=dim,
                         hidden_sizes=[hidden_width] * depth,
                         out_features=dim*2,
                         num_masks=num_masks,
                         natural_ordering=natural_ordering,
                         device=device, dtype=dtype)
        self.parity = parity

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # Evaluate all xs in parallel; fast density estimation
        st = self.net(x) # invoke MADE
        s, T = st.split(self.dim, dim=1)
        z = x * t.exp(s) + T
        z = z.flip(dims=(1,)) if self.parity else z
        log_det = t.sum(s, dim=1)
        return z, log_det

    def backward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        # we need to decode the x one at a time
        x = t.zeros_like(z)  # prepare a buffer
        log_det = t.zeros(z.size(0), device=z.device, dtype=z.dtype)
        z = z.flip(dims=(1,)) if self.parity else z
        for i in range(self.dim):
            st = self.net(x.clone())  # TODO remove clone if not IAF?
            s, T = st.split(self.dim, dim=1)
            x[:, i] = (z[:, i] - T[:, i]) * t.exp(-s[:, i])
            log_det += -s[:, i]
        return x, log_det


def logistic_prior(dim: int, device: t.device = None,
                   dtype: t.dtype = None) -> Distribution:
    return TransformedDistribution(Uniform(t.zeros(dim, device=device, dtype=dtype),
                                           t.ones(dim, device=device, dtype=dtype)),
                                   SigmoidTransform().inv)


class NormalisingFlow(nn.Module):
    """Normalising flows can be composed as a sequence by repeated applications
    of the change of variables rule.k"""

    def __init__(self, flows: list[nn.Module]):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        m, _ = x.shape
        log_det = t.zeros(m, dtype=x.dtype, device=x.device)
        zs = [x]
        for flow in self.flows:
            x, ld = flow.forward(x)
            log_det += ld
            zs.append(x)
        return zs, log_det

    def backward(self, z) -> tuple[Tensor, Tensor]:
        m, _ = z.shape
        log_det = t.zeros(m, dtype=z.dtype, device=z.device)
        xs = [z]
        for flow in self.flows[::-1]:  # iterate through flows in reverse order
            z, ld = flow.backward(z)
            log_det += ld
            xs.append(z)
        return xs, log_det


class NormalisingFlowModel(nn.Module):
    """A normalising flow is a (prior, flow) pair."""

    def __init__(self, prior: Distribution, flows: list[nn.Module]):
        super().__init__()
        self.prior = prior
        self.flow = NormalisingFlow(flows)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        zs, log_det = self.flow.forward(x)
        prior_logprob = self.prior.log_prob(zs[-1]).view(x.size(0), -1).sum(1)
        return zs, prior_logprob, log_det

    def backward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        xs, log_det = self.flow.backward(z)
        return xs, log_det

    def sample(self, num_samples: int) -> Tensor:
        z = self.prior.sample((num_samples,))
        xs, _ = self.flow.backward(z)
        return xs


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
