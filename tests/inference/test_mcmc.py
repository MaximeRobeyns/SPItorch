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
"""Tests for MCMC samplers"""

import torch as t

from spt.types import Tensor
from spt.inference import HMC, HMC_sampler

# Constants -------------------------------------------------------------------

dtype = t.float32
device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")

# Utilities -------------------------------------------------------------------


def mixture_logpdf(x: Tensor) -> Tensor:
    loc = t.tensor([[-4, 0, 3.2, 2.5]], dtype=dtype, device=device).T
    scale = t.tensor([[1.2, 1, 5, 2.8]], dtype=dtype, device=device).T
    weights = t.tensor([[0.3, 0.3, 0.1, 0.3]], dtype=dtype, device=device).T  # sums to 1
    log_probs = t.distributions.Normal(loc, scale).log_prob(x.squeeze())
    return t.logsumexp(t.log(weights) + log_probs, 0).unsqueeze(0)  # unsqueeze to emulate batch 1


def test_HMC_single_batch():
    # 1 batch dimension

    N = 200
    chains = 400

    dim = 1
    find_max = True  # TODO test without this option

    rho = 0.2
    L = 6
    alpha = 1.01

    # individual samples are size [1, chains, dim] = [1, 400, 1]
    pos = t.randn((1, chains, dim), device=device, dtype=dtype)
    sampler = HMC(mixture_logpdf, pos, rho, L, alpha)

    for (tmp, i) in zip(sampler, range(N)):
        assert tmp.shape == t.Size((1, chains, dim))
        break

    # TODO: complete this
