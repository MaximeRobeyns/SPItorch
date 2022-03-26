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
Implements some very simple MCMC samplers.
"""

import math
import time
import logging
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F

from abc import abstractmethod
from typing import Any, Callable, Generator, Optional, Tuple, Type, Union
from torch.utils.data import DataLoader
from rich.progress import Progress

import spt.config as cfg

from spt.types import Tensor

__all__ = ["RWMH", "RWMH_sampler", "HMC", "HMC_sampler"]


def RWMH(logpdf: Callable[[Tensor], Tensor], initial_pos: Tensor,
         sigma: float = 0.1, bounds: Optional[Tensor] = None
         ) -> Generator[Tensor, None, None]:
    """
    Random walk Metropolis-Hastings.

    Args:
        logpdf: log probability density function
        initial     _pos: where to initialise the chains
        sigma: purturbation variance for proposals
        bounds: optional bounds on the acceptance volume
    """
    size = initial_pos.shape
    device, dtype = initial_pos.device, initial_pos.dtype
    pos = initial_pos
    log_prob = logpdf(pos)

    if bounds is not None:
        assert bounds.shape == (size[-1], 2), "bounds must have shape [dim, 2]"

    while True:
        eps = (t.randn(size)*sigma).to(device, dtype)
        proposal = pos + eps
        proposal_log_prob = logpdf(proposal)

        log_unif = t.randn(size[:-1]).log().to(device, dtype)
        accept = log_unif < proposal_log_prob - log_prob

        if bounds is not None:
            OOB = (proposal.gt(bounds[:, 0] & proposal.lt(bounds[:, 1]))).all(-1)
            accept = accept & OOB

        pos = t.where(accept.unsqueeze(-1), proposal, pos)
        log_prob = t.where(accept, proposal_log_prob, log_prob)

        yield pos

# class MCMC_sampler:
#
#     def __init__(f: Callable[[Tensor], Tensor],
#                  N: int = 1, B: int = 1, chains: int = 1, dim: int = 1,
#                  initial_pos: Tensor = None,
#                  burn: int = 0, burn_chains: int = None,
#                  find_max: bool = False, logging_freq: int = 10,
#                  bounds: Optional[Tensor] = None,
#                  device: t.device = None, dtype: t.dtype = None):
#         """Batched MCMC sampler"""
#         pass
#
#     def __call__(self) -> Union[Tensor, Tuple[Tensor, Tensor]]:
#         return self.sample()
#
#     def sample():
#         """Draw N samples"""
#
#     def __iter__(self) -> Tensor:
#         raise NotImplementedError
#
# TODO finish this


def RWMH_sampler(f: Callable[[Tensor], Tensor],
                 N: int = 1000, B: int = 1, chains: int = 100000, dim: int = 1,
                 initial_pos: Tensor = None,
                 burn: int = 1000, burn_chains: int = None,
                 sigma: float = 0.1, find_max: bool = False,
                 logging_freq: int = 10,
                 device: t.device = None, dtype: t.dtype = None
                 ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    # WARNING: this sampler is incorrect
    # TODO: create an abstract MCMC sampler (using HMC_sampler as reference)
    """Random walk Metropolis Hasings sampler

    Args:
        f: log probability density function / function to maximise
        N: the number of samples to return _per chain_
        B: the batch shape
        chains: the number of chains to run concurrently during sampling
        dim: number of dimensions for samples / the target distribution
        initial_pos: optional tensor of starting positions, size [B, chains, dim]
        burn: the number of 'burn-in' steps.
        burn_chains: the number of chains to use while burning in. Defaults to
            `chains`.
        sigma: variance of Gaussian random step size

    Returns: either
        - a tensor of shape [N, B, chains, dim] if find_max is False, else
          additionally a tensor of shape [B, dim] with the maximum for each
          batch.
    """
    logging.info('Beginning RWMH sampling')
    start = time.time()

    init = initial_pos
    assert N is not None, "number of samples to draw omitted"
    if initial_pos is None:
        assert B is not None, "batch size omitted"
        assert chains is not None, "number of chains omitted"
        assert dim is not None, "number of dimensions omitted"
        init = t.randn((B, chains, dim), device=device, dtype=dtype)
    else:
        B, chains, dim = initial_pos.shape
    assert init is not None

    if burn_chains is None:
        burn_chains = chains

    samples = t.empty((N, B, chains, dim), device=device, dtype=dtype)
    max_pos, prev_obj = None, None

    with Progress() as prog:
        burn_t = prog.add_task("Burning-in...", total = burn)

        # burn in phase
        pos = init.clone()
        burn_sampler = RWMH(f, pos, sigma)
        for (tmp, b) in zip(burn_sampler, range(burn)):
            lb = (b * burn_chains) % chains
            ub = min(((b + 1) * burn_chains) % chains, chains)
            init[lb:ub] = tmp[:ub-lb]
            if b % logging_freq == 0:
                prog.update(burn_t, advance=logging_freq)

        # sampling phase
        sampler = RWMH(f, init, sigma)
        sample_t = prog.add_task("Sampling...", total=N)
        for (pos, i) in zip(sampler, range(N)):
            samples[i] = pos.unsqueeze(0)

            if find_max:
                pos = pos.reshape(B, -1, dim)
                obj = f(pos)
                idx = t.argmax(obj, dim=-1)[:, None]
                oidx = obj.gather(1, idx)
                pidx = pos.squeeze(-1).gather(1, dim)
                if max_pos is None:
                    max_pos, prev_obj = pidx, oidx
                    continue
                max_pos = t.where(oidx > prev_obj, pidx, max_pos)

            if i % 10 == 0:
                prog.update(sample_t, advance=10)

    duration = time.time() - start
    logging.info(f'Completed {N * chains:,} samples in {duration:.2f} seconds')
    return samples, max_pos if find_max else samples  # type: ignore


def HMC(f: Callable[[Tensor], Tensor], initial_pos: Tensor,
        rho: float = 1e-2, L: int = 10, alpha: float = 1.1,
        bounds: Optional[Tensor] = None,
        ) -> Generator[Tensor, None, None]:
    """
    A simple Hamiltonian Monte Carlo implementation.

    Args:
        f: log probability density function / a function to be _maximised_
        initial_pos: where to initialise the chains
        rho: a learning rate / step size
        L: the number of 'leapfrog' steps to complete per iteration
        alpha: momentum tempering term
        bounds: optional bounds on the samples, shape [dim, 2] (for lower,
            upper; unfortunately we don't support different bounds for each
            batch currently.)
    """
    size = initial_pos.shape
    device, dtype = initial_pos.device, initial_pos.dtype
    pos = initial_pos

    if bounds is not None:
        assert bounds.shape == (size[-1], 2), "bounds must have shape [dim, 2]"

    l2 = math.floor(L/2.)
    salpha = t.ones(size).to(device, dtype) * math.sqrt(alpha)
    talpha = t.ones(size).to(device, dtype) * alpha

    def dfd(x: Tensor) -> Tensor:  # df/dx |_x
        tmpx = x.detach().requires_grad_(True)
        f(tmpx).backward(t.ones(x.shape[:-1], device=x.device, dtype=x.dtype))
        return -tmpx.grad

    def K(u: Tensor) -> Tensor:  # kinetic energy
        return -t.distributions.Normal(0, 1).log_prob(u).sum(-1)

    def U(x: Tensor) -> Tensor:  # potential
        return -f(x)

    while True:
        xl = pos.clone()
        u = t.randn(size).to(device, dtype)

        u *= salpha
        for i in range(L):
            up = u - (rho / 2) * dfd(xl)
            xl = xl + rho * up
            up = up - (rho / 2) * dfd(xl)
            up = up * talpha if i < l2 else up / talpha

        H_prop = K(up) + U(xl)
        H_curr = K(u) + U(pos)
        A = t.exp(-H_prop + H_curr)
        accept = t.rand(size[:-1]).to(device, dtype) < A
        if bounds is not None:
            OOB = (xl.gt(bounds[:, 0]) & xl.lt(bounds[:, 1])).all(-1)
            accept = accept & OOB
        pos = t.where(accept.unsqueeze(-1), xl, pos)

        yield pos


def HMC_sampler(f: Callable[[Tensor], Tensor],
                N: int = 1000, B: int = 1, chains: int = 100000, dim: int = 1,
                initial_pos: Tensor = None,
                burn: int = 1000, burn_chains: int = None,
                rho: float = 1e-2, L: int = 10, alpha: float = 1.1,
                find_max: bool = False, logging_freq: int = 10,
                bounds: Optional[Tensor] = None,
                device: t.device = None, dtype: t.dtype = None
                ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Batched Hamiltonian Monte Carlo sampler

    We work with tensors of shape [N, B, C, D], for N the number of samples, B
    the batch shape, C the number of chains, and D the dimension of the target
    distribution.

    You may either specify each of N, B, chains and dim, or specify initial_pos,
    which must match these.

    Args:
        f: log probability density function / function to maximise
        N: the number of samples to return for each individual chain.
        B: the batch shape
        chains: the number of chains to run concurrently during sampling
        dim: number of dimensions for samples / the target distribution
        initial_pos: optional tensor of starting positions, size [B, C, D]
        burn: the number of 'burn-in' steps.
        burn_chains: the number of chains to use while burning in. Defaults to
            `chains`.
        rho: a learning rate / step size
        L: the number of 'leapfrog' steps to complete per iteration
        alpha: momentum tempering term
        find_max: whether to additionally return the sampled position with the
            maximum f value.
        bounds: optional bounds on the samples, shape [dim, 2] (for lower,
            upper; unfortunately we don't support different bounds for each
            batch currently.)

    Returns: a  tensor of shape [N, B, chains, dim] if find_max is False, else
        also a tensor of shape [B, dim] with the maximum for each batch.
    """
    logging.info('Beginning HMC sampling')
    start = time.time()

    init = initial_pos  # used as a circular buffer for burning in
    assert N is not None, "number of samples to draw omitted"

    if initial_pos is None:
        assert B is not None, "batch size omitted"
        assert chains is not None, "number of chains omitted"
        assert dim is not None, "number of dimensions omitted"
        ichains = burn_chains if burn_chains is not None and burn_chains > 0\
            else chains
        init = t.randn((B, ichains, dim), device=device, dtype=dtype)
    else:
        B, chains, dim = initial_pos.shape
    assert init is not None

    if bounds is not None:
        init = init.clamp(bounds[:, 0], bounds[:, 1])

    if burn_chains is None:
        burn_chains = chains

    samples = t.empty((N, B, chains, dim), device=device, dtype=dtype)
    max_pos, prev_obj = None, None
    assert init is not None

    with Progress() as prog:
        burn_t = prog.add_task("Burning-in...", total=burn)

        # burn in phase
        pos = init.clone()
        burn_sampler = HMC(f, pos, rho, L, alpha, bounds)
        for (tmp, b) in zip(burn_sampler, range(burn)):
            lb = (b * burn_chains) % chains
            ub = min(((b + 1) * burn_chains) % chains, chains)
            init[:, lb:ub] = tmp[:, :ub-lb]
            if b % logging_freq == 0:
                prog.update(burn_t, advance=logging_freq)

        # sampling phase
        sampler = HMC(f, init, rho, L, alpha, bounds)
        sample_t = prog.add_task("Sampling...", total=N)

        for (pos, i) in zip(sampler, range(N)):
            samples[i] = pos.unsqueeze(0)  # [1, B, chains, dim]

            if find_max:
                pos = pos.reshape(B, chains, dim)
                with t.no_grad():
                    obj = f(pos)  # [B, C]
                assert obj.shape == (B, chains), "objective must be scalar-valued"
                idx = t.argmax(obj, dim=-1).unsqueeze(-1)
                pidx = idx.unsqueeze(-1).expand(B, 1, dim)
                this_obj = obj.gather(1, idx)
                this_pos = pos.gather(1, pidx).squeeze(-2)
                if max_pos is None:
                    max_pos, prev_obj = this_pos, this_obj
                    continue
                max_pos = t.where(this_obj > prev_obj, this_pos, max_pos)

            if i % logging_freq == 0:
                prog.update(sample_t, advance=logging_freq)

    if max_pos is not None:
        assert max_pos.shape == (B, dim)
    assert samples.shape == (N, B, chains, dim)

    duration = time.time() - start
    logging.info(f'Completed {N * chains:,} samples in {duration:.2f} seconds')
    return samples, max_pos if find_max else samples  # type: ignore
