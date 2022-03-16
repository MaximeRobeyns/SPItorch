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
         sigma: float = 0.1) -> Generator[Tensor, None, None]:
    """
    Random walk Metropolis-Hastings.

    Args:
        logpdf: log probability density function
        initial     _pos: where to initialise the chains
        sigma: purturbation variance for proposals
    """
    size = initial_pos.shape
    device, dtype = initial_pos.device, initial_pos.dtype
    pos = initial_pos
    log_prob = logpdf(pos)
    yield pos

    while True:
        eps = (t.randn(size)*sigma).to(device, dtype)
        proposal = pos + eps
        proposal_log_prob = logpdf(proposal)

        log_unif = t.randn(size[0]).log().to(device, dtype)
        accept = log_unif < proposal_log_prob - log_prob

        pos = t.where(accept.unsqueeze(-1), proposal, pos)
        log_prob = t.where(accept, proposal_log_prob, log_prob)

        yield pos


def RWMH_sampler(f: Callable[[Tensor], Tensor], N: int = 1000,
                 chains: int = 100000, burn: int = 1000,
                 burn_chains: int = None, initial_pos: Tensor = None,
                 dim: int = 1, sigma: float = 0.1, find_max: bool = False,
                 device: t.device = None, dtype: t.dtype = None
                 ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Random walk Metropolis Hasings sampler

    Args:
        f: log probability density function / function to maximise
        N: the number of samples to return _per chain_
        chains: the number of chains to run concurrently during sampling
        burn: the number of 'burn-in' steps.
        burn_chains: the number of chains to use while burning in. Defaults to
            `chains`.
        initial_pos: optional tensor of starting positions, size [chains, dim]
        dim: number of dimensions for samples / the target distribution
        sigma: variance of Gaussian random step size

    Returns: either
        - a tensor of shape [N, chains, dim], for `dim` the number of dimensions
        per sample, if find_max == False
        - a tuple of samples, and max points
    """
    logging.info('Beginning RWMH sampling')
    start = time.time()
    if burn_chains is None:
        burn_chains = chains

    samples = t.empty((N, chains, dim), device=device, dtype=dtype)
    max_pos, prev_obj = None, None

    with Progress() as prog:
        burn_t = prog.add_task("Burning-in...", total = burn)

        pos = t.randn((burn_chains, dim), device=device, dtype=dtype)
        # used as a circular buffer
        init = initial_pos if initial_pos is not None else \
            t.randn((chains, dim), device=device, dtype=dtype)

        # burn in phase
        burn_sampler = RWMH(f, pos, sigma)
        for (tmp, b) in zip(burn_sampler, range(burn)):
            lb = (b*burn_chains)%chains
            ub = min(((b+1)*burn_chains)%chains, chains)
            init[lb:ub] = tmp[:ub-lb]
            if b % 10 == 0:
                prog.update(burn_t, advance=10)

        # sampling phase
        sampler = RWMH(f, init, sigma)
        sample_t = prog.add_task("Sampling...", total=N)
        for (pos, i) in zip(sampler, range(N)):
            samples[i] = pos.unsqueeze(0)

            if find_max:
                pos = pos.reshape(-1, dim)
                obj = f(pos)
                idx = t.argmax(obj, dim=-1)
            if max_pos is None:
                max_pos, prev_obj = pos[idx], obj[idx]
                continue
            max_pos = t.where(obj[idx] > prev_obj, pos[idx], max_pos)

            if i % 10 == 0:
                prog.update(sample_t, advance=10)

    duration = time.time() - start
    logging.info(f'Completed {N * chains:,} samples in {duration:.2f} seconds')
    if find_max:
        return samples, max_pos
    return samples


def HMC(f: Callable[[Tensor], Tensor], initial_pos: Tensor,
        rho: float = 1e-2, L: int = 10, alpha: float = 1.1
        ) -> Generator[Tensor, None, None]:
    """
    A simple Hamiltonian Monte Carlo implementation.

    Args:
        f: log probability density function / a function to be _maximised_
        initial_pos: where to initialise the chains
        rho: a learning rate / step size
        L: the number of 'leapfrog' steps to complete per iteration
        alpha: momentum tempering term

    """
    size = initial_pos.shape
    device, dtype = initial_pos.device, initial_pos.dtype
    pos = initial_pos

    l2 = math.floor(L/2.)
    salpha = t.ones(size).to(device, dtype) * math.sqrt(alpha)
    talpha = t.ones(size).to(device, dtype) * alpha

    def dfd(x: Tensor) -> Tensor:
        tmpx = x.detach().requires_grad_(True)
        f(tmpx).backward(t.ones(x.shape[0], device=x.device, dtype=x.dtype))
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
        accept = t.rand(size[0]).to(device, dtype) < A
        pos = t.where(accept.unsqueeze(-1), xl, pos)

        yield pos


def HMC_sampler(f: Callable[[Tensor], Tensor], N: int = 1000,
                chains: int = 100000, burn: int = 1000,
                burn_chains: int = None, initial_pos: Tensor = None,
                dim: int = 1, rho: float = 1e-2, L: int = 10,
                alpha: float = 1.1, find_max: bool = False,
                device: t.device = None, dtype: t.dtype = None
                ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Hamiltonian Monte Carlo sampler

    Args:
        f: log probability density function / function to maximise
        N: the number of samples to return _per chain_
        chains: the number of chains to run concurrently during sampling
        burn: the number of 'burn-in' steps.
        burn_chains: the number of chains to use while burning in. Defaults to
            `chains`.
        initial_pos: optional tensor of starting positions, size [chains, dim]
        dim: number of dimensions for samples / the target distribution
        rho: a learning rate / step size
        L: the number of 'leapfrog' steps to complete per iteration
        alpha: momentum tempering term
        find_max: whether to additionally return the sampled position with the
            maximum f value.

    Returns: a tensor of shape [N, chains, dim], for `dim` the number of
        dimensions per sample.
    """
    logging.info('Beginning HMC sampling')
    start = time.time()
    if burn_chains is None:
        burn_chains = chains

    samples = t.empty((N, chains, dim), device=device, dtype=dtype)
    max_pos, prev_obj = None, None

    with Progress() as prog:
        burn_t = prog.add_task("Burning-in...", total=burn)

        pos = t.randn((burn_chains, dim), device=device, dtype=dtype)
        # used as a circular buffer
        init = initial_pos if initial_pos is not None else \
            t.randn((chains, dim), device=device, dtype=dtype)

        # burn in phase
        burn_sampler = HMC(f, pos, rho, L, alpha)
        for (tmp, b) in zip(burn_sampler, range(burn)):
            lb = (b*burn_chains)%chains
            ub = min(((b+1)*burn_chains)%chains, chains)
            init[lb:ub] = tmp[:ub-lb]
            if b % 10 == 0:
                prog.update(burn_t, advance=10)

        # sampling phase
        sampler = HMC(f, init, rho, L, alpha)
        sample_t = prog.add_task("Sampling...", total=N)
        for (pos, i) in zip(sampler, range(N)):
            samples[i] = pos.unsqueeze(0)

            if find_max:
                pos = pos.reshape(-1, dim)
                with t.no_grad():
                    obj = f(pos)
                    idx = t.argmax(obj, dim=-1)
                if max_pos is None:
                    max_pos, prev_obj = pos[idx], obj[idx]
                    continue
                max_pos = t.where(obj[idx] > prev_obj, pos[idx], max_pos)

            if i % 10 == 0:
                prog.update(sample_t, advance=10)

    duration = time.time() - start
    logging.info(f'Completed {N * chains:,} samples in {duration:.2f} seconds')
    if find_max:
        return samples, max_pos
    return samples
