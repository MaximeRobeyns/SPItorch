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

import time
import logging
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F

from abc import abstractmethod
from typing import Any, Callable, Generator, Optional, Type
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
        initial_pos: where to initialise the chains
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


def RWMH_sampler(logpdf, N: int = 1000, chains: int = 100000, burn: int = 1000,
                 burn_chains: int = None, initial_pos: Tensor = None,
                 dim: int = 1, sigma: float = 0.1, device: t.device = None,
                 dtype: t.dtype = None) -> Tensor:
    """Random walk Metropolis Hasings sampler

    Args:
        logpdf: log probability density function
        N: the number of samples to return _per chain_
        chains: the number of chains to run concurrently during sampling
        burn: the number of 'burn-in' steps.
        burn_chains: the number of chains to use while burning in. Defaults to
            `chains`.
        initial_pos: optional tensor of starting positions, size [chains, dim]
        dim: number of dimensions for samples / the target distribution
        sigma: variance of Gaussian random step size

    Returns: a tensor of shape [N, chains, dim], for `dim` the number of
        dimensions per sample.
    """
    logging.info('Beginning RWMH sampling')
    start = time.time()
    if burn_chains is None:
        burn_chains = chains

    samples = t.empty((N, chains, dim), device=device, dtype=dtype)

    with Progress() as prog:
        burn_t = prog.add_task("Burning-in...", total = burn)

        pos = t.randn((burn_chains, dim), device=device, dtype=dtype)
        # used as a circular buffer
        init = initial_pos if initial_pos is not None else \
            t.randn((chains, dim), device=device, dtype=dtype)

        # burn in phase
        burn_sampler = RWMH(logpdf, pos.T, sigma)
        b = 0
        for tmp in burn_sampler:
            lb = (b*burn_chains)%chains
            ub = min(((b+1)*burn_chains)%chains, chains)
            init[lb:ub] = tmp[:ub-lb]
            b += 1
            if b % 10 == 0:
                prog.update(burn_t, advance=10)
            if b == burn:
                prog.update(burn_t, completed=burn)
                break

        # sampling phase
        sampler = RWMH(logpdf, init)
        sample_t = prog.add_task("Sampling...", total=N)
        i = 0
        for pos in sampler:
            samples[i] = pos.unsqueeze(0)
            if i % 10 == 0:
                prog.update(sample_t, advance=10)
            if i == N:
                prog.update(sample_t, completed=N)
                break

    duration = time.time() - start
    logging.info(f'Completed {N * chains:,} samples in {duration:.2f} seconds')
    return samples


def HMC(logpdf: Callable[[Tensor], Tensor], initial_pos: Tensor,
        rho: float = 1e-2, L: int = 10) -> Generator[Tensor, None, None]:
    """
    A simple Hamiltonian Monte Carlo implementation.

    Args:
        logpdf: log probability density function
        initial_pos: where to initialise the chains
        rho: a learning rate / step size
        L: the number of 'leapfrog' steps to complete per iteration

    TODO: implement more sophisticated learning rate schedule (e.g. Adam), and
    samplers (e.g. NUTS).
    """
    size = initial_pos.shape
    device, dtype = initial_pos.device, initial_pos.dtype
    pos = initial_pos
    log_prob = logpdf(pos)
    logpxl = None

    while True:
        momentum = t.randn(size).to(device, dtype)
        xl = pos.clone().detach().requires_grad_(True)
        logpdf(xl).backward(t.ones(size[0], device=device, dtype=dtype))
        ul = momentum * rho * xl.grad / 2

        for l in range(L):
            rho_l = rho if l < L-1 else rho / 2
            xl = xl + rho_l * ul
            xl = xl.detach().requires_grad_(True)
            logpxl = logpdf(xl)
            logpxl.backward(t.ones(size[0], device=device, dtype=dtype))
            ul = ul + rho_l * xl.grad

        log_uniform = t.rand(size).log().to(device, dtype)
        ul1, ul2 = ul.unsqueeze(-1), ul.unsqueeze(-2)
        m1, m2 = momentum.unsqueeze(-1), momentum.unsqueeze(-2)
        A = logpxl - log_prob - (t.bmm(ul2, ul1).squeeze() - t.bmm(m2, m1).squeeze())/2
        A = t.where(A > 0, t.zeros_like(A), A)
        accept = log_uniform < A

        pos = t.where(accept.unsqueeze(-1), xl, pos)
        log_prob = t.where(accept, logpxl, log_prob)
        yield pos


def HMC_sampler(logpdf, N: int = 1000, chains: int = 100000, burn: int = 1000,
                burn_chains: int = None, initial_pos: Tensor = None,
                dim: int = 1, rho: float = None, sigma: float = None,
                device: t.device = None, dtype: t.dtype = None
                 ) -> Tensor:
    """Hamiltonian Monte Carlo sampler

    Args:
        logpdf: log probability density function
        N: the number of samples to return _per chain_
        chains: the number of chains to run concurrently during sampling
        burn: the number of 'burn-in' steps.
        burn_chains: the number of chains to use while burning in. Defaults to
            `chains`.
        initial_pos: optional tensor of starting positions, size [chains, dim]
        dim: number of dimensions for samples / the target distribution
        rho: a learning rate / step size
        L: the number of 'leapfrog' steps to complete per iteration

    Returns: a tensor of shape [N, chains, dim], for `dim` the number of
        dimensions per sample.
    """
    logging.info('Beginning HMC sampling')
    start = time.time()
    if burn_chains is None:
        burn_chains = chains

    samples = t.empty((N, chains, dim), device=device, dtype=dtype)

    with Progress() as prog:
        burn_t = prog.add_task("Burning-in...", total=burn)

        pos = t.randn((burn_chains, dim), device=device, dtype=dtype)
        # used as a circular buffer
        init = initial_pos if initial_pos is not None else \
            t.randn((chains, dim), device=device, dtype=dtype)

        # burn in phase
        burn_sampler = HMC(logpdf, pos, rho, sigma)
        b = 0
        for tmp in burn_sampler:
            lb = (b*burn_chains)%chains
            ub = min(((b+1)*burn_chains)%chains, chains)
            init[lb:ub] = tmp[:ub-lb]
            b += 1
            if b % 10 == 0:
                prog.update(burn_t, advance=10)
            if b == burn:
                prog.update(burn_t, completed=burn)
                break

        # sampling phase
        sampler = HMC(logpdf, init, rho, sigma)
        sample_t = prog.add_task("Sampling...", total=N)
        i = 0
        for pos in sampler:
            samples[i] = pos.unsqueeze(0)
            if i % 10 == 0:
                prog.update(sample_t, advance=10)
            if i == N:
                prog.update(sample_t, completed=N)
                break

    duration = time.time() - start
    logging.info(f'Completed {N * chains:,} samples in {duration:.2f} seconds')
    return samples
