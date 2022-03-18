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
"""Inference utilities"""


import math
import logging
import torch as t
import torch.nn as nn

from typing import Union
from numbers import Number
from torch.distributions import Distribution, constraints, \
    TransformedDistribution, AffineTransform
from torch.distributions.utils import broadcast_all

from spt.types import Tensor


class Squareplus(nn.Module):
    def __init__(self, a=2):
        super().__init__()
        self.a = a
    def forward(self, x):
        """The 'squareplus' activation function: has very similar properties to
        softplus, but is computationally cheaper and more configurable.
            - squareplus(0) = 1 (softplus(0) = ln 2)
            - gradient diminishes more slowly for negative inputs.
            - ReLU = (x + sqrt(x^2))/2
            - 'squareplus' becomes smoother with higher 'a'
        """
        return (x + t.sqrt(t.square(x)+self.a*self.a))/2


def squareplus_f(x, a=2):
    """Functional version of the 'squareplus' activation function. See
    `Squareplus` for more information.
    """
    return (x + t.sqrt(t.square(x)+a*a))/2


# Truncated normal distribution ------------------------------------------------
# Used in the SAN and MCMC steps.
# Implementation from github.com/toshas/torch_truncnorm


# Constants
CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)


class TruncatedStandardNormal(Distribution):
    """A truncated standard normal distribution."""

    arg_constraints = {
        'a': constraints.real,
        'b': constraints.real,
    }

    has_rsample = True

    def __init__(self, a: Union[Tensor, Number], b: Union[Tensor, Number],
                 validate_args: bool = None):

        self.a, self.b = broadcast_all(a, b)
        self.a, self.b = self.a.detach(), self.b.detach()
        if isinstance(a, Number) and isinstance(b, Number):
            batch_shape = t.Size()
        else:
            batch_shape = self.a.size()
        super(TruncatedStandardNormal, self).__init__(batch_shape, validate_args=validate_args)

        if self.a.dtype != self.b.dtype:
            raise ValueError('Truncation bound types are different')
        if any((self.a >= self.b).view(-1,).tolist()):
            raise ValueError('Incorrect truncation range')
        eps = t.finfo(self.a.dtype).eps
        self._dtype_min_gt_0 = eps
        self._dtype_max_lt_1 = 1 - eps
        self._little_phi_a = self._little_phi(self.a)
        self._little_phi_b = self._little_phi(self.b)
        self._big_phi_a = self._big_phi(self.a)
        self._big_phi_b = self._big_phi(self.b)
        self._Z = (self._big_phi_b - self._big_phi_a).clamp_min(eps)
        self._log_Z = self._Z.log()
        little_phi_coeff_a = t.nan_to_num(self.a, nan=math.nan)
        little_phi_coeff_b = t.nan_to_num(self.b, nan=math.nan)
        self._lpbb_m_lpaa_d_Z = (self._little_phi_b * little_phi_coeff_b -
                                 self._little_phi_a * little_phi_coeff_a) / self._Z
        self._mean = -(self._little_phi_b - self._little_phi_a) / self._Z
        self._variance = 1 - self._lpbb_m_lpaa_d_Z - \
            ((self._little_phi_b - self._little_phi_a) / self._Z) ** 2
        self._entropy = CONST_LOG_SQRT_2PI_E + \
            self._log_Z - 0.5 * self._lpbb_m_lpaa_d_Z

    @constraints.dependent_property
    def support(self):
        return constraints.interval(self.a, self.b)

    @property
    def mean(self) -> Tensor:
        return self._mean

    @property
    def variance(self) -> Tensor:
        return self._variance

    @property
    def entropy(self) -> Tensor:
        return self._entropy

    @property
    def auc(self) -> Tensor:
        return self._Z

    @staticmethod
    def _little_phi(x):
        return (-(x ** 2) * 0.5).exp() * CONST_INV_SQRT_2PI

    @staticmethod
    def _big_phi(x):
        return 0.5 * (1 + (x * CONST_INV_SQRT_2).erf())

    @staticmethod
    def _inv_big_phi(x):
        return CONST_SQRT_2 * (2 * x - 1).erfinv()

    def cdf(self, value):
        # If value < lower bound, then return 0
        # if value > upper bound, return 1

        if self._validate_args:
            self._validate_sample(value)
        return ((self._big_phi(value) - self._big_phi_a) / self._Z).clamp(0, 1)

    def icdf(self, value):
        return self._inv_big_phi(self._big_phi_a + value * self._Z)

    def log_prob(self, value):
        # TODO: if value < lower bound or > upper bound, then return -inf?
        if self._validate_args:
            self._validate_sample(value)
        return CONST_LOG_INV_SQRT_2PI - self._log_Z - (value ** 2) * 0.5

    def rsample(self, sample_shape=t.Size()):
        shape = self._extended_shape(sample_shape)
        p = t.empty(shape, device=self.a.device).uniform_(
            self._dtype_min_gt_0, self._dtype_max_lt_1)
        return self.icdf(p)


class TruncatedNormal(TruncatedStandardNormal):
    """A truncated normal distribution, **for use in a mixture distribution(!)**

    Could also obtain this by using AffineTransform on TruncatedStandardNormal,
    although this is a little simpler and thus probably more efficient.
    """

    arg_constraints = {
        'loc': constraints.real, 'scale': constraints.positive,
        'a': constraints.real, 'b': constraints.real,
    }

    has_rsample = True

    @constraints.dependent_property
    def support(self):
        # WARNING: this is specific to use in a mixture distribution
        return constraints.interval(self.a[..., 0], self.b[..., 0])

    def __init__(self, loc, scale, a, b, validate_args=None):
        self.loc, self.scale, a, b = broadcast_all(loc, scale, a, b)

        # loc = loc.clamp(a, b)
        # TODO write unit tests to verify that all this behaviour is as expected...
        a = (a - self.loc) / self.scale
        b = (b - self.loc) / self.scale

        # TODO: unnecessary
        aa = t.where(a > b, b, a)
        bb = t.where(a > b, a, b)

        super(TruncatedNormal, self).__init__(aa, bb, validate_args=validate_args)
        self._log_scale = self.scale.log()
        self._mean = self._mean * self.scale + self.loc
        self._variance = self._variance * self.scale ** 2
        self._entropy += self._log_scale

    def _to_std_rv(self, value):
        return (value - self.loc) / self.scale

    def _from_std_rv(self, value):
        return value * self.scale + self.loc

    def cdf(self, value):
        return super(TruncatedNormal, self).cdf(self._to_std_rv(value))

    def icdf(self, value):
        return self._from_std_rv(super(TruncatedNormal, self).icdf(value))

    def log_prob(self, value):
        return super(TruncatedNormal, self).log_prob(self._to_std_rv(value)) - self._log_scale
