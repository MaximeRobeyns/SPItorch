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
"""Tests for inference utilities and distributions"""

import torch as t
import pytest


from spt.inference.utils import TruncatedNormal, TruncatedStandardNormal

def test_truncated_standard_normal_support():
    """Tests that we can initialise the truncated normal distribution correctly"""

    # test single (not-batched) inputs

    a, b = 0.25, 0.75
    tsn = TruncatedStandardNormal(a, b)
    supp = tsn.support
    assert supp.lower_bound == t.tensor(a)
    assert supp.upper_bound == t.tensor(b)
    assert tsn.batch_shape == t.Size()
    assert tsn.event_shape == t.Size()

    # mix of batched and non-batched parameters
    tsn2 = TruncatedStandardNormal(0.25, t.tensor([0.7, 0.8]))
    supp = tsn2.support
    assert list(supp.lower_bound) == [0.25, 0.25]
    assert list(supp.upper_bound) == [0.7, 0.8]
    assert tsn2.batch_shape == t.Size([2])
    assert tsn2.event_shape == t.Size([])

    # batched parameters
    A, B = t.rand((100, 50)), 1 + t.rand((100, 50))
    tsn3 = TruncatedStandardNormal(A, B)
    assert tsn3.a.shape == t.Size([100, 50])
    assert tsn3.b.shape == t.Size([100, 50])

    # lower bound < upper_bound
    A, B = t.rand((100, 50)), 1 + t.rand((100, 50))
    with pytest.raises(ValueError):
        tsn3 = TruncatedStandardNormal(B, A)

def test_truncated_standard_normal_mean():
    a, b = 0.25, 0.75
    tsn = TruncatedStandardNormal(a, b)
    eps = 0.02
    assert tsn.mean >= (t.tensor(0.5) - eps) and tsn.mean <= (t.tensor(0.5) + eps)

    # batched
    A, B = t.ones((100, 50)) * 0.25, t.ones((100, 50)) * 0.75
    tsn2 = TruncatedStandardNormal(A, B)
    assert (tsn2.mean >= (t.tensor(0.5) - eps)).all() and \
           (tsn2.mean <= (t.tensor(0.5) + eps)).all()


def test_truncated_standard_normal_cdf():

    a, b = 0.25, 0.75
    tsn = TruncatedStandardNormal(a, b)
    eps = 0.02
    assert tsn.cdf(t.tensor([0.251])) <= t.tensor(0) + eps
    assert tsn.cdf(t.tensor([0.749])) >= t.tensor(1) - eps

    # out of bounds
    with pytest.raises(ValueError):
        assert tsn.cdf(t.tensor([0.2])) == t.tensor(0)
        assert tsn.cdf(t.tensor([0.8])) == t.tensor(1)

def test_truncated_standard_normal_lp():

    a, b = 0.25, 0.75
    tsn = TruncatedStandardNormal(a, b)
    pts = a + t.rand(100000) * (b - a)
    assert (a <= pts).all()
    assert (b >= pts).all()
    # should not raise out-of-support error
    tsn.log_prob(pts)


    aa = 0.24
    pts = aa + t.rand(100000) * (b - aa)
    with pytest.raises(ValueError):
        tsn.log_prob(pts)

    # batched distribution, 4 dimensions
    A = t.arange(0, 1, 0.2).expand(100, 10, 5)
    B = t.arange(0.2, 1.1, 0.2).expand(100, 10, 5)
    tsn2 = TruncatedStandardNormal(A, B)
    assert tsn2.batch_shape == t.Size([100, 10, 5])

    pts = A[0, 0] + t.rand(10, 100, 10, 5) * (B[0, 0] - A[0, 0])
    for i in range(5):
        assert (.2 * i <= pts[..., i]).all() and \
               (pts[..., i] <= .2 * (i+1)).all()

    tsn2.log_prob(pts)

def test_truncated_standard_normal_rsample():

    a, b = 0.25, 0.75
    tsn = TruncatedStandardNormal(a, b)
    samples = tsn.sample((100000,))
    assert (a <= samples).all()
    assert (b >= samples).all()

    aa = 0.24
    tsn = TruncatedStandardNormal(aa, b)
    samples = tsn.sample((100000,))
    assert not (a <= samples).all()
    assert (b >= samples).all()

    # batched distribution, 4 dimensions
    A = t.arange(0, 1, 0.2).expand(100, 10, 5)
    B = t.arange(0.2, 1.1, 0.2).expand(100, 10, 5)
    tsn2 = TruncatedStandardNormal(A, B)
    assert tsn2.batch_shape == t.Size([100, 10, 5])

    samples = tsn2.sample((10,))
    assert samples.shape == t.Size([10, 100, 10, 5])
    for i in range(5):
        assert (.2 * i <= samples[..., i]).all() and \
               (samples[..., i] <= .2 * (i+1)).all()
