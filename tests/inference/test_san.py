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
"""Tests for sequential autoregressive network and associated likelihoods"""

import torch as t
import torch.nn as nn

from typing import Any, Optional, Type

import spt.inference as inf

from spt.config import ForwardModelParams
from spt.inference import san
from spt.load_photometry import load_simulated_data, get_norm_theta, \
    get_norm_theta_t, new_sample


class InferenceParams(inf.InferenceParams):
    model: inf.model_t = san.SAN
    logging_frequency: int = 1000000


class InferenceParams(inf.InferenceParams):
    model: inf.model_t = san.SAN
    split_ratio: float = 0.9
    logging_frequency: int = 10000000
    dataset_loc: str = './data/dsets/testing/'
    retrain_model: bool = False
    use_existing_checkpoints: bool = True
    ident: str = 'test'
    catalogue_loc: str = ''
    # catalogue_loc: str = './data/DES_VIDEO_v1.0.1.fits'


class MoGSANParams(san.SANParams):
    epochs: int = 10
    batch_size: int = 1024
    dtype: t.dtype = t.float32
    cond_dim: int = 7
    data_dim: int = 6
    module_shape: list[int] = [16, 32]
    sequence_features: int = 8
    likelihood: Type[san.SAN_Likelihood] = san.MoG

    likelihood_kwargs: Optional[dict[str, Any]] = {
        'K': 8, 'mult_eps': 1e-4, 'abs_eps': 1e-4,
    }

    layer_norm: bool = True
    train_rsample: bool = False
    opt_lr: float = 1e-3
    limits: Any = t.tensor([[0., 1.]]).repeat((6, 1))  # [6, 2]

def test_san_initialisation():
    sp = MoGSANParams()
    model = san.SAN(sp)
    assert model.module_shape == sp.module_shape
    assert model.sequence_features == sp.sequence_features


def test_san_MoG_likelihood():
    dtype: t.float32
    device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")

    fp = ForwardModelParams()
    ip = InferenceParams()
    sp = MoGSANParams()

    train_loader, test_loader = load_simulated_data(
        path=ip.dataset_loc, split_ratio=ip.split_ratio,
        batch_size=sp.batch_size, # test_batch_size=1,
        phot_transforms=[lambda x: t.from_numpy(x).log()],
        theta_transforms=[get_norm_theta(fp)])

    model = san.SAN(sp)

    assert isinstance(model.likelihood, san.MoG)
    assert model.likelihood.name == "MoG"
    # 8 mixture components per output dimension
    assert model.likelihood.K == 8
    # number of parameters required for each dimension of the output
    # (loc, scale, mixture_weight) * number of mixture components
    assert model.likelihood.n_params() == 24
    assert model.last_params is None

    # TODO: test the _extract_params method
    # TODO draw test samples and verify dimensions
    # TODO evaluate point likelihoods


def test_san_sequential_blocks():

    model = san.SAN(MoGSANParams())
    test_block: nn.Module
    test_heads: nn.ModuleList

    # 0th block has no additional conditioning parameters
    test_block, test_heads = model._sequential_block(
        cond_dim=8, d=0, module_shape=[16, 32], out_shapes=[4, 2],
        out_activations=[nn.ReLU, None])

    sizes = [8, 16, 32]

    for j, (name, m) in enumerate(test_block.named_modules()):
        m: nn.Module
        if j == 0 or j == 7:
            continue
        i = j-1
        if i % 3 == 0:
            # linear
            assert name == f'B0L{i//3}'
            assert m.in_features == sizes[i//3]
            assert m.out_features == sizes[(i//3)+1]
        elif i+1 % 3 == 0:
            assert name == f'B0LN{i//3}'  # Layer norm
        elif i+2 % 3 == 0:
            assert name == f'B0A{i//3}'  # Activation

    assert len(test_heads) == 2
    h1, h2 = test_heads
    assert h1.get_submodule("H0:0H0").in_features == 32
    assert h1.get_submodule("H0:0H0").out_features == 4
    assert h2.get_submodule("H0:1H1").in_features == 32
    assert h2.get_submodule("H0:1H1").out_features == 2

    #n>0th blocks have additional conditioning parameters
    test_block, test_heads = model._sequential_block(
        cond_dim=8, d=1, module_shape=[16, 32], out_shapes=[4, 2],
        out_activations=[nn.ReLU, None])
    sizes = [8 + 4 + 1, 16, 32]

    for j, (name, m) in enumerate(test_block.named_modules()):
        m: nn.Module
        if j == 0 or j == 7:
            continue
        i = j-1
        if i % 3 == 0:
            # Linear
            assert name == f'B1L{i//3}'
            assert m.in_features == sizes[i//3]
            assert m.out_features == sizes[(i//3)+1]
        elif i+1 % 3 == 0:
            assert name == f'B1LN{i//3}'  # Layer norm
        elif i+2 % 3 == 0:
            assert name == f'B1A{i//3}'  # Activation
    assert len(test_heads) == 2
    h1, h2 = test_heads
    assert h1.get_submodule("H1:0H0").in_features == 32
    assert h1.get_submodule("H1:0H0").out_features == 4
    assert h2.get_submodule("H1:1H1").in_features == 32
    assert h2.get_submodule("H1:1H1").out_features == 2

    test_block, test_heads = model._sequential_block(
            cond_dim=8, d=40, module_shape=[16, 32], out_shapes=[4, 2],
            out_activations=[nn.ReLU, None])
    sizes = [8 + 4 + 40, 16, 32]
    for j, (name, m) in enumerate(test_block.named_modules()):
        m: nn.Module
        if j == 0 or j == 7:
            continue
        i = j-1
        if i % 3 == 0:
            # Linear
            assert name == f'B40L{i//3}'
            assert m.in_features == sizes[i//3]
            assert m.out_features == sizes[(i//3)+1]
        elif i+1 % 3 == 0:
            assert name == f'B40LN{i//3}'  # Layer norm
        elif i+2 % 3 == 0:
            assert name == f'B40A{i//3}'  # Activation

def test_san_fpath():
    model = san.SAN(MoGSANParams())
    assert model.fpath() == './results/sanmodels/lMoG_cd7_dd6_ms16_32_8_lp24_lnTrue_lr0.001_e10_bs1024_trsampleFalse_lim_.pt'


def test_san_forward():
    """Tests that all tensor shapes and parameter values are as we expect during
    a forward pass,"""

    dtype = t.float32
    device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")

    ip = InferenceParams()
    sp = MoGSANParams()
    fp = ForwardModelParams()

    _, test_loader = load_simulated_data(
        path=ip.dataset_loc, split_ratio=ip.split_ratio,
        batch_size=sp.batch_size, # test_batch_size=1,
        phot_transforms=[t.from_numpy, t.log],
        theta_transforms=[t.from_numpy, get_norm_theta_t(fp)])

    model = san.SAN(sp)

    # Multiple samples

    N = 100
    xs, true_ys = new_sample(test_loader, N)  # photometry, params
    assert xs.shape == (N, sp.cond_dim)

    xs, true_ys = model.preprocess(xs, true_ys)
    assert isinstance(xs, t.Tensor) and isinstance(true_ys, t.Tensor)

    assert xs.shape == (N, sp.cond_dim)
    assert true_ys.shape == (N, sp.data_dim)

    # Explicitly step through the lines in the `forward` function
    B = xs.size(0)
    assert B == N
    ys = t.empty((B, 0), dtype=dtype, device=device)

    # last parameters are used to evaluate a likelihood after a forward pass
    last_params = t.ones((B, sp.data_dim, model.likelihood.n_params()),
                         dtype=dtype, device=device)
    assert last_params.shape == (N, 6, 24)

    seq_features = t.empty((B, 0), dtype=dtype, device=device)

    for d in range(model.data_dim):
        d_input = t.cat((xs, seq_features, ys), 1)
        if d == 0:
            assert d_input.shape == (N, sp.cond_dim)
        elif d > 0:
            assert d_input.shape == (N, sp.cond_dim + sp.sequence_features + d)

        H = model.network_blocks[d](d_input)
        assert H.shape == (N, sp.module_shape[-1])

        seq_features = model.block_heads[d][0](H)
        assert seq_features.shape == (N, sp.sequence_features)

        params = model.block_heads[d][1](H)
        assert params.shape == (N, model.likelihood.n_params())
        y_d = model.likelihood.sample(params)# .unsqueeze(-1)
        assert y_d.shape == (N, 1)

        ys = t.cat((ys, y_d), -1)
        assert ys.shape == (N, d+1)

        last_params[:, d] = params
        assert (last_params[:, d+1:] == 1.).all()
        assert (last_params[:, :d+1] != 1.).all()  # _highly_ unlikely that a param is exactly 1.

    assert ys.shape == (N, sp.data_dim)


def test_san_forward_single():
    """Same as above, but for a single point."""

    dtype = t.float32
    device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")

    ip = InferenceParams()
    sp = MoGSANParams()
    fp = ForwardModelParams()

    _, test_loader = load_simulated_data(
        path=ip.dataset_loc, split_ratio=ip.split_ratio,
        batch_size=sp.batch_size, # test_batch_size=1,
        phot_transforms=[t.from_numpy, t.log],
        theta_transforms=[t.from_numpy, get_norm_theta_t(fp)])

    model = san.SAN(sp)

    # Multiple samples

    xs, true_ys = new_sample(test_loader, 1)  # photometry, params
    assert xs.shape == (sp.cond_dim,)
    assert true_ys.shape == (sp.data_dim,)

    xs, true_ys = model.preprocess(xs.unsqueeze(0), true_ys.unsqueeze(0))
    assert isinstance(xs, t.Tensor) and isinstance(true_ys, t.Tensor)
    assert xs.shape == (1, sp.cond_dim)
    assert true_ys.shape == (1, sp.data_dim)

    # Explicitly step through the lines in the `forward` function
    B = xs.size(0)
    assert B == 1
    ys = t.empty((B, 0), dtype=dtype, device=device)

    # last parameters are used to evaluate a likelihood after a forward pass
    last_params = t.ones((B, sp.data_dim, model.likelihood.n_params()),
                         dtype=dtype, device=device)
    assert last_params.shape == (1, 6, 24)

    seq_features = t.empty((B, 0), dtype=dtype, device=device)

    for d in range(model.data_dim):
        d_input = t.cat((xs, seq_features, ys), 1)
        if d == 0:
            assert d_input.shape == (1, sp.cond_dim)
        elif d > 0:
            assert d_input.shape == (1, sp.cond_dim + sp.sequence_features + d)

        H = model.network_blocks[d](d_input)
        assert H.shape == (1, sp.module_shape[-1])

        seq_features = model.block_heads[d][0](H)
        assert seq_features.shape == (1, sp.sequence_features)

        params = model.block_heads[d][1](H)
        assert params.shape == (1, model.likelihood.n_params())
        y_d = model.likelihood.sample(params)# .unsqueeze(-1)
        assert y_d.shape == (1, 1)

        ys = t.cat((ys, y_d), -1)
        assert ys.shape == (1, d+1)

        last_params[:, d] = params
        assert (last_params[:, d+1:] == 1.).all()
        assert (last_params[:, :d+1] != 1.).all()  # _highly_ unlikely that a param is exactly 1.

    assert ys.shape == (1, sp.data_dim)
