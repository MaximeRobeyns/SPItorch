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
import prospect.models.priors as priors

from typing import Any, Optional, Type

import spt
import spt.inference as inf

from spt.utils import ConfigClass
from spt.filters import FilterCheck, FilterLibrary, Filter
from spt.modelling import ParamConfig, Parameter,\
    build_model_fn_t, build_obs_fn_t, build_sps_fn_t
from spt.inference import san
from spt.load_photometry import load_simulated_data, get_norm_theta, \
    get_norm_theta_t, new_sample


class ForwardModelParams(FilterCheck, ParamConfig, ConfigClass):
    """Testing parameters"""
    filters: list[Filter] = FilterLibrary['des']
    build_obs_fn: build_obs_fn_t = spt.modelling.build_obs
    model_param_templates: list[str] = ['parametric_sfh']
    model_params: list[Parameter] = [
        Parameter('zred', 0., 0.1, 4., units='redshift, $z$'),
        Parameter('mass', 10**6, 10**8, 10**10, priors.LogUniform,
                  units='$log(M/M_\\odot)$', disp_floor=10**6.),
        Parameter('logzsol', -2, -0.5, 0.19, units='$\\log (Z/Z_\\odot)$'),
        Parameter('dust2', 0., 0.05, 2., units='optical depth at 5500AA'),
        Parameter('tage', 0.001, 13., 13.8, units='Age, Gyr', disp_floor=1.),
        # Parameter('tau', 0.1, 1, 100, priors.LogUniform, units='Gyr^{-1}'),
        Parameter('tau', 10**(-1), 10**0, 10**2, priors.LogUniform, units='Gyr^{-1}'),]
    build_model_fn: build_model_fn_t = spt.modelling.build_model
    sps_kwargs: dict[str, Any] = {'zcontinuous': 1}
    build_sps_fn: build_sps_fn_t = spt.modelling.build_sps


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

    # HMC update parameters ----------------------------------------------------

    hmc_update_N: int = 5  # number of HMC steps
    hmc_update_C: int = 100  # number of chains to use in HMC
    hmc_update_D: int = len(ForwardModelParams().filters)  # data dimensions
    hmc_update_L: int = 2  # leapfrog integrator steps per iteration
    hmc_update_rho: float = 0.1  # step size
    hmc_update_alpha: float = 1.1  # momentum

    # simulated data update procedure:

    # The number of samples to use in each update step.
    # (note: quickly increases memory requirements)
    hmc_update_sim_K: int = 1
    hmc_update_sim_ident: str = 'update_sim_example'  # saving / checkpointing
    hmc_update_sim_epochs: int = 5

    # real data update procedure:

    hmc_update_real_K: int = 1
    hmc_update_real_epochs: int = 5
    hmc_update_real_ident: str = 'update_real_example'


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
    # limits: Any = t.tensor([[0., 1.]]).repeat((6, 1))  # [6, 2]

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
    assert model.fpath() == './results/sanmodels/lMoG_cd7_dd6_ms16_32_8_lp24_lnTrue_lr0.001_ld0.0001_e10_bs1024_trsampleFalse_.pt'


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
        y_d = model.likelihood.sample(params).unsqueeze(-1)  # TODO: check unsqueeze here
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
        y_d = model.likelihood.sample(params).unsqueeze(-1)  # TODO: verify unsqueeze here
        assert y_d.shape == (1, 1)

        ys = t.cat((ys, y_d), -1)
        assert ys.shape == (1, d+1)

        last_params[:, d] = params
        assert (last_params[:, d+1:] == 1.).all()
        assert (last_params[:, :d+1] != 1.).all()  # _highly_ unlikely that a param is exactly 1.

    assert ys.shape == (1, sp.data_dim)


# Truncated Gaussian tests -----------------------------------------------------


class TSANParams(san.SANParams):
    epochs: int = 10
    batch_size: int = 1024
    dtype: t.dtype = t.float32
    cond_dim: int = 7
    data_dim: int = 6
    module_shape: list[int] = [16, 32]
    sequence_features: int = 8
    likelihood: Type[san.SAN_Likelihood] = san.TruncatedMoG

    likelihood_kwargs: Optional[dict[str, Any]] = {
        'lims': t.tensor(ForwardModelParams().free_param_lims(normalised=True)),
        'K': 8, 'mult_eps': 1e-4, 'abs_eps': 1e-4,
    }

    layer_norm: bool = True
    train_rsample: bool = False
    opt_lr: float = 1e-3
    limits: Any = t.tensor([[0., 1.]]).repeat((6, 1))  # [6, 2]


def test_tsan_initialisation():
    sp = TSANParams()
    model = san.SAN(sp)
    assert model.module_shape == sp.module_shape
    assert model.sequence_features == sp.sequence_features


def test_san_tmog_likelihood():
    dtype: t.float32
    device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")

    fp = ForwardModelParams()
    ip = InferenceParams()
    sp = TSANParams()

    train_loader, test_loader = load_simulated_data(
        path=ip.dataset_loc, split_ratio=ip.split_ratio,
        batch_size=sp.batch_size, # test_batch_size=1,
        phot_transforms=[lambda x: t.from_numpy(x).log()],
        theta_transforms=[get_norm_theta(fp)])

    model = san.SAN(sp)

    assert isinstance(model.likelihood, san.TruncatedMoG)
    assert model.likelihood.name == "TMoG"
    # 8 mixture components per output dimension
    assert model.likelihood.K == 8
    # number of parameters required for each dimension of the output
    # (loc, scale, mixture_weight) * number of mixture components
    assert model.likelihood.n_params() == 24
    assert model.last_params is None

    assert (model.likelihood.lower == 0.).all()
    assert (model.likelihood.upper == 1.).all()


# def test_truncated_mog_likelihood():
#     # What is there to test?
#     # - one dimensional and batch dimensions
#     # - whether the support is being applied correctly
#     # - whether samples are within the expected range
#     # - whether you can call log prob on valid points
#     #
#     # Note: the limits are the same for each mixture component; they just change
#     # per dimension. They are even the same for different batches...
#
#     K = 10
#     lims = t.tensor(())



    # parameters are [B, n_params] big. We need to incorporate information about
    # the limits on each dimension. Note that the parameter shape currently
    # decomposes to [B, (dims), K, NP] where dims is the number of dimensions of
    # the data vector we're interested in, and NP are the number of parameters
    # required for an individual mixture component (e.g. loc, scale, weight).
    #
    # In particular, the same limits will be shared among the B, K, NP
    # dimensions.
    #
    # It may just be simpler to accept an additional dimension parameter which
    # is used to determine which dimension we're interested in when calling
    # log_prob, sample etc.

    # we are just off to test the extract params thing to see the structure of
    # the parameter blob
    #
    # Question: how can we provide the upper and lower limits along with the
    # rest of the parameters without them getting all jumbled up?
    #
    # batches don't get re-shaped
    # single parameters for the $n$th variable (of num_params) are strided by $n$
