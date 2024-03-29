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
"""Project configuration file

Edit the classes in this file to configure SPItorch before running one of the
targets in the Makefile.
"""

import os
import math
import logging
import torch as t
import prospect.models.priors as priors

from typing import Any, Optional, Type

import spt
import spt.modelling
import spt.inference as inference

from spt.utils import ConfigClass
from spt.types import MCMCMethod, FittingMethod, ConcurrencyMethod, Tensor
from spt.filters import Filter, FilterLibrary, FilterCheck
from spt.inference import san
from spt.inference import maf
from spt.modelling import Parameter, ParamConfig
from spt.modelling import build_obs_fn_t, build_model_fn_t, build_sps_fn_t


class ForwardModelParams(FilterCheck, ParamConfig, ConfigClass):
    """Default parameters for the forward model"""

    # Observations ------------------------------------------------------------

    filters: list[Filter] = FilterLibrary["des"]
    build_obs_fn: build_obs_fn_t = spt.modelling.build_obs

    # Model parameters --------------------------------------------------------

    # Begin with a library of parameters.
    model_param_templates: list[str] = ["parametric_sfh"]

    # Manually override some properties of the SedModel parameters from the
    # template above. You can also leave out the template entirely and define
    # all the parameters below.
    #
    # Note:
    # - the 'name' attribute must be some FSPS name.
    # - 'units' describes notation (you can use LaTeX)
    # - the order matters for ML models: if you reorder them, retrain the model
    # - set 'model_this=False' to define a fixed parameter
    # - range_min and range_max delimit the allowable range, and serve as bounds
    #       for the Uniform distribution (the default if the prior distribution
    #       is omitted)
    model_params: list[Parameter] = [
        Parameter("zred", 0.0, 0.1, 4.0, units="redshift, $z$"),
        Parameter(
            "mass",
            10**6,
            10**8,
            10**10,
            priors.LogUniform,
            units="$log(M/M_\\odot)$",
            disp_floor=10**6.0,
        ),
        Parameter("logzsol", -2, -0.5, 0.19, units="$\\log (Z/Z_\\odot)$"),
        Parameter("dust2", 0.0, 0.05, 2.0, units="optical depth at 5500AA"),
        Parameter("tage", 0.001, 13.0, 13.8, units="Age, Gyr", disp_floor=1.0),
        # Parameter('tau', 0.1, 1, 100, priors.LogUniform, units='Gyr^{-1}'),
        Parameter(
            "tau", 10 ** (-1), 10**0, 10**2, priors.LogUniform, units="Gyr^{-1}"
        ),
    ]

    # Estimate the redshift first ----------------------------------------------

    @property
    def ordered_params(self) -> list[str]:
        params = list(self.all_params.keys())
        params.sort()
        params.insert(0, params.pop(params.index("zred")))
        return params

    @property
    def ordered_free_params(self) -> list[str]:
        fp = list(self.free_params.keys())
        fp.sort()
        fp.insert(0, fp.pop(fp.index("zred")))
        return fp

    build_model_fn: build_model_fn_t = spt.modelling.build_model

    # SPS parameters ----------------------------------------------------------

    sps_kwargs: dict[str, Any] = {"zcontinuous": 1}
    build_sps_fn: build_sps_fn_t = spt.modelling.build_sps


# ==================== Sampling (offline dset) Parameters =====================


class SamplingParams(ConfigClass):

    n_samples: int = int(10e6)
    # n_samples: int = int(10e4)
    concurrency: int = 10
    observation: bool = False  # use real observation as obs... should have no effect
    save_dir: str = "./data/dsets/example/"
    combine_samples: bool = True  # combine partial samples into one big file?
    cmethod: ConcurrencyMethod = ConcurrencyMethod.MPI  # how to multithread


# ============================ Inference Parameters ===========================


class InferenceParams(inference.InferenceParams):

    # The model to use
    model: inference.model_t = san.SAN

    # Train / test split ratio (offline training only)
    split_ratio: float = 0.9

    # Number of iterations bewteen logs
    logging_frequency: int = 1000

    # Filepath to hdf5 file or directory of files to use as offline dataset
    # dataset_loc: str = SamplingParams().save_dir
    dataset_loc: str = "./data/dsets/example/photometry_sim_10000000.h5"

    # Force re-training of an existing model
    retrain_model: bool = False

    # Attempt to use checkpoints (if any) or start training from scratch. If
    # set to False, any previous checkpoints are deleted!
    use_existing_checkpoints: bool = True

    # ident: str = "zred1"
    # ident: str = "discrete_l"
    # ident: str = "discrete_l_rand"
    ident: str = "input_whitening"

    # Ensure that the forward model description in ForwardModelParams matches
    # the data below (e.g. number / types of filters etc)
    catalogue_loc: str = "./data/catalogues/DES_VIDEO_v1.0.1.fits"

    # HMC update parameters ----------------------------------------------------

    hmc_update_batch_size: int = 500  # HMC update consumes more memory than ML

    hmc_update_N: int = 5  # number of HMC steps
    hmc_update_C: int = 100  # number of chains to use in HMC
    hmc_update_D: int = len(ForwardModelParams().filters)  # data dimensions
    hmc_update_L: int = 2  # leapfrog integrator steps per iteration
    hmc_update_rho: float = 0.1  # step size
    hmc_update_alpha: float = 1.1  # momentum

    # simulated data update procedure:

    # The number of samples to use in each update step.
    # (note: quickly increases memory requirements)
    hmc_update_sim_K: int = 5
    hmc_update_sim_ident: str = f"update_sim_{ident}"  # saving / checkpointing
    hmc_update_sim_epochs: int = 5

    # real data update procedure:

    hmc_update_real_K: int = 5
    hmc_update_real_epochs: int = 5
    hmc_update_real_ident: str = f"update_real_{ident}"


# Baseline MCMC fitting parameters (Prospector) -------------------------------


class FittingParams(ConfigClass):
    """
    Parameters controlling an optional numerical optimisation procedure to
    initialise the model parameters to sensible values / 'burn' them in in
    preparation for later MCMC sampling.
    """

    # Whether or not to do the fitting in the first place?
    do_fitting: bool = True

    min_method: FittingMethod = FittingMethod.LM

    # Start minimisation at n different places to guard against local minima:
    nmin: int = 2


class EMCEEParams(ConfigClass):
    nwalkers: int = 128
    nburn: list[int] = [16, 32, 64]
    niter: int = 512

    # Whether to use numerical optimisation
    optimise: bool = True
    min_method: FittingMethod = FittingMethod.LM
    nmin: int = 10
    pool: ConcurrencyMethod = ConcurrencyMethod.none  # MPI recommended
    workers = 6
    results_dir = "./results/mcmc/emcee_samples/"


class DynestyParams(ConfigClass):
    nested_method: str = "rwalk"
    nlive_init: int = 400
    nlive_batch: int = 200
    nested_dlogz_init: float = 0.05
    nested_posterior_thresh: float = 0.05
    nested_maxcall: int = int(1e7)

    # Whether to use numerical optimisation
    optimise: bool = True
    min_method: FittingMethod = FittingMethod.LM
    nmin: int = 10
    results_dir = "./results/mcmc/dynesty_samples/"


# Machine learning inference --------------------------------------------------


class MAFParams(maf.MAFParams):
    """Parameters for masked autoregressive flow CNDE for approx posterior"""

    epochs: int = 10

    batch_size: int = 1024

    dtype: t.dtype = t.float32

    cond_dim: int = len(ForwardModelParams().filters)

    data_dim: int = len(ForwardModelParams().free_params)

    prior: t.distributions.Distribution = maf.LogisticPrior

    depth: int = 6

    maf_depth: int = 4

    maf_hidden_width: int = 256

    maf_num_masks: int = 1

    natural_ordering: bool = True

    layer_norm: bool = True

    opt_lr: float = 3e-3

    opt_decay: float = 1e-4

    limits = None


class SANParams(san.SANParams):
    """These are parameters for the approximate posterior."""

    # Number of epochs to train for (offline training)
    epochs: int = 10

    batch_size: int = 1024

    dtype: t.dtype = t.float32

    # Dimension of observed photometry
    cond_dim: int = len(ForwardModelParams().filters)

    # Number of free parameters to predict
    data_dim: int = len(ForwardModelParams().free_params)

    # Shape of first module
    # This is used to generate the first sequence features and should be an
    # accurate density estimator for the redshift.
    first_module_shape: list[int] = [1024, 2048, 1024]

    # shape of the network 'modules'
    module_shape: list[int] = [1024, 1024]

    # features passed between sequential blocks
    sequence_features: int = 16

    # Mixture-of-betas distribution
    # likelihood: Type[san.SAN_Likelihood] = san.MoB
    # likelihood_kwargs: Optional[dict[str, Any]] = {
    #     'lims': t.tensor(ForwardModelParams().free_param_lims(normalised=True)),
    #     'K': 10, 'mult_eps': 1e-4, 'abs_eps': 1e-4
    # }

    likelihood: Type[san.SAN_Likelihood] = san.TruncatedMoG
    likelihood_kwargs: Optional[dict[str, Any]] = {
        "lims": t.tensor(ForwardModelParams().free_param_lims(normalised=True)),
        "K": 10,
        "mult_eps": 1e-4,
        "abs_eps": 1e-4,
        "trunc_eps": 1e-4,
        "validate_args": False,
    }

    # Whether to use layer normalisation
    layer_norm: bool = True

    # Whether to use reparametrised sampling during training (leave to False)
    train_rsample: bool = False

    # Optimiser (Adam) learning rate
    opt_lr: float = 3e-3

    # Optimiser (Adam) weight decay
    opt_decay: float = 1e-4


class SANv2Params(san.SANv2Params):

    epochs: int = 4

    # batch_size: int = 5000
    batch_size: int = 4016

    # dtype: t.dtype = t.float32
    dtype: t.dtype = t.bfloat16

    cond_dim: int = len(ForwardModelParams().filters)

    latent_features: int = 100

    data_dim: int = len(ForwardModelParams().free_params)

    # layers in the main encoder block
    # encoder_layers: list[int] = [7000, 7000]
    encoder_layers: list[int] = [5000, 5000]

    # shape of the sequence modules
    # module_shape: list[int] = [2000]
    module_shape: list[int] = [1000]

    # features passed between sequential blocks
    sequence_features: int = 4

    # Mixture-of-betas distribution
    # likelihood: Type[san.SAN_Likelihood] = san.MoB
    # likelihood_kwargs: Optional[dict[str, Any]] = {
    #     'lims': t.tensor(ForwardModelParams().free_param_lims(normalised=True)),
    #     'K': 10, 'mult_eps': 1e-4, 'abs_eps': 1e-4
    # }

    # Truncated MoG distribution
    # likelihood: Type[san.SAN_Likelihood] = san.TruncatedMoG
    # likelihood_kwargs: Optional[dict[str, Any]] = {
    #     "lims": t.tensor(ForwardModelParams().free_param_lims(normalised=True)),
    #     "K": 5,
    #     "mult_eps": 1e-4,
    #     "abs_eps": 1e-4,
    #     "trunc_eps": 1e-4,
    #     "validate_args": False,
    # }

    # Softmax marginals
    likelihood: Type[san.SAN_Likelihood] = san.Softmax
    likelihood_kwargs: Optional[dict[str, Any]] = {
        "lims": t.tensor(ForwardModelParams().free_param_lims(normalised=True)),
        "atoms": 150,
    }

    # Whether to use layer normalisation
    layer_norm: bool = True

    # Whether to use reparametrised sampling during training (leave to False)
    train_rsample: bool = False

    # Optimiser (Adam) learning rate
    # opt_lr: float = 7e-4
    opt_lr: float = 1e-3

    # Optimiser (Adam) weight decay
    opt_decay: float = 1e-4


class SANLikelihoodParams(san.SANParams):
    """Parameters for the neural likelihood.

    Note: this network can be much smaller than the approximate posterior.
    """

    # Number of epochs to train for (offline training)
    epochs: int = 10

    batch_size: int = 1024

    dtype: t.dtype = t.float32

    # Dimension of observed photometry
    data_dim: int = len(ForwardModelParams().filters)

    # Dimension of physical parameters
    cond_dim: int = len(ForwardModelParams().free_params)

    # shape of the first network block
    first_module_shape: list[int] = [200, 200]

    # shape of subsequent network 'modules'
    module_shape: list[int] = [
        200,
    ]

    # features passed between sequential blocks
    sequence_features: int = 8

    likelihood: Type[san.SAN_Likelihood] = san.MoG

    likelihood_kwargs: Optional[dict[str, Any]] = {
        "K": 3,
        "mult_eps": 1e-4,
        "abs_eps": 1e-4,
    }

    # Whether to use layer normalisation
    layer_norm: bool = True

    # Whether to use reparametrised sampling during training
    train_rsample: bool = False

    # Optimiser (Adam) learning rate
    opt_lr: float = 3e-3

    # Optimiser (Adam) weight decay
    opt_decay: float = 1e-4


class SANv2LikelihoodParams(san.SANv2Params):
    """Parameters for the neural likelihood.

    Note: this network can be much smaller than the approximate posterior.
    """

    # Number of epochs to train for (offline training)
    epochs: int = 10

    batch_size: int = 2048

    dtype: t.dtype = t.float32

    # Dimension of observed photometry
    data_dim: int = len(ForwardModelParams().filters)

    # Dimension of latent feature vector
    latent_features: int = 100

    # Dimension of physical parameters
    cond_dim: int = len(ForwardModelParams().free_params)

    # shape of the first network block
    encoder_layers: list[int] = [
        512,
    ]

    # shape of subsequent network 'modules'
    module_shape: list[int] = [
        128,
    ]

    # features passed between sequential blocks
    sequence_features: int = 8

    likelihood: Type[san.SAN_Likelihood] = san.MoG

    likelihood_kwargs: Optional[dict[str, Any]] = {
        "K": 3,
        "mult_eps": 1e-4,
        "abs_eps": 1e-4,
    }

    # Whether to use layer normalisation
    layer_norm: bool = True

    # Whether to use reparametrised sampling during training
    train_rsample: bool = False

    # Optimiser (Adam) learning rate
    opt_lr: float = 3e-3

    # Optimiser (Adam) weight decay
    opt_decay: float = 1e-4


# =========================== Logging Parameters ==============================


class LoggingParams(ConfigClass):
    """Logging parameters

    For reference, the logging levels are:

    CRITICAL (50) > ERROR (40) > WARNING (30) > INFO (20) > DEBUG (10) > NOTSET

    Logs are output for the given level and higher (e.g. logging.WARNING
    returns all warnings, errors and critical logs).
    """

    file_loc: str = "./logs.txt"

    log_to_file: bool = True
    file_level: int = logging.INFO

    log_to_console: bool = True
    console_level: int = logging.INFO

    # NOTE: if set to true, you _should_ set log_to_console = False above.
    debug_logs: bool = False
    debug_level: int = logging.DEBUG
