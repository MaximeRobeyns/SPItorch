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
"""Project configuration file

Edit the classes in this file to configure SPItorch before running one of the
targets in the Makefile.
"""

import os
import logging
import torch as t
import prospect.models.priors as priors

from typing import Any, Optional, Type

import spt
import spt.modelling
import spt.inference as inference

from spt.utils import ConfigClass
from spt.types import MCMCMethod, FittingMethod, ConcurrencyMethod
from spt.filters import Filter, FilterLibrary, FilterCheck
from spt.inference import san
from spt.modelling import Parameter, ParamConfig
from spt.modelling import build_obs_fn_t, build_model_fn_t, build_sps_fn_t


class ForwardModelParams(FilterCheck, ParamConfig, ConfigClass):
    """Default parameters for the forward model"""

    # Observations ------------------------------------------------------------

    filters: list[Filter] = FilterLibrary['des']
    build_obs_fn: build_obs_fn_t = spt.modelling.build_obs

    # Model parameters --------------------------------------------------------

    # List of templates
    model_param_templates: list[str] = ['parametric_sfh']

    # Manually defined SedModel parameters:
    # - parameters below with model_this=True are modelled (in Prospector & ML)
    # - the 'name' attribute must be some FSPS name
    # - use the 'units' to describe and notate the param (you can use LaTeX!)
    # - the order matters for ML models: if you reorder them, retrain the model
    model_params: list[Parameter] = [
        Parameter('zred', 0., 0.1, 4., units='redshift, $z$'),
        Parameter('mass', 6, 8, 10, priors.LogUniform, units='log_mass',
                  disp_floor=6.),
        Parameter('logzsol', -2, -0.5, 0.19, units='$\\log (Z/Z_\\odot)$'),
        Parameter('dust2', 0., 0.05, 2., units='optical depth at 5500AA'),
        Parameter('tage', 0.001, 13., 13.8, units='Age, Gyr', disp_floor=1.),
        Parameter('tau', 0.1, 1, 100, priors.LogUniform, units='Gyr^{-1}'),
    ]

    build_model_fn: build_model_fn_t = spt.modelling.build_model

    # SPS parameters ----------------------------------------------------------

    sps_kwargs: dict[str, Any] = {'zcontinuous': 1}
    build_sps_fn: build_sps_fn_t = spt.modelling.build_sps


# ==================== Sampling (offline dset) Parameters =====================


class SamplingParams(ConfigClass):

    # n_samples: int = int(50e6)
    n_samples: int = int(1e6)
    concurrency: int = 8
    observation: bool = False  # use real observation as obs... should have no effect
    save_dir = './data/dsets/dev/'  # Make this unique
    combine_samples: bool = True  # combine partial samples into one big file?
    cmethod: ConcurrencyMethod = ConcurrencyMethod.MPI  # how to multithread


# ============================ Inference Parameters ===========================


class InferenceParams(inference.InferenceParams):

    # The model to use
    model: inference.Model = san.SAN

    # Train / test split ratio (offline training only)
    split_ratio: float = 0.9

    # Number of iterations bewteen logs
    logging_frequency: int = 1000

    # Filepath to hdf5 file or directory of files to use as offline dataset
    dataset_loc: str = SamplingParams().save_dir

    # Force re-train an existing model
    retrain_model: bool = False

    # Attempt to use checkpoints (if any) or start training from scratch. If
    # set to False, any previous checkpoints are deleted!
    use_existing_checkpoints: bool = True

    ident: str = 'test_model'

    # Ensure that the forward model description in ForwardModelParams matches
    # the data below (e.g. number / types of filters etc)
    catalogue_loc: str = './data/DES_VIDEO_v1.0.1.fits'


# Prospector fitting parameters -----------------------------------------------


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
    results_dir = './results/mcmc/emcee_samples/'


class DynestyParams(ConfigClass):

    nested_method: str = 'rwalk'
    nlive_init: int = 400
    nlive_batch: int = 200
    nested_dlogz_init: float = 0.05
    nested_posterior_thresh: float = 0.05
    nested_maxcall: int = int(1e7)

    # Whether to use numerical optimisation
    optimise: bool = True
    min_method: FittingMethod = FittingMethod.LM
    nmin: int = 10
    results_dir = './results/mcmc/dynesty_samples/'


# Machine learning inference --------------------------------------------------


class SANParams(san.SANParams):

    # Number of epochs to train for (offline training)
    epochs: int = 20

    batch_size: int = 1024

    dtype: t.dtype = t.float32

    # Dimension of observed photometry
    cond_dim: int = len(ForwardModelParams().filters)

    # Number of free parameters to predict
    data_dim: int = len(ForwardModelParams().free_params)

    # shape of the network 'modules'
    module_shape: list[int] = [512, 512]

    # features passed between sequential blocks
    sequence_features: int = 8

    likelihood: Type[san.SAN_Likelihood] = san.MoG

    likelihood_kwargs: Optional[dict[str, Any]] = {
        'K': 10, 'mult_eps': 1e-4, 'abs_eps': 1e04
    }

    # Whether to use batch norm
    batch_norm: bool = True

    # Optimiser (Adam) learning rate
    opt_lr: float = 1e-4


# =========================== Logging Parameters ==============================


class LoggingParams(ConfigClass):
    """Logging parameters

    For reference, the logging levels are:

    CRITICAL (50) > ERROR (40) > WARNING (30) > INFO (20) > DEBUG (10) > NOTSET

    Logs are output for the given level and higher (e.g. logging.WARNING
    returns all warnings, errors and critical logs).
    """

    file_loc: str = './logs.txt'

    log_to_file: bool = True
    file_level: int = logging.INFO

    log_to_console: bool = True
    console_level: int = logging.INFO

    # NOTE: if set to true, you _should_ set log_to_console = False above.
    debug_logs: bool = False
    debug_level: int = logging.DEBUG
