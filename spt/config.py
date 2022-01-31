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

import logging
import prospect.models.priors as priors

from typing import Any

import spt
import spt.modelling

from spt.utils import ConfigClass
from spt.modelling import Parameter, build_obs_fn_t, build_model_fn_t,\
        build_sps_fn_t
from spt.filters import Filter, FilterLibrary


class ForwardModelParams(ConfigClass):
    """Default parameters for the forward model"""

    # Observations ------------------------------------------------------------

    filters: list[Filter] = FilterLibrary['des']
    build_obs_fn: build_obs_fn_t = spt.modelling.build_obs

    # Model parameters --------------------------------------------------------

    # List of templates
    param_templates: list[str] = ['parametric_sfh']

    # Manually defined SedModel parameters:
    # - parameters below with model_this=True are modelled (in Prospector & ML)
    # - the 'name' attribute must be some FSPS name
    # - use the 'units' to describe and notate the param (you can use LaTeX!)
    # - the order matters for ML models: if you reorder them, retrain the model
    params: list[Parameter] = [
        Parameter('zred', 0., 0.1, 4., units='redshift, $z$'),
        Parameter('mass', 6, 8, 10, priors.LogUniform, units='log_mass',
                  disp_floor=6.),
        Parameter('logzsol', -2, -0.5, 0.19, units='$\\log (Z/Z_\\odot)$'),
        Parameter('dust2', 0., 0.05, 2., units='optical depth at 5500AA'),
        Parameter('tage', 0.001, 13., 13.8, units='Age, Gyr', disp_floor=1.),
        Parameter('tau', 0.1, 1, 100, priors.LogUniform, units='Gyr^{-1}'),
    ]

    model_kwargs: dict[str, Any] = {}
    build_model_fn: build_model_fn_t = spt.modelling.build_model

    build_sps_fn: build_sps_fn_t = spt.modelling.build_sps


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
