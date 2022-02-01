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
"""Contains base builders for prospector."""

import numpy as np
import pandas as pd
import logging

from sedpy import observate
from typing import Any, Callable, Optional, Union
from prospect.models.sedmodel import SedModel
from prospect.models.templates import TemplateLibrary
from prospect.utils.obsutils import fix_obs
from prospect.sources import SSPBasis, CSPSpecBasis

from spt import load_photometry
from spt.filters import Filter
from spt.modelling.parameter import Parameter, pdict_t


# Types -----------------------------------------------------------------------


obs_dict_t = dict[str, Union[np.ndarray, list[observate.Filter], bool, None]]

build_obs_fn_t = Callable[[list[Filter], Optional[pd.Series]], obs_dict_t]
build_sps_fn_t = Callable[[Any], SSPBasis]
build_model_fn_t = Callable[[list[Parameter], list[str]], SedModel]


@staticmethod
def build_obs(filters: list[Filter], galaxy: Optional[pd.Series]) -> obs_dict_t:
    """Build a dictionary of photometry (and perhaps eventually spectra).

    Args:
        filters: The SPS filter list to use
        galaxy: An optional galaxy to use while inferring parameters with MCMC
            and other sampling methods. When we only want to use Prospector for
            the forward models, we can leave it out and a 'dummy' galaxy will
            be created.

    Returns:
        obs_dict_t: A dictionary of observational data to use in the fit.
    """

    obs: obs_dict_t = {}

    if galaxy is not None:
        f, m, mu = load_photometry.load_galaxy_for_prospector(
            galaxy, filters)
        obs['_fake_galaxy'] = False
    else:
        f, m, mu = load_photometry.load_dummy_galaxy(filters)
        obs['_fake_galaxy'] = True

    obs['filters'] = f
    obs['maggies'] = m
    obs['maggies_unc'] = mu

    # This mask tells us which flux values to conisder in the likelihood.
    # Mask values are True for values that you want to fit.
    obs['phot_mask'] = np.array([True for _ in obs['filters']])

    # This is an array of the effective wavelengths for each of the filters.
    # Unnecessary, but useful for plotting so it's stored here for convenience
    obs['phot_wave'] = np.array([f.wave_effective for f in obs['filters']])

    # Since we don't have a spectrum, we set some required elements of the obs
    # directory to None. (This would be a vector of vacuum wavelengths in
    # angstroms.) NOTE: Could use the SDSS spectra here for truth label fitting
    # This is inelegant wrt. the types of cpz_obs_dict_t but since this is a
    # Prospector API, it's not trivial to get aroud...
    obs['wavelength'] = None
    obs['spectrum'] = None
    obs['unc'] = None
    obs['mask'] = None

    return fix_obs(obs)


def combine_params(parameters: list[Parameter] = [],
                   templates: list[str] = []) -> pdict_t:
    """A utility method to combine template and manually specified model
    parameters.

    Returns:
        pdict_t: [TODO:description]
    """

    model_params: pdict_t = {}

    # Begin by applying the templates...
    for t in templates:
        if t not in TemplateLibrary._entries.keys():
            logging.warning(f'Template library {t} is not recognized.')
        else:
            model_params |= TemplateLibrary[t]

    # Such that we can override parameters with the manually-defined
    # parameters:
    for p in parameters:
        model_params |= p.to_dict()

    return model_params


@staticmethod
def build_model(parameters: list[Parameter],
                templates: list[str] = ['parametric_sfh']) -> SedModel:

    model_params = combine_params(parameters, templates)

    return SedModel(model_params)


@staticmethod
def build_sps(zcontinuous: int = 1) -> SSPBasis:
    """An extremely simple function to build an SPS model.
    """
    return CSPSpecBasis(zcontinuous=zcontinuous)
