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

from sedpy import observate
from typing import Any, Callable, Optional, Union
from prospect.models.sedmodel import SedModel
from prospect.utils.obsutils import fix_obs
from prospect.sources import SSPBasis, CSPSpecBasis

from spt import load_photometry
from spt.filters import Filter
from spt.modelling.parameter import pdict_t


# Types -----------------------------------------------------------------------


obs_dict_t = dict[str, Union[np.ndarray, list[observate.Filter], bool, None]]

build_obs_fn_t = Callable[[Any, list[Filter], Optional[pd.Series]], obs_dict_t]
build_sps_fn_t = Callable[[Any, Any], SSPBasis]
build_model_fn_t = Callable[[Any, dict[str, pdict_t], Optional[list[str]]], SedModel]


def build_obs(_, filters: list[Filter], observation: Optional[pd.Series]) -> obs_dict_t:
    """Build a dictionary of photometry (and perhaps eventually spectra).

    Note: the first argument is _ to emulate a static method (`self` is
    implicitly passed to this method when it is called).

    Args:
        filters: The SPS filter list to use
        observation: An optional observation to use while inferring parameters with MCMC
            and other sampling methods. When we only want to use Prospector for
            the forward models, we can leave it out and a 'dummy' observation will
            be created.

    Returns:
        obs_dict_t: A dictionary of observational data to use in the fit.
    """

    obs: obs_dict_t = {}

    if observation is not None:
        f, m, mu = load_photometry.load_observation_for_prospector(
            observation, filters)
        if 'idx' in observation:
            obs['_index'] = observation['idx']
            obs['_fake_observation'] = False

        if 'survey' in observation:
            obs['_survey'] = observation['survey']
    else:
        f, m, mu = load_photometry.load_dummy_observation(filters)
        # A fake observations is one that clearly can't be fitted (e.g. all
        # ones). For this purpose, a simulated observation is not a 'fake'
        # observation...
        obs['_fake_observation'] = True


    obs['filters'] = f
    obs['maggies'] = m
    obs['maggies_unc'] = mu

    assert isinstance(obs['filters'], list)

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


def build_model(_, params: dict[str, pdict_t], param_order: Optional[list[str]] = None
                ) -> SedModel:
    return SedModel(params, param_order=param_order)


def build_sps(_, zcontinuous: int = 1) -> SSPBasis:
    """An extremely simple function to build an SPS model."""
    return CSPSpecBasis(zcontinuous=zcontinuous)
