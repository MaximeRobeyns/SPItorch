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
"""A light wrapper around Prospector"""

import logging
import numpy as np
import pandas as pd

from typing import Callable, Optional

import spt.visualisation as vis

from spt.types import tensor_like
from spt.config import ForwardModelParams


def get_forward_model(galaxy: Optional[pd.Series] = None,
                      mp = ForwardModelParams) -> Callable[[np.ndarray], np.ndarray]:
    """Returns a (photometry-only) forward model.

    Useful for predicting the photometry for a given theta sample (e.g.
    validating ML model outputs).

    Args:
        galaxy: galaxy to initialise obs dict with. If left out, a dummy galaxy
            will be used.
            TODO: remove this if it has no effect on predictions.
        mp: The forward model parameters.

    Returns:
        Callable[[np.ndarray], np.ndarray]: A callable forward model.
    """
    logging.info('Initialising new forward model')
    mp = mp()
    _obs = mp.build_obs_fn(mp.filters, galaxy)
    _model = mp.build_model_fn(mp.all_params, mp.ordered_params)
    _sps = mp.build_sps_fn(**mp.sps_kwargs)

    def f(theta: np.ndarray) -> np.ndarray:
        _, phot, _ = _model.sed(theta, obs=_obs, sps=_sps)
        return phot

    logging.info('Forward model created')
    return f


class Prospector:

    def __init__(self, galaxy: Optional[pd.Series] = None, mp = ForwardModelParams):
        """Construct a prospector instance for simulation and MCMC-based
        parameter inference.

        Args:
            galaxy: an optional galaxy. If None, a dummy galaxy will be used to
                make prospector happy (e.g. for sampling).
        """
        logging.debug('Initialising prospector class')

        mp = mp()

        self.obs = mp.build_obs_fn(mp.filters, galaxy)
        logging.debug(f'Created obs dict: {self.obs}')

        self.model = mp.build_model_fn(mp.all_params, mp.ordered_free_params)
        logging.debug(f'Created model: {self.model}')

        self.sps = mp.build_sps_fn(**mp.sps_kwargs)
        logging.debug(f'Created sps: {self.sps}')

    def __call__(self, theta: Optional[np.ndarray] = None,
                 ) -> tuple[np.ndarray, np.ndarray]:
        """Compute spectroscopy and photometry for the (optional)
        parameter array.

        Args:
            theta: An optional array of (denormalised)

        Returns:
            tuple[np.ndarray, np.ndarray]: spectroscopy, photoemtry
        """
        if theta is None:
            theta = self.model.theta
        spec, phot, _ = self.model.sed(theta, obs=self.obs, sps=self.sps)
        # TODO check that these are correct
        return spec, phot

    # def photometry(self, theta: Optional[np.ndarray] = self.model.theta
    #               ) -> np.ndarray:
    #     """Return the simulated photometric observations

    #     Args:
    #         theta: [TODO:description]

    #     Returns:
    #         np.ndarray: [TODO:description]
    #     """
    #     raise NotImplementedError

    def visualise_obs(self, show: bool=True, save: bool=False,
                      path: str = None):
        logging.info('[bold]Visualising observations')
        if self.obs['_fake_galaxy']:
            logging.warning((
                'Calling visualise_obs with fake galaxy observations.\n'
                'Please construct the Prospector object with a real galaxy '
                'instead.'))
        vis.visualise_obs(self.obs, show, save, path)


    def visualise_model(self, theta: Optional[tensor_like] = None,
                        no_obs: bool = False, show: bool = True,
                        save: bool = False, path: str = None):
        """Visualise predicted photometry from a theta vector.

        Args:
            theta: An optionally speciifed parameter vector. If omitted, the
                model's current parameter vector is used.
            no_obs: Whether to omit the photometric observations. This is
                useful when using a dummy 'obs' dictionary when using this
                Prospector class in 'forward-model' mode.
            show: show the plot?
            save: save the plot?
            path: where to save the plot.
        """
        logging.info('[bold]Visualising model predictions')

        # If theta is not specified, get the current model's theta values:
        if theta is None:
            theta = self.model.theta.copy()

        if not no_obs and self.obs['_fake_galaxy']:
            logging.warning((
                'Plotting fake galaxy observations.\n'
                'To avoid plotting observations in model predictions, '
                'set no_obs=True.'))

        vis.visualise_model(self.model, self.sps, theta,
                            None if no_obs else self.obs, show,
                            save, path)
