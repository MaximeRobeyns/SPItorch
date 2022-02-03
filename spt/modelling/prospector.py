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

import os
import logging
import numpy as np
import pandas as pd

from rich.console import Console
from typing import Any, Callable, Optional, Union
from prospect.fitting import fit_model, lnprobfn
from prospect.io import write_results as writer

import spt.visualisation as vis

from spt.types import tensor_like, FittingMethod, ConcurrencyMethod
from spt.config import ForwardModelParams, FittingParams, EMCEEParams,\
                       DynestyParams

prun_params_t = dict[str, Union[int, bool, float, None, list[int], str, Any]]


def get_forward_model(observtion: Optional[pd.Series] = None,
                      mp = ForwardModelParams) -> Callable[[np.ndarray], np.ndarray]:
    """Returns a (photometry-only) forward model.

    Useful for predicting the photometry for a given theta sample (e.g.
    validating ML model outputs).

    Args:
        observtion: observtion to initialise obs dict with. If left out, a dummy observtion
            will be used.
            TODO: remove this if it has no effect on predictions.
        mp: The forward model parameters.

    Returns:
        Callable[[np.ndarray], np.ndarray]: A callable forward model.
    """
    logging.info('Initialising new forward model')
    mp = mp()
    _obs = mp.build_obs_fn(mp.filters, observtion)
    _model = mp.build_model_fn(mp.all_params, mp.ordered_params)
    _sps = mp.build_sps_fn(**mp.sps_kwargs)

    def f(theta: np.ndarray) -> np.ndarray:
        _, phot, _ = _model.sed(theta, obs=_obs, sps=_sps)
        return phot

    logging.info('Forward model created')
    return f


class Prospector:

    def __init__(self, observtion: Optional[pd.Series] = None, mp = ForwardModelParams):
        """Construct a prospector instance for simulation and MCMC-based
        parameter inference.

        Args:
            observtion: an optional observtion. If None, a dummy observtion will be used to
                make prospector happy (e.g. for sampling).
        """
        logging.info('Initialising prospector class')

        mp = mp()
        if observtion is not None and 'idx' in observtion:
            self.index = int(observtion.idx)

        self.obs = mp.build_obs_fn(mp.filters, observtion)
        logging.info(f'Created obs dict with filters\n{[f.name for f in self.obs["filters"]]}')
        logging.debug(f'Created obs dict: {self.obs}')

        self.model = mp.build_model_fn(mp.all_params, mp.ordered_params)
        logging.info(f'Created model:\n\tfree params {self.model.free_params},\n\tfixed: {self.model.fixed_params})')
        logging.debug(f'Created model: {self.model}')

        logging.info(f'Creating sps object...')
        self.sps = mp.build_sps_fn(**mp.sps_kwargs)
        logging.info(f'Done.')
        logging.debug(f'Created sps: {self.sps}')

        self.mp = mp
        self.has_fit: bool = False

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

    def __repr__(self) -> str:
        c = Console(record=True, width=80)
        c.begin_capture()
        c.rule('Prospector Instance')
        c.print('Model is:')
        c.print(self.model)
        c.print(f'Filters:\n{[f.name for f in self.obs["filters"]]}')
        c.rule()
        return c.end_capture()

    def _fake_obs_warning(self, method: str = 'prospector method'):
            logging.warning((
                f'Calling {method} with a fake observation.\n'
                'Please construct the Prospector object with a real observtion '
                'instead.'))

    def set_theta(self, theta: np.ndarray):
        # TODO implement this.
        raise NotImplementedError

    def numerical_fit(self, fp: FittingParams = FittingParams()) -> list[Any]:
        """'Burn in' for layer MCMC sampling using a numerical method.

        This initialisation could also be done by a machine learning algorithm.
        """
        logging.info(f'Running {fp.min_method.value} fitting')
        if fp.min_method == FittingMethod.ML:
            raise NotImplementedError("MCMC initialisation with ML results not yet implemented.")

        run_params: prun_params_t = {
                'dynesty': False, 'emcee': False, 'optimize': True}
        run_params["min_method"] = fp.min_method.value
        run_params["nmin"] = fp.min_n

        output = fit_model(self.obs, self.model, self.sps, lnprobfn=lnprobfn,
                           **run_params)
        (results, time) = output['optimization']
        logging.info(f'Fitting took {time:.2f}s')
        assert results is not None
        return results

    def emcee_fit(self, ep: EMCEEParams = EMCEEParams()):
        """Runs MCMC method to update the value of self.model.theta
        """
        logging.info(f'Running EMCEE fitting with parameters:')
        logging.info(ep)
        if self.obs['_fake_observtion']:
            self._fake_obs_warning('emcee_fit')

        run_params: prun_params_t = {'dynesty': False, 'emcee': True}

        run_params['optimize'] = ep.optimise
        if ep.min_method == FittingMethod.ML:
            raise NotImplementedError("MCMC initialisation with ML results not yet implemented.")
        run_params['min_method'] = ep.min_method.value
        run_params['nmin'] = ep.min_n

        run_params['nwalkers'] = ep.nwalkers
        run_params['niter'] = ep.niter
        run_params['nburn'] = ep.nburn

        if ep.pool == ConcurrencyMethod.MPI:
            logging.info('running this for some reason')
            from schwimmbad.mpi import MPIPool
            run_params['pool'] = MPIPool()
        elif ep.pool == ConcurrencyMethod.native:
            from multiprocessing import Pool
            run_params['pool'] = Pool(ep.workers)

        output = fit_model(self.obs, self.model, self.sps, lnprobfn=lnprobfn,
                           **run_params)
        logging.info(f'Finished EMCEE sampling in {output["sampling"][1]:.2f}s')

        survey = os.path.basename(self.mp.catalogue_loc).split('.')[0]
        hfile = os.path.join(ep.results_dir, f'{survey}_{self.index}.h5')

        writer.write_hdf5(hfile, run_params, self.model, self.obs,
                          output["sampling"][0], output["optimization"][0],
                          tsample=output["sampling"][1],
                          toptimize=output["optimization"][1])
        logging.info(f'Saved EMCEE results to {hfile}')


    def dynesty_fit(self, dp: DynestyParams = DynestyParams()):
        """Runs Dynesty (nested) sampling to update the value of self.model.theta"""
        logging.info('Running Dynesty fitting with parameters:')
        logging.info(dp)
        if self.obs['_fake_observtion']:
            self._fake_obs_warning('dynesty_fit')

        run_params: prun_params_t = {'dynesty': True, 'emcee': False}

        run_params['optimize'] = dp.optimise
        if dp.min_method == FittingMethod.ML:
            raise NotImplementedError("MCMC initialisation with ML results not yet implemented.")
        run_params['min_method'] = dp.min_method.value
        run_params['nmin'] = dp.min_n

        run_params['nested_method'] = dp.nested_method
        run_params['nlive_init'] = dp.nlive_init
        run_params['nlive_batch'] = dp.nlive_batch
        run_params['nested_dlogz_init'] = dp.nested_dlogz_init
        run_params['nested_posterior_thresh'] = dp.nested_posterior_thresh
        run_params['nested_maxcall'] = dp.nested_maxcall

        output = fit_model(self.obs, self.model, self.sps, lnprobfn=lnprobfn,
                           **run_params)
        logging.info(f'Finished Dynesty sampling in {output["sampling"][1]:.2f}s')

        survey = os.path.basename(self.mp.catalogue_loc).split('.')[0]
        hfile = os.path.join(dp.results_dir, f'{survey}_{self.index}.h5')

        writer.write_hdf5(hfile, run_params, self.model, self.obs,
                          output["sampling"][0], output["optimization"][0],
                          tsample=output["sampling"][1],
                          toptimize=output["optimization"][1])
        logging.info(f'Saved Dynesty results to {hfile}')


    # def photometry(self, theta: Optional[np.ndarray] = self.model.theta
    #               ) -> np.ndarray:
    #     """Return the simulated photometric observations

    #     Args:
    #         theta: [TODO:description]

    #     Returns:
    #         np.ndarray: [TODO:description]
    #     """
    #     raise NotImplementedError

    def visualise_obs(self, show: bool=False, save: bool=True,
                      path: str = './results/obs.png'):
        logging.info('[bold]Visualising observations')
        if self.obs['_fake_observtion']:
            self._fake_obs_warning('visualise_obs')
        vis.visualise_obs(self.obs, show, save, path)


    def visualise_model(self, theta: Optional[tensor_like] = None,
                        no_obs: bool = False, show: bool = False,
                        save: bool = True, path: str = './results/model.png'):
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

        if not no_obs and self.obs['_fake_observtion']:
            self._fake_obs_warning('visualise_model')

        vis.visualise_model(self.model, self.sps, theta,
                            None if no_obs else self.obs, show,
                            save, path)
