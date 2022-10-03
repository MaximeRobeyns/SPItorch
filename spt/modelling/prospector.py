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
from prospect.io import read_results as reader
from prospect.models.sedmodel import SedModel
from prospect.sources import SSPBasis

import spt.visualisation as vis

from spt.types import tensor_like, FittingMethod, ConcurrencyMethod, MCMCMethod
from spt.modelling.builders import obs_dict_t
from spt.config import (
    ForwardModelParams,
    FittingParams,
    EMCEEParams,
    DynestyParams,
    InferenceParams,
)


fit_params_t = Union[EMCEEParams, DynestyParams]
prun_params_t = dict[str, Union[int, bool, float, None, list[int], str, Any]]
ffit_t = Callable[["Prospector", fit_params_t, prun_params_t], None]


def get_forward_model(
    observation: Optional[pd.Series] = None, mp=ForwardModelParams
) -> Callable[[np.ndarray], np.ndarray]:
    """Returns a (photometry-only) forward model.

    Useful for predicting the photometry for a given theta sample (e.g.
    validating ML model outputs).

    Args:
        observation: observation to initialise obs dict with. If left out, a dummy observation
            will be used.
            TODO: remove this if it has no effect on predictions.
        mp: The forward model parameters.

    Returns:
        Callable[[np.ndarray], np.ndarray]: A callable forward model.
    """
    logging.info("Initialising new forward model")
    mp = mp()
    _obs = mp.build_obs_fn(mp.filters, observation)
    _model = mp.build_model_fn(mp.all_params, mp.ordered_params)
    _sps = mp.build_sps_fn(**mp.sps_kwargs)

    def f(theta: np.ndarray) -> np.ndarray:
        _, phot, _ = _model.sed(theta, obs=_obs, sps=_sps)
        return phot

    logging.info("Forward model created")
    return f


def save_and_load(
    mm: MCMCMethod,
) -> Callable[[ffit_t], Callable[["Prospector", fit_params_t], None]]:
    """Decorator for fitting methods to first check whether the sampling has
    already been run (and load it if necessary), and if not to ensure that the
    samples are saved to disk after sampling does proceed.
    """

    def g(fit_func: ffit_t) -> Callable[["Prospector", fit_params_t], None]:
        def f(self: "Prospector", *args, **kwargs) -> None:

            if self.obs["_fake_observation"]:
                self._fake_obs_warning(f"fit_model on {mm.value}")

            if "always_fit" in kwargs:
                always_fit = kwargs["always_fit"]
            else:
                always_fit = False

            if "_survey" in self.obs:
                assert isinstance(self.obs["_survey"], str)
                hfile = self.results_fpath(self.index, mm, survey=self.obs["_survey"])
            else:
                hfile = self.results_fpath(self.index, mm)

            if not always_fit:
                if os.path.exists(hfile):
                    logging.info(f"Found results file ({hfile}) for fit: skipping.")
                    self.load_fit_results(hfile)
                    return

            run_params: prun_params_t = {}
            kwargs |= {"run_params": run_params}
            fit_func(self, *args, **kwargs)  # type: ignore

            assert self.fit_output is not None

            writer.write_hdf5(
                hfile,
                run_params,
                self.model,
                self.obs,
                self.fit_output["sampling"][0],
                self.fit_output["optimization"][0],
                tsample=self.fit_output["sampling"][1],
                toptimize=self.fit_output["optimization"][1],
            )  # , sps=self.sps)
            logging.info(f"Saved {mm.value} results to {hfile}")
            self.fit_results, _, _ = reader.results_from(hfile)

        return f

    return g


class Prospector:
    def __init__(self, observation: Optional[pd.Series] = None, mp=ForwardModelParams):
        """Construct a prospector instance for simulation and MCMC-based
        parameter inference.

        Args:
            observation: an optional observation. If None, a dummy observation will be used to
                make prospector happy (e.g. for sampling).
        """
        logging.info("Initialising prospector class")

        mp = mp()
        if observation is not None and "idx" in observation:
            self.index = int(observation.idx)
        else:
            self.index = -1

        self.obs: obs_dict_t = mp.build_obs_fn(mp.filters, observation)
        logging.info(
            (
                f"Created obs dict with filters:"
                f'{[f.name for f in self.obs["filters"]]}'
            )
        )  # type: ignore
        logging.debug(f"Created obs dict: {self.obs}")

        self.model: SedModel = mp.build_model_fn(mp.all_params, mp.ordered_params)
        logging.info(
            (
                f"Created model:\n\tfree params {self.model.free_params}"
                f"\tfixed: {self.model.fixed_params})"
            )
        )
        logging.debug(f"Created model: {self.model}")

        logging.info(f"Creating sps object...")
        self.sps: SSPBasis = mp.build_sps_fn(**mp.sps_kwargs)
        logging.info(f"Done.")
        logging.debug(f"Created sps: {self.sps}")

        self.mp = mp

        # output from fit_model. Contains 'optimization' and 'sampling' keys.
        self.fit_output = None

    def __call__(
        self,
        theta: Optional[np.ndarray] = None,
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
        c.rule("Prospector Instance")
        c.print("Model is:")
        c.print(self.model)
        assert isinstance(self.obs["filters"], list)
        c.print(f'Filters:\n{[f.name for f in self.obs["filters"]]}')
        c.rule()
        return c.end_capture()

    def set_new_obs(self, obs: pd.Series) -> None:
        """Allows us to re-use this prospector instance with a new observation"""
        self.index = int(obs.idx) if "idx" in obs else -1
        self.obs = self.mp.build_obs_fn(self.mp.filters, obs)

    def _fake_obs_warning(self, method: str = "prospector method"):
        logging.warning(
            (
                f"Calling {method} with a fake observation.\n"
                "Please construct the Prospector object with a real observation "
                "instead."
            )
        )

    def set_theta(self, theta: np.ndarray):
        """Convenience method to set model parameters."""
        self.model.set_parameters(theta)

    def results_fpath(
        self,
        index: int = None,
        method: MCMCMethod = None,
        p: fit_params_t = None,
        survey: str = None,
    ) -> str:
        if p is None:
            if method == MCMCMethod.EMCEE:
                p = EMCEEParams()
            else:
                p = DynestyParams()
        if survey is not None:
            return os.path.join(p.results_dir, f"{survey}_{index}.h5")
        else:
            return os.path.join(p.results_dir, f"{index}.h5")

    def numerical_fit(self, fp: FittingParams = FittingParams()) -> list[Any]:
        """'Burn in' for layer MCMC sampling using a numerical method.

        This initialisation could also be done by a machine learning algorithm.
        """
        logging.info(f"Running {fp.min_method.value} fitting")
        if fp.min_method == FittingMethod.ML:
            raise NotImplementedError(
                "MCMC initialisation with ML results not yet implemented."
            )

        run_params: prun_params_t = {"dynesty": False, "emcee": False, "optimize": True}
        run_params["min_method"] = fp.min_method.value
        run_params["nmin"] = fp.nmin

        self.fit_output = fit_model(
            self.obs, self.model, self.sps, lnprobfn=lnprobfn, **run_params
        )
        assert self.fit_output is not None
        (results, time) = self.fit_output["optimization"]
        logging.info(f"Fitting took {time:.2f}s")
        assert results is not None
        return results

    @save_and_load(MCMCMethod.EMCEE)
    def emcee_fit(
        self,
        ep: fit_params_t = EMCEEParams(),
        run_params: prun_params_t = {},
        always_fit: bool = False,
    ) -> None:
        """Runs MCMC method to update the value of self.model.theta"""
        assert isinstance(ep, EMCEEParams)
        logging.info(f"Running EMCEE fitting with parameters:\n{ep}")

        if ep.min_method == FittingMethod.ML:
            raise NotImplementedError(
                "MCMC initialisation with ML results not yet implemented."
            )
        run_params |= ep.to_dict() | {"dynesty": False, "emcee": True}

        if ep.pool == ConcurrencyMethod.MPI:
            raise NotImplementedError("MPI parallelism for EMCEE not implemented")
            # from schwimmbad.mpi import MPIPool
            # run_params['pool'] = MPIPool()
        elif ep.pool == ConcurrencyMethod.native:
            from multiprocessing import Pool

            run_params["pool"] = Pool(ep.workers)
        elif ep.pool == ConcurrencyMethod.none:
            run_params["pool"] = None

        self.fit_output = fit_model(
            self.obs, self.model, self.sps, lnprobfn=lnprobfn, **run_params
        )
        assert self.fit_output is not None
        logging.info(
            f'Finished EMCEE sampling in {self.fit_output["sampling"][1]:.2f}s'
        )

    @save_and_load(MCMCMethod.Dynesty)
    def dynesty_fit(
        self,
        dp: fit_params_t = DynestyParams(),
        run_params: prun_params_t = {},
        always_fit: bool = False,
    ) -> None:
        """Runs Dynesty (nested) sampling to update the value of self.model.theta"""
        assert isinstance(dp, DynestyParams)
        logging.info("Running Dynesty fitting with parameters:\n{dp}")

        if dp.min_method == FittingMethod.ML:
            raise NotImplementedError(
                "MCMC initialisation with ML results not yet implemented."
            )

        run_params |= dp.to_dict() | {"dynesty": True, "emcee": False}

        self.fit_output = fit_model(
            self.obs, self.model, self.sps, lnprobfn=lnprobfn, **run_params
        )
        assert self.fit_output is not None
        logging.info(
            f'Finished Dynesty sampling in {self.fit_output["sampling"][1]:.2f}s'
        )

    def load_fit_results(
        self,
        file: str = None,
        index: int = None,
        method: MCMCMethod = None,
        survey: str = None,
    ) -> None:
        """Attempt to load the results of fitting; either by providing a path
        to the hdf5 results file directly, or by specifying both the index and
        fitting method.

        Args:
            file: file path to a hdf5 file of results.
            index: index of galaxy to try to load
            method: MCMC method of results to find.

        # TODO refactor to take [method, obs_dict] as argument to infer file name.

        Implicit Returns:
            Sets the self.fit_output property with the results.
        """
        if file is None:
            if index is None or method is None:
                raise ValueError("Please specify both the index and fitting method")
            file = self.results_fpath(index, method, survey=survey)
        if not os.path.exists(file):
            logging.error(f"File {file} cannot be found.")
            raise ValueError(f"Bad file path {file}")
        self.fit_results, self.obs, tmp_model = reader.results_from(file)
        if tmp_model is not None:
            self.model = tmp_model
        # self.sps = reader.get_sps(self.fit_output)
        logging.info("Loaded fitting results.")

    # def photometry(self, theta: Optional[np.ndarray] = self.model.theta
    #               ) -> np.ndarray:
    #     """Return the simulated photometric observations
    #     """
    #     raise NotImplementedError

    def visualise_obs(
        self,
        show: bool = False,
        save: bool = True,
        path: str = "./results/obs.png",
        title: str = None,
    ):
        logging.info("[bold]Visualising observations")
        if self.obs["_fake_observation"]:
            self._fake_obs_warning("visualise_obs")
        vis.visualise_obs(self.obs, show, save, path, title)

    def visualise_model(
        self,
        theta: Optional[list[tensor_like]] = None,
        theta_labels: list[str] = [],
        no_obs: bool = False,
        show: bool = False,
        save: bool = True,
        path: str = "./results/model.png",
        title: str = None,
    ):
        """Visualise predicted photometry from a theta vector.

        Args:
            theta: An optionally speciifed list of parameter vectors. If
                omitted, the model's current parameter vector is used.
            no_obs: Whether to omit the photometric observations. This is
                useful when using a dummy 'obs' dictionary when using this
                Prospector class in 'forward-model' mode.
            show: show the plot?
            save: save the plot?
            path: where to save the plot.
            title: An optional title (defaults to Modelled Photometry)
        """
        logging.info("[bold]Visualising model predictions")

        if not no_obs and self.obs["_fake_observation"]:
            self._fake_obs_warning("visualise_model")

        # for backward compatability: calling this function without a list of theta
        if theta is not None and not isinstance(theta, list):
            theta = [theta]

        vis.visualise_model(
            self.model,
            self.sps,
            theta,
            None if no_obs else self.obs,
            show,
            save,
            theta_labels,
            path,
            title,
        )
