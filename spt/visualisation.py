# SPItorch: Inference of Stellar Population Properties in PyTorch
#
# Copyright (C) 2022 Maxime Robeyns <dev@maximerobeyns.com>
# Copyright (C) 2019-20 Mike Walmsley <walmsleymk1@gmail.com>
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
"""Project-wide utilities file"""

import numpy as np
import logging
import matplotlib.pyplot as plt

from typing import Optional
from prospect.models.sedmodel import SedModel
from prospect.sources.ssp_basis import SSPBasis

from spt.types import tensor_like
from spt.modelling.builders import obs_dict_t

colours = {
    "b": "#025159",  # blue(ish)
    "o": "#F28705",  # orange
    "lb": "#03A696", # light blue
    "do": "#F25D27", # dark orange
    "r": "#F20505"   # red.
}


# Utility functions -----------------------------------------------------------


def _plot_obs_photometry(obs: obs_dict_t):
    """Plots observations, in observer frame"""

    wphot = obs["phot_wave"]
    assert isinstance(wphot, np.ndarray)
    assert isinstance(obs["maggies"], np.ndarray)
    assert isinstance(obs["maggies_unc"], np.ndarray)

    # Plot all the data
    plt.plot(wphot, obs['maggies'], label='All observed photometry',
             marker='o', markersize=12, alpha=0.8, ls='', lw=3,
             color=colours['b'])

    # overplot only the data we intend to fit
    mask = obs["phot_mask"]
    assert isinstance(mask, np.ndarray)
    plt.errorbar(wphot[mask], obs['maggies'][mask],
                 yerr=obs['maggies_unc'][mask],
                 label='Photometry to fit',
                 marker='o', markersize=8, alpha=0.8, ls='', lw=3,
                 ecolor=colours['r'], markerfacecolor='none',
                 markeredgecolor=colours['r'], markeredgewidth=3)


def _plot_filters(obs: obs_dict_t, ymin: float, ymax: float):
    assert isinstance(obs['filters'], list)
    for f in obs['filters']:
        w, t = f.wavelength.copy(), f.transmission.copy()
        t = t / t.max()
        t = 10**(0.2*(np.log10(ymax/ymin)))*t * ymin
        plt.loglog(w, t, lw=3, color=colours['b'], alpha=0.7)


def _get_observer_frame_wavelengths(model: SedModel, sps: SSPBasis
                                    ) -> np.ndarray:
    # initial_phot is y-values (maggies) as observed at obs['phot_wave']
    # wavelengths, in observer frame?

    # cosmological redshifting w_new = w_old * (1+z)
    a = 1.0 + model.params.get('zred', 0.0)

    # redshift the *restframe* sps spectral wavelengths:
    # wavelengths of (source frame) fluxes
    source_wavelengths = sps.wavelengths
    # redshift them via w_observed = w_source * (1+z), using z of model
    observer_wavelengths = source_wavelengths * a
    # wspec is now *observer frame* wavelengths of source fluxes
    return observer_wavelengths


def _get_bounds(obs: obs_dict_t, wspec: Optional[np.ndarray] = None,
               initial_spec: Optional[np.ndarray] = None
               ) -> tuple[tuple[float, float], tuple[float, float]]:
    """Gets appropriate bounds on the figure (both x and y).

    Args:
        obs: observation dictionary from prospector class
        wspec: monotonically increasing sequence of x data points
        initial_spec: monotonically increasing sequence of y points

    Returns:
        tuple[tuple[float, float], tuple[float, float]]:
            (xmin, xmax), (ymin, ymax)

    Raises:
        ValueError: If `initial_spec` is not set while `wspec` is.
    """

    wphot = obs['phot_wave']
    assert isinstance(wphot, np.ndarray)
    xmin, xmax = np.min(wphot)*0.8, np.max(wphot)/0.8

    if wspec is not None:  # interpolate sed to calculate y bounds
        if initial_spec is None:
            logging.error(f'initial_spec cannot be None if wspec is defined')
            raise ValueError('initial_spec not set')
        # evaluate wspec (x) vs. initial spec (y), along new x grid
        tmp = np.interp(np.linspace(xmin, xmax, 10000), wspec, initial_spec)
        ymin, ymax = tmp.min()*0.8, tmp.max()/0.4
    else:
        assert isinstance(obs['maggies'], np.ndarray)
        ymin, ymax = obs['maggies'].min()*0.8, obs['maggies'].max()/0.4
        # ymin, ymax = obs['maggies'].min()*0.4, obs['maggies'].max()/0.4
    return (xmin, xmax), (ymin, ymax)


def _style_plot(fig: plt.Figure, ax: plt.Axes, xmin: float, xmax: float,
                ymin: float, ymax: float):
    ax.set_xlabel('Wavelength [A]', size=20)
    ax.set_ylabel('Flux Density [Maggies]', size=20)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.legend(loc='best', fontsize=15)
    fig.patch.set_facecolor('white')

    fig.tight_layout()


def _describe_model(model: SedModel):
    return ', '.join(["{}={}".format(p, model.params[p][0])
                    for p in model.free_params])

# -----------------------------------------------------------------------------


def visualise_obs(obs: obs_dict_t, show: bool = True, save: bool = False,
                  path: str = None):
    """Visualise a loaded observation dictionary.

    Args:
        obs: The obs dictionary
        show: Whether to show the plot
        save: Whether to save the plot
        path: Filepath to save the plot at (default './results/obs.png')
    """
    logging.info(f'Visualising observations')
    xbounds, ybounds = _get_bounds(obs)
    fig, ax = plt.subplots(figsize=(16, 8), dpi=300)
    _plot_obs_photometry(obs)
    _plot_filters(obs, *ybounds)

    fig.text(0.06, 1.035, s="Observed Photometry", fontfamily='sans-serif',
             fontweight='demibold', fontsize=25)
    fig.text(0.06, 1.007, s=f'Filters: {str(obs["filternames"])}',
             fontfamily='sans-serif', fontweight='normal', fontsize=10,
             alpha=0.7)
    _style_plot(fig, ax, *xbounds, *ybounds)

    if save:
        if path is None:
            path = './results/obs.png'
        plt.savefig(path, bbox_inches='tight')
        logging.info(f'Saved observations plot to {path}')
    if show:
        plt.show()
    plt.close()


def visualise_model(model: SedModel, sps: SSPBasis,
                    theta: Optional[tensor_like] = None,
                    obs: Optional[obs_dict_t] = None,
                    show: bool = True, save: bool = False,
                    path: str = None):
    """Visualise output from SedPy model at optionally specified parameters.

    Args:
        model: The SedPy model
        sps: SPS object
        theta: Optional parameter dictionary: if omitted, current model theta
            is used.
        obs: Observation dictionary (note; this may be a 'dummy' dictionary)
        show: Whether to show the plot
        save: Whether to save the plot
        path: Filepath to save the plot at (default './results/model.png')
    """
    fig, ax = plt.subplots(figsize=(16, 8), dpi=300)
    omit_obs = False

    lg = 'Visualising model predictions'
    if obs is None:
        from spt.config import ForwardModelParams
        mp = ForwardModelParams()
        obs = mp.build_obs_fn(mp.filters, None)
        lg += ' without observations'
        omit_obs = True
    else:
        _plot_obs_photometry(obs)

    if theta is None:
        theta = model.theta.copy()
        lg += ' from [bold]model.theta[/bold]'
    else:
        lg += ' from custom theta.'

    logging.info(lg)

    spec, phot, _ = model.sed(theta, obs, sps)
    wphot = obs['phot_wave']
    wspec = _get_observer_frame_wavelengths(model, sps)
    xbounds, ybounds = _get_bounds(obs, wspec, spec)

    ax.loglog(wspec, spec, label='Model spectrum', lw=0.7, color=colours['lb'],
              alpha=0.7)
    ax.errorbar(wphot, phot, label='Model photometry', marker='s',
                markersize=10, alpha=0.8, ls='', lw=3, markerfacecolor='none',
                markeredgecolor=colours['b'], markeredgewidth=3)
    if not omit_obs:
        ax.errorbar(wphot, obs['maggies'], yerr=obs['maggies_unc'],
                    label='Observed photometry', marker='o', markersize=10,
                    alpha=0.8, ls='', lw=3, ecolor=colours['r'],
                    markerfacecolor='none', markeredgecolor=colours['r'],
                    markeredgewidth=3)

    fig.text(0.07, 1.035, s="Modelled Photometry", fontfamily='sans-serif',
             fontweight='demibold', fontsize=25)
    fig.text(0.07, 1.007, s=f'{_describe_model(model)}',
             fontfamily='sans-serif', fontweight='normal', fontsize=10,
             alpha=0.7)
    _plot_filters(obs, *ybounds)
    _style_plot(fig, ax, *xbounds, *ybounds)

    if save:
        if path is None:
            path = './results/model.png'
        plt.savefig(path, bbox_inches='tight')
        logging.info(f'Saved model plot to {path}')
    if show:
        plt.show()
    plt.close()
