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
"""Loads filters and other data."""


import os
import sys
import h5py
import random
import logging
import torch as t
import numpy as np
import pandas as pd

from sedpy import observate
from typing import Any, Callable, Optional, Union
from astropy.io import fits
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms

from spt.filters import Filter
from spt.types import Tensor, tensor_like
from spt.config import ForwardModelParams


__all__ = ["load_observation", "load_catalogue", "ObsDataset", "load_simulated_data"]

# "get_simulated_observation", "GalaxyDataset", "load_real_data"


def mags_to_maggies(mags: tensor_like) -> tensor_like:
    # mags should be apparent AB magnitudes
    # The units of the fluxes need to be maggies (Jy/3631)
    return 10**(-0.4*mags)


def calculate_maggie_uncertainty(mag_error: tensor_like, maggie: tensor_like,
                                 ) -> tensor_like:
    # http://slittlefair.staff.shef.ac.uk/teaching/phy217/lectures/stats/L18/index.html#magnitudes
    return maggie * mag_error / 1.09


def add_maggies_cols(input_df: Union[pd.DataFrame, pd.Series],
                     filters: list[Filter]) -> Union[pd.DataFrame, pd.Series]:
    """Add maggies column to calalogue of real galaxies.

    Args:
        input_df: either full dataframe observation catalogue, or single row (pd.Series)
        fset: the FilterSet used to make the observations.
    """
    df = input_df.copy()  # we don't modify the df inplace
    logging.info(f'Adding maggies cols for {filters}')
    for f in filters:
        if isinstance(df, pd.DataFrame):
            df[f.maggie_col] = df[f.mag_col].apply(mags_to_maggies)
        else:
            df[f.maggie_col] = mags_to_maggies(input_df[f.mag_col])

        if isinstance(df, pd.DataFrame):
            mec = df[[f.mag_error_col, f.maggie_col]]
            df[f.maggie_error_col] = mec.apply(
                    lambda x: calculate_maggie_uncertainty(*x), axis=1)
        else:
            mec = [input_df[f.mag_error_col], df[f.maggie_col]]
            df[f.maggie_error_col] = calculate_maggie_uncertainty(*mec)
    logging.info('Finished adding maggies cols.')
    return df


def load_catalogue(catalogue_loc: str, filters: list[Filter],
                   compute_maggies_cols: bool = False) -> pd.DataFrame:
    """Load a catalogue of photometric observations.

    Regrettably, this function is a little brittle in the sense that we expect
    certain catalogues with certain columns. If this function fails on a new
    catalogue, then alter the *_required_cols (perhaps provide them as an
    argument to this function).

    Args:
        catalogue_loc: file path to the catalogue on disk
        filters: filters used
        compute_maggies_cols: whether to compute maggie_* columns from mag_*
            columns.

    Returns:
        pd.DataFrame: the loaded catalogue.
    """
    logging.info(f'Using {catalogue_loc} as catalogue')

    maggie_required_cols = [f.maggie_col for f in filters] + \
                           [f.maggie_error_col for f in filters] + \
                           ['redshift']
    mag_required_cols = [f.mag_col for f in filters] + \
                        [f.mag_error_col for f in filters] + \
                        ['redshift']

    if catalogue_loc.endswith('.fits'):
        with fits.open(catalogue_loc) as f:
            df = pd.DataFrame(f[1].data)
            if compute_maggies_cols:
                df = add_maggies_cols(df, filters)
                return df[maggie_required_cols]
            else:
                return df[mag_required_cols]
    elif catalogue_loc.endswith('.csv'):
        df = pd.read_csv(catalogue_loc, usecols=mag_required_cols)
        assert isinstance(df, pd.DataFrame)
        if compute_maggies_cols:
            df = add_maggies_cols(df, filters)
    elif catalogue_loc.endswith('.parquet'):
        df = pd.read_parquet(catalogue_loc)
        if compute_maggies_cols:
            df = add_maggies_cols(df, filters)
    else:
        logging.exception((
            'Unhandled catalogue type.'
            'Please implement your own load_catalogue function.'))
        raise ValueError("Unhandled catalogue file type.")

    assert df is not None
    assert isinstance(df, pd.DataFrame)

    if compute_maggies_cols:
        df = df.dropna(subset=maggie_required_cols)
    else:
        df = df.dropna(subset=mag_required_cols)

    assert isinstance(df, pd.DataFrame)
    df_with_spectral_z = df[
        ~pd.isnull(df['redshift'])
    ].query('redshift > 1e-2').query('redshift < 4').reset_index()
    return df_with_spectral_z


def load_observation(index: Optional[int] = None,
                     catalogue_loc: Optional[str] = None,
                     filters: Optional[list[Filter]] = None,
                     ) -> pd.Series:
    """Load an observation from a catalogue of real-world observations.

    Args:
        index: the optional index of the observation to return. If omitted, index is
            random. Must be within range.
        catalogue_loc: the filepath to the .fits, .csv or .parquet file. By
            default the catalogue configured in the InferenceParams will be
            used.
        filters: the list of filters used in the survey. By default the filter
            list in the ForwardModelParams will be used.

    Returns:
        pd.Series: the observation
    """
    if catalogue_loc is None or filters is None:
        from spt.config import ForwardModelParams, InferenceParams
        catalogue_loc = InferenceParams().catalogue_loc
        filters = ForwardModelParams().filters

    df = load_catalogue(catalogue_loc, filters=filters, compute_maggies_cols=False)

    assert df is not None
    df.reset_index(drop=True)
    if index is None:
        index = random.randint(0, len(df))
        logging.info(f'No index specified: using random index {index}')

    df_series = add_maggies_cols(df.iloc[index], filters)
    df_series['idx'] = index
    assert isinstance(df_series, pd.Series)
    return df_series


def sim_observation(filters: list[Filter], phot: np.ndarray,
                    phot_unc: Optional[np.ndarray] = None) -> pd.Series:
    """Build an observation from (e.g. a simulated) observation supplied
    directly as numpy array.

    Args:
        filters: The SPS filter list to use
        phot: photometric observations (maggies) to use.
        phot_unc: (optional) observation uncertainty, if you have it. Otherwise
            it will be faked.

    Returns:
        pd.Series: the observation
    """

    m = pd.Series(phot, [f.maggie_col for f in filters])
    pu = calculate_maggie_uncertainty(phot/100, phot) if phot_unc is None else phot_unc
    mu = pd.Series(pu, [f.maggie_error_col for f in filters])
    return m.combine_first(mu)


# TODO port get_simulated_observation
# TODO port observationDataset
# TODO port load_real_data


def filter_has_valid_data(f: Filter, observation: pd.Series) -> bool:
    """Ensures that observation data series has maggie cols"""
    filter_value = observation[f.maggie_col]
    assert isinstance(filter_value, np.floating) or isinstance(filter_value, float)
    valid_value = not pd.isnull(filter_value) \
                  and filter_value > -98 \
                  and filter_value < 98
    filter_error = observation[f.maggie_error_col]
    assert isinstance(filter_error, np.floating) or isinstance(filter_error, float)
    valid_error = not pd.isnull(filter_error) \
                  and filter_error > 0  # <0 if -99 (unknown) or -1 (only upper bound)
    return bool(valid_value and valid_error)


def load_maggies_to_array(observation: pd.Series, filters: list[Filter]
                         ) -> tuple[np.ndarray, np.ndarray]:
    maggies = np.array([observation[f.maggie_col] for f in filters])
    maggies_unc = np.array([observation[f.maggie_error_col] for f in filters])
    return maggies, maggies_unc


def load_observation_for_prospector(
        observation: pd.Series, filter_sel: list[Filter]
    ) -> tuple[list[observate.Filter], np.ndarray, np.ndarray]:
    valid_filters = [f for f in filter_sel if filter_has_valid_data(f, observation)]
    maggies, maggies_unc = load_maggies_to_array(observation, valid_filters)
    filters = observate.load_filters([f.bandpass_file for f in valid_filters])
    return filters, maggies, maggies_unc


def load_dummy_observation(filter_sel: list[Filter]
        ) -> tuple[list[observate.Filter], np.ndarray, np.ndarray]:
    """Loads a dummy observation for prospector. This is useful for running forward
    models to simulate theta -> photometry mappings.

    Args:
        filter_sel: The selected list of filters.

    Returns:
        tuple[list[observate.Filter], np.ndarray, np.ndarray]: A list of the
            loaded sedpy filters, the dummy flux (ones; maggies) and dummy
            uncertainty (ones)
    """
    loaded_filters = observate.load_filters(
            [f.bandpass_file for f in filter_sel])
    maggies = np.ones(len(loaded_filters))
    maggies_unc = np.ones(len(loaded_filters))
    return loaded_filters, maggies, maggies_unc


# Transformations =============================================================


def denormalise_theta(norm_theta: np.ndarray, limits: np.ndarray) -> np.ndarray:
    """Rescale norm_theta lying within [0, 1] range; not altering the
    distribution of points.
    """
    assert norm_theta.shape[1] == limits.shape[0]
    return limits[:,0] + (limits[:,1] - limits[:,0]) * norm_theta # type: ignore


def denormalise_unif_theta(norm_theta: np.ndarray, limits: np.ndarray,
                           log_mask: list[bool]) -> np.ndarray:
    """Denormalise photometry (lying in the [0-1] range) to lie within the min
    and max limits (either explicitly set in ForwardModelParams.model_params,
    or inerred from a prior; perhaps in a template): additionally accounting
    for logarithmic parameters.

    Args:
        norm_theta: matrix of [0, 1] normalised theta values: 1 theta sample
            per row
        limits: 2D array of min-max limits, given in the standard theta order.
        log_mask: a boolean list with True where we have a logarithmic
            parameter.

    Returns:
        denormalised theta values.
    """
    # assert columns of theta matrix == number of limits
    assert norm_theta.shape[1] == limits.shape[0]
    limits[log_mask] = np.log10(limits[log_mask, :]) # type: ignore
    dtheta = limits[:,0] + (limits[:,1] - limits[:,0]) * norm_theta # type: ignore

    return np.where(np.array(log_mask), 10**np.clip(dtheta, -10, 20), dtheta)


def normalise_theta(theta: np.ndarray, limits: np.ndarray) -> np.ndarray:
    """Rescale theta values to lie within [0, 1] range; not altering the
    distribution of points within this range."""
    assert theta.shape[1] == limits.shape[0]
    offset = theta - limits[:, 0] # type: ignore
    return offset / limits[:, 1] - limits[:, 0]


def get_norm_theta(fp: ForwardModelParams) -> Callable[[np.ndarray], np.ndarray]:
    lims = np.array(fp.free_param_lims())
    islog = np.array(fp.is_log())

    def f(y: np.ndarray) -> np.ndarray:
        yy = normalise_theta(y, lims)
        # TODO: this doesn't feel great... Look into the impact of clipping
        # (esp min.)
        min = np.finfo(yy.dtype).eps
        max = np.finfo(yy.dtype).max
        return np.where(islog, np.log10(np.clip(yy, min, max)), yy)

    return f

def get_denorm_theta(fp: ForwardModelParams) -> Callable[[np.ndarray], np.ndarray]:
    lims = np.array(fp.free_param_lims())
    islog = np.array(fp.is_log())

    def f(yy: np.ndarray) -> np.ndarray:
        y = denormalise_theta(yy, lims)
        return np.where(islog, 10**(y), y)

    return f


# We would never use this type of function...
# def normalise_unif_theta(theta: np.ndarray, limits: np.ndarray,
#                          log_mask: list[bool]) -> np.ndarray:
#     assert theta.shape[1] == limits.shape[0]
#     limits[log_mask] = np.log10(limits[log_mask, :])
#     theta[log_mask] = np.log10()
#     offset = theta - limits[:, 0]
#     return offset / limits[:, 1] - limits[:, 0]


# HDF5 file IO ----------------------------------------------------------------


def save_sim(path: str, theta: np.ndarray, cols: list[str], phot: np.ndarray,
             pw: np.ndarray):
    """Saves simulated data points to hdf5 file.

    Args:
        path: File path
        theta: Array of physical parameters used in forward model
        cols: list of free parameter names
        phot: Simulated photometric observations (flux)
        obs: Effective wavelengths for each of the filters
    """

    with h5py.File(path, 'w') as f:
        grp = f.create_group('samples')

        ds_x = grp.create_dataset('theta', data=theta, maxshape=(None,theta.shape[1]), chunks=True)
        ds_x.attrs['columns'] = cols
        ds_x.attrs['description'] = 'Parameters used by simulator'

        ds_y = grp.create_dataset('simulated_y', data=phot, maxshape=(None,phot.shape[1]), chunks=True)
        ds_y.attrs['description'] = 'Response of simulator'

        # Wavelengths at for each of the simulated_y
        ds_wl = grp.create_dataset('wavelengths', data=pw)
        ds_wl.attrs['description'] = 'Effective wavelengths for each of the filters'


def _must_get_grp(f: h5py.File, key: str) -> h5py.Group:
    g = f.get(key)
    assert g is not None and isinstance(g, h5py.Group)
    return g


def _must_get_dset(g: h5py.Group, key: str) -> h5py.Dataset:
    d = g.get(key)
    assert d is not None and isinstance(d, h5py.Dataset)
    return d


def join_partial_results(save_dir: str, n_samples: int, concurrency: int) -> None:
    base = os.path.join(save_dir, f'photometry_sim_{n_samples//concurrency}_0.h5')
    allf = os.path.join(save_dir, f'photometry_sim_{n_samples}.h5')
    with h5py.File(base, 'a') as f:
        fgrp = _must_get_grp(f, 'samples')
        ds_theta = _must_get_dset(fgrp, 'theta')
        ds_sim_y = _must_get_dset(fgrp, 'simulated_y')

        for i in range(1, concurrency):
            tmp_f = os.path.join(save_dir, f'photometry_sim_{n_samples//concurrency}_{i}.h5')
            with h5py.File(tmp_f, 'r') as rf:
                tmp_grp = _must_get_grp(rf, 'samples')
                tmp_theta = _must_get_dset(tmp_grp, 'theta')
                tmp_sim_y = _must_get_dset(tmp_grp, 'simulated_y')

                ds_theta.resize((ds_theta.shape[0] + tmp_theta.shape[0]), axis=0)
                ds_sim_y.resize((ds_sim_y.shape[0] + tmp_sim_y.shape[0]), axis=0)

                ds_theta[-tmp_theta.shape[0]:] = tmp_theta
                ds_sim_y[-tmp_sim_y.shape[0]:] = tmp_sim_y


    # Rename file with accumulated results, and remove all others.
    os.rename(base, allf)

    # Delete other partial files /after/ renaming, so that if something
    # fails above, we don't lose data!
    for i in range(1, concurrency):
        os.remove(os.path.join(save_dir, f'photometry_sim_{n_samples//concurrency}_{i}.h5'))


class InMemoryObsDataset(Dataset):
    """Loads one or more hdf5 files containing simulated (theta, photometry)
    pairs into system memory simultaneously.

    For larger datasets this may well easily exceed system memorty: if this is
    the case, then implement GalaxyDataset (see below) to incrementally load
    data into memory from disk. Since we have not yet come against this issue,
    we leave this to future work.
    """

    def __init__(self, path: str, phot_transforms: list[Callable[[Any], Any]],
                 theta_transforms: list[Callable[[Any], Any]]):
        """Loads the simulated observation dataset.

        Note: see the load_simulated_data convenience method if you need
        PyTorch data loaders.

        Args:
            path: either the path of a hdf5 file or directory containing hdf5
                files, with (theta, photometry) pairs, output from a simulation
                run.
            phot_transforms: transformations to apply to the (un-normalised)
                photometry (model inputs)
            theta_transforms: transformations to apply to the physical
                parameters (model outputs)
        """
        self.phot_transforms = phot_transforms
        self.theta_transforms = theta_transforms
        files = []

        if os.path.isdir(path):
            for file in os.listdir(path):
                if file.endswith(".h5"):
                    files.append(os.path.join(path, file))
            logging.info(f'Loading {len(files)} file(s) into dataset')
        elif os.path.isfile(path):
            files = [path]
            logging.info(f'Loading 1 file into dataset.')
        else:
            logging.error((f'Provided path ({path}) is neither a directory or '
                           f'path to hdf5 file.'))
            sys.exit()

        xs, ys = self._get_x_y_from_file(files[0])

        for f in files[1:]:
            tmp_xs, tmp_ys = self._get_x_y_from_file(f)
            xs = np.concatenate((xs, tmp_xs), 0)
            ys = np.concatenate((ys, tmp_ys), 0)

        self._x_dim, self._y_dim = xs.shape[-1], ys.shape[-1]
        self.dataset = np.concatenate((xs, ys), -1)


        logging.info(f'Galaxy dataset loaded.')

    def get_xs(self) -> Any:
        """Just return all the xs (photometric measurements) in the dataset

        Returns type Any (not Tensor, or np.ndarray) because the
        transformations could be arbitrary.
        """
        xs = self.dataset[:, :self._x_dim]
        for tr in self.phot_transforms:
            xs = tr(xs)
        return xs.squeeze()

    def get_ys(self) -> Any:
        """Return all the y values (physical parameters) in the dataset

        Returns type Any (not Tensor, or np.ndarray) because the
        transformations could be arbitrary.
        """
        ys = self.dataset[:, self._x_dim:]
        for tr in self.theta_transforms:
            ys = tr(ys)
        return ys.squeeze()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: Union[int, list[int], Tensor]) -> tuple[tensor_like, tensor_like]:

        if isinstance(idx, Tensor):
            idx = idx.to(dtype=t.int).tolist()

        data = self.dataset[idx]
        if data.ndim == 1:
            data = np.expand_dims(data, 0)
        xs, ys = data[:, :self._x_dim], data[:, self._x_dim:]

        # eagerly compute transformations (rather than during __getitem__,
        # since this allows us to use vector operations and is more efficient)
        for xtr in self.phot_transforms:
            xs = xtr(xs)
        for ytr in self.theta_transforms:
            ys = ytr(ys)

        # both np.ndarray and torch.Tensor implement `squeeze`
        return (xs.squeeze(), ys.squeeze())

    def _get_x_y_from_file(self, file: str) -> tuple[np.ndarray, np.ndarray]:
        assert os.path.exists(file)
        with h5py.File(file, 'r') as f:
            samples = _must_get_grp(f, 'samples')

            # Simulated photometry (model inputs)
            xs = np.array(_must_get_dset(samples, 'simulated_y'))

            # physical parameters (model outputs)
            ys = np.array(_must_get_dset(samples, 'theta'))

            # Ensure that the ys are in the same order as the free model
            # parameters
            ofp = ForwardModelParams().ordered_free_params
            ncols = _must_get_dset(samples, 'theta').attrs['columns']
            assert isinstance(ncols, np.ndarray)
            cols: list[str] = ncols.tolist()
            permlist = [cols.index(cn) for cn in ofp] # type: ignore
            # can be expensive for big dsets; skip if possible...
            if permlist != list(range(len(ofp))):
                ys = ys[:, np.ndarray(permlist)]

            return xs, ys


# For backwards compatability
ObsDataset = InMemoryObsDataset


def load_simulated_data(
        path: str, split_ratio: float = 0.8, batch_size: int = 1024,
        test_batch_size: int = None,
        phot_transforms: list[Callable[[Any], Any]] = [],
        theta_transforms: list[Callable[[Any], Any]] = [],
        split_seed: int = None
        ) -> tuple[DataLoader, DataLoader]:
    """Load simulated (theta, photometry) data as train and test loaders.

    Args:
        path: file path to the hdf5 file containing the simulated data.
        split_ratio: train / test split ratio
        batch_size: training batch size
        test_batch_size: optionally specify a different batch size for the
            training data loader (defaults to batch_size if omitted)
        phot_transforms: any transformations to apply to the inputs
            (photometry, maggies)
        theta_transforms: any transfoms to apply to theta (physical parameters
            to estiamte)
        split_seed: optional random seed to initialise PRNG for train/test split

    Returns:
        tuple[DataLoader, DataLoader]: Train and test DataLoaders, respectively.
    """
    tbatch_size = test_batch_size if test_batch_size is not None else batch_size

    cuda_kwargs = {'num_workers': 8, 'pin_memory': True}
    train_kwargs: dict[str, Any] = {
        'batch_size': batch_size, 'shuffle': False} | cuda_kwargs
    test_kwargs: dict[str, Any] = {
        'batch_size': tbatch_size, 'shuffle': False} | cuda_kwargs

    dataset = InMemoryObsDataset(path, phot_transforms, theta_transforms)

    n_train = int(len(dataset) * split_ratio)
    n_test = len(dataset) - n_train

    rng = t.Generator().manual_seed(split_seed) if split_seed is not None else None
    train_set, test_set = random_split(dataset, [n_train, n_test], rng)

    train_loader = DataLoader(train_set, **train_kwargs)
    test_loader = DataLoader(test_set, **test_kwargs)

    return train_loader, test_loader
