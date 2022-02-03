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


import random
import logging
import numpy as np
import pandas as pd

from sedpy import observate
from typing import Optional, Union
from astropy.io import fits

from spt.filters import Filter
from spt.types import tensor_like


__all__ = ["load_galaxy", "load_catalogue"] # "get_simulated_galaxy", "GalaxyDataset", "load_real_data"


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
        input_df: either full dataframe galaxy catalogue, or single row (pd.Series)
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


def load_galaxy(index: Optional[int] = None,
                catalogue_loc: Optional[str] = None,
                filters: Optional[list[Filter]] = None,
                ) -> tuple[pd.Series, int]:
    """Load a galaxy from a catalogue of real-world observations.

    Args:
        index: the optional index of the galaxy to return. If omitted, index is
            random. Must be within range.
        catalogue_loc: the filepath to the .fits, .csv or .parquet file. By
            default the catalogue configured in the InferenceParams will be
            used.
        filters: the list of filters used in the survey. By default the filter
            list in the ForwardModelParams will be used.

    Returns:
        tuple[pd.Series, int]: the galaxy's photometry, and catalogue index used
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
    assert isinstance(df_series, pd.Series)
    return df_series, index


# TODO port get_simulated_galaxy
# TODO port GalaxyDataset
# TODO port load_real_data


def filter_has_valid_data(f: Filter, galaxy: pd.Series) -> bool:
    """Ensures that galaxy data series has maggie cols"""
    filter_value = galaxy[f.maggie_col]
    assert isinstance(filter_value, np.floating) or isinstance(filter_value, float)
    valid_value = not pd.isnull(filter_value) \
                  and filter_value > -98 \
                  and filter_value < 98
    filter_error = galaxy[f.maggie_error_col]
    assert isinstance(filter_error, np.floating) or isinstance(filter_error, float)
    valid_error = not pd.isnull(filter_error) \
                  and filter_error > 0  # <0 if -99 (unknown) or -1 (only upper bound)
    return bool(valid_value and valid_error)


def load_maggies_to_array(galaxy: pd.Series, filters: list[Filter]
                         ) -> tuple[np.ndarray, np.ndarray]:
    maggies = np.array([galaxy[f.maggie_col] for f in filters])
    maggies_unc = np.array([galaxy[f.maggie_error_col] for f in filters])
    return maggies, maggies_unc


def load_galaxy_for_prospector(
        galaxy: pd.Series, filter_sel: list[Filter]
    ) -> tuple[list[observate.Filter], np.ndarray, np.ndarray]:
    valid_filters = [f for f in filter_sel if filter_has_valid_data(f, galaxy)]
    maggies, maggies_unc = load_maggies_to_array(galaxy, valid_filters)
    filters = observate.load_filters([f.bandpass_file for f in valid_filters])
    return filters, maggies, maggies_unc


def load_dummy_galaxy(filter_sel: list[Filter]
        ) -> tuple[list[observate.Filter], np.ndarray, np.ndarray]:
    """Loads a dummy galaxy for prospector. This is useful for running forward
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
