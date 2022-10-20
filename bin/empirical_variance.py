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
"""Calculates the empirical variance of filter observations, using a real
training dataset.
"""

import numpy as np

import spt.config as cfg

from spt.load_photometry import load_catalogue

if __name__ == '__main__':

    # Load up the real catalogue -----------------------------------------------

    fp = cfg.ForwardModelParams()
    path = "data/DES_VIDEO_v1.0.1.fits"
    df = load_catalogue(path, fp.filters, True)

    # Get the observational data -----------------------------------------------
    # df columns are:
    # Index(['maggie_auto_g', 'maggie_auto_i', 'maggie_auto_r', 'HAUTOMAG',
    #        'JAUTOMAG', 'YAUTOMAG', 'ZAUTOMAG', 'maggieerr_auto_g',
    #        'maggieerr_auto_i', 'maggieerr_auto_r', 'HAUTOMAGERR', 'JAUTOMAGERR',
    #        'YAUTOMAGERR', 'ZAUTOMAGERR', 'redshift'], dtype='object')
    #
    # Get the mean and variance of the error columns:

    err_cols = np.log(df.iloc[:, 7:13].to_numpy())

    err_means = err_cols.mean(0)
    print(f'maggie error means: {err_means}')
    err_vars = err_cols.var(0)
    print(f'maggie error vars: {err_vars}')

    # Means:
    # maggieerr_auto_g    2.822223e-11
    # maggieerr_auto_i    6.756946e-11
    # maggieerr_auto_r    3.781835e-11
    # HAUTOMAGERR         6.686266e-11
    # JAUTOMAGERR         6.268341e-11
    # YAUTOMAGERR         5.360808e-11
    #
    # Variances:
    # maggieerr_auto_g    2.107892e-22
    # maggieerr_auto_i    1.016179e-21
    # maggieerr_auto_r    3.381821e-22
    # HAUTOMAGERR         1.833526e-20
    # JAUTOMAGERR         2.373840e-20
    # YAUTOMAGERR         2.943154e-20
