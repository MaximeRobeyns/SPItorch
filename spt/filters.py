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
"""Classes and datatypes to do with Sedpy filters used for modelling"""

import os
import sedpy
import shutil
import logging

from copy import deepcopy
from typing import Optional, Union


class Filter(object):
    def __init__(self, bandpass_file: str, mag_col: str, error_col: str):
        """Encapsulates a sedpy filter.

        Args:
            bandpass_file: Location of the bandpass_file
            mag_col: Name of the maggies column in the dataframe.
            error_col: Name of the errors column.

        Note:
            The bandpass file is relative to the
            python3.x/site-packages/sedpy/data/filters directory.
            If you have some custom filters, you can still use them, but you
            have to copy them to this location first.
        """
        self.bandpass_file = bandpass_file
        self.mag_col = mag_col
        self.mag_error_col = error_col

        self.maggie_col = mag_col.replace('mag', 'maggie')
        self.maggie_error_col = error_col.replace('mag', 'maggie')

    def __repr__(self) -> str:
        return self.bandpass_file


class FilterSet():
    def __init__(self, name: str, fs: list[Filter]):
        self.value = name
        self.filters = fs

    @property
    def dim(self) -> int:
        return len(self.filters)

    @property
    def description(self) -> str:
        return str(self.filters)

    def __repr__(self) -> str:
        return f'{len(self.filters)} "{self.value}" filters: {self.description}'

    def __eq__(self, other) -> bool:
        return self.value == other

    def __call__(self) -> list[Filter]:
        return self.filters

    def __len__(self) -> int:
        return len(self.filters)


class _FilterDirectory(object):
    """A dict-like that only returns copies of the dictionary values.
    It also includes a dictionary of information describing each entry in the
    directory.
    """

    def __init__(self):
        self._entries: dict[str, FilterSet] = {}
        self._descriptions: dict[str, str] = {}

    def __getitem__(self, k: str) -> list[Filter]:
        # TODO is deepcopy necessary here?
        return deepcopy(self._entries[k].filters)

    def get(self, k: str) -> FilterSet:
        return self._entries[k]

    def __setitem__(self, k: str, v: Union[tuple[FilterSet, str], FilterSet]):
        if isinstance(v, tuple):
            e, d = v
        else:
            e = v
            d = f'{v}'
        self._entries[k] = e
        self._descriptions[k] = d

    def describe(self, k: str) -> None:
        print(self._entries[k])

    def show_contents(self) -> None:
        for k, v in list(self._descriptions.items()):
            print("'{}':\n  {}".format(k, v))


FilterLibrary = _FilterDirectory()

_galex = [
    Filter(
        bandpass_file=f'{b}_galex',
        mag_col=f'mag_auto_galex_{b.lower()}_dr67',
        error_col=f'magerr_auto_galex_{b.lower()}_dr67')
    for b in ['NUV', 'FUV']]

# cfht is awkward due to i filter renaming. For now, we use i = i_new
_cfht = [
    Filter(
        bandpass_file='{}_cfhtl{}'.format(
            b, '_new' if b == 'i' else ''
        ),
        mag_col=f'mag_auto_cfhtwide_{b}_dr7',
        error_col=f'magerr_auto_cfhtwide_{b}_dr7')
    for b in ['g', 'i', 'r', 'u', 'z']]

_des = [
    Filter(
        bandpass_file=f'DES_{b}',
        mag_col=f'mag_auto_{b}',
        error_col=f'magerr_auto_{b}')
    for b in ['g', 'i', 'r']]

_kids = [
    Filter(
        bandpass_file=f'{b}_kids',
        mag_col=f'mag_auto_kids_{b}_dr2',
        error_col=f'magerr_auto_kids_{b}_dr2')
    for b in ['i', 'r']]

_vista = [
    Filter(
        bandpass_file=f'VISTA_{b}',
        mag_col='mag_auto_viking_{}_dr2'.format(b.lower().strip('s')),
        error_col='magerr_auto_viking_{}_dr2'.format(b.lower().strip('s')))
    for b in ['H', 'J', 'Ks', 'Y', 'Z']]

_vista_euclid = [
    Filter(
        bandpass_file=f'VISTA_{b}',
        mag_col='mag_auto_viking_{}_dr2'.format(b.lower().strip('s')),
        error_col='magerr_auto_viking_{}_dr2'.format(b.lower().strip('s')))
    for b in ['H', 'J', 'Y']]

_vista_des = [
    Filter(
        bandpass_file=f'VISTA_{b}',
        mag_col=f'{b}AUTOMAG',
        error_col=f'{b}AUTOMAGERR')
    for b in ['H', 'J', 'Y', 'Z']]

_sdss = [
    Filter(
        bandpass_file=f'{b}_sloan',
        mag_col=f'mag_auto_sdss_{b}_dr12',
        error_col=f'magerr_auto_sdss_{b}_dr12')
    for b in ['u', 'g', 'r', 'i', 'z']]

_wise = [
    Filter(
        bandpass_file=f'wise_{b}',
        mag_col='mag_auto_AllWISE_{b.upper()}',
        error_col=f'magerr_auto_AllWISE_{b.upper()}')
    for b in ['w1', 'w2']]  # exclude w3, w4


FilterLibrary["galex"] = FilterSet("galex", _galex)
FilterLibrary["cfht"] = FilterSet("cfht", _cfht)
FilterLibrary["kids"] = FilterSet("kids", _kids)
FilterLibrary["vista"] = FilterSet("vista", _vista)
FilterLibrary["vista_euclid"] = FilterSet("vista_euclid", _vista_euclid)
FilterLibrary["vista_des"] = FilterSet("vista_des", _vista_des)
FilterLibrary["sdss"] = FilterSet("sdss", _sdss)
FilterLibrary["wise"] = FilterSet("wise", _wise)

FilterLibrary["des"] = FilterSet("des", _des + _vista_des),\
                       "Dark Energey Survey filter set"
FilterLibrary["euclid"] = FilterSet("euclid", _sdss + _vista_euclid),\
                          "Filter set for Euclid"
FilterLibrary["reliable"] = FilterSet("reliable", _sdss + _vista + _wise),\
                            "Reliable filter set"
FilterLibrary["all"] = FilterSet("all", _galex + _sdss + _cfht + _kids + _vista + _wise), \
                       "'All' filters (legacy)"

class FilterCheck:

    def __init__(self):
        self.ensure_filters_installed()

    filters: list[Filter] = []
    filter_loc: Optional[str] = None

    def ensure_filters_installed(self):
        """Looks at all the bandpass files for the provided list of filters, and
        ensures that they are in `sedpy`'s filter set.

        When using a custom filter (placed in `spitorch/custom_filters`) for
        the first time, it will be missing. This function will attempt to copy the
        correspondingly named transmission file to sedpy's filter location.
        """

        loc = '/'.join(sedpy.__file__.split('/')[:-1]) + '/data/filters'
        existing_filters = os.listdir(loc)
        missing = []
        for f in self.filters:
            bp_file = f'{f.bandpass_file}.par'
            if bp_file not in existing_filters:
                missing.append(bp_file)

        if len(missing):
            if self.filter_loc is None:
                import spt
                self.filter_loc = os.path.split(spt.__path__[0])[0] + '/custom_filters'
            assert self.filter_loc is not None
            cf_contents = os.listdir(self.filter_loc)

            has_errors = False

            for f in missing:
                if f not in cf_contents:
                    logging.error(f'Could not install bandpass file [bold]{f}[/bold] from {self.filter_loc}')
                    has_errors = True
                    continue
                shutil.copy(f'{self.filter_loc}/{f}', f'{loc}')

            if has_errors:
                raise ValueError("Please ensure custom filters are in the custom_filters folder.")
