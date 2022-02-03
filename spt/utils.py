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
"""Project-wide utilities file"""

import time
import numpy as np
import pprint
import logging

class ConfigClass():
    """ConfigClass is an abstract base class for all SPItorch configuration
    objects.

    TODO: use rich to provide clearer class representations.
    """

    def __init__(self) -> None:
        logging.debug(f'New configuration object: {self}')

    def __repr__(self) -> str:
        r = f'\n\n{79*"="}\n'
        c = f'Configuration class `{type(self).__name__}`'
        n = len(c)
        nn = int((79 - n) / 2)
        r += nn * ' ' + c + f'\n{79*"-"}\n\n'
        members = [a for a in dir(self) if not callable(getattr(self, a))\
                   and not a.startswith("__")]
        for m in members:
            r += f'{m}: {pprint.pformat(getattr(self, m), compact=True)}\n'
            # r += f'{m}: {pprint(getattr(self, m), max_length=80)}\n'
        r += '\n' + 79 * '=' + '\n\n'
        return r


def denormalise_theta(norm_theta: np.ndarray, limits: np.ndarray) -> np.ndarray:
    """Rescale norm_theta lying within [0, 1] range; not altering the
    distribution of points.
    """
    assert norm_theta.shape[1] == limits.shape[0]
    return limits[:,0] + (limits[:,1] - limits[:,0]) * norm_theta


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
    limits[log_mask] = np.log10(limits[log_mask, :])
    dtheta = limits[:,0] + (limits[:,1] - limits[:,0]) * norm_theta

    return np.where(np.array(log_mask), 10**np.clip(dtheta, -10, 20), dtheta)


def normalise_theta(theta: np.ndarray, limits: np.ndarray) -> np.ndarray:
    """Rescale theta values to lie within [0, 1] range; not altering the
    distribution of points within this range."""
    assert theta.shape[1] == limits.shape[0]
    offset = theta - limits[:, 0]
    return offset / limits[:, 1] - limits[:, 0]


# We would never use this type of function...
# def normalise_unif_theta(theta: np.ndarray, limits: np.ndarray,
#                          log_mask: list[bool]) -> np.ndarray:
#     assert theta.shape[1] == limits.shape[0]
#     limits[log_mask] = np.log10(limits[log_mask, :])
#     theta[log_mask] = np.log10()
#     offset = theta - limits[:, 0]
#     return offset / limits[:, 1] - limits[:, 0]


colours = {
    "b": "#025159",  # blue(ish)
    "o": "#F28705",  # orange
    "lb": "#03A696", # light blue
    "do": "#F25D27", # dark orange
    "r": "#F20505"   # red.
}


# Splash screen ==============================================================


def splash_screen():
    import logging.handlers as lhandlers
    from spt import __version__
    from rich.padding import Padding
    from rich.console import Console

    console = Console(width=80)
    console.rule()
    info = Padding(f'''
    SPItorch

    Version: {__version__}, {time.ctime()}
    Copyright (C) 2019-20 Mike Walmsley <walmsleymk1@gmail.com>
    Copyright (C) 2022 Maxime Robeyns <dev@maximerobeyns.com>
            ''', (1, 8))
    console.print(info, highlight=False, markup=False)
    console.rule()

    lc = Console(record=True, force_terminal=False, width=80)
    lc.begin_capture()
    lc.rule()
    lc.print(info, highlight=False, markup=False)
    lc.rule()
    for h in logging.getLogger().handlers:
        if isinstance(h, lhandlers.RotatingFileHandler):
            r = logging.makeLogRecord({
                "msg": '\n'+lc.end_capture(),
                "level": logging.INFO,
                })
            h.handle(r)
