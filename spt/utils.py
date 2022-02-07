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
import pprint
import logging

from enum import Enum
from typing import Any

class ConfigClass():
    """ConfigClass is an abstract base class for all SPItorch configuration
    objects.

    TODO: use rich to provide clearer class representations.
    """

    def __init__(self) -> None:
        logging.debug(f'New configuration object: {self}')

    def __repr__(self) -> str:
        width = 60
        r = f'\n\n{width*"="}\n'
        c = f'Configuration class `{type(self).__name__}`'
        n = len(c)
        nn = int((width - n) / 2)
        r += nn * ' ' + c + f'\n{width*"-"}\n\n'
        members = [a for a in dir(self) if not callable(getattr(self, a))\
                   and not a.startswith("__")]
        for m in members:
            r += f'{m}: {pprint.pformat(getattr(self, m), compact=True)}\n'
            # r += f'{m}: {pprint(getattr(self, m), max_length=80)}\n'
        r += '\n' + width * '=' + '\n\n'
        return r

    def to_dict(self) -> dict[str, Any]:
        members = [a for a in dir(self) if not callable(getattr(self, a))\
                   and not a.startswith("__")]
        d = {}
        for m in members:
            d[m] = getattr(self, m)
            if isinstance(d[m], Enum):
                d[m] = d[m].value
        return d


# Miscellaneous ==============================================================


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
