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
"""spt is the main module for SPItorch, a package of inferring the properties
of stellar populations in PyTorch.
"""

__version__ = "0.0.1"

import os, sys

os.environ['SPS_HOME'] = os.path.split(__path__[0])[0] + '/deps/fsps'

from .load_photometry import *
from .modelling.prospector import Prospector
from .utils import splash_screen

from spt.logs import configure_logging
configure_logging()

# If user is running in interactive tty, print splash screen
# Will not print when used as library import, which would get annoying.
# if os.isatty(sys.stdout.fileno()):
#     splash_screen()
