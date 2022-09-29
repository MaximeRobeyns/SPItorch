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
"""
spt is the main module for SPItorch, a package of inferring the properties
of stellar populations in PyTorch.
"""

import os, sys

from dataclasses import dataclass

from .__version__ import __version__
from .load_photometry import *
from .modelling.prospector import Prospector
from .utils import splash_screen
from spt.logs import configure_logging

os.environ["SPS_HOME"] = os.path.split(__path__[0])[0] + "/deps/fsps"  # type: ignore
configure_logging()

# If user is running in interactive tty, print splash screen
# Will not print when used as library import, which would get annoying.
# if os.isatty(sys.stdout.fileno()):
#     splash_screen()


@dataclass
class RunConfig:
    """Configuration class for runs

    Parameters:
        id: semantically meaningful identifier for this task configuration.
        description: a useful description of this particular run. This is
            output at the root of the results directory (in README.txt)
        task: the name of the task to run. This will determine what is run in
            ``run.py``.
        for_version: ensures that the configuration is compatible with the
            current source code version.
    """

    id: str
    description: str
    task: str
    for_version: str


@dataclass
class DistConfig:
    """Configuration class for DDP parameters

    Parameters:
        world_size: the number of GPUs / processes, across all nodes
        rank: the index of the current process
        master_addr: the hostname of the master node (this is usually rank 0)
        master_port: the port number used by PyTorch DDP
        comm_port: the port used by IPC for synchronisation / sharing Python
            objects between objects.
        backend: the distributed backend (nccl | gloo | mpi)
    """

    world_size: int
    rank: int
    master_addr: int
    master_port: int
    comm_port: int
    backend: str = "nccl"
