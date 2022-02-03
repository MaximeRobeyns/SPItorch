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
"""Some custom types"""

import numpy as np
import torch as t
import pandas as pd

from enum import Enum
from typing import Union

# Neural network related ------------------------------------------------------


# A PyTorch Tensor
Tensor = t.Tensor
# One or more tensors used to parametrise a distribution
DistParams = list[Tensor]
# NumPy array or PyTorch tensor
tensor_like = Union[np.ndarray, Tensor, pd.Series, pd.DataFrame]


class MCMCMethod(Enum):
    EMCEE = 'EMCEE'
    Dynesty = 'Dynesty'


class FittingMethod(Enum):
    LM = 'lm'  # Levenberg-Marquardt
    Powell = 'powell'
    ML = 'ml'  # use machine learning predictions to initialise theta.


class ConcurrencyMethod(Enum):
    MPI = 1  # mpi-based concurrency
    native = 2  # multiprocessing based concurrency
