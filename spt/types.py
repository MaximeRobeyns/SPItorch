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

from sedpy import observate
from typing import Callable, Optional, Union
from prospect.models.priors import Prior

# Note: to avoid circular imports, do not import any spt.* modules here.

# Type for CPz model parameter description
pdict_t = dict[str, Union[float, bool, str, Prior]]

# Remove if unused.
# # Type for the limits on the free parameters.
# paramspace_t = dict[str, tuple[float, float]]
#
# # Type for prospector run parameters
# prun_params_t = dict[str, Union[int, bool, float, None, list[int], str]]


# Neural network related ------------------------------------------------------


# A PyTorch Tensor
Tensor = t.Tensor
# One or more tensors used to parametrise a distribution
DistParams = list[Tensor]
# NumPy array or PyTorch tensor
tensor_like = Union[np.ndarray, Tensor, pd.Series, pd.DataFrame]


# Inference Procedures --------------------------------------------------------


class MCMCMethod():
    def __init__(self) -> None:
        self.name = 'MCMC'

    def __repr__(self) -> str:
        return self.name

class EMCEE(MCMCMethod):
    def __init__(self) -> None:
        self.name = 'EMCEE'

class Dynesty(MCMCMethod):
    def __init__(self) -> None:
        self.name = 'Dynesty'

