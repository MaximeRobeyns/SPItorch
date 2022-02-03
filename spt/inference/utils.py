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
"""Inference utilities"""


import torch as t
import torch.nn as nn


class Squareplus(nn.Module):
    def __init__(self, a=2):
        super().__init__()
        self.a = a
    def forward(self, x):
        """The 'squareplus' activation function: has very similar properties to
        softplus, but is computationally cheaper and more configurable.
            - squareplus(0) = 1 (softplus(0) = ln 2)
            - gradient diminishes more slowly for negative inputs.
            - ReLU = (x + sqrt(x^2))/2
            - 'squareplus' becomes smoother with higher 'a'
        """
        return (x + t.sqrt(t.square(x)+self.a*self.a))/2


def squareplus_f(x, a=2):
    """Functional version of the 'squareplus' activation function. See
    `Squareplus` for more information.
    """
    return (x + t.sqrt(t.square(x)+a*a))/2


# TODO complete (e.g. loading offline observation datasets)
