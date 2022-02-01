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
"""Classes to describe modelling parameters"""

from typing import Any, Optional, Type, Union
from prospect.models.priors import Prior, Uniform

from rich import print_json
from rich.console import Console


__all__ = ["Parameter", "pdict_t"]


# Type for CPz model parameter description
pdict_t = dict[str, Union[float, bool, str, Prior]]


class Parameter:

    def __init__(self, name: str, range_min: float, init: float,
                 range_max: float, prior: Type[Prior] = Uniform,
                 prior_kwargs: dict[str, Any] = {},
                 model_this: bool = True,
                 units: str = '', dim: int = 1,
                 disp_floor: Optional[float] = None):
        """Describes a Prospector parameter.

        NOTE: if 'units' begins with 'log' e.g. log_mass, then the
        range_min, init and range_max and disp_floor parameters are
        automatically exponentiated (base 10). The prior_kwargs are *not*
        modified however: it is up to you to transform them.

        A default Uniform prior is applied. For all priors, the `mini` and
        `maxi` parameters are automatically set from the (range_min, range_max)
        arguments.

        Args:
            name: Name of this parameter: this must be recognized by FSPS. See
                <https://dfm.io/python-fsps/current/stellarpop_api/> for these.
            range_min: Lower bound on the range you expect this parameter to
                lie in
            init: Initial value / guess of this parameter
            range_max: Upper bound on the range you expet this paramter to lie
                in.
            prior: Prior distribution (from prospect.models.priors). NOTE: mini
                and maxi parameters are automatically populated based on
                range_min, range_max.
            prior_kwargs: A dictionary of keyword arguments accepted by your
                prior of choice.
            model_this: whether or not to model this parameter with ML / MCMC
                (else it is treated as a 'fixed' parameter)
            units: Description of units (you can use $<expr>$ delimited LaTeX
                maths in here.)
            dim: The number of dimensions (defaults to 1; scalar)
            disp_floor: Sets the initial dispersion to use when using clouds of
                EMCEE walkers (only for MCMC sampling).
        """
        self.log = units.startswith('log')

        self.N = dim
        self.name = name
        self.units = units
        self.init = 10**init if self.log else init
        self.isfree = model_this
        if disp_floor is not None:
            self.disp_floor = 10**disp_floor if self.log else disp_floor
        else:
            self.disp_floor = None

        self.min = 10**range_min if self.log else range_min
        self.max = 10**range_max if self.log else range_max

        # This is poor form since not all priors accept mini and maxi; however
        # the prospect.models.priors don't check for redundant kwargs...
        self.prior = prior(mini=self.min, maxi=self.max, **prior_kwargs)

    def to_dict(self) -> pdict_t:
        values = {
            'units': self.units,
            'init': self.init,
            'prior': self.prior,
            'isfree': self.isfree,
            'N': self.N,
        }
        if self.disp_floor is not None:
            values['disp_floor']  = self.disp_floor
        ret = {}
        ret[self.name] = values
        return ret

    def __repr__(self) -> str:
        pn = str(self.prior.__class__).split('.')[-1].split("'")[0]
        if self.isfree:
            return f'\n\tFixed {self.name} ({self.units}) at {self.init}'
        else:
            return f'\n\tLearned {self.name} ({self.units}) with {pn} prior'
