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

from abc import abstractmethod
import logging

from typing import Any, Optional, Type, Union
from prospect.models.priors import Prior, Uniform
from prospect.models.templates import TemplateLibrary


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

    def to_dict(self) -> dict[str, pdict_t]:
        values = {
            'units': self.units,
            'init': self.init,
            'prior': self.prior,
            'isfree': self.isfree,
            'N': self.N,
            '_is_log': self.log,
        }
        if self.disp_floor is not None:
            values['disp_floor']  = self.disp_floor
        return {self.name: values}

    def __repr__(self) -> str:
        pn = str(self.prior.__class__).split('.')[-1].split("'")[0]
        if self.isfree:
            return f'\n\tFixed {self.name} ({self.units}) at {self.init}'
        else:
            return f'\n\tLearned {self.name} ({self.units}) with {pn} prior'


class ParamConfig:
    """Base class for forward model parameters"""

    @property
    @abstractmethod
    def model_param_templates(self) -> list[str]:
        return []

    @property
    @abstractmethod
    def model_params(self) -> list[Parameter]:
        return []

    @property
    def all_params(self) -> dict[str, pdict_t]:
        """A utility method to combine template and manually specified model
        parameters.

        Returns:
            pdict_t: The combined model parameters.
        """

        tmp_params: dict[str, pdict_t] = {}

        # Begin by applying the templates...
        for t in self.model_param_templates:
            if t not in TemplateLibrary._entries.keys():
                logging.warning(f'Template library {t} is not recognized.')
            else:
                tmp_params |= TemplateLibrary[t]

        # Such that we can override parameters with the manually-defined
        # parameters:
        for p in self.model_params:
            tmp_params |= p.to_dict()

        return tmp_params

    @property
    def free_params(self) -> dict[str, pdict_t]:
        fp: dict[str, pdict_t] = {}
        ap = self.all_params
        for k, v in zip(ap.keys(), ap.values()):
            if v['isfree']:
                fp |= {k: v}
        return fp

    @property
    def ordered_params(self) -> list[str]:
        params = list(self.all_params.keys())
        params.sort()
        return params

    @property
    def ordered_free_params(self) -> list[str]:
        # fp = [p[0] for p in self.all_params.items() if p[1]['isfree']].sort()
        fp = list(self.free_params.keys())
        fp.sort()
        return fp

    def free_param_lims(self) -> list[tuple[float, float]]:
        lims: list[tuple[float, float]] = []
        fp = self.free_params
        ofp = self.ordered_free_params
        for p in ofp:
            prior = fp[p]['prior']
            assert isinstance(prior, Prior)
            if 'mini' not in prior.params.keys() or \
               'maxi' not in prior.params.keys():
                logging.error('Free parameters must have a speciried range')
                raise RuntimeError((
                    f'Could not find "mini" and "maxi" attributes of free '
                    'parameter prior {prior}'))
            lims.append((prior.params['mini'], prior.params['maxi']))
        return lims

    def is_log(self) -> list[bool]:
        # Note: we do not try to 'auto detect' whether a template parameter is
        # logarithmic e.g. from its name. Only manually defined parameters will
        # be considered.

        log = []
        fp, ofp = self.free_params, self.ordered_free_params
        for p in ofp:
            if '_is_log' in fp[p]:
                log.append(fp[p]['_is_log'])
            # Otherwise, this must be a template parameter. Skip it.
        return log
