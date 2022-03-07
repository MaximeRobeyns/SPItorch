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

import math
import logging
import torch as t
import numpy as np
import prospect.models.priors as ppr
import torch.distributions as tdist

from abc import abstractmethod
from typing import Any, Optional, Type, Union
from torch.distributions import Distribution
from prospect.models.priors import Prior, Uniform
from prospect.models.templates import TemplateLibrary

from spt.types import Tensor


__all__ = ["Parameter", "pdict_t"]


# Type for CPz model parameter description
pdict_t = dict[str, Union[float, bool, str, Prior]]


class Parameter:

    def __init__(self, name: str, range_min: float, init: float,
                 range_max: float, prior: Type[Prior] = Uniform,
                 prior_kwargs: dict[str, Any] = {},
                 model_this: bool = True, units: str = '',
                 dim: int = 1, disp_floor: Optional[float] = None):
        """Describes a Prospector parameter.

        This should be defined for direct use in the forward model (no
        normalisations, no other transformations).

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
            units: Human-readable description of units (mostly for plotting; you
                can use $<expr>$-delimited LaTeX expressions in here.)
            dim: The number of dimensions (defaults to 1; scalar)
            disp_floor: Sets the initial dispersion to use when using clouds of
                EMCEE walkers (only for MCMC sampling).
        """
        self.N = dim
        self.name = name
        self.units = units
        self.init = init
        self.isfree = model_this
        self.disp_floor = disp_floor
        self._range_min = range_min
        self._range_max = range_max


        # This is poor form since not all priors accept mini and maxi; however
        # the prospect.models.priors don't check for redundant kwargs...
        #
        # TODO: for LogUniform distribution, should we additionally add
        # loc = (range_max-range_min)/2, scale = (range_min-range_max)?
        self.prior = prior(mini=range_min, maxi=range_max, **prior_kwargs)

    def to_dict(self) -> dict[str, pdict_t]:
        values = {
            'units': self.units,
            'init': self.init,
            'prior': self.prior,
            'isfree': self.isfree,
            'N': self.N,
            '_range_min': self._range_min,
            '_range_max': self._range_max,
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
            pdict_t: A dictionary of combined model parameters ready for use in
                Prospector. The prior distributions are in their denormalised
                range; that is, they follow a log scale if is_log is true,

        """

        tmp_params: dict[str, pdict_t] = {}

        # Begin by applying the templates...
        for t in self.model_param_templates:
            if t not in TemplateLibrary._entries.keys():
                logging.warning(f'Template library {t} is not recognized.')
            else:
                tmp_params |= TemplateLibrary[t]

        # Allows us to override parameters with the manually-defined parameters:
        for param in self.model_params:
            tmp_params |= param.to_dict()

        # Identify parameters defined on a logarithmic scale for normalisation.
        for tmp_param in tmp_params.keys():
            try:
                tpp = tmp_params[tmp_param]['prior']
                tmp_params[tmp_param]['_log_scale'] = \
                    isinstance(tpp, (ppr.LogNormal, ppr.LogUniform))
            except KeyError:  # not all parameters have a prior.
                continue

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
        fp = list(self.free_params.keys())
        fp.sort()
        return fp

    def free_param_lims(self, log_scaled: bool = True,
                        normalised: bool = False,
                        ) -> list[tuple[float, float]]:
        """Get the (prior) limits on the free parameters

        Args:
            log_scaled: whether to apply (natural) logarithm to log-scaled
                parameters.
            normalise: Whether to return the limits for the normalised
                parameters. This is not redundant, since we don't always do
                [0, 1] normalisation.

        Returns:
            list[tuple[float, float]]: list of tuples in standard parameter
                order, with min_max values.

        Raises:
            RuntimeError: upon missing prior parameters.
        """
        lims: list[tuple[float, float]] = []
        fp = self.free_params
        ofp = self.ordered_free_params

        for k in ofp:
            tmp = fp[k]

            prior = tmp['prior']
            assert isinstance(prior, Prior)

            tmp_lim = prior.range

            if log_scaled and tmp['_log_scale']: # uses natural logarithm
                tmp_lim = (math.log(float(tmp_lim[0])), math.log(float(tmp_lim[1])))

            if normalised:
                tmp_lim = (0., 1.)

            lims.append(tmp_lim)

        return lims

    def log_scale(self) -> list[bool]:
        fp, ks = self.free_params, self.ordered_free_params
        return [isinstance(fp[k]['prior'], (ppr.LogNormal, ppr.LogUniform))
                for k in ks]

    def to_torch_priors(self, dtype: t.dtype = None, device: t.device = None
            ) -> list[Distribution]:
        """Returns the free parameter's priors as pytorch distributions."""
        priors: list[Distribution] = []
        fp, ofp = self.free_params, self.ordered_free_params
        for p in ofp:
            P = fp[p]['prior']
            assert isinstance(P, Prior)
            priors.append(prospector_to_torch_dist(P, dtype, device))
        return priors


# Prior conversions -----------------------------------------------------------


class LogUniform(tdist.TransformedDistribution):
    def __init__(self, lb, ub):
        super().__init__(tdist.Uniform(lb.log(), ub.log()),
                         tdist.ExpTransform())


class AffineBeta(tdist.TransformedDistribution):
    def __init__(self, alpha, beta, lb, ub):
        super().__init__(tdist.Beta(alpha, beta),
                         tdist.AffineTransform(lb, ub-lb))


def prospector_to_torch_dist(P: Prior, dtype: t.dtype = None,
                             device: t.device = None) -> Distribution:

    def _t(args: float) -> Tensor:
        return t.tensor(args, dtype=dtype, device=device)

    if isinstance(P, ppr.Uniform):
        return tdist.Uniform(_t(P.range[0]), _t(P.range[1]))
    elif isinstance(P, ppr.TopHat):
        return tdist.Uniform(_t(P.range[0]), _t(P.range[1]))
    elif isinstance(P, ppr.Normal):
        return tdist.Normal(_t(P.loc), _t(P.scale))
    elif isinstance(P, ppr.ClippedNormal):
        raise NotImplementedError(
            'TODO: https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/20)')
    elif isinstance(P, ppr.LogUniform):
        return LogUniform(_t(P.range[0]), _t(P.range[1]))
    elif isinstance(P, ppr.Beta):
        return AffineBeta(_t(P.loc), _t(P.scale), _t(P.range[0]), _t(P.range[1]))
    elif isinstance(P, ppr.LogNormal):
        # TODO account for mini and maxi here too?
        return tdist.LogNormal(_t(P.params['mode']), _t(P.params['sigma']))
    elif isinstance(P, ppr.LogNormalLinpar):
        raise NotImplementedError()
    elif isinstance(P, ppr.SkewNormal):
        raise NotImplementedError()
    elif isinstance(P, ppr.StudentT):
        return tdist.StudentT(_t(P.params['df']), _t(P.params['mean']), _t(P.params['scale']))
    else:
        raise ValueError(f'Unexpected distribution {P}')
