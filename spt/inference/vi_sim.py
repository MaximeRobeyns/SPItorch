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
"""Simulation-Based Inference with VI.
"""

import os
import sys
import torch as t
import numpy as np
import logging

from typing import Any, Optional, Type

import spt.config as cfg
import spt.inference.san as san

from spt.load_photometry import load_simulated_data, load_real_data, get_norm_theta
from spt.types import Tensor

if __name__ == '__main__':

    # 1. Train a SAN approximate posterior using maximum likelihood (following
    # the same training procedure as previously)

    ip = cfg.InferenceParams()
    fp = cfg.ForwardModelParams()
    sp = cfg.SANParams()

    if not os.path.exists(ip.dataset_loc):
        logging.error(f'Could not locate training dataset {ip.dataset_loc}')
        sys.exit(1)

    Q = san.SAN(sp)
    logging.info(f'Initialised {Q.name} model for approximate posterior')
    train_loader, test_loader = load_simulated_data(
        path=ip.dataset_loc,
        split_ratio=ip.split_ratio,
        batch_size=Q.params.batch_size,
        phot_transforms=[lambda x: t.from_numpy(np.log(x))],
        theta_transforms=[get_norm_theta(fp)],
    )
    logging.info('Created data loaders')

    ip.ident = "ML_approx_post"
    Q.offline_train(train_loader, ip)

    # 2. Train a neural likelihood (either another SAN, or another simple MLP
    # described with arch_t; try both) to learn the likelihood P(x | theta)

    # likelihood parameters (for SAN)
    class LP(san.SANParams):
        # Number of epochs to train for (offline training)
        epochs: int = 10
        batch_size: int = 1024
        dtype: t.dtype = t.float32
        # We are modelling observations
        data_dim: int = len(cfg.ForwardModelParams().filters)
        # likelihood conditions on physical parameters
        cond_dim: int = len(cfg.ForwardModelParams().free_params)
        module_shape: list[int] = [512, 512]
        sequence_features: int = 8
        likelihood: Type[san.SAN_Likelihood] = san.MoG
        likelihood_kwargs: Optional[dict[str, Any]] = {
            'K': 10, 'mult_eps': 1e-4, 'abs_eps': 1e-4
        }
        batch_norm: bool = True
        opt_lr: float = 1e-4

    class PModel(san.SAN):
        def preprocess(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
            """Need to overload this method since the data loaders treat x as
            photometry and y as physical parameters:
            """
            return y.to(self.device, self.dtype), x.to(self.device, self.dtype)

    P = PModel(LP())
    ip.ident = "ML_likelihood"
    P.offline_train(train_loader, ip)

    # 3. VI tuning to real distribution ---------------------------------------

    # 4. Load up a catalogue of real observations as a data loader, and
    # optimise an ELBO between our approximate posterior from step 2, and the
    # product of the neural likelihood and the prior.
    #
    # => Need to make evaluating the prior more efficient; pull out the prior
    # distributions from Prospector and implement them yourself.
    # => Need to use reparametrised sampling in the SAN in order to optimise
    # the ELBO. Provide a 'toggle' in the SAN architecture to optionally stop
    # gradients at the sampling steps.

    if not os.path.exists(ip.catalogue_loc):
        logging.error(f'Could not find the catalogue {ip.catalogue_loc}')
        sys.exit(1)

    real_train_loader, real_test_loader = load_real_data(
        path=ip.catalogue_loc,
        filters=cfg.ForwardModelParams().filters,
        split_ratio=ip.split_ratio,
        batch_size=1024,
        transforms=[lambda x: t.from_numpy(x)],
        x_transforms=[lambda x: np.log(x)], #centre_phot_np],
    )
    logging.info('Finished loading real catalogue.')

    # TODO: put this in its own configuration class in config.py
    # VIParams
    epochs: int = 10
    K: int = 100

    # TODO develop this part in a notebook, then write it up in here
    # Q.train()
    # for e in range(epochs):
    #     for i, (x, y) in enumerate(real_train_loader):
    #         x, y = Q.preprocess(x, y)

    #         # MC estimate of the ELBO.
    #         theta = Q.sample(x, K)

    #         TODO assert the shape of this...

    #         obj = -ELBO
    #         Q.opt.zero_grad
    #         obj.backward()
    #         Q.opt.step()

    #         if i % ip.logging_frequency == 0 or i == len(real_train_loader)-1:
    #             logging.info(
    #                 "VI Epoch: {:02d}/{:02d}, Batch: {:05d}/{:d}, Loss {:9.4f}"
    #                 .format(e+1, epochs, i, len(real_train_loader)-1,
    #                         obj.item()))


    # Create a data loader containing real observations
