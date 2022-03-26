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
"""Implements the full training, and subsequent inference procedure for a
catalogue."""

import logging
import torch as t
import numpy as np

from torch.utils.data import DataLoader

from spt.types import Tensor
from spt.inference.san import SAN, PModel
from spt.load_photometry import load_simulated_data, get_norm_theta


# def tuning_procedure(Q: SAN, P: SAN, train_loader: DataLoader, epochs: int,
#                      K: int, logging_frequency: int = 1000) -> SAN:
#     """Run the training procedure on the real data"""
#
#     # Training start time: 17:03
#     Q.train()
#     P.eval()
#
#     fp = cfg.ForwardModelParams()
#
#     dtt = get_denorm_theta_t(fp, dtype=mp.dtype, device=mp.device)
#     priors = fp.to_torch_priors(mp.dtype, mp.device)
#
#     Q.opt = t.optim.Adam(Q.parameters(), lr=3e-4, weight_decay=1e-4)
#
#     # update on in-distribution points first
#     for e in range(epochs):
#         for i, (x, y) in enumerate(train_loader):
#             x, _ = Q.preprocess(x, y)
#
#             # xshape = x.shape
#             # xs = x.unsqueeze(-2).expand(xshape[0], K, xshape[1])
#             xs = x.repeat_interleave(K, 0)
#
#             theta_hat = Q(xs, rsample=False).detach()
#
#             with t.no_grad():
#                 x_hat = P(theta_hat, rsample=False).detach()
#
#             _ = Q(x_hat, rsample=False)
#             post_prob = Q.likelihood.log_prob(theta_hat, Q.last_params)
#
#             loss = -post_prob.sum(-1).mean(0)
#
#             Q.opt.zero_grad()
#             loss.backward()
#             Q.opt.step()
#
#             if i % ip.logging_frequency == 0:
#                 logging.info((f'Objective at epoch: {e:02d}/{epochs:02d}'
#                               f' iter: {i:04d}/{len(train_loader):04d} is '
#                               f'{loss.sum(-1).detach().cpu().item()}'))
#     return Q


if __name__ == '__main__':

    import spt.config as cfg

    ip = cfg.InferenceParams()
    fp = cfg.ForwardModelParams()

    # Maximum-likelihood training of approximate posterior --------------------

    mp = cfg.SANParams()
    Q = SAN(mp)
    logging.info(f'Initialised {Q.name} model')

    train_loader, test_loader = load_simulated_data(
        path=ip.dataset_loc,
        split_ratio=ip.split_ratio,
        batch_size=Q.params.batch_size,
        phot_transforms=[lambda x: t.from_numpy(np.log(x))],
        theta_transforms=[get_norm_theta(fp)],
    )
    logging.info('Created data loaders')

    Q.offline_train(train_loader, ip)
    logging.info('ML training of approximate posterior complete.')

    # Maximum-likelihood training of neural likelihood ------------------------

    lp = cfg.SANLikelihoodParams()
    P = PModel(lp)
    logging.info(f'Initialised neural likelihood: {P.name}')
    ip.ident = "ML_likelihood"

    P.offline_train(train_loader, ip)
    logging.info('ML training of neural likelihood complete.')

    # Update procedure with simulated data ------------------------------------

    ip.ident = ip.update_sim_ident
    Q.retrain_procedure(train_loader, ip, P=P, epochs=ip.update_sim_epochs,
                        K=ip.update_sim_K, lr=3e-4, decay=1e-4)
    logging.info('Updated on simulated data')

    # Update procedure with real data -----------------------------------------

    ip.ident = ip.update_real_ident
    Q.retrain_procedure(train_loader, ip, P=P, epochs=ip.update_real_epochs,
                        K=ip.update_real_K, lr=3e-4, decay=1e-4)
    logging.info('Updated on real data')
