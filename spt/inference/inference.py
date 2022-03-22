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
from spt.inference.san import SAN
from spt.load_photometry import load_simulated_data, get_denorm_theta_t


class PModel(SAN):
    """A SAN which is slightly adapted to act as a likelihood / forward model
    emulator by switching the xs and thetas in the preprocessing step."""
    def preprocess(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        return y.to(self.device, self.dtype), x.to(self.device, self.dtype)


def match_posteriors(Q: SAN, P: SAN, train_loader: DataLoader, epochs: int,
                     K: int, logging_frequency: int = 1000) -> SAN:
    """Run the posterior matching training procedure"""
    Q.train()
    P.eval()

    fp = cfg.ForwardModelParams()

    dtt = get_denorm_theta_t(fp, dtype=mp.dtype, device=mp.device)
    priors = fp.to_torch_priors(mp.dtype, mp.device)

    Q.opt = t.optim.Adam(Q.parameters(), lr=3e-4, weight_decay=1e-4)

    # update on in-distribution points first
    for e in range(epochs):
        for i, (x, y) in enumerate(train_loader):
            x, _ = Q.preprocess(x, y)

            xshape = x.shape
            xs = x.unsqueeze(-2).expand(xshape[0], K, xshape[1])

            theta_hat = Q(xs, rsample=False).detach()
            x_hat = P(theta_hat, rsample=False).detach()

            _ = Q(x_hat, rsample=False)
            approx_posterior = Q.likelihood.log_prob(theta_hat, Q.last_params)
            approx_posterior = approx_posterior.sum(-1)

            _ = P(theta_hat, rsample=False)
            log_likelihood = P.likelihood.log_prob(x_hat, P.last_params).sum(-1)
            log_likelihood = log_likelihood.sum(-1)[..., None]
            denorm = dtt(theta_hat)
            prior_dims = [priors[d].log_prob(denorm[..., d])[..., None]
                          for d in range(len(priors))]
            log_prior = t.cat(prior_dims, -1).sum(-1)[..., None]
            posterior = t.cat((log_likelihood, log_prior), -1).logsumexp(-1)
            posterior = posterior.detach()

            X1, X2 = t.sort(approx_posterior)[0], t.sort(posterior)[0]
            loss = ((X1 - X2) ** 2.).sum(-1).mean(0)

            Q.opt.zero_grad()
            loss.backward()
            Q.opt.step()

            if i % ip.logging_frequency == 0:
                logging.info((f'Objective at epoch: {e:02d}/{epochs:02d}'
                              f' iter: {i:04d}/{len(train_loader):04d} is '
                              f'{loss.sum(-1).detach().cpu().item()}'))
    return Q


if __name__ == '__main__':

    import spt.config as cfg

    ip = cfg.InferenceParams()
    fp = cfg.ForwardModelParams()

    # Maximum-likelihood training of approximate posterior --------------------

    # TODO load model parameters dynamically based on the model type
    mp = cfg.SANParams()
    Q = ip.model(mp)
    logging.info(f'Initialised {Q.name} model')

    train_loader, test_loader = load_simulated_data(
        path=ip.dataset_loc,
        split_ratio=ip.split_ratio,
        batch_size=Q.params.batch_size,
        phot_transforms=[],
        theta_transforms=[],
    )
    logging.info('Created data loaders')

    Q.offline_train(train_loader, ip)
    logging.info('ML training of approximate posterior complete.')

    # Maximum-likelihood training of neural likelihood ------------------------

    lp = cfg.SANLikelihoodParams()
    P = SAN(lp)
    logging.info(f'Initialised neural likelihood: {P.name}')
    ip.ident = "ML_likelihood"
    P.offline_train(train_loader, ip)
    logging.info('ML training of neural likelihood complete.')

    # "Posterior matching" procedure ------------------------------------------

    Q = match_posteriors(Q, P, train_loader, ip.update_epochs, ip.update_K,
                         ip.logging_frequency)

    real_train_loader, real_test_loader = load_real_data(
        path=ip.catalogue_loc,
        filters=fp.filters,
        split_ratio=ip.split_ratio,
        batch_size=1024,
        transforms=[t.from_numpy],
        x_transforms=[np.log],
    )

    Q = match_posteriors(Q, P, real_train_loader, ip.update_real_epochs,
                         ip.update_K, ip.logging_frequency)

    t.save(Q.state_dict(), Q.fpath(f'{ip.ident}_fulltrained'))
