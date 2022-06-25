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
"""Full training run."""

import logging
import torch as t
import numpy as np

from spt.types import Tensor
from spt import config as cfg
from spt.inference.san import SANv2, PModelv2
from spt.load_photometry import load_simulated_data, get_norm_theta, \
                                load_real_data


def dcn(x: Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


if __name__ == '__main__':

    device = "cuda" if t.cuda.is_available() else "cpu"
    dtype = t.float32

    logging.info(f'Beginning SAN (approximate posterior) training')

    ip = cfg.InferenceParams()
    fp = cfg.ForwardModelParams()
    sp = cfg.SANv2Params()

    # Train the approximate posterior -----------------------------------------
    ip.retrain_model = True
    ip.use_existing_checkpoints = False

    Q = SANv2(sp)
    logging.info(f'Initialised approximate posterior: {Q}')

    train_loader, test_loader = load_simulated_data(
        path=ip.dataset_loc,
        split_ratio=ip.split_ratio,
        batch_size=sp.batch_size,
        phot_transforms=[lambda x: t.from_numpy(np.log(x))],
        theta_transforms=[get_norm_theta(fp)],
    )

    Q.offline_train(train_loader, ip)

    logging.info('Finished training approximate posterior')

    # Train the neural likelihood ---------------------------------------------

    slp = cfg.SANv2LikelihoodParams()
    P = PModelv2(slp)
    ip.ident = "ML_likelihood"

    P.offline_train(train_loader, ip)
    logging.info(f'Initialised neural likelihood: {P}')

    # Run the HMC update procedure on simulated data --------------------------

    # Create new data loaders with smaller batch sizes (for memory consumption)
    train_loader, test_loader = load_simulated_data(
        path=ip.dataset_loc,
        split_ratio=ip.split_ratio,
        batch_size=ip.hmc_update_batch_size,
        phot_transforms=[np.log, t.from_numpy],
        theta_transforms=[get_norm_theta(fp)],
    )
    logging.info('Created data loaders with HMC update batch size')

    ip.ident = ip.hmc_update_sim_ident
    Q.hmc_retrain_procedure(train_loader, ip, P=P,
                            epochs=ip.hmc_update_sim_epochs,
                            K=ip.hmc_update_sim_K, lr=3e-4, decay=1e-4)
    logging.info('Updated on simulated data')

    # # Run the HMC update procedure on real data (no augmentation) -------------

    # real_train_loader, real_test_loader = load_real_data(
    #     path=ip.catalogue_loc, filters=fp.filters, split_ratio=ip.split_ratio,
    #     batch_size=ip.hmc_update_batch_size,
    #     transforms=[t.from_numpy], x_transforms=[np.log],
    # )

    # ip.ident = ip.hmc_update_real_ident
    # Q.hmc_retrain_procedure(real_train_loader, ip, P=P,
    #                         epochs=ip.hmc_update_real_epochs,
    #                         K=ip.hmc_update_real_K, lr=3e-4, decay=1e-4)
    # logging.info('Updated on real data')

    logging.info('Exiting')
