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
"""Full training for san with logging"""

import logging
import torch as t
import numpy as np

from spt.types import Tensor
from spt import config as cfg
from spt.inference.san import SANv2
from spt.load_photometry import load_simulated_data, get_norm_theta, load_catalogue


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
    ip.logging_frequency = 100

    P = SANv2(sp)
    logging.info(f'Initialised {P.name} model as approximate posterior')
    logging.info(P)

    train_loader, test_loader = load_simulated_data(
        path=ip.dataset_loc,
        split_ratio=ip.split_ratio,
        batch_size=sp.batch_size,
        phot_transforms=[lambda x: t.from_numpy(np.log(x))],
        theta_transforms=[get_norm_theta(fp)],
    )

    P.offline_train(train_loader, ip)

    logging.info(f'Exiting')


    # Train the neural likelihood ---------------------------------------------
    #
    # Run the HMC update procedure on simulated data --------------------------
    #
    # Run the HMC update procedure on real data (no augmentation) -------------
    #
    # Run the full parameter estimation procedure -----------------------------

