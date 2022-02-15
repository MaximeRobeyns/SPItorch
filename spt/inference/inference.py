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
"""Actually run the inference procedure."""

import logging

from spt.load_photometry import load_simulated_data

if __name__ == '__main__':

    import spt.config as cfg

    ip = cfg.InferenceParams()

    # TODO load model parameters dynamically based on the model type
    mp = cfg.SANParams()
    model = ip.model(mp)
    logging.info(f'Initialised {model.name} model')

    train_loader, test_loader = load_simulated_data(
        path=ip.dataset_loc,
        split_ratio=ip.split_ratio,
        batch_size=model.params.batch_size,
        phot_transforms=[],
        theta_transforms=[],
    )

    logging.info('Created data loaders')

    model.offline_train()
    logging.info('Trained model')

    # TODO:
    #
    # 1. Train the model on a full dataset
    # 1.1 Draw a corner plot from one of the observations from the test set;
    # draw the 'real' parameters on top of the plot; this is to show that we
    # have been able to learn from the synthetic training data.
    #
    # => Run EMCEE inference on one of the real galaxies from a survey
    # => Draw the corner plot for these results (and estimate the real parameter values)
    #
    # 2. Run the SAN on the very same galaxy. Get the estimated parameters from
    # the samples, initialise a model, set the theta values to be mean
    # estimated parameters from the network, and plot the SEDs.
