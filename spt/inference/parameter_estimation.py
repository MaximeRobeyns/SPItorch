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
"""Estimate parameters for a catalogue of real observations."""

import sys
import h5py
import math
import logging
import torch as t
import numpy as np

from tqdm import tqdm

from spt.types import Tensor
from spt.inference.san import SAN
from spt.load_photometry import get_denorm_theta, load_catalogue

def dcn(x: Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()

if __name__ == '__main__':
    """
    - Load a trained model
    - load a catalogue
    - estimate median and mode of parameters
    - save as hdf5 file
    """

    import spt.config as cfg

    # Load a trained model -----------------------------------------------------

    ip = cfg.InferenceParams()
    mp = cfg.SANParams()
    fp = cfg.ForwardModelParams()

    Q = SAN(mp)

    savepath: str = Q.fpath(ip.hmc_update_real_ident)
    try:
        logging.info(
            f'Attempting to load {Q.name} model from {savepath}')
        Q.load_state_dict(t.load(savepath))
        Q.is_trained = True
        logging.info('Successfully loaded')
        Q.to(Q.device, Q.dtype)
        Q.eval()
    except ValueError:
        logging.info(
            f'Could not load model at {savepath}. Exiting...')
        sys.exit()

    # Load the catalogue from file --------------------------------------------
    catalogue = load_catalogue(ip.catalogue_loc, fp.filters, True)
    required_cols = [f.maggie_col for f in fp.filters]
    xs_np = np.log(catalogue[required_cols].values)
    xs = t.from_numpy(xs_np)

    N: int = 1000     # samples per posterior
    bs: int = 1200    # batch size
    batches: int = math.ceil((xs.shape[0] / bs))

    medians = []
    modes = []

    logging.info(f'{batches} batches to process')

    with t.inference_mode():
        for b in tqdm(range(batches)):
            batch_xs, _ = Q.preprocess(xs[b*bs: (b+1)*bs], t.empty(0))
            tmp_sample = Q.forward(batch_xs.repeat_interleave(N, 0)).reshape(-1, N, mp.data_dim)
            medians.append(dcn(tmp_sample.median(1)[0]))
            # medians.append(dcn(Q.sample(batch_xs, N).median(-1)[0]))
            modes.append(dcn(Q.mode(batch_xs, N)))

    median = np.concatenate(medians, 0)
    mode = np.concatenate(modes, 0)

    logging.info(f'median shape: {median.shape}, mode shape: {mode.shape}')

    t.cuda.empty_cache()

    values = np.concatenate((median, mode), 1)
    cat = ip.catalogue_loc.split('/')[-1].split('.')[0]
    t.save(t.from_numpy(values), f'./results/params/{cat}_{ip.ident}_norm.pt')

    # Save results to HDF5 ----------------------------------------------------


    pcols = [f'{c}_median' for c in fp.ordered_free_params] + \
            [f'{c}_mode' for c in fp.ordered_free_params]

    dt = get_denorm_theta(fp)
    denorm_median = dt(values[:, :mp.data_dim])
    denorm_mode = dt(values[:, mp.data_dim:])

    denorm_params = np.concatenate((denorm_median, denorm_mode), 1)

    save_path = f'./results/params/{cat}_{ip.ident}.h5'
    with h5py.File(save_path, 'w') as f:
        grp = f.create_group(cat)

        phot = grp.create_dataset('photometry', data=catalogue.values)
        phot.attrs['columns'] = list(catalogue.columns)
        phot.attrs['description'] = f'''
    Photometric observations from {ip.catalogue_loc}
'''

        pars = grp.create_dataset('parameter_estimates', data=denorm_params)
        pars.attrs['columns'] = pcols
        pars.attrs['description'] = f'''
    Median and mode estimates for the galaxy parameters.

    Model parameter configuration used:

    {mp}

    Free parameter configuration used:

    {fp}
'''
