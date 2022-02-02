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

import os
import h5py
import logging
import numpy as np

from enum import Enum
from rich.progress import Progress
from typing import Type

from spt.load_photometry import load_galaxy
from multiprocessing import Process, Queue

from spt.config import ForwardModelParams, SamplingParams


class Simulator:

    def __init__(self, fmp: Type[ForwardModelParams] = ForwardModelParams,
                 use_galaxy: bool = True):

        mp = fmp()
        g = None
        if use_galaxy:
            # Uses catalogue in InferenceParams
            g, _ = load_galaxy(filters=mp.filters)

        self.obs = mp.build_obs_fn(mp.filters, g)
        self.model = mp.build_model_fn(mp.all_params, mp.ordered_params)
        self.sps = mp.build_sps_fn(**mp.sps_kwargs)

        self.dim = len(mp.free_params)
        self.phot_dim = len(fmp.filters)

        self.priors = []
        for p in self.model.free_params:
            self.priors.append(self.model.config_dict[p]['prior'])

    def draw_random_theta(self) -> np.ndarray:
        """Draws a theta parameter vector from the priors

        Inexplicably, the priors don't allow you to draw more than one sample
        at a time...

        TODO: remove this loop from the hot path
        """
        samples = []
        for p in self.priors:
            samples.append(p.sample())
        return np.hstack(samples)

    def simulate_sample(self) -> tuple[np.ndarray, np.ndarray]:
        theta = self.draw_random_theta()
        _, phot, _ = self.model.sed(theta, obs=self.obs, sps=self.sps)
        return theta, phot


class Status(Enum):
    STARTING = 1
    LOADING = 2
    SAMPLED = 3
    SAVING = 4
    DONE = 5


def work_func(idx: int, n: int, q: Queue, fmp: Type[ForwardModelParams],
              save_dir: str, logging_freq: int = 10) -> None:
    q.put((idx, Status.STARTING, 0))

    sim = Simulator(fmp, False)

    theta = np.zeros((n, sim.dim))
    phot = np.zeros((n, sim.phot_dim))

    q.put((idx, Status.LOADING, 0))

    for i in range(n):
        theta[i], phot[i] = sim.simulate_sample()
        if i % logging_freq == 1:
            q.put((idx, Status.SAMPLED, i))

    q.put((idx, Status.SAVING, n))

    save_path = os.path.join(save_dir, f'photometry_sim_{n}_{idx}.h5')
    with h5py.File(save_path, 'w') as f:
        grp = f.create_group('samples')

        ds_x = grp.create_dataset('theta', data=theta)
        ds_x.attrs['columns'] = sim.model.free_params
        ds_x.attrs['description'] = 'Parameters used by simulator'

        ds_y = grp.create_dataset('simulated_y', data=phot)
        ds_y.attrs['description'] = 'Response of simulator'

        # Wavelengths at for each of the simulated_y
        ds_wl = grp.create_dataset('wavelengths', data=sim.obs['phot_wave'])
        ds_wl.attrs['description'] = 'Effective wavelengths for each of the filters'

    q.put((idx, Status.DONE, n))


if __name__ == '__main__':

    sp = SamplingParams()

    C = sp.concurrency
    N = sp.n_samples // C
    status_q = Queue()

    logging.info(f'Creating a dataset size {sp.n_samples} across {C} workers')
    logging.info(f'Note: process setup can take a few minutes')

    for p in range(C):
        Process(target=work_func,
                args=(p, N, status_q, ForwardModelParams, sp.save_dir))

    # silence all non-error logs:
    log = logging.getLogger()
    l = log.getEffectiveLevel()
    log.setLevel(logging.ERROR)
    with Progress() as progress:

        tasks = []

        for p in range(C):
            Process(target=work_func,
                    args=(p, N, status_q, ForwardModelParams, sp.save_dir)).start()
            tasks.append(progress.add_task(f'[red]Starting {p:02}', total=N))

        done = 0
        while done < C:
            (idx, status, n) = status_q.get()
            if status == Status.LOADING:
                progress.update(tasks[idx], completed=n,
                                description=f'[blue]Loading  {idx:02}')
            elif status == Status.SAMPLED:
                progress.update(tasks[idx], completed=n,
                                description=f'[green]Running  {idx:02}')
            elif status == Status.SAVING:
                progress.update(tasks[idx], completed=n,
                                description=f'[green]Saving   {idx:02}')
            elif status == Status.DONE:
                progress.update(tasks[idx], completed=n,
                                description=f'[white]Done     {idx:02}')
                done += 1
    log.setLevel(l)

    # join and save samples
    # utils.join_partial_samples(sp)
