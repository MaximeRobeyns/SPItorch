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
"""Run simulation with MPI"""

import os
import numpy as np
import signal
import logging

from time import sleep
from typing import Union, Optional
from threading import Lock
from rich.progress import Progress
from multiprocessing import Manager, Process

from spt.config import ForwardModelParams, SamplingParams
from spt.modelling.simulation import Simulator, Status
from spt.load_photometry import save_sim, join_partial_results


status_t = tuple[int, Status, int]  # rank, Status, N


class TSStatus:
    def __init__(self, val: Status = Status.STARTING):
        self._lock = Lock()
        self.value: Status = val
        self.n = 0

    def update(self, new: Status, n: Optional[int] = None):
        self._lock.acquire()
        try:
            self.value = new
            if n is not None:
                self.n = n
        finally:
            self._lock.release()


def sim_func(status, idx: int):
    # status is a ListProxy

    # silence all non-error logs:
    log = logging.getLogger()
    log.setLevel(logging.ERROR)

    sp = SamplingParams()
    N = sp.n_samples // sp.concurrency

    sim = Simulator(ForwardModelParams, sp.observation)

    status[0] = Status.LOADED

    theta = np.zeros((N, sim.dim))
    phot = np.zeros((N, sim.phot_dim))

    for i in range(N):
        theta[i], phot[i] = sim.simulate_sample()
        if i % 10 == 0:
            status[0] = Status.SAMPLED
            status[1] = i

    status[0] = Status.SAVING
    status[1] = N

    save_path = os.path.join(sp.save_dir, f'photometry_sim_{N}_{idx}.h5')
    assert isinstance(sim.obs['phot_wave'], np.ndarray)
    save_sim(save_path, theta, sim.model.free_params, phot, sim.obs['phot_wave'])

    # set the status to done before exiting.
    status[0] = Status.DONE
    status[1] = N


def main():

    from mpi4py import MPI
    from spt.utils import splash_screen

    MPIComm = Union[MPI.Intracomm, MPI.Intercomm]

    def root_func(mpi_comm: MPIComm):

        splash_screen()
        sp = SamplingParams()
        C = sp.concurrency
        N = sp.n_samples // C

        if not os.path.exists(sp.save_dir):
            os.makedirs(sp.save_dir)
            logging.info(f'Created results directory {sp.save_dir}')

        sm = Manager()
        status = sm.list([Status.STARTING, 0])

        p = Process(target=sim_func, args=(status, mpi_comm.Get_rank()))
        p.start()

        def interrupt(signum, _):
            logging.error(f'Rank {mpi_comm.Get_rank} received signal {signum}')
            p.terminate()
            mpi_comm.bcast(True)

        map((lambda x: signal.signal(x, interrupt)),
                [signal.SIGTERM, signal.SIGINT, signal.SIGKILL, signal.SIGTRAP])

        logging.info(f'Creating a dataset of size {sp.n_samples} across {C} workers')

        if not os.path.exists(sp.save_dir):
            os.makedirs(sp.save_dir)
            logging.info(f'Created results directory {sp.save_dir}')

        with Progress() as progress:
            tasks = []
            for t in range(sp.concurrency):
                tasks.append(progress.add_task(f'[red]Starting    {t:02}', total=N, start=False))

            done: int = 0
            while done < sp.concurrency:
                sleep(1)

                mpi_comm.bcast(False)

                done = 0
                # all_status = mpi_comm.gather(None)
                all_status = mpi_comm.gather((0, status[0], status[1]))
                assert isinstance(all_status, list)

                for s in all_status:
                    assert len(s) == 3, "Bad status update"
                    tidx, ts, tn = s
                    if ts == Status.STARTING:
                        progress.update(tasks[tidx], completed=tn,
                                        description= f'[red]Starting    {tidx:02}')
                    elif ts == Status.LOADED:
                        progress.reset(tasks[tidx], completed=tn, start=True,
                                       description=f'[dark_orange]Loading SPS {tidx:02}')
                    elif ts == Status.SAMPLED:
                        progress.update(tasks[tidx], completed=tn,
                                        description=f'[green]Running     {tidx:02}')
                    elif ts == Status.SAVING:
                        progress.update(tasks[tidx], completed=tn,
                                        description=f'[green]Saving       {tidx:02}')
                    elif ts == Status.DONE:
                        progress.update(tasks[tidx], completed=tn,
                                        description=f'[white]Done        {tidx:02}')
                        done += 1

        # send TERM signal
        mpi_comm.bcast(True)

        logging.info(f'Joining {sp.concurrency} partial result files')

        join_partial_results(sp.save_dir, sp.n_samples, sp.concurrency)

    def work_func(mpi_comm: MPIComm):

        sm = Manager()
        status = sm.list([Status.STARTING, 0])

        idx = mpi_comm.Get_rank()
        p = Process(target=sim_func, args=(status, idx))
        p.start()

        TERM = False
        while not TERM:
            # block on poll request from root rank
            TERM = mpi_comm.bcast(None)

            # return the status. Note, we may send DONE multiple times.
            mpi_comm.gather((idx, status[0], status[1]))

        if p.is_alive():
            p.terminate()

    mpi_comm: MPIComm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    if mpi_size == 1:
        logging.warning((
            'Running MPI with only 1 rank.\n'
            'Try increasing SamplingParams.concurrency or check the mpi launch'
            ' command (e.g. mpirun or similar).'))
        raise RuntimeError('Only one MPI rank.')

    if mpi_rank == 0:
        root_func(mpi_comm)
    else:
        work_func(mpi_comm)


if __name__ == '__main__':

    main()
