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
"""Project-wide utilities file"""

import os
import sys
import time
import torch as t
import numpy as np
import pprint
import semver
import logging

from enum import Enum
from typing import Any, Sized, Optional
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, random_split, Subset

from spt.types import Tensor
from spt.__version__ import __version__
from spt.ddp_utils import DDP_IPC


class ConfigClass:
    """ConfigClass is an abstract base class for all SPItorch configuration
    objects.

    TODO: use rich to provide clearer class representations.
    """

    def __init__(self) -> None:
        logging.debug(f"New configuration object: {self}")

    def __repr__(self) -> str:
        width = 60
        r = f'\n\n{width*"="}\n'
        c = f"Configuration class `{type(self).__name__}`"
        n = len(c)
        nn = int((width - n) / 2)
        r += nn * " " + c + f'\n{width*"-"}\n\n'
        members = [
            a
            for a in dir(self)
            if not callable(getattr(self, a)) and not a.startswith("__")
        ]
        for m in members:
            r += f"{m}: {pprint.pformat(getattr(self, m), compact=True)}\n"
            # r += f'{m}: {pprint(getattr(self, m), max_length=80)}\n'
        r += "\n" + width * "=" + "\n\n"
        return r

    def to_dict(self) -> dict[str, Any]:
        members = [
            a
            for a in dir(self)
            if not callable(getattr(self, a)) and not a.startswith("__")
        ]
        d = {}
        for m in members:
            d[m] = getattr(self, m)
            if isinstance(d[m], Enum):
                d[m] = d[m].value
        return d


# Miscellaneous ==============================================================


colours = {
    "b": "#025159",  # blue(ish)
    "o": "#F28705",  # orange
    "lb": "#03A696",  # light blue
    "do": "#F25D27",  # dark orange
    "r": "#F20505",  # red.
}


# Splash screen ==============================================================


def splash_screen():
    import logging.handlers as lhandlers
    from spt import __version__
    from rich.padding import Padding
    from rich.console import Console

    console = Console(width=80)
    console.rule()
    info = Padding(
        f"""
    SPItorch

    Version: {__version__}, {time.ctime()}
    Copyright (C) 2019-20 Mike Walmsley <walmsleymk1@gmail.com>
    Copyright (C) 2022 Maxime Robeyns <dev@maximerobeyns.com>
            """,
        (1, 8),
    )
    console.print(info, highlight=False, markup=False)
    console.rule()

    lc = Console(record=True, force_terminal=False, width=80)
    lc.begin_capture()
    lc.rule()
    lc.print(info, highlight=False, markup=False)
    lc.rule()
    for h in logging.getLogger().handlers:
        if isinstance(h, lhandlers.RotatingFileHandler):
            r = logging.makeLogRecord(
                {
                    "msg": "\n" + lc.end_capture(),
                    "level": logging.INFO,
                }
            )
            h.handle(r)


def log_run_description(cfg: DictConfig):
    """Prints run description at root of output directory.

    Args:
        cfg: the Hydra config
    """
    if cfg.print_config:
        print(OmegaConf.to_yaml(cfg))

    if semver.compare(cfg.run.for_version, __version__) < 0:
        log = logging.getLogger(__name__)
        log.error(
            (
                f"\n\n\n\tConfiguration was written for version {cfg.run.for_version},"
                f"\n\tyet the current code version is {__version__}!"
                "\n\n\tPlease check that everything is still compatible,\n\tthen update "
                "the version number in your config.\n\n\n"
            )
        )
        sys.exit(1)

    with open("README.txt", "w") as f:
        f.write(cfg.run.description)


def init_hydra_run(cfg: DictConfig, comm: Optional[DDP_IPC]):
    """Send the current working directory (newly created on rank 0) to the
    other ranks to share logging / tensorboard directories."""
    if cfg.dist.world_size > 1:
        assert comm is not None
        if cfg.dist.rank == 0:
            comm.bcast(os.getcwd(), "CWD")
        else:
            os.chdir(comm.accept("CWD"))

    if cfg.dist.rank == 0:
        log_run_description(cfg)


def new_sample(dloader: DataLoader, n: int = 1) -> tuple[Tensor, Tensor]:
    dset: Dataset = dloader.dataset
    rand_idxs = t.randperm(len(dset))[:n]
    logging.debug("Random test index :", rand_idxs)
    # [n, data_dim]; concatenate along rows: dim 0
    xs, ys = [], []
    for i in rand_idxs:
        tmp_xs, tmp_ys = dset.__getitem__(i)
        if isinstance(tmp_xs, np.ndarray):
            tmp_xs = t.from_numpy(tmp_xs)
        if isinstance(tmp_ys, np.ndarray):
            tmp_ys = t.from_numpy(tmp_ys)
        xs.append(tmp_xs.unsqueeze(0))
        ys.append(tmp_ys.unsqueeze(0))
    return t.cat(xs, 0).squeeze(), t.cat(ys, 0).squeeze()


def train_test_split(
    dataset: Dataset, split_ratio: float = 0.9
) -> tuple[Subset, Subset]:
    assert isinstance(dataset, Sized)
    n_train = int(len(dataset) * split_ratio)
    n_test = len(dataset) - n_train
    train_set, test_set = random_split(dataset, [n_train, n_test])
    return train_set, test_set


def get_median_mode(
    samples: np.ndarray, nbins: int = 1000
) -> tuple[np.ndarray, np.ndarray]:
    # sample shape: (N x free_params)

    # compute median
    median = np.median(samples, 0)
    mode: list[np.ndarray] = []

    for i in range(samples.shape[1]):
        n, b = np.histogram(samples[:, i], nbins)
        m = np.argmax(n)
        mode.append((b[m] + b[m + 1]) / 2)
    np_mode = np.array(mode)

    return median, np_mode
