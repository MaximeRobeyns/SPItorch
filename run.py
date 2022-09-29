# Entrypoint for the application.

import hydra
import logging
import torch.distributed as dist

from omegaconf import DictConfig

from spt import utils
from spt import ddp_utils

log = logging.getLogger(__name__)


# def setup(cfg: DictConfig) -> None:
#     dist.init_process_group(
#         backend=cfg.dist.backend, rank=cfg.dist.rank, world_size=cfg.dist.world_size
#     )


@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:

    # Initialis ddp
    comm = ddp_utils.setup(cfg)
    # optional synchronisation
    if comm is not None:
        comm.synchronise()

    # Initialise hydra run
    utils.init_hydra_run(cfg, comm)

    # Dispatch a task
    if cfg.run.task == "smoke_test":
        print(
            f"Running smoke test on rank {dist.get_rank()} ",
            f"with variable `foo: {cfg.smoketest.foo}`",
        )
    else:
        raise ValueError(f"Unrecognized task in configuration: {cfg.run.task}")

    ddp_utils.teardown(cfg, comm)


if __name__ == "__main__":
    main()
