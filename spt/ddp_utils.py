import os
import time
import logging
import torch as t
import torch.distributed as dist

from typing import Any, Callable, Optional, TypeVar, Union
from omegaconf import DictConfig
from multiprocessing.connection import Listener, Client, Connection


def log_dist(message: str, ranks: list[int] = [0], level: int = logging.INFO) -> None:
    """Logs messages on specified ranks only"""
    logger = logging.getLogger()
    my_rank = dist.get_rank()
    if my_rank in ranks:
        if level == logging.INFO:
            logger.info(f"[Rank {my_rank}] {message}")
        elif level == logging.ERROR:
            logger.error(f"[Rank {my_rank}] {message}")
        elif level == logging.DEBUG:
            logger.debug(f"[Rank {my_rank}] {message}")


T = TypeVar("T")


class DDP_IPC:
    def __init__(self, comms: list[Connection] | Connection, rank: int = 0) -> None:
        self.is_master = isinstance(comms, list)
        if isinstance(comms, list):
            self._comms: list[Connection] = comms
        else:
            self._comm: Connection = comms
        self.rank = rank

    @property
    def comms(self) -> list[Connection]:
        if self._comms is None:
            raise RuntimeError("Attempted to access comms list from worker")
        return self._comms

    @property
    def comm(self) -> Connection:
        if self._comm is None:
            raise RuntimeError("Attempted to access comm from master")
        return self._comm

    def close(self):
        self.synchronise()
        if not self.is_master:
            self.comm.close()

    def bcast(self, msg: Any, opcode: str = "") -> None:
        """Broadcast the object using the provided opcode to all ranks in the
        process group.

        Args:
            msg: any pickleable object
            opcode: an identifier for the message sent.

        Raises:
            RuntimeError: if called on a worker node.
        """
        if not self.is_master:
            raise RuntimeError("Only the master rank can broadcast to DDP workers")
        for c in self.comms:
            c.send((opcode, msg))

    def reduce_0(self, msg: T, op: Callable[[T, T], T]) -> Optional[T]:
        """Reduce values to rank 0.

        Returns reduced result if rank is 0, and the unmodified input otherwise.
        """
        if self.is_master:
            result: T = msg
            ts = time.time_ns()
            for c in self.comms:
                c.send(("REDUCE_REQ", ts))
            for c in self.comms:
                result = op(result, self.accept(str(ts), c))
            return result
        else:
            ts = self.accept("REDUCE_REQ")
            self.comm.send((str(ts), msg))
            return msg

    def accept(self, opcode: str = "", comm: Optional[Connection] = None):
        """Accept a message with a particular opcode from either the
        communication channel provided as argument (master, worker), or the
        default communication channel (worker only)."""
        if comm is None:
            comm = self.comm
        while True:
            msg_op, msg = comm.recv()
            if msg_op == opcode:
                return msg
            log_dist(
                "Unexpected opcode {msg_op} received on rank {self.rank}",
                [self.rank],
                logging.WARNING,
            )

    def synchronise(self):
        """Synchronises all the workers
        TODO: add timeouts and loop limits.
        """
        if self.is_master:
            # 1. receive timestamps from all workers, and send back to them.
            tss = {}
            for c in self.comms:
                rank, ts = self.accept("SYNC_REQ", c)
                tss[rank] = ts
            # 2. send acknowledgement to workers
            self.bcast(tss, "SYNC_ACK")

        else:
            # 1. Send the current timestamp to the master node
            ts = time.time_ns()
            self.comm.send(("SYNC_REQ", (self.rank, ts)))
            # 2. Await a response with opcode "SYNC_ACK" and the correct time
            while True:
                tss = self.accept("SYNC_ACK")
                if tss[self.rank] == ts:
                    break


def validate_dist_config(cfg: DictConfig) -> None:
    """
    Validates the DDP configuration variables.
    """
    assert dist.is_available(), "PyTorch distributed is not available."
    if cfg.dist.backend == "nccl":
        assert dist.is_nccl_available()
    elif cfg.dist.backend == "gloo":
        assert dist.is_gloo_available()
    elif cfg.dist.backend == "mpi":
        assert dist.is_mpi_available()


def setup(cfg: DictConfig) -> Optional[DDP_IPC]:
    """
    Setup the DDP run. This initialises a PyTorch process group, using TCP
    connections to rank 0.
    """

    validate_dist_config(cfg)

    dist.init_process_group(
        backend=cfg.dist.backend, rank=cfg.dist.rank, world_size=cfg.dist.world_size
    )

    # device = min(cfg.dist.rank, t.cuda.device_count() - 1)
    # t.cuda.set_device(device)
    assert dist.is_initialized(), "distributed backend not initialised!"

    if cfg.dist.rank == 0:
        # Get connections to all other ranks
        listener = Listener((cfg.dist.master_addr, cfg.dist.comm_port))
        return DDP_IPC([listener.accept() for _ in range(cfg.dist.world_size - 1)])
    else:
        for _ in range(1800):  # timeout of 30 mins
            try:
                addr = (cfg.dist.master_addr, cfg.dist.comm_port)
                return DDP_IPC(Client(addr), cfg.dist.rank)
            except:
                time.sleep(1)
        raise RuntimeError("IPC connection timeout")


def teardown(cfg: DictConfig, comm: Optional[DDP_IPC]) -> None:

    if comm is not None:
        comm.synchronise()
        if cfg.dist.rank == 0:
            for c in comm._comms:
                c.close()
            dist.destroy_process_group()

    print(f"Successfully finished program on rank {cfg.dist.rank}")
