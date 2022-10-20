#!/bin/bash

# Copyright (C) 2022 Maxime Robeyns <ez18285@bristol.ac.uk>
#
# Script to launch the ray worker node.
# Args:
#  1. the number of tasks to run on this worker node
#  2. the name of the host node so that it can be excluded from the worker
#     pool.

cd $SLURM_SUBMIT_DIR

module load lang/python/anaconda/3.9.7-2021.12-tensorflow.2.7.0
source /user/work/ez18285/spivenv/bin/activate

srun --nodes=$1 --ntasks=$1 --cpus-per-task=${SLURM_CPUS_PER_TASK} --exclude=$2 \
    ray start --address $ip_head --block --num-cpus ${SLURM_CPUS_PER_TASK} &
# TODO: specify --num-gpus here?

sleep 5
