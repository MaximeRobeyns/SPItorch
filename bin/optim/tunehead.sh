#!/bin/bash

# Copyright (C) 2022 Maxime Robeyns <ez18285@bristol.ac.uk>
#
# Script to launch the ray head node.

cd $SLURM_SUBMIT_DIR

module load lang/python/anaconda/3.9.7-2021.12-tensorflow.2.7.0
source /user/work/ez18285/spivenv/bin/activate

srun --nodes=1 --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK} --nodelist=$1 \
    ray start --head --block --port=6379 --num-cpus ${SLURM_CPUS_PER_TASK} &

# TODO: --num-gpus here?

# Sleep to give SLURM the time to allocate resources, and for the head node to
# register before trying to launch the worker processes.
sleep 5
