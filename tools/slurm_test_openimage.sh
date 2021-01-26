#!/usr/bin/env bash

set -x

PARTITION=$1
JOB_NAME=$2
CONFIG=$3
CHECKPOINT=$4
GPUS=${GPUS:-16}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-3}
PY_ARGS=${@:5}
SRUN_ARGS=${SRUN_ARGS:-""}

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/test_openimages.py ${CONFIG} ${CHECKPOINT} --launcher="slurm" ${PY_ARGS} --eval bbox --label_dir=./data/OpenImages/challenge2019 --det_file=./work_dir/r50-FPN-1x_classsampling_TSD/dump.txt --tmpdir=./tmp1
