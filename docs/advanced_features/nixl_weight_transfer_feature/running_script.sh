#!/bin/bash
 
if [ "$1" == "network" ]; then
    PORTFOLIO=network_research_advdev
elif [ "$1" == "trtllm" ]; then
    PORTFOLIO=coreai_comparch_trtllm
elif [ "$1" == "triton" ]; then
    PORTFOLIO=coreai_tritoninference_triton3
else
    echo "Error: Invalid portfolio. Use 'network', 'trtllm' or 'triton'"
    exit 1
fi
 
PARTITION=interactive
TIME=02:00:00
 
USER_TRIMMED=$USER
BASEDIR=/lustre/fsw/portfolios/network/users/$USER/
WORKSPACE=/sgl-workspace/
SUBPROJECT=weight_transfer
JOB_NAME=$PORTFOLIO-$SUBPROJECT.dev
C_NAME=$SUBPROJECT.dev_$USER
C_IMAGE=docker://lmsysorg/sglang:nightly-dev-cu13-20260709-074bb928
 

C_CONTAINER_SAVED_NAME=/lustre/fsw/portfolios/network/users/amitw/sglang/sglang_amit.sqsh

SLURM_PATHS+=/lib/x86_64-linux-gnu/libmunge.so.2,
SLURM_PATHS+=/run/munge,
SLURM_PATHS+=/etc/slurm,
SLURM_PATHS+=/cm/shared/apps/slurm/current:/opt/slurm
 
 
MOUNTS+=$SLURM_PATHS,
MOUNTS+=$BASEDIR:$WORKSPACE/lustre,
MOUNTS+=/home/$USER:$WORKSPACE/home,
#MOUNTS+=$BASEDIR/llm_models:$WORKSPACE/llm_models,
MOUNTS+=/lustre/fsw/portfolios/network/users/bbiber/llm_models:$WORKSPACE/llm_models,
MOUNTS+=/lustre:/lustre
srun \
    -A $PORTFOLIO \
    -N 1 \
    -p $PARTITION \
    --gpus-per-node=8 \
    -J $JOB_NAME \
    --container-image=$C_IMAGE \
    --container-save=$C_CONTAINER_SAVED_NAME \
    --container-mounts=$MOUNTS \
    --time=$TIME \
    --pty bash