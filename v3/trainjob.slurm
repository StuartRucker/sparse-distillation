#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --nodes=2
#SBATCH --mem=200GB
####SBATCH --exclusive
#SBATCH --ntasks-per-node=1

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
echo "Run started at:- "
date

export MASTER_PORT=12340
export WORLD_SIZE=2

## User python environment
HOME2=/nobackup/users/$(whoami)
PYTHON_VIRTUAL_ENVIRONMENT=wmlce-1.7.0
CONDA_ROOT=$HOME2/anaconda3

# Activate WMLCE virtual environment
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
export NCCL_DEBUG=DEBUG
export NCCL_DEBUG_SUBSYS=ALL

echo $SLURM_LAUNCH_NODE_IPADDR


srun python main.py

echo "Run completed at:- "
date