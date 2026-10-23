#!/bin/bash
#SBATCH -c 4  # Number of Cores per MPI Task
#SBATCH --job-name=kalypso-build  # Number of Cores per MPI Task
#SBATCH -N 1 # Number of Nodes
#SBATCH --account=pi_ece590d_690d_umass_edu
#SBATCH --mem=30G  # Requested Memory Per Node
#SBATCH -p cpu  # Partition
#SBATCH -t 01:00:00  # Job time limit
#SBATCH -o /scratch4/workspace/ypathak_umass_edu-acads/sysfml/kalypso-cpu/slurm-scripts/output/slurm-%j.out  # %j = job ID

cd /scratch4/workspace/ypathak_umass_edu-acads/sysfml/kalypso-cpu/
source .venv/bin/activate
VLLM_TARGET_DEVICE=cpu uv pip install -e . --no-build-isolation
