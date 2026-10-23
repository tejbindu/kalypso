#!/bin/bash
#SBATCH -c 4  # Number of Cores per MPI Task
#SBATCH --job-name=kalypso-run  # Number of Cores per MPI Task
#SBATCH -N 1 # Number of Nodes
#SBATCH --account=pi_ece590d_690d_umass_edu
#SBATCH --mem=30G  # Requested Memory Per Node
#SBATCH -p cpu  # Partition
#SBATCH -t 00:10:00  # Job time limit
#SBATCH -o /scratch4/workspace/ypathak_umass_edu-acads/sysfml/kalypso-cpu/slurm-scripts/output/slurm-%x-%j.out  # %j = job ID

cd /scratch4/workspace/ypathak_umass_edu-acads/sysfml/kalypso-cpu/
source .venv/bin/activate

VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_CPU_OMP_THREADS_BIND=nobind \
OMP_NUM_THREADS=4 \
VLLM_CPU_KVCACHE_SPACE=4 \
vllm serve meta-llama/Llama-3.2-1B \
  --tensor-parallel-size 4 \
  --enable-prefix-caching \
  --max-model-len 32768 \
  --port 8003

