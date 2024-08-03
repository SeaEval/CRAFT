#!/bin/bash
#SBATCH --job-name=craft
#SBATCH --account=project_462000514
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --gpus-per-node=1
#SBATCH --partition=small-g


srun singularity exec -B /scratch/project_462000514:/scratch/project_462000514 /scratch/project_462000514/wangbin/workspaces/container/singularity_seaeval_v4.sif bash scripts/keyword_filter.sh singapore $1
