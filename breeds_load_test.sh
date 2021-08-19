#!/bin/bash
#SBATCH -p rise # partition (queue)
#SBATCH -N 1 # number of nodes requested
#SBATCH -n 1 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=12 # number of cores per task
#SBATCH --gres=gpu:1
##SBATCH -o /work/vsrivatsa/slurm_logs/simclr_out.%N.%j.out # STDOUT
#SBATCH --nodelist=luigi # if you need specific nodes
##SBATCH --exclude=luigi
#SBATCH -t 13-00:00 # time requested (D-HH:MM)

#SBATCH -o breeds_basic_1.%N.%j.out # STDOUT
#SBATCH -D /data/vsrivatsa/overparam_spur_corr

pwd
hostname
date

echo starting job...

source ~/.bashrc
conda activate breeds

#pip install tensorboard_logger torch
export PYTHONUNBUFFERED=1

python3 run_expt.py --shift_type confounder --dataset Breeds --train_from_scratch --lr .1 --target_name breeds --save_step 50 --n_epochs 300 --batch_size 128 --weight_decay 1.00E-04 --model resnet18 --subsample_to_minority --log_dir ./log/basic_breeds --reload_breeds_groups True
