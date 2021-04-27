#!/bin/bash
#SBATCH -p rise # partition (queue)
#SBATCH -N 1 # number of nodes requested
#SBATCH -n 1 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=16 # number of cores per task
#SBATCH --gres=gpu:1
#SBATCH --nodelist=como # if you need specific nodes
##SBATCH --exclude=ace,blaze,bombe,flaminio,freddie,luigi,pavia,r[10,16],atlas,como,havoc
#SBATCH -t 0-10:00 # time requested (D-HH:MM)

#SBATCH -D /work/alanpham/overparam_spur_corr

pwd
hostname
date

echo starting job...

source ~/.bashrc
conda activate overparam_spur_corr

export PYTHONUNBUFFERED=1

# python run_expt.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --batch_size 16 --model barlowtwins_resnet50 --n_epochs 300 --save_step 100 --reweight_groups --gamma 0.1 --generalization_adjustment 0 --log_dir ./logs_cutmix/test2
python run_expt.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --batch_size 16 --model resnet101 --n_epochs 300 --save_step 100 --reweight_groups --gamma 0.1 --generalization_adjustment 0 --log_dir ./logs_cutmix/test2
# python run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --lr 0.01 --batch_size 128 --weight_decay 0.0001 --model resnet152 --n_epochs 50 --reweight_groups --train_from_scratch --resnet_width 32

date
