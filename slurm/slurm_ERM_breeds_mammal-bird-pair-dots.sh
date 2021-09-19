#!/bin/bash
#SBATCH -p rise # partition (queue)
#SBATCH -N 1 # number of nodes requested
#SBATCH -n 5 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=5 # number of cores per task
#SBATCH --gres=gpu:5
#SBATCH --nodelist=luigi # if you need specific nodes
##SBATCH --exclude=ace,blaze,bombe,flaminio,freddie,luigi,pavia,r[10,16],atlas,como,havoc,steropes
#SBATCH -t 7-00:00 # time requested (D-HH:MM)

#SBATCH -o slurm.%j.%N.ERM_breeds.mammal-bird-pair-dots.out # STDOUT
#SBATCH -e slurm.%j.%N.ERM_breeds.mammal-bird-pair-dots.err # STDERR

#SBATCH -D /data/vsrivatsa/overparam_spur_corr/slurm/output
cd /data/vsrivatsa/overparam_spur_corr

pwd
hostname
date

echo starting job...

source ~/.bashrc
conda activate breeds

export PYTHONUNBUFFERED=1

job_limit () {
    # author: https://stackoverflow.com/posts/33048123/timeline#history_be44a66e-9fa4-48ea-b4dd-b66a14ffef03
    # Test for single positive integer input
    if (( $# == 1 )) && [[ $1 =~ ^[1-9][0-9]*$ ]]
    then

        # Check number of running jobs
        joblist=($(jobs -rp))
        while (( ${#joblist[*]} >= $1 ))
        do

            # Wait for any job to finish
            command='wait '${joblist[0]}
            for job in ${joblist[@]:1}
            do
                command+=' || wait '$job
            done
            eval $command
            joblist=($(jobs -rp))
        done
   fi
}

srun -N 1 -n 1 --gres=gpu:1 python run_expt.py --shift_type confounder --dataset Breeds --target_name breeds --save_step 10 --save_last --save_best --lr .1 --step_scheduler --scheduler_step 120 --n_epochs 300 --batch_size 128 --weight_decay 1.00E-04 --model resnet18 --train_from_scratch --gamma 0.1 --breeds_color_dots --generalization_adjustment 0 --breeds_dataset_type entity13 --log_dir ./logs/ERM_breeds/mammal-bird-pair-dots/fbe0dbce-16ac-441b-b3e2-88be1707e2a8 &
job_limit 5

srun -N 1 -n 1 --gres=gpu:1 python run_expt.py --shift_type confounder --dataset Breeds --target_name breeds --save_step 10 --save_last --save_best --lr .1 --step_scheduler --scheduler_step 120 --n_epochs 300 --batch_size 128 --weight_decay 1.00E-04 --model resnet34 --train_from_scratch --gamma 0.1 --breeds_color_dots --generalization_adjustment 0 --breeds_dataset_type entity13 --log_dir ./logs/ERM_breeds/mammal-bird-pair-dots/19707476-ad73-4856-9007-e9db777adaf9 & 
job_limit 5

srun -N 1 -n 1 --gres=gpu:1 python run_expt.py --shift_type confounder --dataset Breeds --target_name breeds --save_step 10 --save_last --save_best --lr .1 --step_scheduler --scheduler_step 30 --n_epochs 100 --batch_size 128 --weight_decay 1.00E-04 --model resnet18 --train_from_scratch --gamma 0.1 --breeds_color_dots --generalization_adjustment 0 --breeds_dataset_type entity13 --log_dir ./logs/ERM_breeds/mammal-bird-pair-dots/0dfc15e9-13c3-4101-a19e-0acdfba68692 & 
job_limit 5

srun -N 1 -n 1 --gres=gpu:1 python run_expt.py --shift_type confounder --dataset Breeds --target_name breeds --save_step 10 --save_last --save_best --lr 0.5 --step_scheduler --scheduler_step 60 --n_epochs 300 --batch_size 128 --weight_decay 1.00E-04 --model resnet34 --train_from_scratch --gamma 0.1 --breeds_color_dots --generalization_adjustment 0 --breeds_dataset_type entity13 --log_dir ./logs/ERM_breeds/mammal-bird-pair-dots/3730bc2f-314a-42c5-b3a4-af9902a16e17 & 
job_limit 5

srun -N 1 -n 1 --gres=gpu:1 python run_expt.py --shift_type confounder --dataset Breeds --target_name breeds --save_step 10 --breeds_proportions 3498,184,56,1057,467,466,133,133,2255,2255,642,642 --save_last --save_best --lr .1 --step_scheduler --scheduler_step 120 --n_epochs 300 --batch_size 128 --weight_decay 1.00E-04 --model resnet18 --train_from_scratch --gamma 0.1 --breeds_color_dots --generalization_adjustment 0 --breeds_dataset_type entity13 --log_dir ./logs/ERM_breeds/mammal-bird-pair-dots/ef1a309a-09b5-4925-a473-4857474dc8ee & 
job_limit 5

srun -N 1 -n 1 --gres=gpu:1 python run_expt.py --shift_type confounder --dataset Breeds --target_name breeds --save_step 10 --breeds_proportions 3498,184,56,1057,467,466,133,133,2255,2255,642,642 --save_last --save_best --lr .1 --step_scheduler --scheduler_step 120 --n_epochs 300 --batch_size 128 --weight_decay 1.00E-04 --model resnet34 --train_from_scratch --gamma 0.1 --breeds_color_dots --generalization_adjustment 0 --breeds_dataset_type entity13 --log_dir ./logs/ERM_breeds/mammal-bird-pair-dots/1b7fe9a2-cc76-43cf-9197-fe8faceed81a & 
job_limit 5

srun -N 1 -n 1 --gres=gpu:1 python run_expt.py --shift_type confounder --dataset Breeds --target_name breeds --save_step 10 --breeds_proportions 441,10676,9918,8258,1270,1528,1439,1147,2531,3050,3846,2341 --save_last --save_best --lr .1 --step_scheduler --scheduler_step 120 --n_epochs 300 --batch_size 128 --weight_decay 1.00E-04 --model resnet18 --train_from_scratch --gamma 0.1 --breeds_color_dots --generalization_adjustment 0 --breeds_dataset_type entity13 --log_dir ./logs/ERM_breeds/mammal-bird-pair-dots/18277309-f687-4ef5-ab8b-18b542265609 & 
job_limit 5

srun -N 1 -n 1 --gres=gpu:1 python run_expt.py --shift_type confounder --dataset Breeds --target_name breeds --save_step 10 --breeds_proportions 441,10676,9918,8258,1270,1528,1439,1147,2531,3050,3846,2341 --save_last --save_best --lr .1 --step_scheduler --scheduler_step 120 --n_epochs 300 --batch_size 128 --weight_decay 1.00E-04 --model resnet34 --train_from_scratch --gamma 0.1 --breeds_color_dots --generalization_adjustment 0 --breeds_dataset_type entity13 --log_dir ./logs/ERM_breeds/mammal-bird-pair-dots/1871ad8e-7b94-481c-a12e-fd9c405d7d49 & 
job_limit 5

srun -N 1 -n 1 --gres=gpu:1 python run_expt.py --shift_type confounder --dataset Breeds --target_name breeds --save_step 10 --breeds_proportions 0,10676,9918,8258,1270,1528,1439,1147,2531,3050,3846,2341 --save_last --save_best --lr .1 --step_scheduler --scheduler_step 120 --n_epochs 300 --batch_size 128 --weight_decay 1.00E-04 --model resnet18 --train_from_scratch --gamma 0.1 --breeds_color_dots --generalization_adjustment 0 --breeds_dataset_type entity13 --log_dir ./logs/ERM_breeds/mammal-bird-pair-dots/6b5d2406-8729-46da-9dfb-8d9e23976c94 & 
job_limit 5

srun -N 1 -n 1 --gres=gpu:1 python run_expt.py --shift_type confounder --dataset Breeds --target_name breeds --save_step 10 --breeds_proportions 40,10676,9918,8258,1270,1528,1439,1147,2531,3050,3846,2341 --save_last --save_best --lr .1 --step_scheduler --scheduler_step 120 --n_epochs 300 --batch_size 128 --weight_decay 1.00E-04 --model resnet34 --train_from_scratch --gamma 0.1 --breeds_color_dots --generalization_adjustment 0 --breeds_dataset_type entity13 --log_dir ./logs/ERM_breeds/mammal-bird-pair-dots/b213ffea-d74d-43b5-9655-cc0de3c0a00c 

wait
date
