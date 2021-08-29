#!/bin/bash
#SBATCH -p rise # partition (queue)
#SBATCH -N 1 # number of nodes requested
#SBATCH -n 5 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=5 # number of cores per task
#SBATCH --gres=gpu:5
#SBATCH --nodelist=luigi # if you need specific nodes
##SBATCH --exclude=ace,blaze,bombe,flaminio,freddie,luigi,pavia,r[10,16],atlas,como,havoc,steropes
#SBATCH -t 3-00:00 # time requested (D-HH:MM)

#SBATCH -o slurm.%j.%N.ERM_breeds.mammal-bird-pair-all.out # STDOUT
#SBATCH -e slurm.%j.%N.ERM_breeds.mammal-bird-pair-all.err # STDERR

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

srun -N 1 -n 1 --gres=gpu:1 python run_expt.py --shift_type confounder --dataset Breeds --target_name breeds --save_step 10 --save_last --save_best --lr .1 --step_scheduler --scheduler_step 120 --n_epochs 300 --batch_size 128 --weight_decay 1.00E-04 --model resnet18 --train_from_scratch --gamma 0.1 --generalization_adjustment 0 --breeds_dataset_type entity13 --log_dir ./logs/ERM_breeds/mammal-bird-pair-all/fac56640-fc8d-49f5-8950-d40478c4b5f8 & 
job_limit 5

srun -N 1 -n 1 --gres=gpu:1 python run_expt.py --shift_type confounder --dataset Breeds --target_name breeds --save_step 10 --save_last --save_best --lr .1 --step_scheduler --scheduler_step 120 --n_epochs 300 --batch_size 128 --weight_decay 1.00E-04 --model resnet34 --train_from_scratch --gamma 0.1 --generalization_adjustment 0 --breeds_dataset_type entity13 --log_dir ./logs/ERM_breeds/mammal-bird-pair-all/794c3a63-1f48-4a29-a8f5-e22161cdbe6d & 
job_limit 5

srun -N 1 -n 1 --gres=gpu:1 python run_expt.py --shift_type confounder --dataset Breeds --target_name breeds --save_step 10 --save_last --save_best --lr .1 --step_scheduler --scheduler_step 120 --n_epochs 300 --batch_size 128 --weight_decay 1.00E-04 --model resnet50 --train_from_scratch --gamma 0.1 --generalization_adjustment 0 --breeds_dataset_type entity13 --log_dir ./logs/ERM_breeds/mammal-bird-pair-all/d9f3a914-50a5-40dd-b699-9512a5e80daf & 
job_limit 5

srun -N 1 -n 1 --gres=gpu:1 python run_expt.py --shift_type confounder --dataset Breeds --target_name breeds --save_step 10 --save_last --save_best --lr .1 --step_scheduler --scheduler_step 120 --n_epochs 300 --batch_size 128 --weight_decay 1.00E-04 --model resnet101 --train_from_scratch --gamma 0.1 --generalization_adjustment 0 --breeds_dataset_type entity13 --log_dir ./logs/ERM_breeds/mammal-bird-pair-all/d7cd573e-692a-4a02-a92b-5bb4bdf6ced2 & 
job_limit 5

srun -N 1 -n 1 --gres=gpu:1 python run_expt.py --shift_type confounder --dataset Breeds --target_name breeds --save_step 10 --save_last --save_best --lr .1 --step_scheduler --scheduler_step 120 --n_epochs 300 --batch_size 128 --weight_decay 1.00E-04 --model resnet152 --train_from_scratch --gamma 0.1 --generalization_adjustment 0 --breeds_dataset_type entity13 --log_dir ./logs/ERM_breeds/mammal-bird-pair-all/66b926fa-600b-4fb5-a5c8-7e235fdbcb4b & 
job_limit 5

srun -N 1 -n 1 --gres=gpu:1 python run_expt.py --shift_type confounder --dataset Breeds --target_name breeds --save_step 10 --breeds_proportions 3498,184,56,1057,467,466,133,133,2255,2255,642,642 --save_last --save_best --lr .1 --step_scheduler --scheduler_step 120 --n_epochs 100 --batch_size 128 --weight_decay 1.00E-04 --model resnet18 --train_from_scratch --gamma 0.1 --generalization_adjustment 0 --breeds_dataset_type entity13 --log_dir ./logs/ERM_breeds/mammal-bird-pair-all/65f02539-aac1-4fa0-b3c0-22c04051cbf1 & 
job_limit 5

srun -N 1 -n 1 --gres=gpu:1 python run_expt.py --shift_type confounder --dataset Breeds --target_name breeds --save_step 10 --breeds_proportions 3498,184,56,1057,467,466,133,133,2255,2255,642,642 --save_last --save_best --lr .1 --step_scheduler --scheduler_step 120 --n_epochs 100 --batch_size 128 --weight_decay 1.00E-04 --model resnet34 --train_from_scratch --gamma 0.1 --generalization_adjustment 0 --breeds_dataset_type entity13 --log_dir ./logs/ERM_breeds/mammal-bird-pair-all/ed208bfc-dc12-4a11-b87b-df46cca7924e & 
job_limit 5

srun -N 1 -n 1 --gres=gpu:1 python run_expt.py --shift_type confounder --dataset Breeds --target_name breeds --save_step 10 --breeds_proportions 3498,184,56,1057,467,466,133,133,2255,2255,642,642 --save_last --save_best --lr .1 --step_scheduler --scheduler_step 120 --n_epochs 100 --batch_size 128 --weight_decay 1.00E-04 --model resnet50 --train_from_scratch --gamma 0.1 --generalization_adjustment 0 --breeds_dataset_type entity13 --log_dir ./logs/ERM_breeds/mammal-bird-pair-all/1e7c027b-4ce7-4b01-9a60-95ff0b12cf1f & 
job_limit 5

srun -N 1 -n 1 --gres=gpu:1 python run_expt.py --shift_type confounder --dataset Breeds --target_name breeds --save_step 10 --breeds_proportions 3498,184,56,1057,467,466,133,133,2255,2255,642,642 --save_last --save_best --lr .1 --step_scheduler --scheduler_step 120 --n_epochs 100 --batch_size 128 --weight_decay 1.00E-04 --model resnet101 --train_from_scratch --gamma 0.1 --generalization_adjustment 0 --breeds_dataset_type entity13 --log_dir ./logs/ERM_breeds/mammal-bird-pair-all/b45ef10a-8eb4-4b44-93ff-bdd8c2006507 & 
job_limit 5

srun -N 1 -n 1 --gres=gpu:1 python run_expt.py --shift_type confounder --dataset Breeds --target_name breeds --save_step 10 --breeds_proportions 3498,184,56,1057,467,466,133,133,2255,2255,642,642 --save_last --save_best --lr .1 --step_scheduler --scheduler_step 120 --n_epochs 100 --batch_size 128 --weight_decay 1.00E-04 --model resnet152 --train_from_scratch --gamma 0.1 --generalization_adjustment 0 --breeds_dataset_type entity13 --log_dir ./logs/ERM_breeds/mammal-bird-pair-all/887e4ffa-85fc-4dd9-963c-73082c01e7ab & 
job_limit 5

srun -N 1 -n 1 --gres=gpu:1 python run_expt.py --shift_type confounder --dataset Breeds --target_name breeds --save_step 10 --breeds_proportions 441,10676,9918,8258,1270,1528,1439,1147,2531,3050,3846,2341 --save_last --save_best --lr .1 --step_scheduler --scheduler_step 120 --n_epochs 100 --batch_size 128 --weight_decay 1.00E-04 --model resnet18 --train_from_scratch --gamma 0.1 --generalization_adjustment 0 --breeds_dataset_type entity13 --log_dir ./logs/ERM_breeds/mammal-bird-pair-all/431f6b33-22a4-49fc-baaa-66728e84b831 & 
job_limit 5

srun -N 1 -n 1 --gres=gpu:1 python run_expt.py --shift_type confounder --dataset Breeds --target_name breeds --save_step 10 --breeds_proportions 441,10676,9918,8258,1270,1528,1439,1147,2531,3050,3846,2341 --save_last --save_best --lr .1 --step_scheduler --scheduler_step 120 --n_epochs 100 --batch_size 128 --weight_decay 1.00E-04 --model resnet34 --train_from_scratch --gamma 0.1 --generalization_adjustment 0 --breeds_dataset_type entity13 --log_dir ./logs/ERM_breeds/mammal-bird-pair-all/cfe42a87-b5fc-4a29-bc66-1b81b6a0d68a & 
job_limit 5

srun -N 1 -n 1 --gres=gpu:1 python run_expt.py --shift_type confounder --dataset Breeds --target_name breeds --save_step 10 --breeds_proportions 441,10676,9918,8258,1270,1528,1439,1147,2531,3050,3846,2341 --save_last --save_best --lr .1 --step_scheduler --scheduler_step 120 --n_epochs 100 --batch_size 128 --weight_decay 1.00E-04 --model resnet50 --train_from_scratch --gamma 0.1 --generalization_adjustment 0 --breeds_dataset_type entity13 --log_dir ./logs/ERM_breeds/mammal-bird-pair-all/69259e82-0c2f-4812-acae-29ceed425e02 & 
job_limit 5

srun -N 1 -n 1 --gres=gpu:1 python run_expt.py --shift_type confounder --dataset Breeds --target_name breeds --save_step 10 --breeds_proportions 441,10676,9918,8258,1270,1528,1439,1147,2531,3050,3846,2341 --save_last --save_best --lr .1 --step_scheduler --scheduler_step 120 --n_epochs 100 --batch_size 128 --weight_decay 1.00E-04 --model resnet101 --train_from_scratch --gamma 0.1 --generalization_adjustment 0 --breeds_dataset_type entity13 --log_dir ./logs/ERM_breeds/mammal-bird-pair-all/1d72d98a-2008-4951-9973-834107c7acc4 & 
job_limit 5

srun -N 1 -n 1 --gres=gpu:1 python run_expt.py --shift_type confounder --dataset Breeds --target_name breeds --save_step 10 --breeds_proportions 441,10676,9918,8258,1270,1528,1439,1147,2531,3050,3846,2341 --save_last --save_best --lr .1 --step_scheduler --scheduler_step 120 --n_epochs 100 --batch_size 128 --weight_decay 1.00E-04 --model resnet152 --train_from_scratch --gamma 0.1 --generalization_adjustment 0 --breeds_dataset_type entity13 --log_dir ./logs/ERM_breeds/mammal-bird-pair-all/c2204093-2010-425d-803f-c0e2c802f96e 

wait
date
