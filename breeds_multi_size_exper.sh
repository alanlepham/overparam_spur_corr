#!/bin/bash
#SBATCH -p rise # partition (queue)
#SBATCH -N 1 # number of nodes requested
#SBATCH -n 1 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=16 # number of cores per task
#SBATCH --gres=gpu:1
#SBATCH --nodelist=luigi # if you need specific nodes
##SBATCH --exclude=ace,blaze,bombe,flaminio,freddie,luigi,pavia,r[10,16],atlas,como,havoc,steropes
#SBATCH -t 3-00:00 # time requested (D-HH:MM)

#SBATCH -o slurm.%j.%N.ERM_breeds.mammal-bird-pair.out # STDOUT
#SBATCH -e slurm.%j.%N.ERM_breeds.mammal-bird-pair.err # STDERR

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

srun -N 1 -n 1 --gres=gpu:1 python run_expt.py --shift_type confounder --dataset Breeds --target_name breeds --save_step 10 --save_last --save_best --lr .1 --step_scheduler --scheduler_step 30 --n_epochs 100 --batch_size 128 --weight_decay 1.00E-04 --model resnet18 --train_from_scratch --gamma 0.1 --generalization_adjustment 0 --breeds_dataset_type entity13 --log_dir ./logs/ERM_breeds/mammal-bird-pair/ffa20516-4e35-4a08-863f-d2edbf7118202 --reload_breeds_groups & 
#job_limit 1

#srun -N 1 -n 1 --gres=gpu:1 python run_expt.py --shift_type confounder --dataset Breeds --target_name breeds --save_step 10 --save_last --save_best --lr .1 --step_scheduler --scheduler_step 30 --n_epochs 100 --batch_size 128 --weight_decay 1.00E-04 --model resnet34 --train_from_scratch --gamma 0.1 --generalization_adjustment 0 --breeds_dataset_type entity13 --log_dir ./logs/ERM_breeds/mammal-bird-pair/3d14a437-2090-4c60-894d-01a304425a39 & 
#job_limit 1

#srun -N 1 -n 1 --gres=gpu:1 python run_expt.py --shift_type confounder --dataset Breeds --target_name breeds --save_step 10 --save_last --save_best --lr .1 --step_scheduler --scheduler_step 30 --n_epochs 100 --batch_size 128 --weight_decay 1.00E-04 --model resnet50 --train_from_scratch --gamma 0.1 --generalization_adjustment 0 --breeds_dataset_type entity13 --log_dir ./logs/ERM_breeds/mammal-bird-pair/72aaed17-368f-4867-a81d-958c145086b7 & 
#job_limit 1

#srun -N 1 -n 1 --gres=gpu:1 python run_expt.py --shift_type confounder --dataset Breeds --target_name breeds --save_step 10 --save_last --save_best --lr .1 --step_scheduler --scheduler_step 30 --n_epochs 100 --batch_size 128 --weight_decay 1.00E-04 --model resnet101 --train_from_scratch --gamma 0.1 --generalization_adjustment 0 --breeds_dataset_type entity13 --log_dir ./logs/ERM_breeds/mammal-bird-pair/291dbbf7-48af-4d99-a8c9-6458b49fea60 & 
#job_limit 1

#srun -N 1 -n 1 --gres=gpu:1 python run_expt.py --shift_type confounder --dataset Breeds --target_name breeds --save_step 10 --save_last --save_best --lr .1 --step_scheduler --scheduler_step 30 --n_epochs 100 --batch_size 128 --weight_decay 1.00E-04 --model resnet152 --train_from_scratch --gamma 0.1 --generalization_adjustment 0 --breeds_dataset_type entity13 --log_dir ./logs/ERM_breeds/mammal-bird-pair/76d4996b-fdc9-4a06-9409-9f00571289a3 

wait
date

