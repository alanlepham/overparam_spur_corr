#!/bin/bash
#SBATCH -p rise # partition (queue)
#SBATCH -N 1 # number of nodes requested
#SBATCH -n 5 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=5 # number of cores per task
#SBATCH --gres=gpu:5
#SBATCH --nodelist=luigi # if you need specific nodes
##SBATCH --exclude=ace,blaze,bombe,flaminio,freddie,luigi,pavia,r[10,16],atlas,como,havoc,steropes
#SBATCH -t 3-00:00 # time requested (D-HH:MM)

#SBATCH -o slurm.%j.%N.ERM_breeds.mammal-bird-pair-waterbirds-size.out # STDOUT
#SBATCH -e slurm.%j.%N.ERM_breeds.mammal-bird-pair-waterbirds-size.err # STDERR

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

srun -N 1 -n 1 --gres=gpu:1 python run_expt.py --shift_type confounder --dataset Breeds --target_name breeds --save_step 10 --breeds_proportions 3498,184,56,1057,467,466,133,133,2255,2255,642,642 --save_last --save_best --lr .1 --step_scheduler --scheduler_step 30 --n_epochs 100 --batch_size 128 --weight_decay 1.00E-04 --model resnet18 --train_from_scratch --gamma 0.1 --generalization_adjustment 0 --breeds_dataset_type entity13 --log_dir ./logs/ERM_breeds/mammal-bird-pair-waterbirds-size/86c212a5-7024-4431-91d8-7e90e1040a6c & 
job_limit 5

srun -N 1 -n 1 --gres=gpu:1 python run_expt.py --shift_type confounder --dataset Breeds --target_name breeds --save_step 10 --breeds_proportions 3498,184,56,1057,467,466,133,133,2255,2255,642,642 --save_last --save_best --lr .1 --step_scheduler --scheduler_step 30 --n_epochs 100 --batch_size 128 --weight_decay 1.00E-04 --model resnet34 --train_from_scratch --gamma 0.1 --generalization_adjustment 0 --breeds_dataset_type entity13 --log_dir ./logs/ERM_breeds/mammal-bird-pair-waterbirds-size/4dc4c797-16cb-4cf0-ad05-0b20a1352d9f & 
job_limit 5

srun -N 1 -n 1 --gres=gpu:1 python run_expt.py --shift_type confounder --dataset Breeds --target_name breeds --save_step 10 --breeds_proportions 3498,184,56,1057,467,466,133,133,2255,2255,642,642 --save_last --save_best --lr .1 --step_scheduler --scheduler_step 30 --n_epochs 100 --batch_size 128 --weight_decay 1.00E-04 --model resnet50 --train_from_scratch --gamma 0.1 --generalization_adjustment 0 --breeds_dataset_type entity13 --log_dir ./logs/ERM_breeds/mammal-bird-pair-waterbirds-size/ccbd7d79-7797-4f29-a336-c053035f32ee & 
job_limit 5

srun -N 1 -n 1 --gres=gpu:1 python run_expt.py --shift_type confounder --dataset Breeds --target_name breeds --save_step 10 --breeds_proportions 3498,184,56,1057,467,466,133,133,2255,2255,642,642 --save_last --save_best --lr .1 --step_scheduler --scheduler_step 30 --n_epochs 100 --batch_size 128 --weight_decay 1.00E-04 --model resnet101 --train_from_scratch --gamma 0.1 --generalization_adjustment 0 --breeds_dataset_type entity13 --log_dir ./logs/ERM_breeds/mammal-bird-pair-waterbirds-size/485fecb1-460f-4226-b573-0c2dda6aca78 & 
job_limit 5

srun -N 1 -n 1 --gres=gpu:1 python run_expt.py --shift_type confounder --dataset Breeds --target_name breeds --save_step 10 --breeds_proportions 3498,184,56,1057,467,466,133,133,2255,2255,642,642 --save_last --save_best --lr .1 --step_scheduler --scheduler_step 30 --n_epochs 100 --batch_size 128 --weight_decay 1.00E-04 --model resnet152 --train_from_scratch --gamma 0.1 --generalization_adjustment 0 --breeds_dataset_type entity13 --log_dir ./logs/ERM_breeds/mammal-bird-pair-waterbirds-size/6f847fc1-5901-4b34-b16b-0feff145eefd 

wait
date

