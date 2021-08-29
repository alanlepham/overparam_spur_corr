#!/bin/bash
#SBATCH -p rise # partition (queue)
#SBATCH -N 1 # number of nodes requested
#SBATCH -n 1 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=5 # number of cores per task
#SBATCH --gres=gpu:1
#SBATCH --nodelist=luigi # if you need specific nodes
##SBATCH --exclude=ace,blaze,bombe,flaminio,freddie,luigi,pavia,r[10,16],atlas,como,havoc,steropes
#SBATCH -t 3-00:00 # time requested (D-HH:MM)

#SBATCH -o slurm.%j.%N.ERM_breeds.mammal-bird-pair-100_percent_tests.out # STDOUT
#SBATCH -e slurm.%j.%N.ERM_breeds.mammal-bird-pair-100_percent_tests.err # STDERR

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

#srun -N 1 -n 1 --gres=gpu:1 python run_expt.py --shift_type confounder --dataset Breeds --target_name breeds --save_step 10 --save_last --save_best --lr .1 --scheduler_step 30 --n_epochs 300 --batch_size 128 --weight_decay 1.00E-04 --model resnet18 --train_from_scratch --gamma 0.1 --generalization_adjustment 0 --breeds_dataset_type entity13 --log_dir ./logs/ERM_breeds/mammal-bird-pair-100_percent_tests/1f560257-0569-4486-9eef-8684a4123a89 & 
#job_limit 5

#srun -N 1 -n 1 --gres=gpu:1 python run_expt.py --shift_type confounder --dataset Breeds --target_name breeds --save_step 10 --save_last --save_best --lr .1 --step_scheduler --scheduler_step 60 --n_epochs 300 --batch_size 128 --weight_decay 1.00E-04 --model resnet18 --train_from_scratch --gamma 0.1 --generalization_adjustment 0 --breeds_dataset_type entity13 --log_dir ./logs/ERM_breeds/mammal-bird-pair-100_percent_tests/29f735fd-b2f2-4926-9948-3d8668c13af4 & 
#job_limit 5

#srun -N 1 -n 1 --gres=gpu:1 python run_expt.py --shift_type confounder --dataset Breeds --target_name breeds --save_step 10 --save_last --save_best --lr .1 --step_scheduler --scheduler_step 30 --n_epochs 300 --batch_size 128 --weight_decay 1.00E-04 --model resnet18 --resnet_width 96 --train_from_scratch --gamma 0.1 --generalization_adjustment 0 --breeds_dataset_type entity13 --log_dir ./logs/ERM_breeds/mammal-bird-pair-100_percent_tests/14d20cff-23b3-4afb-a1e1-286469535aab & 
#job_limit 5

#srun -N 1 -n 1 --gres=gpu:1 python run_expt.py --shift_type confounder --dataset Breeds --target_name breeds --save_step 10 --save_last --save_best --lr .1 --step_scheduler --scheduler_step 30 --n_epochs 300 --batch_size 128 --weight_decay 1.00E-04 --model resnet18 --resnet_width 192 --train_from_scratch --gamma 0.1 --generalization_adjustment 0 --breeds_dataset_type entity13 --log_dir ./logs/ERM_breeds/mammal-bird-pair-100_percent_tests/4f846d58-8a82-4677-a250-7ddda9454293 & 
#job_limit 5

srun -N 1 -n 1 --gres=gpu:1 python run_expt.py --shift_type confounder --dataset Breeds --target_name breeds --save_step 10 --save_last --save_best --lr .1 --step_scheduler --scheduler_step 60 --n_epochs 300 --batch_size 128 --weight_decay 1.00E-04 --model resnet18 --train_from_scratch --gamma 0.5 --generalization_adjustment 0 --breeds_dataset_type entity13 --log_dir ./logs/ERM_breeds/mammal-bird-pair-100_percent_tests/195d881f-4f8f-41e4-805e-378afaaa76ad 
job_limit 1


srun -N 1 -n 1 --gres=gpu:1 python run_expt.py --shift_type confounder --dataset Breeds --target_name breeds --save_step 10 --save_last --save_best --lr .1 --step_scheduler --scheduler_step 120 --n_epochs 300 --batch_size 128 --weight_decay 1.00E-04 --model resnet18 --train_from_scratch --gamma 0.5 --generalization_adjustment 0 --breeds_dataset_type entity13 --log_dir ./logs/ERM_breeds/mammal-bird-pair-100_percent_tests/196d881f-4f8f-41e4-805e-378afaaa76ad
job_limit 1

wait
date

