#!/bin/bash

source ~/.bashrc

repo_dir=/work/$USER/overparam_spur_corr
env_name=overparam_spur_corr

# ==================
# General setup
# ==================
# conda env remove -n $env_name
# conda create -n $env_name python=3.6.8
# conda activate $env_name

# echo $(python --version)
# pip install -r $repo_dir/requirements.txt


# ==================
# Waterbirds setup
# ==================

# mkdir -p $repo_dir/cub/data/

# cd /data/$USER
# wget -nc https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz
# tar -xzvf waterbird_complete95_forest2water2.tar.gz
# ln -s /data/$USER/waterbird_complete95_forest2water2 $repo_dir/cub/data/waterbird_complete95_forest2water2


# ==================
# CelebA setup
# ==================

mkdir -p $repo_dir/celebA

mkdir -p /data/$USER/celebA
cd /data/$USER/celebA

# install
kaggle datasets download -d jessicali9530/celeba-dataset
unzip celeba-dataset.zip

# move nested folder to not be nested
mv /data/$USER/celebA/img_align_celeba/img_align_celeba /data/$USER/celebA/TEMP
rm -r /data/$USER/celebA/img_align_celeba
mv /data/$USER/celebA/TEMP /data/$USER/celebA/img_align_celeba

# link
ln -s /data/$USER/celebA/data $repo_dir/celebA/data #ln -s /data/$USER/celebA/data /work/$USER/overparam_spur_corr/celebA/data