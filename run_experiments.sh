#!/bin/bash

#SBATCH --job-name=20C
#SBATCH --output=/research/d3/drdeng/continual_learning/IDEA/0t.txt

SEED=2

PMNIST='--dataset mnist_permutations --samples_per_task 1000 --workers 4 --batch_size 10 --n_epochs 1 --glances 5 --mem_batch_size 300 --thres 0.99 --thres_add 0.0005 --cuda'
CIFAR='--dataset cifar100 --n_tasks 10 --pc_valid 0.05 --batch_size 64 --test_batch_size 64 --n_epochs 50 --mem_batch_size 125 --thres 0.97 --thres_add 0.003 --cuda --second_order --earlystop'
CSUPER='--dataset cifar100_superclass --pc_valid 0.05 --batch_size 64 --test_batch_size 64 --n_epochs 50 --mem_batch_size 125 --thres 0.98 --thres_add 0.001 --cuda --freeze_bn --second_order --earlystop'
IMGNET='--data_path ../data/tiny-imagenet-200/ --dataset tinyimagenet --pc_valid 0.1 --loader class_incremental_loader --increment 5 --class_order random --workers 8 --batch_size 10 --test_batch_size 64 --n_epochs 10 --mem_batch_size 200 --thres 0.9 --thres_add 0.0025 --cuda'

FSDGPM='--model fsdgpm --inner_batches 2 --sharpness --method xdgpm'

## PMNIST
python main.py $PMNIST --seed $SEED $FSDGPM --memories 200 --lr 0.01 --eta1 0.05 --eta2 0.01 --expt_name fs-dgpm

## 10-Split CIFAR-100 DATASET
#python main.py $CIFAR --seed $SEED $FSDGPM --memories 1000 --lr 0.01 --eta1 0.001 --eta2 0.01 --expt_name fs-dgpm

## 20-Split CIFAR-100 SuperClass DATASET
#python main.py $CSUPER --seed $SEED $FSDGPM --memories 1000 --lr 0.01 --eta1 0.01 --eta2 0.01 --expt_name fs-dgpm

## 40-Split TinyImageNet DATASET
#python main.py $IMGNET --seed $SEED $FSDGPM --memories 400 --lr 0.01 --eta1 0.001 --eta2 0.01 --expt_name fs-dgpm



###### VISUALIZATION OF LOSS LANDSCAPE ######

#VISUAL='--visual_landscape --step_size 0.02 --dir_num 10'
#DATASET='--dataset pmnist --memories 1000 --batch_size 10 --n_epochs 10 --glances 1 --lr_factor 10 --lr_min 0.01 --lr_patience 1 --cuda'

# ER
##python main.py $DATASET $VISUAL --seed $SEED --model ER --lr 0.005 --expt_name er

# ER+FS
##python main.py $DATASET $VISUAL --seed $SEED --model ER --sharpness --inner_batches 1 --lr 0.001 --eta1 0.01 --expt_name fs-er
