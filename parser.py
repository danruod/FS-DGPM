# coding=utf-8
import os
import argparse
from utils.utils import str2bool

def get_parser():
    parser = argparse.ArgumentParser(description='Continual learning')
    parser.add_argument('--expt_name', type=str, default='debug', help='name of the experiment')

    # experiment parameters
    parser.add_argument('--model', default='fsdgpm', type=str, required=True, choices=['fsdgpm', 'ER'], help='')
    parser.add_argument('--cuda', default=False, action='store_true', help='Use GPU')
    parser.add_argument('--seed', type=int, default=0, help='random seed of model')
    parser.add_argument('--device', default='cuda:0', type=str, help='gpu id')

    # data parameters
    parser.add_argument('--data_path', default='./data/', help='path where data is located')
    parser.add_argument("--dataset", default='mnist_permutations', type=str, required=True,
                        choices=['mnist_permutations', 'cifar100', 'cifar100_superclass', 'tinyimagenet', 'pmnist'],
                        help="Dataset to train and test on.")
    parser.add_argument('--pc_valid', type=float, default=0.1, help='percentage for validation')

    # For mnist_permutations & tinyimagenet
    parser.add_argument('--loader', type=str, default='task_incremental_loader',
                        help='data loader to use')
    parser.add_argument('--samples_per_task', type=int, default=-1,
                        help='training samples per task (all if negative)')
    parser.add_argument("--increment", default=5, type=int,
                        help="number of classes to increment by in class incremental loader")
    parser.add_argument("--class_order", default="random", type=str, choices=["random", "chrono", "old", "super"],
                        help="define classes order of increment ")
    parser.add_argument("--workers", default=4, type=int, help="Number of workers preprocessing the data.")

    # For cifar100
    parser.add_argument('--n_tasks', type=int, default=10,
                        help='total number of tasks, invalid for cifar100_superclass')
    parser.add_argument('--shuffle_task', default=False, action='store_true',
                        help='Invalid for cifar100_superclass')

    # For cifar100_superclass
    parser.add_argument('--t_order', type=int, default=0, help='0-4, just valid for cifar100_superclass')

    # optimizer parameters influencing all models
    parser.add_argument('--n_epochs', type=int, default=5,
                        help='Number of epochs per task')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='the amount of items received by the algorithm at one time.')
    parser.add_argument('--test_batch_size', type=int, default=100000,
                        help='the amount of items for testing at one time.')
    parser.add_argument("--glances", default=1, type=int,
                        help="# of times the model is allowed to train over a set of samples in the single pass setting")

    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--grad_clip_norm', type=float, default=2.0,
                        help='Clip the gradients by this value')

    parser.add_argument('--earlystop', default=False, action='store_true', help='')
    parser.add_argument('--lr_factor', type=float, default=2, help='')
    parser.add_argument('--lr_min', type=float, default=1e-5, help='')
    parser.add_argument('--lr_patience', type=int, default=6, help='')

    # log parameters
    parser.add_argument('--log_dir', type=str, default='logs/',
                        help='the directory where the logs will be saved')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint/',
                        help='the directory where the logs will be saved')
    parser.add_argument('--log_every', type=int, default=20,
                        help='frequency of checking the validation accuracy, in minibatches')

    # FSDGPM parameters
    parser.add_argument('--memories', type=int, default=1000,
                        help='number of total memories stored in a reservoir sampling based buffer')
    parser.add_argument('--inner_batches', type=int, default=2, help='the # of inner/sharpness steps')
    parser.add_argument('--sharpness', default=False, action='store_true', help='')
    parser.add_argument('--method', default='xdgpm', type=str, required=False, choices=['xdgpm', 'dgpm', 'xgpm'],
                        help="If sharpness is True, x means FS(flatten sharpness), else La(look ahead)")

    parser.add_argument('--eta1', type=float, default=1e-2, help='update step size of weight perturbation')
    parser.add_argument('--eta2', type=float, default=1e-2, help='learning rate of lambda(soft weight for basis)')

    parser.add_argument('--lam_init', type=float, default=1.0, help='temperature for sigmoid')
    parser.add_argument('--tmp', type=float, default=10, help='temperature for sigmoid')

    parser.add_argument('--mem_batch_size', type=int, default=300,
                        help='the amount of items selected to update feature spaces.')
    parser.add_argument('--thres', type=float, default=0.90, help='thres')
    parser.add_argument('--thres_add', type=float, default=0.003, help='thres_add')
    parser.add_argument('--thres_last', type=float, default=0.99999999999, help='thres_last')

    parser.add_argument('--use_track', type=str2bool, default=True)
    parser.add_argument('--freeze_bn', default=False, action='store_true', help='')
    parser.add_argument('--second_order', default=False, action='store_true', help='')

    # visualization of loss landscape
    parser.add_argument('--visual_landscape', default=False, action='store_true', help='')
    parser.add_argument('--step_min', type=float, default=-1.0)
    parser.add_argument('--step_max', type=float, default=1.0)
    parser.add_argument('--step_size', type=float, default=0.04)
    parser.add_argument('--dir_num', type=int, default=1)

    return parser
