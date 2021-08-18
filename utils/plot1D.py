import sys
import os
import time
import math
import copy
import importlib
import torch
import numpy as np
from matplotlib import pyplot as plt


def eval_acc(model, bx, by, bt):
    model.eval()
    accs = np.zeros(model.n_tasks, dtype=np.float32)

    with torch.no_grad():
        for task in np.unique(bt.data.cpu().numpy()):
            task = int(task)
            idx = torch.nonzero(bt == task).view(-1)

            x = bx[idx]
            y = by[idx]

            if model.cuda:
                x = x.cuda()
                y = y.cuda()

            outputs = model(x, task)

            _, pb = torch.max(outputs.data.cpu(), 1, keepdim=False)
            acc = (pb == y.cpu()).float().sum() / len(y)

            accs[task] = acc
            
    return accs


def eval_loss(model, bx, by, bt):
    model.eval()
    losses = np.zeros(model.n_tasks, dtype=np.float32)

    with torch.no_grad():
        for task in np.unique(bt.data.cpu().numpy()):
            task = int(task)
            idx = torch.nonzero(bt == task).view(-1)

            x = bx[idx]
            y = by[idx]

            if model.cuda:
                x = x.cuda()
                y = y.cuda()

            outputs = model(x, task)
            lss = model.loss_ce(outputs, y)
            losses[task] = lss

    return losses

# https://github.com/tomgoldstein/loss-landscape/blob/master/net_plotter.py
def normalize_direction(direction, weights, norm='filter'):
    """
        Rescale the direction so that it has similar norm as their corresponding
        model in different levels.
        Args:
          direction: a variables of the random direction for one layer
          weights: a variable of the original model for one layer
          norm: normalization method, 'filter' | 'layer'
    """
    if norm == 'filter':
        # Rescale the filters (weights in group) in 'direction' so that each
        # filter has the same norm as its corresponding filter in 'weights'.
        for d, w in zip(direction, weights):
            d.mul_(w.norm()/(d.norm() + 1e-10))
    else:
        # Rescale the layer variables in the direction so that each layer has
        # the same norm as the layer variables in weights.
        direction.mul_(weights.norm()/direction.norm())


# https://github.com/tomgoldstein/loss-landscape/blob/master/net_plotter.py
def create_random_direction(weights, args, ignore='biasbn', norm='filter', model=None):
    """
        Setup a random (normalized) direction with the same dimension as the weights.
        Args:
          weights: the given trained model
          ignore: 'biasbn', ignore biases and BN parameters.
        Returns:
          direction: a random direction with the same dimension as weights.
    """

    # random direction
    direction = []
    for w in weights:
        d = torch.randn(w.size())
        if args.cuda:
            d = d.cuda()
        if d.dim() <= 1:
            if ignore == 'biasbn':
                d.fill_(0)  # ignore directions for weights with 1 dimension
            else:
                d.copy_(w)  # keep directions for weights/bias that are only 1 per node
        else:
            normalize_direction(d, w, norm)

        direction.append(d)

    return direction


# https://github.com/tomgoldstein/loss-landscape/blob/master/net_plotter.py
def set_weights(net, weights, directions=None, step=None):
    """
        Overwrite the network's weights with a specified list of tensors
        or change weights along directions with a step size.
    """
    if directions is None:
        # You cannot specify a step length without a direction.
        for (p, w) in zip(net.parameters(), weights):
            p.data.copy_(w.type(type(p.data)))
    else:
        assert step is not None, 'If a direction is specified then step must be specified as well'

        for (p, w, d) in zip(net.parameters(), weights, directions):
            p.data = w + d * step

    return net


def calculate_loss(model, data, ids, task, steps, args):

    val_acc = np.zeros(args.n_tasks, dtype=np.float32)
    train_lss = np.zeros((args.n_tasks, args.dir_num, len(steps)), dtype=np.float32)

    with torch.no_grad():
        # calculate val_acc
        for t in range(task + 1):
            xvalid = data[ids[t]]['valid']['x']
            yvalid = data[ids[t]]['valid']['y']
            tvalid = torch.ones_like(yvalid) * t

            if args.cuda:
                xvalid = xvalid.cuda()
                yvalid = yvalid.cuda()
                tvalid = tvalid.cuda()

            acc = eval_acc(model, xvalid, yvalid, tvalid)

            val_acc[t] = acc[t]

        # calculate train_loss and train_acc
        trained_weights = copy.deepcopy(list(model.net.parameters()))

        train_x = data[ids[0]]['train']['x']
        train_y = data[ids[0]]['train']['y']
        train_t = torch.zeros_like(train_y)

        for t in range(1, task + 1):
            train_x = torch.cat((train_x, data[ids[t]]['train']['x']), 0)
            train_y = torch.cat((train_y, data[ids[t]]['train']['y']), 0)
            train_t = torch.cat((train_t, torch.ones_like(data[ids[t]]['train']['y']) * t), 0)

        train_acc = eval_acc(model, train_x, train_y, train_t)

        for di in range(args.dir_num):
            direction = create_random_direction(model.net.parameters(), args=args, model=model)
            for s, step in enumerate(steps):
                model.net = set_weights(model.net, trained_weights, direction, step)

                train_lss[:, di, s] = eval_loss(model, train_x, train_y, train_t)

        set_weights(model.net, trained_weights)

    return train_lss, val_acc, train_acc


def save_visual_results(val_acc, train_acc, test_acc, file_name):
    assert val_acc.ndim == train_acc.ndim and val_acc.ndim == test_acc.ndim

    val_acc_flat = np.zeros((val_acc.reshape([-1, 1]).shape[0], 3), dtype=np.float32)
    val_acc_flat[:, 2] = val_acc.reshape([-1])

    train_acc_flat = np.zeros((train_acc.reshape([-1, 1]).shape[0], 3), dtype=np.float32)
    train_acc_flat[:, 2] = train_acc.reshape([-1])

    test_acc_flat = np.zeros((test_acc.reshape([-1, 1]).shape[0], 3), dtype=np.float32)
    test_acc_flat[:, 2] = test_acc.reshape([-1])

    # flatten results
    r = 0
    for i in range(val_acc.shape[0]):
        for j in range(val_acc.shape[1]):
            val_acc_flat[r, 0] = i
            train_acc_flat[r, 0] = i
            test_acc_flat[r, 0] = i
            val_acc_flat[r, 1] = j
            train_acc_flat[r, 1] = j
            test_acc_flat[r, 1] = j
            r += 1

    val_acc_file = './visual/val_acc_%s.csv' % file_name
    np.savetxt(val_acc_file, val_acc_flat)

    train_acc_file = './visual/train_acc_%s.csv' % file_name
    np.savetxt(train_acc_file, train_acc_flat)

    test_acc_file = './visual/test_acc_%s.csv' % file_name
    np.savetxt(test_acc_file, test_acc_flat)


def plot_1d_loss_all(loss, steps, file_name, show=False):
    # loss = n_tasks * n_tasks * direction_num * steps

    print("train_loss:")
    print(loss)

    save_lss = np.ones((loss.reshape((-1, 1)).shape[0], 5))
    r = 0

    # loss map
    fig, axes = plt.subplots(nrows=loss.shape[0], ncols=loss.shape[1], sharex='all', sharey='all',
                             figsize=(loss.shape[0] * 2.5, loss.shape[1] * 2.5))

    for i in range(loss.shape[0]):
        axes[i, 0].set_ylim(0, 5)
        for j in range(loss.shape[1]):
            for k in range(loss.shape[2]):
                axes[i, j].plot(steps, loss[j, i, k], 'b-', linewidth=1)
                axes[i, j].set_title('Task %d' % j)
                axes[i, j].set_ylabel('loss of Task %d' % i)

                for m, s in enumerate(steps):
                    save_lss[r, 0] = j
                    save_lss[r, 1] = i
                    save_lss[r, 2] = k
                    save_lss[r, 3] = s
                    save_lss[r, 4] = loss[j, i, k, m]
                    r += 1

    for ax in axes.flat:
        ax.set(xlabel='alpha')
        ax.label_outer()

    plt.tight_layout()

    plt.savefig('./visual/1d_loss_%s.pdf' % file_name, dpi=300, bbox_inches='tight', format='pdf')

    csv_file = open('./visual/1d_loss_%s.csv' % file_name, 'ab')
    np.savetxt('./visual/1d_loss_%s.csv' % file_name, save_lss)
    csv_file.close()

    if show:
        plt.show()
