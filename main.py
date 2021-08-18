import time
import importlib
import numpy as np
import torch
import copy
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
from sklearn.utils import shuffle
from torch.utils.tensorboard import SummaryWriter
import parser as file_parser
import utils.utils as utils
import utils.plot1D as plot


def eval(model, x, y, t, args):
    model.net.eval()

    total_loss = 0
    total_acc = 0
    idx = np.arange(x.size(0))
    np.random.shuffle(idx)
    idx = torch.LongTensor(idx)

    with torch.no_grad():
        # Loop batches
        for i in range(0, len(idx), args.test_batch_size):
            if i + args.test_batch_size <= len(idx):
                pos = idx[i: i + args.test_batch_size]
            else:
                pos = idx[i:]

            images = x[pos]
            targets = y[pos]
            if args.cuda:
                images = images.cuda()
                targets = targets.cuda()

            outputs = model(images, t)
            if model.net.multi_head:
                offset1, offset2 = model.compute_offsets(t)
                loss = model.loss_ce(outputs[:, offset1:offset2], targets - offset1)
            else:
                loss = model.loss_ce(outputs, targets)

            _, p = torch.max(outputs.data.cpu(), 1, keepdim=False)
            total_loss += loss.detach() * len(pos)
            total_acc += (p == targets.cpu()).float().sum()

    return total_loss / len(x), total_acc / len(x)


def life_experience(model, data, ids, args):
    time_start = time.time()

    # store accuravy & loss for all tasks
    acc = np.zeros((args.n_tasks, args.n_tasks), dtype=np.float32)
    lss = np.zeros((args.n_tasks, args.n_tasks), dtype=np.float32)
    tasks = np.arange(args.n_tasks, dtype=np.int32)

    # visual landscape
    if args.visual_landscape:
        steps = np.arange(args.step_min, args.step_max, args.step_size)
        visual_lss = np.zeros((args.n_tasks, args.n_tasks, args.dir_num, len(steps)), dtype=np.float32)
        visual_val_acc = np.zeros((args.n_tasks, args.n_tasks), dtype=np.float32)
        visual_train_acc = np.zeros((args.n_tasks, args.n_tasks), dtype=np.float32)

    # tensorboard & checkpoint
    args.log_dir, args.checkpoint_dir = utils.log_dir(args)
    writer = SummaryWriter(args.log_dir)

    # train/val/test order by ids
    # t: the real task id
    for i, t in enumerate(ids):
        # Get data
        xtrain = data[t]['train']['x']
        ytrain = data[t]['train']['y']
        xvalid = data[t]['valid']['x']
        yvalid = data[t]['valid']['y']
        task = t

        assert xtrain.shape[0] == ytrain.shape[0] and xvalid.shape[0] == yvalid.shape[0]

        if args.cuda:
            xtrain = xtrain.cuda()
            ytrain = ytrain.cuda()
            xvalid = xvalid.cuda()
            yvalid = yvalid.cuda()

        print('*' * 100)
        print('>>>Task {:2d}({:s}) | Train: {:5d}, Val: {:5d}, Test: {:5d}<<<'.format(i, data[t]['name'],
                                   ytrain.shape[0], yvalid.shape[0], data[t]['test']['y'].shape[0]))
        print('*' * 100)

        # Train
        clock0 = time.time()
        # bn's parameters are only learned for the first task
        if args.freeze_bn and i == 1:
            for m in model.net.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

        # reset the learning rate
        lr = args.lr
        model.update_optimizer(lr)
        if args.model == 'fsdgpm':
            model.eta1 = args.eta1
            if len(model.M_vec) > 0 and args.method in ['dgpm', 'xdgpm']:
                # reset lambda
                model.eta2 = args.eta2
                model.define_lambda_params()
                model.update_opt_lambda(model.eta2)

        # if use early stop, then start training new tasks from the optimal model
        if args.earlystop:
            best_loss = np.inf
            patience = args.lr_patience
            best_model = copy.deepcopy(model.net.state_dict())

        prog_bar = tqdm(range(args.n_epochs))
        for ep in prog_bar:
            # train
            model.epoch += 1
            model.real_epoch = ep

            model.net.train()
            idx = np.arange(xtrain.size(0))
            np.random.shuffle(idx)
            idx = torch.LongTensor(idx)
            train_loss = 0.0

            # Loop batches
            for bi in range(0, len(idx), args.batch_size):
                if bi + args.batch_size <= len(idx):
                    pos = idx[bi: bi + args.batch_size]
                else:
                    pos = idx[bi:]

                v_x = xtrain[pos]
                v_y = ytrain[pos]

                loss = model.observe(v_x, v_y, t)
                train_loss += loss * len(v_x)

            train_loss = train_loss / len(xtrain)
            writer.add_scalar(f"1.Train-LOSS/{data[t]['name']}", round(train_loss.item(), 5), model.epoch)

            # if use early stop, we need to adapt lr and store the best model
            if args.earlystop:
                # Valid
                valid_loss, valid_acc = eval(model, xvalid, yvalid, t, args)
                writer.add_scalar(f"2.Val-LOSS/{data[t]['name']}", round(valid_loss.item(), 5), model.epoch)
                writer.add_scalar(f"2.Val-ACC/{data[t]['name']}", 100 * valid_acc, model.epoch)

                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_model = copy.deepcopy(model.net.state_dict())
                    patience = args.lr_patience
                else:
                    patience -= 1
                    if patience <= 0:
                        lr /= args.lr_factor
                        print(' lr={:.1e} |'.format(lr), end='')
                        if lr < args.lr_min:
                            break
                        patience = args.lr_patience
                        model.update_optimizer(lr)
                        if args.model == 'fsdgpm':
                            model.eta1 = model.eta1 / args.lr_factor
                            if len(model.M_vec) > 0 and args.method in ['dgpm', 'xdgpm']:
                                model.eta2 = model.eta2 / args.lr_factor
                                model.update_opt_lambda(model.eta2)

                    prog_bar.set_description(
                        "Task: {} | Epoch: {}/{} | time={:2.2f}s | Train: loss={:.3f} | Valid: loss={:.3f}, acc={:5.1f}% |".format(
                            i, ep + 1, model.n_epochs, time.time() - clock0, round(train_loss.item(), 5),
                            round(valid_loss.item(), 5), 100 * valid_acc)
                    )
            else:
                prog_bar.set_description("Task: {} | Epoch: {}/{} | time={:2.2f}s | Train: loss={:.3f} |".format(
                        i, ep + 1, model.n_epochs, time.time() - clock0, round(train_loss.item(), 5))
                    )

        if args.earlystop:
            model.net.load_state_dict(copy.deepcopy(best_model))

        print('-' * 60)
        print('Total Epoch: {}/{} | Training Time: {:.2f} min | Last Lr: {}'.format(ep + 1, model.n_epochs,
                                                                                    (time.time() - clock0) / 60, lr))
        print('-' * 60)

        # Test
        clock1 = time.time()
        for u in range(i + 1):
            xtest = data[ids[u]]['test']['x']
            ytest = data[ids[u]]['test']['y']

            if args.cuda:
                xtest = xtest.cuda()
                ytest = ytest.cuda()

            test_loss, test_acc = eval(model, xtest, ytest, ids[u], args)

            acc[i, u] = test_acc
            lss[i, u] = test_loss

            writer.add_scalar(f"0.Test-LOSS/{data[ids[u]]['name']}", test_loss, i)
            writer.add_scalar(f"0.Test-ACC/{data[ids[u]]['name']}", 100 * test_acc, i)
            writer.add_scalar(f"0.Test-BWT/{data[ids[u]]['name']}", 100 * (test_acc - acc[u, u]), i)

        avg_acc = sum(acc[i]) / (i + 1)
        bwt = np.mean((acc[i]-np.diag(acc)))

        writer.add_scalar(f"0.Test/Avg-ACC", 100 * avg_acc, i)
        writer.add_scalar(f"0.Test/Avg-BWT", 100 * bwt, i)

        print('-' * 60)
        print('Test Result: ACC={:5.3f}%, BWT={:5.3f}%, Elapsed time = {:.2f} s'.format(100 * avg_acc, 100 * bwt,
                                                                                        time.time() - clock1))
        print('-' * 60)

        # Update Memory of Feature Space
        if args.model in ['fsdgpm']:
            clock2 = time.time()

            # Get Thres
            thres_value = min(args.thres + i * args.thres_add, args.thres_last)
            thres = np.array([thres_value] * model.net.n_rep)
            print('-' * 60)
            print('Threshold: ', thres)

            # Update basis of Feature Space
            model.set_gpm_by_svd(thres)

            # Get the info of mem
            for p in range(len(model.M_vec)):
                writer.add_scalar(f"3.MEM-Total/Layer_{p}", model.M_vec[p].shape[1], i)

            print('Spend Time = {:.2f} s'.format(time.time() - clock2))
            print('-' * 60)

        if args.visual_landscape:
            visual_lss[i], visual_val_acc[i], visual_train_acc[i] = plot.calculate_loss(model, data, ids, i, steps, args)

    time_end = time.time()
    time_spent = time_end - time_start

    print('*' * 100)
    print('>>> Final Test Result: ACC={:5.3f}%, BWT={:5.3f}%, Total time = {:.2f} min<<<'.format(
        100 * avg_acc, 100 * bwt, time_spent / 60))
    print('*' * 100)

    # plot & save
    if args.visual_landscape:
        timestamp = utils.get_date_time()
        file_name = '%s_ep_%d_task_%d_%s' % (args.model, args.n_epochs, model.n_tasks, timestamp)
        plot.plot_1d_loss_all(visual_lss, steps, file_name, show=True)
        plot.save_visual_results(visual_val_acc, visual_train_acc, acc, file_name)

    return torch.from_numpy(tasks), torch.from_numpy(acc), time_spent


def eval_class_tasks(model, tasks, args, idx=-1):
    model.eval()

    result_acc = []
    result_lss = []

    with torch.no_grad():
        # Loop batches
        for t, task_loader in enumerate(tasks):
            if idx == -1 or idx == t:
                lss = 0.0
                acc = 0.0

                for (i, (x, y)) in enumerate(task_loader):
                    if args.cuda:
                        x = x.cuda()
                        y = y.cuda()

                    outputs = model(x, t)

                    offset1, offset2 = model.compute_offsets(t)
                    loss = model.loss_ce(outputs[:, offset1:offset2], y - offset1)

                    _, p = torch.max(outputs.data.cpu(), 1, keepdim=False)

                    lss += loss.detach() * len(x)
                    acc += (p == y.cpu()).float().sum()

                result_lss.append(lss / len(task_loader.dataset))
                result_acc.append(acc / len(task_loader.dataset))

    return result_lss, result_acc


def eval_tasks(model, tasks, args, idx=-1):
    model.eval()
    result_acc = []
    result_lss = []

    with torch.no_grad():
        for i, task in enumerate(tasks):
            if idx == -1 or idx == i:
                t = i
                x = task[1]
                y = task[2]
                lss = 0.0
                acc = 0.0

                eval_bs = min(x.size(0), args.test_batch_size)

                for b_from in range(0, x.size(0), eval_bs):
                    b_to = min(b_from + eval_bs, x.size(0) - 1)
                    if b_from == b_to:
                        xb = x[b_from].view(1, -1)
                        yb = torch.LongTensor([y[b_to]]).view(1, -1)
                    else:
                        xb = x[b_from:b_to]
                        yb = y[b_from:b_to]

                    if args.cuda:
                        xb = xb.cuda()
                        yb = yb.cuda()

                    outputs = model(xb, t)
                    loss = model.loss_ce(outputs, yb)
                    lss += loss.detach() * xb.size(0)

                    _, pb = torch.max(outputs.data.cpu(), 1, keepdim=False)
                    acc += (pb == yb.cpu()).float().sum()

                result_acc.append(acc / x.size(0))
                result_lss.append(lss / x.size(0))

    return result_lss, result_acc

def life_experience_loader(model, inc_loader, args):
    time_start = time.time()

    result_test_a = []
    result_test_t = []
    test_tasks = inc_loader.get_tasks("test")
    val_tasks = inc_loader.get_tasks("val")

    evaluator = eval_tasks
    if args.loader == "class_incremental_loader":
        evaluator = eval_class_tasks

    # tensorboard
    args.log_dir, args.checkpoint_dir = utils.log_dir(args)
    writer = SummaryWriter(args.log_dir)

    for i in range(inc_loader.n_tasks):
        task_info, train_loader, _, _ = inc_loader.new_task()

        # Train
        clock0 = time.time()
        # bn's parameters are only learned for the first task
        if args.freeze_bn and i == 1:
            for m in model.net.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
            for m in model.net.vars.parameters():
                if m.ndim == 1:
                    m.requires_grad = False

        lr = args.lr
        model.update_optimizer(lr)
        if args.model == 'fsdgpm':
            model.eta1 = args.eta1
            if len(model.M_vec) > 0 and args.method in ['dgpm', 'xdgpm']:
                # reset lambda
                model.eta2 = args.eta2
                model.define_lambda_params()
                model.update_opt_lambda(model.eta2)

        if args.earlystop:
            best_loss = np.inf
            patience = args.lr_patience
            best_model = copy.deepcopy(model.net.state_dict())

        for ep in range(args.n_epochs):
            model.epoch += 1
            model.real_epoch = ep
            train_loss = 0.0

            prog_bar = tqdm(train_loader)
            for (k, (v_x, v_y)) in enumerate(prog_bar):
                if args.cuda:
                    v_x = v_x.cuda()
                    v_y = v_y.cuda()

                loss = model.observe(v_x, v_y, task_info['task'])
                train_loss += loss * len(v_x)

            train_loss = train_loss / len(train_loader.dataset)
            writer.add_scalar(f"1.Train-LOSS/Task_{task_info['task']}", round(train_loss.item(), 5), model.epoch)

            if args.earlystop:
                val_loss, val_acc = evaluator(model, val_tasks, args, task_info['task'])
                valid_loss = val_loss[-1].item()

                writer.add_scalar(f"2.Val-LOSS/Task_{task_info['task']}", round(valid_loss, 5), model.epoch)
                writer.add_scalar(f"2.Val-ACC/Task_{task_info['task']}", 100 * val_acc[-1], model.epoch)

                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_model = copy.deepcopy(model.net.state_dict())
                    patience = args.lr_patience
                else:
                    patience -= 1
                    if patience <= 0:
                        lr /= args.lr_factor
                        print('** lr={:.1e} **|'.format(lr), end='')
                        if lr < args.lr_min:
                            break
                        patience = args.lr_patience
                        model.update_optimizer(lr)
                        if args.model == 'fsdgpm':
                            model.eta1 = model.eta1 / args.lr_factor
                            if len(model.M_vec) > 0 and args.method in ['dgpm', 'xdgpm']:
                                model.eta2 = model.eta2 / args.lr_factor
                                model.update_opt_lambda(model.eta2)

                prog_bar.set_description(
                    "Task: {} | Epoch: {}/{} | time={:2.2f}s | Train: loss={:.3f} | Valid: loss={:.3f}, acc={:5.1f}% |".format(
                        task_info['task'], ep + 1, model.n_epochs, model.iter, time.time() - clock0,
                        round(train_loss.item(), 5), round(valid_loss, 5), 100 * val_acc[-1])
                )
            else:
                prog_bar.set_description(
                    "Task: {} | Epoch: {}/{} | time={:2.2f}s | Train: loss={:.3f} |".format(task_info['task'],
                        ep + 1, model.n_epochs, model.iter, time.time() - clock0, round(train_loss.item(), 5))
                )

        if args.earlystop:
            model.net.load_state_dict(copy.deepcopy(best_model))

        # Test
        clock1 = time.time()
        t_loss, t_acc = evaluator(model, test_tasks, args)
        result_test_a.append(t_acc)
        result_test_t.append(task_info["task"])

        avg = sum(t_acc[:(i + 1)]) / (i + 1)
        bwt = np.mean((np.array(t_acc[:(i+1)]) - np.diag(result_test_a[:(i+1)])))

        writer.add_scalar(f"0.Test/Avg-ACC", 100 * avg, i)
        writer.add_scalar(f"0.Test/Avg-BWT", 100 * bwt, i)

        for j in range(len(result_test_a)):
            writer.add_scalar(f"0.Test-LOSS/Task_{j}", t_loss[j].item(), i)
            writer.add_scalar(f"0.Test-ACC/Task_{j}", 100 * t_acc[j].item(), i)
            writer.add_scalar(f"0.Test-BWT/Task_{j}", 100 * (t_acc[j] - result_test_a[j][j]), i)

        print('-' * 60)
        print('Test Result: ACC={:5.3f}%, BWT={:5.3f}%, Elapsed time = {:.2f} s'.format(100 * avg, 100 * bwt,
                                                                                        time.time() - clock1))
        print('-' * 60)

        # Update Memory of Feature Space
        if args.model in ['fsdgpm']:
            clock2 = time.time()

            # Get threshold
            thres_value = min(args.thres + i * args.thres_add, args.thres_last)
            thres = np.array([thres_value] * model.net.n_rep)

            print('-' * 60)
            print('Threshold: ', thres)

            # Update basis of Feature Space
            model.set_gpm_by_svd(thres)

            # Get the info of GPM
            for p in range(len(model.M_vec)):
                writer.add_scalar(f"3.MEM-Total/Layer_{p}", model.M_vec[p].shape[1], i)

            print('Spend Time = {:.2f} s'.format(time.time() - clock2))
            print('-' * 60)

    time_end = time.time()
    time_spent = time_end - time_start

    print('*' * 100)
    print('>>> Final Test Result: ACC={:5.3f}%, BWT={:5.3f}%, Total time = {:.2f} min<<<'.format(
        100 * avg, 100 * bwt, time_spent / 60))
    print('*' * 100)

    return torch.Tensor(result_test_t), torch.Tensor(result_test_a), time_spent


def main():
    parser = file_parser.get_parser()
    args = parser.parse_args()

    utils.print_arguments(args)
    print("Starting at :", datetime.now().strftime("%Y-%m-%d %H:%M"))

    # initialize seeds
    utils.init_seed(args.seed)

    # Setup DataLoader
    print('Load data...')
    print("Dataset: ", args.dataset, args.data_path)

    if args.dataset in ['tinyimagenet', 'mnist_permutations']:
        Loader = importlib.import_module('dataloaders.' + args.loader)
        loader = Loader.IncrementalLoader(args, seed=args.seed)
        n_inputs, n_outputs, n_tasks, input_size = loader.get_dataset_info()

        # input_size: ch * size * size = n_inputs
        print('Input size =', input_size, '\nOutput number=', n_outputs, '\nTotal task=', n_tasks)
        print('-' * 100)
    else:
        dataloader = importlib.import_module('dataloaders.' + args.dataset)
        if args.dataset == 'cifar100_superclass':
            task_order = [np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
                          np.array([15, 12, 5, 9, 7, 16, 18, 17, 1, 0, 3, 8, 11, 14, 10, 6, 2, 4, 13, 19]),
                          np.array([17, 1, 19, 18, 12, 7, 6, 0, 11, 15, 10, 5, 13, 3, 9, 16, 4, 14, 2, 8]),
                          np.array([11, 9, 6, 5, 12, 4, 0, 10, 13, 7, 14, 3, 15, 16, 8, 1, 2, 19, 18, 17]),
                          np.array([6, 14, 0, 11, 12, 17, 13, 4, 9, 1, 7, 19, 8, 10, 3, 15, 18, 5, 2, 16])]

            ids = task_order[args.t_order]
            data, output_info, input_size, n_tasks, n_outputs = dataloader.get(data_path=args.data_path, task_order=ids,
                                                                               seed=args.seed, pc_valid=args.pc_valid)
            args.n_tasks = n_tasks
            args.samples_per_task = int(data[0]['train']['y'].shape[0] / (1.0 - args.pc_valid))
        else:
            data, output_info, input_size, n_tasks, n_outputs = dataloader.get(data_path=args.data_path, args=args,
                                                                               seed=args.seed, pc_valid=args.pc_valid,
                                                                               samples_per_task=args.samples_per_task)
            args.samples_per_task = int(data[0]['train']['y'].shape[0] / (1.0 - args.pc_valid))
            # Shuffle tasks
            if args.shuffle_task:
                ids = list(shuffle(np.arange(args.n_tasks), random_state=args.seed))
            else:
                ids = list(np.arange(args.n_tasks))

        print('Task info =', output_info)
        print('Input size =', input_size, '\nOutput number=', n_outputs, '\nTotal task=', n_tasks)
        print('Task order =', ids)
        print('-' * 100)

    # Setup Model
    Model = importlib.import_module('model.' + args.model)
    model = Model.Net(input_size, n_outputs, n_tasks, args)
    print("Model:", model.net)
    if args.cuda:
        model.net.cuda()

    # Train & Test
    try:
        if args.dataset in ['tinyimagenet', 'mnist_permutations']:
            result_test_t, result_test_a, spent_time = life_experience_loader(model, loader, args)
        else:
            result_test_t, result_test_a, spent_time = life_experience(model, data, ids, args)

        # save results in checkpoint_dir
        utils.save_results(args, result_test_t, result_test_a, model, spent_time)

    except KeyboardInterrupt:
        print()

if __name__ == "__main__":
    main()
