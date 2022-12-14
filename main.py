import os
import csv
import sys
import time
import shutil
import pickle
import argparse

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from sklearn import metrics
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from tsdnn.data import split_bagging, aggregate_bagging, CIFData
from tsdnn.data import collate_pool, get_pos_unl_val_test_loader
from tsdnn.model import CrystalGraphConvNet


parser = argparse.ArgumentParser(
    description='Semi-Supervised Crystal Graph Convolutional Neural Networks')
parser.add_argument('data_options', metavar='OPTIONS', nargs='+',
                    help='dataset options, started with the path to root dir, then other options')
parser.add_argument('--graph', type=str, metavar='N',
                    help='Folder name for preloaded crystal graph files')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--uds', '--udsteps', default=0, type=int, metavar='N',
                    help='number of unsupervised PU learning iterations to perform')
parser.add_argument('-j', '--workers', default=0, type=int,
                    metavar='N', help='number of data loading workers (default: 0)')
parser.add_argument('-g', '--gpu', default=0, type=int,
                    metavar='N', help='gpu to run on (default: 0)')
parser.add_argument('--teacher', default='', type=str, metavar='PATH',
                    help='path to latest teacher checkpoint (default: none)')
parser.add_argument('--student', default='', type=str, metavar='PATH',
                    help='path to latest student checkpoint (default: none)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run (default: 30)')
parser.add_argument('--start-epoch', default=0, type=int,
                    metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--lr-milestones', default=[100], nargs='+', type=int,
                    metavar='N', help='milestones for scheduler (default: [100])')
parser.add_argument('--momentum', default=0.9, type=float,
                    metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0,
                    type=float, metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--iter', default=0, type=int, metavar='N',
                    help='iteration number to save as (default=0)')
train_group = parser.add_mutually_exclusive_group()
train_group.add_argument('--train-ratio', default=0.7, type=float,
                         metavar='N', help='number of training data to be loaded (default none)')
train_group.add_argument('--train-size', default=None, type=int, metavar='N',
                         help='number of training data to be loaded (default none)')
valid_group = parser.add_mutually_exclusive_group()
valid_group.add_argument('--val-ratio', default=0.1, type=float, metavar='N',
                         help='percentage of validation data to be loaded (default 0.1)')
valid_group.add_argument('--val-size', default=None, type=int, metavar='N',
                         help='number of validation data to be loaded (default 1000)')
test_group = parser.add_mutually_exclusive_group()
test_group.add_argument('--test-ratio', default=0.2, type=float, metavar='N',
                        help='percentage of test data to be loaded (default 0.1)')
test_group.add_argument('--test-size', default=None, type=int,
                        metavar='N', help='number of test data to be loaded (default 1000)')
parser.add_argument('--optim', default='SGD', type=str, metavar='SGD',
                    help='choose an optimizer, SGD or Adam, (default: SGD)')
parser.add_argument('--atom-fea-len', default=90, type=int, metavar='N',
                    help='number of hidden atom features in conv layers')
parser.add_argument('--h-fea-len', default=180, type=int,
                    metavar='N', help='number of hidden features after pooling')
parser.add_argument('--n-conv', default=3, type=int,
                    metavar='N', help='number of conv layers')
parser.add_argument('--n-h', default=1, type=int, metavar='N',
                    help='number of hidden layers after pooling')

args = parser.parse_args(sys.argv[1:])

args.cuda = not args.disable_cuda and torch.cuda.is_available()

root_dir = args.data_options[0]


modern = True


def preload(preload_folder, id_prop_file):
    data = []
    with open(id_prop_file) as g:
        reader = csv.reader(g)
        cif_list = [row[0] for row in reader]

    for cif_id in tqdm(cif_list):
        with open(preload_folder + '/' + cif_id + '.pickle', 'rb') as f:
            data.append(pickle.load(f))

    return data


def main():
    global t_best_mae_error, s_best_mae_error
    t_best_mae_error = 0.
    s_best_mae_error = 0.

    if args.cuda:
        torch.cuda.set_device(args.gpu)

    # load data
    if args.uds:
        split_bagging(args.data_options[0], os.path.join(
            args.data_options[0], 'bagging'), gen_test=not modern)

    collate_fn = collate_pool

    num_to_train = 1
    if args.uds:
        num_to_train = args.uds

    for mx in range(num_to_train):
        if args.uds:
            labeled_dataset = preload(preload_folder=args.graph, id_prop_file=os.path.join(
                args.data_options[0], 'bagging/data_labeled_' + str(mx)))
            unlabeled_dataset = preload(preload_folder=args.graph, id_prop_file=os.path.join(
                args.data_options[0], 'bagging/data_unlabeled_' + str(mx)))
            test_dataset = preload(preload_folder=args.graph, id_prop_file=os.path.join(
                args.data_options[0], 'bagging/data_test_' + str(mx)))
        else:
            labeled_dataset = preload(preload_folder=args.graph, id_prop_file=os.path.join(
                args.data_options[0], 'data_labeled.csv'))
            unlabeled_dataset = preload(preload_folder=args.graph, id_prop_file=os.path.join(
                args.data_options[0], 'data_unlabeled.csv'))
            test_dataset = preload(preload_folder=args.graph, id_prop_file=os.path.join(
                args.data_options[0], 'data_test.csv'))

        labeled_loader = DataLoader(labeled_dataset, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.workers,
                                    collate_fn=collate_fn, pin_memory=args.cuda)

        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.workers,
                                      collate_fn=collate_fn, pin_memory=args.cuda)

        val_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.workers,
                                collate_fn=collate_fn, pin_memory=args.cuda)

        # target value normalizer
        t_normalizer = Normalizer(torch.zeros(2))
        t_normalizer.load_state_dict({'mean': 0., 'std': 1.})
        s_normalizer = Normalizer(torch.zeros(2))
        s_normalizer.load_state_dict({'mean': 0., 'std': 1.})

        # build model
        structures, _, _ = labeled_dataset[0]
        orig_atom_fea_len = structures[0].shape[-1]
        nbr_fea_len = structures[1].shape[-1]
        t_model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                      atom_fea_len=args.atom_fea_len,
                                      n_conv=args.n_conv,
                                      h_fea_len=args.h_fea_len,
                                      n_h=args.n_h)
        s_model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                      atom_fea_len=args.atom_fea_len,
                                      n_conv=args.n_conv,
                                      h_fea_len=args.h_fea_len,
                                      n_h=args.n_h)
        if args.cuda:
            t_model.cuda()
            s_model.cuda()

        # define loss func and optimizer
        criterion = nn.CrossEntropyLoss()
        if args.optim == 'SGD':
            t_optimizer = optim.SGD(t_model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
            s_optimizer = optim.SGD(s_model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        elif args.optim == 'Adam':
            t_optimizer = optim.Adam(t_model.parameters(), args.lr,
                                     weight_decay=args.weight_decay)
            s_optimizer = optim.Adam(s_model.parameters(), args.lr,
                                     weight_decay=args.weight_decay)
        else:
            raise NameError('Only SGD or Adam is allowed as --optim')

        # optionally resume from a checkpoint
        if args.teacher and args.student:
            if os.path.isfile(args.teacher) and os.path.isfile(args.student):
                assert str(
                    args.iter) in args.teacher, 'Current iteration != teacher input file iteration'
                assert str(
                    args.iter) in args.student, 'Current iteration != student input file iteration'

                print("=> loading teacher checkpoint '{}'".format(args.teacher))
                t_checkpoint = torch.load(args.teacher)
                args.start_epoch = t_checkpoint['epoch']
                t_best_mae_error = t_checkpoint['best_mae_error']
                t_model.load_state_dict(t_checkpoint['state_dict'])
                t_optimizer.load_state_dict(t_checkpoint['optimizer'])
                t_normalizer.load_state_dict(t_checkpoint['normalizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.teacher, t_checkpoint['epoch']))

                print("=> loading student checkpoint '{}'".format(args.student))
                s_checkpoint = torch.load(args.student)
                args.start_epoch = s_checkpoint['epoch']
                s_best_mae_error = s_checkpoint['best_mae_error']
                s_model.load_state_dict(s_checkpoint['state_dict'])
                s_optimizer.load_state_dict(s_checkpoint['optimizer'])
                s_normalizer.load_state_dict(s_checkpoint['normalizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.student, s_checkpoint['epoch']))
                if t_checkpoint['epoch'] != s_checkpoint['epoch']:
                    raise Exception("=> teacher and student out of sync\nTeacher at epoch: {}\nStudent at epoch: {}"
                                    .format(t_checkpoint['epoch'], s_checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
        elif args.teacher or args.student:
            raise Exception("Must have both teacher and student or neither")

        t_scheduler = MultiStepLR(t_optimizer, milestones=args.lr_milestones,
                                  gamma=0.1)
        s_scheduler = MultiStepLR(s_optimizer, milestones=args.lr_milestones,
                                  gamma=0.1)

        # train models
        mpl(labeled_loader, unlabeled_loader, val_loader, t_model, s_model, criterion,
            t_optimizer, s_optimizer, t_scheduler, s_scheduler, t_normalizer, s_normalizer, mx)

        # test best model
        print('---------Evaluate Model on Test Set---------------')
        if os.path.isfile(f'checkpoints/student_best_{args.iter+mx}.pth.tar'):
            best_checkpoint = torch.load(
                f'checkpoints/student_best_{args.iter+mx}.pth.tar')
        else:
            best_checkpoint = torch.load(
                f'checkpoints/s_checkpoint_{args.iter+mx}.pth.tar')
        s_model.load_state_dict(best_checkpoint['state_dict'])
        # validate(labeled_loader, s_model, criterion, s_normalizer, test=True, predict=False, append=False, mx=mx)
        validate(unlabeled_loader, s_model, criterion, s_normalizer,
                 test=True, predict=False, append=False, mx=mx)
        # validate(test_loader, s_model, criterion, s_normalizer, test=True, predict=True, append=True, mx=mx)


def mpl(labeled_loader, unlabeled_loader, val_loader, t_model, s_model, criterion, t_optimizer, s_optimizer, t_scheduler, s_scheduler, t_normalizer, s_normalizer, mx):
    global t_best_mae_error, s_best_mae_error

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    epoch = -1
    epoch_size = max(len(labeled_loader), len(unlabeled_loader))
    print('Epoch Size: ', epoch_size)
    start_step = args.start_epoch * epoch_size
    num_steps = args.epochs * epoch_size
    print('Start: ', start_step)
    print('Number of Steps: ', num_steps)

    for i in range(start_step, num_steps):
        if i % epoch_size == 0:
            batch_time = AverageMeter()
            data_time = AverageMeter()
            s_losses = AverageMeter()
            t_losses = AverageMeter()
            t_losses_l = AverageMeter()
            t_losses_mpl = AverageMeter()
            accuracies = AverageMeter()
            precisions = AverageMeter()
            recalls = AverageMeter()
            fscores = AverageMeter()
            auc_scores = AverageMeter()
            epoch += 1

        # switch to train mode
        t_model.train()
        s_model.train()
        end = time.time()

        try:
            (l_input, l_target, _) = labeled_iter.next()
        except:
            labeled_iter = iter(labeled_loader)
            (l_input, l_target, _) = labeled_iter.next()
        try:
            (u_input, _, _) = unlabeled_iter.next()
        except:
            unlabeled_iter = iter(unlabeled_loader)
            (u_input, _, _) = unlabeled_iter.next()

        # measure data loading time
        data_time.update(time.time() - end)

        if args.cuda:
            input_var_l = [Variable(l_input[0].cuda(non_blocking=True)),
                           Variable(l_input[1].cuda(non_blocking=True)),
                           l_input[2].cuda(non_blocking=True),
                           [crys_idx.cuda(non_blocking=True) for crys_idx in l_input[3]]]
            input_var_u = [Variable(u_input[0].cuda(non_blocking=True)),
                           Variable(u_input[1].cuda(non_blocking=True)),
                           u_input[2].cuda(non_blocking=True),
                           [crys_idx.cuda(non_blocking=True) for crys_idx in u_input[3]]]
        else:
            input_var_l = [Variable(l_input[0]),
                           Variable(l_input[1]),
                           l_input[2],
                           l_input[3]]
            input_var_u = [Variable(u_input[0]),
                           Variable(u_input[1]),
                           u_input[2],
                           u_input[3]]

        num_crys_l = len(input_var_l[3])
        num_crys_u = len(input_var_u[3])
        input_var = (torch.cat((input_var_l[0], input_var_u[0])),
                     torch.cat((input_var_l[1], input_var_u[1])),
                     torch.cat((input_var_l[2], torch.stack(
                         list(map(lambda x: x + num_crys_l, input_var_u[2]))))),
                     input_var_l[3] + list(map(lambda x: x + num_crys_l, input_var_u[3])))

        # normalize target
        target_normed = l_target.view(-1).long()
        if args.cuda:
            target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            target_var = Variable(target_normed)

        # all teacher calls
        t_logits = t_model(*input_var)
        t_logits_l, t_logits_u = t_logits.split([num_crys_l, num_crys_u])
        t_loss_l = criterion(t_logits_l, target_var)
        del t_logits

        soft_pseudo_label = torch.softmax(t_logits_u.detach(), dim=-1)
        _, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)

        # first student call
        s_logits = s_model(*input_var)
        s_logits_l, s_logits_u = s_logits.split([num_crys_l, num_crys_u])

        s_loss_l_old = F.cross_entropy(s_logits_l.detach(), target_var)
        s_loss_u = criterion(s_logits_u, hard_pseudo_label)

        # apply student loss
        s_optimizer.zero_grad()
        s_loss_u.backward()
        s_optimizer.step()
        s_scheduler.step()

        moving_dot_product = torch.empty(1).to(f'cuda:{args.gpu}')
        limit = 3.0**(0.5)  # 3 = 6 / (f_in + f_out)
        nn.init.uniform_(moving_dot_product, -limit, limit)

        # second student call
        with torch.no_grad():
            s_logits_l = s_model(*input_var_l)
        s_loss_l_new = F.cross_entropy(s_logits_l.detach(), target_var)

        # calculate teacher loss
        dot_product = s_loss_l_new - s_loss_l_old
        moving_dot_product = moving_dot_product * 0.99 + dot_product * 0.01
        dot_product = dot_product - moving_dot_product
        t_loss_mpl = dot_product * \
            F.cross_entropy(t_logits_u, hard_pseudo_label)
        t_loss = t_loss_l + t_loss_mpl

        # apply teacher loss
        t_model.zero_grad()
        t_loss.backward()
        t_optimizer.step()
        t_scheduler.step()

        # measure accuracy and record loss
        accuracy, precision, recall, fscore, auc_score = class_eval(
            F.log_softmax(s_logits_l, dim=1).data.cpu(), l_target)
        s_losses.update(s_loss_u.data.cpu().item(), l_target.size(0))
        t_losses.update(t_loss.data.cpu().item(), l_target.size(0))
        t_losses_l.update(t_loss_l.data.cpu().item(), l_target.size(0))
        t_losses_mpl.update(t_loss_mpl.data.cpu().item(), l_target.size(0))
        accuracies.update(accuracy, l_target.size(0))
        precisions.update(precision, l_target.size(0))
        recalls.update(recall, l_target.size(0))
        fscores.update(fscore, l_target.size(0))
        auc_scores.update(auc_score, l_target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'S-Loss {s_loss.val:.4f} ({s_loss.avg:.4f})\t'
                  'T-Loss {t_loss.val:.4f} ({t_loss.avg:.4f})\t'
                  'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                  'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                  'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                  'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                  'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                      epoch, i % epoch_size, epoch_size, batch_time=batch_time,
                      data_time=data_time, s_loss=s_losses, t_loss=t_losses, accu=accuracies,
                      prec=precisions, recall=recalls, f1=fscores, auc=auc_scores)
                  )

        if i != 0 and i % epoch_size == 0:
            # evaluate on validation set
            t_mae_error = validate(val_loader, t_model,
                                   criterion, t_normalizer)
            s_mae_error = validate(val_loader, s_model,
                                   criterion, s_normalizer)

            if t_mae_error != t_mae_error:
                print('Exit due to NaN')
                sys.exit(1)

            if s_mae_error != s_mae_error:
                print('Exit due to NaN')
                sys.exit(1)

            # remember the best mae_eror and save checkpoint
            t_is_best = t_mae_error > t_best_mae_error
            s_is_best = s_mae_error > s_best_mae_error

            t_best_mae_error = max(t_mae_error, t_best_mae_error)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': t_model.state_dict(),
                'best_mae_error': t_best_mae_error,
                'optimizer': t_optimizer.state_dict(),
                'normalizer': t_normalizer.state_dict(),
                'args': vars(args)
            }, t_is_best, False, f'checkpoints/t_checkpoint_{args.iter+mx}.pth.tar', mx=mx)

            s_best_mae_error = max(s_mae_error, s_best_mae_error)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': s_model.state_dict(),
                'best_mae_error': s_best_mae_error,
                'optimizer': s_optimizer.state_dict(),
                'normalizer': s_normalizer.state_dict(),
                'args': vars(args)
            }, s_is_best, True, f'checkpoints/s_checkpoint_{args.iter+mx}.pth.tar', mx=mx)


def validate(val_loader, model, criterion, normalizer, test=False, predict=False, append=False, mx=0):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    s_losses = AverageMeter()
    accuracies = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()
    fscores = AverageMeter()
    auc_scores = AverageMeter()
    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
        if args.cuda:
            with torch.no_grad():
                input_var = (Variable(input[0].cuda(non_blocking=True)),
                             Variable(input[1].cuda(non_blocking=True)),
                             input[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
        else:
            with torch.no_grad():
                input_var = (Variable(input[0]),
                             Variable(input[1]),
                             input[2],
                             input[3])
        target_normed = target.view(-1).long()
        if args.cuda:
            with torch.no_grad():
                target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            with torch.no_grad():
                target_var = Variable(target_normed)

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        accuracy, precision, recall, fscore, auc_score = class_eval(
            F.log_softmax(output, dim=1).data.cpu(), target)
        s_losses.update(loss.data.cpu().item(), target.size(0))
        accuracies.update(accuracy, target.size(0))
        precisions.update(precision, target.size(0))
        recalls.update(recall, target.size(0))
        fscores.update(fscore, target.size(0))
        auc_scores.update(auc_score, target.size(0))
        if test:
            output = F.log_softmax(output, dim=1)
            test_pred = torch.exp(output.data.cpu())
            test_target = target
            assert test_pred.shape[1] == 2
            test_preds += test_pred[:, 1].tolist()
            test_targets += test_target.view(-1).tolist()
            test_cif_ids += batch_cif_ids

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                  'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                  'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                  'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                  'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=s_losses,
                      accu=accuracies, prec=precisions, recall=recalls,
                      f1=fscores, auc=auc_scores)
                  )

    if test:
        star_label = '**'
        import csv
        if not predict:
            with open(f'results/validation/test_results_{args.iter+mx}.csv', ('w' if not append else 'a'), newline='') as f:
                writer = csv.writer(f)
                for cif_id, pred in zip(test_cif_ids, test_preds):
                    writer.writerow((cif_id, pred))
        else:
            with open(f'results/predictions/predictions_{args.iter+mx}.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                for cif_id, pred in zip(test_cif_ids, test_preds):
                    writer.writerow((cif_id, pred))
    else:
        star_label = '*'

    print(' {star} AUC {auc.avg:.3f}'.format(star=star_label,
                                             auc=auc_scores))
    return auc_scores.avg


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if not target_label.shape:
        target_label = np.asarray([target_label])
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary', zero_division=0)
        try:
            auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        except ValueError:
            # for pu learning unlabeled dataset
            auc_score = 0.0
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, isStudent, filename, mx):
    torch.save(state, filename)
    if is_best:
        if isStudent:
            shutil.copyfile(
                filename, f'checkpoints/student_best_{args.iter+mx}.pth.tar')
        else:
            shutil.copyfile(
                filename, f'checkpoints/teacher_best_{args.iter+mx}.pth.tar')


def adjust_learning_rate(optimizer, epoch, k):
    """Sets the learning rate to the initial LR decayed by 10 every k epochs"""
    assert type(k) is int
    lr = args.lr * (0.1 ** (epoch // k))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
