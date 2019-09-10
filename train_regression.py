import os
import argparse
import torch
import shutil

import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from models.res_unet_regressor import ResUNet
from models.network import *
from dataloader_regressor import *
from loss import *
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main(args):

    if args.task == 'train':
        if not args.train_image_path:
            raise 'train data path should be specified !'
        train_dataset = RoomDataset(file_path=args.train_image_path,
                                    augment=args.augment)

        if args.val_image_path:
            val_dataset = RoomDataset(file_path=args.val_image_path)

        model = ResUNet(3, 1).to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0003)

        if args.resume:
            if not os.path.isfile(args.resume):
                raise '=> no checkpoint found at %s' % args.resume
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            args.best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> loaded checkpoint %s (epoch %d)' % (args.resume, args.start_epoch))

        train(args, model, optimizer, train_dataset, val_dataset)

    else: # test
        if not args.test_image_path: 
            raise '=> test data path should be specified'
        if not args.resume or not os.path.isfile(args.resume):
            raise '=> resume not specified or no checkpoint found'
        test_dataset = RoomDataset(file_path=args.test_image_path)
        model = ResUNet(3, 1).to(device)
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        test(args, model, test_dataset)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def train(args, model, optimizer, train_dataset, val_dataset):

    criterion = nn.L1Loss()
    
    # epoch-wise losses
    train_losses = []
    eval_losses = []

    #curr_lr = learning_rate
    dataloader = DataLoader(batch_size=args.batch_size, shuffle=True,
                                      dataset=train_dataset, num_workers=args.workers)
    
    df_loss = pd.DataFrame()
    best_loss = args.best_loss

    for epoch in range(args.epochs):
        print('training epoch %d/%s' % (args.start_epoch+epoch+1, args.start_epoch+args.epochs))
        batch_train_losses = []
        data_iterator = tqdm(dataloader, total=len(train_dataset) // args.batch_size + 1)
        for i, (images, labels) in enumerate(data_iterator):
        
            model.train()
        
            images = images.to(device)
            labels = labels.to(device)
        
            outputs = model(images)
            
            # depth loss + gradient loss + normal loss
            train_loss = depth_loss(labels, outputs) + gradient_loss(labels, outputs)
    
            # loss
            #train_loss = criterion(outputs, labels)
            batch_train_losses.append(train_loss.item())
        
            # backward
            model.zero_grad()
            train_loss.backward()
            optimizer.step()
        
        # evaluation per epoch
        epo_train_loss = np.mean(batch_train_losses)
        print('mean train loss: %.4f' % epo_train_loss)
        train_losses.append(epo_train_loss)
        
        epo_eval_loss = evaluate(args, model, criterion, val_dataset, args.start_epoch+epoch+1)
        print('mean val loss: %.4f' % epo_eval_loss)
        eval_losses.append(epo_eval_loss)
        
        # update output loss file after per epoch
        df_loss.assign(train=train_losses, val=eval_losses).to_csv('./loss_regression.csv')

        # save model
        is_best = False
        if epo_eval_loss < best_loss:
            best_loss = epo_eval_loss
            is_best = True
        save_checkpoint({
            'epoch': args.start_epoch+epoch+1,
            'best_loss': best_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best)
        
        # update learning rate
        #if (epoch + 1) % 20 == 0:
            #curr_lr /= 3
            #update_lr(optimizer, curr_lr)

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def evaluate(args, model, criterion, val_dataset, epo_no):
    
    losses = []
    dataloader = DataLoader(batch_size=args.batch_size,
                                        dataset=val_dataset, num_workers=args.workers)
    
    model.eval()
    with torch.no_grad():
        print('evaluating...')
        data_iterator = tqdm(dataloader, total=len(val_dataset) // args.batch_size + 1)
        for images, labels in data_iterator:

            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            # depth loss + gradient loss + normal loss
            loss = depth_loss(labels, outputs) + gradient_loss(labels, outputs)
            
            # loss
            #loss = criterion(outputs, labels)
            losses.append(loss.item())
        
        mean_val_loss = np.mean(losses)

        if epo_no % 10 == 0:
            np.save('./val_images_%d.npy' % epo_no, images.cpu().numpy().astype(np.uint8))
            np.save('./val_labels_%d.npy' % epo_no, labels.cpu().numpy())
            np.save('./val_preds_%d.npy' % epo_no, outputs.cpu().numpy())
            
    return mean_val_loss

def test(args, model, test_dataset):
    
    dataloader = DataLoader(batch_size=1, dataset=test_dataset, num_workers=args.workers)
    
    model.eval()
    with torch.no_grad():
        data_iterator = tqdm(dataloader, total=len(test_dataset) // args.batch_size + 1)
        for images, labels in data_iterator:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            np.save('xxx.npy', outputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch UNet Training')

    # Task setting
    parser.add_argument('--task', default='train', type=str,
                        choices=['train', 'test'], help='task')

    # Dataset setting
    parser.add_argument('--train-image-path', default='', type=str,
                        help='path to training images')

    parser.add_argument('--val-image-path', default='', type=str,
                        help='path to validation images')

    parser.add_argument('--test-image-path', default='', type=str,
                        help='path to test images')


    # Training strategy
    parser.add_argument('--solver', metavar='SOLVER', default='rms',
                        choices=['rms', 'adam'],
                        help='optimizers')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch-size', default=8, type=int, metavar='N',
                        help='mini-batch size')
    parser.add_argument('--lr', '--learning-rate', default=2.5e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--schedule', type=int, nargs='+', default=[60, 90],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--target-weight', dest='target_weight',
                        action='store_true',
                        help='Loss with target_weight')
    # Data processing
    parser.add_argument('--augment', dest='augment', action='store_true',
                        help='augment data for training')
    # Miscs
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        help='show intermediate results')
    parser.add_argument('--best-loss', type=float, default=np.float('inf'),
                        help='best (minimum) loss of current model.')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='trained epoch of current model.')



    main(parser.parse_args())
