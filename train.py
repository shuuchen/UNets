import os
import argparse
import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from models.res_unet import ResUNet
from dataloader import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device, '-'*20)

def main(args):

    if args.task == 'train':
        if not args.train_image_path:
            raise 'train data path should be specified !'
        train_dataset = RoomDataset(file_path=args.train_image_path,
                                    augment=args.augment)

        train_dataloader = DataLoader(batch_size=args.batch_size, shuffle=True,
                                      dataset=train_dataset, num_workers=args.workers)

        if args.val_image_path:
            val_dataset = RoomDataset(file_path=args.val_image_path)
            val_dataloader = DataLoader(batch_size=args.batch_size,
                                        dataset=val_dataset, num_workers=args.workers)

        # if resume TODO

        model = ResUNet(3, 12).to(device)
        train(args, model, train_dataloader, val_dataloader)

    else: # test
        if not args.test_image_path: 
            raise 'test data path should be specified !'
        test_dataset = RoomDataset(file_path=args.test_image_path)
        test_dataloader = DataLoader(batch_size=1, dataset=test_dataset, num_workers=args.workers)
        model = ResUNet(3, 12).to(device)
        model.load_state_dict(torch.load(args.checkpoint + '/checkpoint_100.pth'))
        test(model, test_dataloader)

def train(args, model, train_dataloader, val_dataloader):

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # epoch-wise losses
    train_losses = []
    eval_losses = []

    #curr_lr = learning_rate

    for epoch in range(args.epochs):
        batch_train_losses = []
        for i, (images, labels) in enumerate(train_dataloader):
        
            model.train()
        
            images = images.to(device)
            labels = labels.to(device)
        
            outputs = model(images)
    
            # loss
            train_loss = criterion(outputs, labels)
            batch_train_losses.append(train_loss)
        
            # backward
            model.zero_grad()
            train_loss.backward()
            optimizer.step()
        
        # evaluation per epoch
        eval_loss = evaluate(model, criterion, val_dataloader)
        eval_losses.append(eval_loss)
        train_losses.append(np.mean(batch_train_losses))

    # save model    
    if (epoch + 1) % 100 == 0:
        torch.save(model.state_dict(), args.checkpoint + '/checkpoint_%d.pth' % (epoch + 1))
        torch.save(optimizer.state_dict(), args.checkpoint + '/optim_%d.pth' % (epoch + 1))
        
    # update learning rate
    if (epoch + 1) % 20 == 0:
        curr_lr /= 3
        #update_lr(optimizer, curr_lr)

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def evaluate(model, criterion, val_dataloader):
    
    model.eval()
    
    losses = []

    with torch.no_grad():

        for images, labels in val_dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)
            losses.append(loss)
            
    return np.mean(losses)

def test(model, test_dataloader):
    model.eval()

    with torch.no_grad():
        for images, labels in test_dataloader:

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
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
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


    main(parser.parse_args())
