---
title: pytorch 笔记
date: 2020-02-14 16:03:19
tags: [pytorch, 深度学习]
categories: 学习笔记
top_img:
cover: https://i.loli.net/2020/02/14/YXEOI3Txj7HJqRa.jpg
---

{% aplayer 'Touch off' 'UVERworld' 'http://music.163.com/song/media/outer/url?id=1348625245.mp3' 'https://i.loli.net/2020/02/14/YXEOI3Txj7HJqRa.jpg' autoplay %}

# 自定义 Dataset
```python
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms

class myDataset(Dataset):
    def __init__(self, images, labels, classes=None, transform=None, to_onehot=False):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.classes = classes
        self.num_classes = len(self.classes)
        self.to_onthot = to_onthot

    def int_to_onehot(self, label):
        label = torch.unsqueeze(torch.unsqueeze(label, 0), 1)
        label = torch.zeros(1, self.num_classes).scatter_(1, label, 1)
        label = torch.squeeze(label)
        return label

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        if(self.onehot):
            label = self.int_to_onehot(label)
        if(self.transform):
            image = self.transform(Image.fromarray(np.uint8(image)))
        return image, label

# ===========================================================================================

myTransform = transforms.Compose([
    transforms.ToTensor(),
])
trainDataset = myDataset(trainImage, torch.Tensor(trainLabel).long(), classes, transform=myTransform, to_onehot=True)
trainLoader = DataLoader(trainDataset, batch_size=trainBS, shuffle=True, num_workers=1)

testDataset = myDataset(testImage, torch.Tensor(testLabel).long(), classes, transform=myTransform, to_onehot=True)
testLoader = DataLoader(testDataset, batch_size=testBS, shuffle=False, num_workers=1)
```

# 自定义 Loss
```python
import torch.nn.functional as F
import torch
from torch import nn

class myLoss(nn.Module):
	def __init__(self):
		super(myLoss, self).__init__()
		
	def forward(self, pred, truth):
		pred = F.log_softmax(pred, 1)
		loss = -torch.sum(pred * truth, 1)
		return loss.mean()

# ===========================================================================================

criterion = myLoss()
```
# AverageMeter()
```python
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n                 # 元素总和
        self.count += n                     # 元素总数
        self.avg = self.sum / self.count    # 平均值
```

# 时间转换
```python
def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600*need_hour) / 60)
    need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
    return need_hour, need_mins, need_secs
```

# TopK Accuracy
```python
def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k
    output (tensor): model output
    target (tensor): ground truth label (not onehot)
    """

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k)
    return res

# ===========================================================================================

outputs = net(inputs)
prec1, prec5 = accuracy(outputs, torch.argmax(labels, 1), topk=(1, 5))
```

# Accuracy/Loss 曲线
```python
import matplotlib.pyplot as plt
import os

def plot_acc_loss(log, type, modelPath, prefix='', suffix='', printFlag=False):
    trainAcc = log['acc']['train']
    trainLoss = log['loss']['train']
    valAcc = log['acc']['val']
    valLoss = log['loss']['val']
    if(type == 'loss'):
        plt.figure(figsize=(7, 5))
        plt.plot(trainLoss, label='Train_loss')
        plt.plot(valLoss, label='Test_loss')
        plt.title('Epoch - Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        figName = '{}loss{}.png'.format(prefix, suffix)
    elif(type == 'accuracy'):
        plt.figure(figsize=(7, 5))
        plt.plot(trainAcc, label='Train_acc')
        plt.plot(valAcc, label='Test_acc')
        plt.title('Epoch - Acuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
    #         plt.ylim(0, 1.01)
        plt.legend()
        figName = '{}accuracy{}.png'.format(prefix, suffix)
    elif(type == 'both'):
        fig, ax1 = plt.subplots(figsize=(7, 5)) # 分双轴分别画loss和acc
        ax2 = plt.twinx()
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax2.set_ylabel('Accuracy')
        plt.ylim(0, 1.01)
        plt.title('Loss & Accuracy')

        l_trainLoss, = ax1.plot(trainLoss)
        l_testLoss, = ax1.plot(valLoss)
        l_trainAcc, = ax2.plot(trainAcc, marker='x')
        l_testAcc, = ax2.plot(valAcc, marker='x')
        plt.legend([l_trainLoss, l_testLoss, l_trainAcc, l_testAcc],
                  ['Train_loss', 'Test_loss', 'Train_acc', 'Test_acc'])
        figName = '{}loss_accuracy{}.png'.format(prefix, suffix)

    plt.grid(linewidth=1, linestyle='-.')
    plt.savefig(os.path.abspath(figName), dpi=200, bbox_inches='tight')
    if(printFlag):
        print("Figure saved to : {}".format(os.path.abspath(figName)))
    plt.close()
```

# Finetune torchvision.models
```python
from torchvision import models

net = models.resnet18(pretrained=True)
fc_features = net.fc.in_features
net.fc = nn.Linear(fc_features, len(classes))
net.avgpool = nn.AdaptiveAvgPool2d((1, 1))
net = net.cuda()
```

# Train/Validate Template
```python
from tqdm import tqdm
import pickle as pkl
import torch
import sys
import time
import os

def train_val(net, optimizer, n_epochs, trainBS, valBS, trainDataset, trainLoader, valDataset, valLoader, expLrScheduler, modelPath, modelName):
    """
    net (torchvision.models) : Net model for training/validation
    optimizer (torch.optim) : Optimizer for training
    n_epochs (int) : Training epochs
    trainBS (int) : Training batchsize
    valBS (int) : Validation batchsize
    trainDataset (torch.Dataset)
    trainLoader (torch.DataLoader)
    valDataset (torch.Dataset)
    valLoader (torch.DataLoader)
    expLrScheduler (torch.optim.lr_scheduler) : Learning rate decay strategy
    modelPath (str) : Path to save the learnt model
    modelName (str) : Model name
    """
    lossLog = dict({'train': [], 'val': []})
    accLog = dict({'train': [], 'val': []})
    dataSet = {'train': trainDataset, 'val': valDataset}
    dataLoader = {'train': trainLoader, 'val': valLoader}
    dataSize = {x: dataSet[x].__len__() for x in ['train', 'val']}
    batchSize = {'train': trainBS, 'val': valBS}
    iterNum = {x: np.ceil(dataSize[x] / batchSize[x]).astype('int32') for x in ['train', 'val']}

    print('dataSize: {}'.format(dataSize))
    print('batchSize: {}'.format(batchSize))
    print('iterNum: {}'.format(iterNum))

    best_acc = 0.0
    start = time.time()
    for epoch in tqdm(range(n_epochs), desc='Epoch'):  # loop over the dataset multiple times
        print('Epoch {}/{}, lr = {}  [best_acc = {:.4f}%]'.format(epoch+1, n_epochs, optimizer.param_groups[0]['lr'], best_acc))
        print('-' * 10)
        epochStart = time.time()
        for phase in ['train', 'val']:
            if(phase == 'train'):
                expLrScheduler.step() # Update learning_rate
                net.train()  # Set model to training mode
            else:
                net.eval()   # Set model to evaluate mode
            losses = AverageMeter() # Init AverageMeter for loss
            top1 = AverageMeter() # Init AverageMeter for top1 accuracy

            for i, data in enumerate(dataLoader[phase], 0):
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()
                with torch.set_grad_enabled(phase == 'train'):
                    # Feed forward
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)

                    # Feed backward
                    if(phase == 'train'):
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                losses.update(loss.item()*inputs.size(0), inputs.size(0))

                # 实现刷新的效果 (/r)
                sys.stdout.write('                                                                                                  \r')
                sys.stdout.flush()
                # Calculate top1 accuracy for single-label classification
                prec1 = accuracy(outputs, torch.argmax(labels, 1), topk=(1,))[0]
                top1.update(prec1.item(), inputs.size(0))
                sys.stdout.write('Iter: {} / {} ({:.0f}s)\tLoss= {:.4f} ({:.4f})\tAcc= {:.2f}% ({:.0f}/{:.0f})\r'
                     .format(i+1, iterNum[phase], time.time() - epochStart, loss.item(), losses.avg, prec1/inputs.size(0)*100, top1.sum, top1.count))
                sys.stdout.flush()
            # 实现刷新的效果 (/r)
            sys.stdout.write('                                                                                                  \r')
            sys.stdout.flush()

            epoch_loss = losses.avg
            epoch_acc = top1.avg*100
            accLog[phase].append(epoch_acc/100)
            lossLog[phase].append(epoch_loss)
            epochDuration = time.time() - epochStart
            epochStart = time.time()
            hour, minute, second = convert_secs2time(epochDuration)
            print('[ {} ]  Loss: {:.4f} Acc: {:.3f}% ({:.0f}/{:.0f}) ({:.0f}h {:.0f}m {:.2f}s)'
                .format(phase, epoch_loss, epoch_acc, top1.sum, top1.count, hour, minute, second))

            # Save models
            if(phase == 'val' and epoch_acc > best_acc):
                print('Saving best model to {}'.format(os.path.join(modelPath, modelName)))
                state = {'net': net, 'opt': optimizer, 'acc': epoch_acc, 'epoch': epoch, 'classes': classes}
                torch.save(state, os.path.join(modelPath, modelName))
                best_acc = epoch_acc
            if(phase == 'val' and epoch == n_epochs - 1):
                finalModelName = 'final-{}'.format(modelName)
                print('Saving final model to {}'.format(os.path.join(modelPath, finalModelName)))
                state = {'net': net, 'opt': optimizer, 'acc': epoch_acc, 'epoch': epoch, 'classes': classes}
                torch.save(state, os.path.join(modelPath, finalModelName))
        print('')

        # Save logs
        log = dict({'acc': accLog, 'loss': lossLog})
        with open(os.path.join(modelPath, 'log.pkl'), 'wb') as f:
            pkl.dump(log, f)
            if(epoch + 1 == n_epochs):
                print("Training logs saved to : {}".format(os.path.join(modelPath, 'log.pkl')))
        plot_acc_loss(log, 'both', modelPath, '{}_'.format(modelName), '', (epoch + 1 == n_epochs))
        plot_acc_loss(log, 'loss', modelPath, '{}_'.format(modelName), '', (epoch + 1 == n_epochs))
        plot_acc_loss(log, 'accuracy', modelPath, '{}_'.format(modelName), '', (epoch + 1 == n_epochs))
    duration = time.time() - start
    print('Training complete in {:.0f}h {:.0f}m {:.2f}s'.format(duration // 3600, (duration % 3600) // 60, duration % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return best_acc

# ===========================================================================================

#---- Create Dataset and DataLoader ----#
trainBS = 100
testBS = 100
myTransform = transforms.Compose([
    transforms.ToTensor(),
])

trainDataset = myDataset(trainImage, torch.Tensor(trainLabel).long(), classes, transform=myTransform, to_onehot=True)
trainLoader = DataLoader(trainDataset, batch_size=trainBS, shuffle=True, num_workers=1)
testDataset = myDataset(testImage, torch.Tensor(testLabel).long(), classes, transform=myTransform, to_onehot=True)
testLoader = DataLoader(testDataset, batch_size=testBS, shuffle=False, num_workers=1)

#---- Create model ----#
net = models.resnet18(pretrained=False)
fc_features = net.fc.in_features
net.fc = nn.Linear(fc_features, len(classes))
net.avgpool = nn.AdaptiveAvgPool2d((1, 1))
net = net.cuda()

#---- Set training strategies ----#
lr = 0.1
momentum = 0.9
weightDecay = 5e-4
criterion = myLoss().cuda()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weightDecay, nesterov=True)
expLrScheduler = optim.lr_scheduler.MultiStepLR(optimizer, [50, 75], gamma=0.1)

modelPath = 'xxxxxxxxxxxxxx/pytorch_model_learnt'
modelName = 'resnet18.ckpt'

#---- Start training and validation ----#
best_acc = train_val( net,
                      optimizer, 
                      10, 
                      trainBS,
                      testBS,
                      trainDataset, 
                      trainLoader, 
                      testDataset, 
                      testLoader,
                      expLrScheduler,
                      modelPath,
                      modelName)
```