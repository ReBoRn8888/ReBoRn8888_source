---
title: pytorch 笔记
date: 2020-02-14 16:03:19
tags: [pytorch, 深度学习, python]
categories: 学习笔记
top_img:
cover: https://i.loli.net/2020/02/14/YXEOI3Txj7HJqRa.jpg
---

{% meting "1348625245" "netease" "song" "autoplay" "mutex:false" "listmaxheight:340px" "preload:auto" "theme:#ad7a86"%}

# 选择运算设备(CPU/GPU)
```python
def select_device(device='', apex=False, batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = 'Using CUDA ' + ('Apex ' if apex else '')  # apex for mixed precision https://github.com/NVIDIA/apex
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
            print("%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                  (s, i, x[i].name, x[i].total_memory / c))
    else:
        print('Using CPU')

    print('')  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')

# ===========================================================================================

device = select_device(device='cpu')
>>> Using CPU

device = select_device(device='0')
>>> Using CUDA device0 _CudaDeviceProperties(name='GeForce GTX 1060 6GB', total_memory=6144MB)
```

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
> loss_accuracy.png 如下
> ![](https://i.loli.net/2020/04/09/19laYwPiVAn26ZX.png)

# Finetune torchvision.models
```python
from torchvision import models

# 读取pytorch库模型
net = models.resnet18(pretrained=True)
# 修改池化层为全局平均池化
net.avgpool = nn.AdaptiveAvgPool2d((1, 1))
# 修改最后一层全连接层
fc_features = net.fc.in_features
net.fc = nn.Linear(fc_features, len(classes))
# 将模型加载到GPU中
net = net.cuda()
```

# Model summary
```python
import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np


def summary(model, input_size, batch_size=-1, device="cuda"):
    outString = ""
    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    outString += "{}\n".format("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    outString += "{}\n".format(line_new)
    outString += "{}\n".format("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        outString += "{}\n".format(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    outString += "{}\n".format("================================================================")
    outString += "{}\n".format("Total params: {0:,}".format(total_params))
    outString += "{}\n".format("Trainable params: {0:,}".format(trainable_params))
    outString += "{}\n".format("Non-trainable params: {0:,}".format(total_params - trainable_params))
    outString += "{}\n".format("----------------------------------------------------------------")
    outString += "{}\n".format("Input size (MB): %0.2f" % total_input_size)
    outString += "{}\n".format("Forward/backward pass size (MB): %0.2f" % total_output_size)
    outString += "{}\n".format("Params size (MB): %0.2f" % total_params_size)
    outString += "{}\n".format("Estimated Total Size (MB): %0.2f" % total_size)
    outString += "{}\n".format("----------------------------------------------------------------")
    return outString
```
> 示例结果如下
```python
from torchvision import models

net = models.vgg11(pretrained=False).cuda()
print(summary(net, (3, 32, 32)))

# ===========================================================================================

>>> ----------------------------------------------------------------
>>>         Layer (type)               Output Shape         Param #
>>> ================================================================
>>>             Conv2d-1           [-1, 64, 32, 32]           1,792
>>>               ReLU-2           [-1, 64, 32, 32]               0
>>>          MaxPool2d-3           [-1, 64, 16, 16]               0
>>>             Conv2d-4          [-1, 128, 16, 16]          73,856
>>>               ReLU-5          [-1, 128, 16, 16]               0
>>>          MaxPool2d-6            [-1, 128, 8, 8]               0
>>>             Conv2d-7            [-1, 256, 8, 8]         295,168
>>>               ReLU-8            [-1, 256, 8, 8]               0
>>>             Conv2d-9            [-1, 256, 8, 8]         590,080
>>>              ReLU-10            [-1, 256, 8, 8]               0
>>>         MaxPool2d-11            [-1, 256, 4, 4]               0
>>>            Conv2d-12            [-1, 512, 4, 4]       1,180,160
>>>              ReLU-13            [-1, 512, 4, 4]               0
>>>            Conv2d-14            [-1, 512, 4, 4]       2,359,808
>>>              ReLU-15            [-1, 512, 4, 4]               0
>>>         MaxPool2d-16            [-1, 512, 2, 2]               0
>>>            Conv2d-17            [-1, 512, 2, 2]       2,359,808
>>>              ReLU-18            [-1, 512, 2, 2]               0
>>>            Conv2d-19            [-1, 512, 2, 2]       2,359,808
>>>              ReLU-20            [-1, 512, 2, 2]               0
>>>         MaxPool2d-21            [-1, 512, 1, 1]               0
>>> AdaptiveAvgPool2d-22            [-1, 512, 7, 7]               0
>>>            Linear-23                 [-1, 4096]     102,764,544
>>>              ReLU-24                 [-1, 4096]               0
>>>           Dropout-25                 [-1, 4096]               0
>>>            Linear-26                 [-1, 4096]      16,781,312
>>>              ReLU-27                 [-1, 4096]               0
>>>           Dropout-28                 [-1, 4096]               0
>>>            Linear-29                 [-1, 1000]       4,097,000
>>> ================================================================
>>> Total params: 132,863,336
>>> Trainable params: 132,863,336
>>> Non-trainable params: 0
>>> ----------------------------------------------------------------
>>> Input size (MB): 0.01
>>> Forward/backward pass size (MB): 2.94
>>> Params size (MB): 506.83
>>> Estimated Total Size (MB): 509.78
>>> ----------------------------------------------------------------
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
                state = {'net': net.state_dict(), 'opt': optimizer, 'acc': epoch_acc, 'epoch': epoch, 'classes': classes}
                torch.save(state, os.path.join(modelPath, modelName))
                best_acc = epoch_acc
            if(phase == 'val' and epoch == n_epochs - 1):
                finalModelName = 'final-{}'.format(modelName)
                print('Saving final model to {}'.format(os.path.join(modelPath, finalModelName)))
                state = {'net': net.state_dict(), 'opt': optimizer, 'acc': epoch_acc, 'epoch': epoch, 'classes': classes}
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
trainLoader = DataLoader(trainDataset, batch_size=trainBS, shuffle=True, num_workers=8)
testDataset = myDataset(testImage, torch.Tensor(testLabel).long(), classes, transform=myTransform, to_onehot=True)
testLoader = DataLoader(testDataset, batch_size=testBS, shuffle=False, num_workers=8)

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

# Evaluation Template
```python
def get_labels_and_pres(net, dataLoader):
    l, p = [], []
    with torch.no_grad():
        for data in dataLoader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = net(images)
            predicted = torch.argmax(outputs.data, 1)
            labels = torch.argmax(labels.data, 1)
            p.extend(predicted.cpu().numpy())
            l.extend(labels.cpu().numpy())
    return l, p
    
def eval_total(labels, preds):
    start = time.time()
    
    total = len(labels)
    correct = np.equal(labels, preds).sum()
    
    accTotal = 100 * correct / total
    duration = time.time() - start
    print('Accuracy of the network on the {} test images: {:.2f}% ({:.0f}mins {:.2f}s)'.format(total, accTotal, duration // 60, duration % 60))
    return accTotal

def eval_per_class(labels, preds, classes):
    num_classes = len(classes)
    start = time.time()
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))

    c = np.equal(labels, preds)
    for i in range(len(c)):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1
    print('class_correct\t:\t{}'.format(class_correct))
    print('class_total\t:\t{}'.format(class_total))
    accPerClass = dict()
    for i in range(num_classes):
        accPerClass[classes[i]] = 0 if class_correct[i] == 0 else '{:.2f}%'.format(100 * class_correct[i] / class_total[i])
    duration = time.time() - start
    print('Per class accuracy :')
    print(accPerClass)
    print('Duration for accPerClass : {:.0f}mins {:.2f}s'.format(duration // 60, duration % 60))
    return accPerClass

# ===========================================================================================

ckpt = torch.load(os.path.join(modelPath, modelName))
net = ResNet18(num_classes=len(classes), input_shape=(3, imageSize, imageSize))
net.load_state_dict(ckpt['net'])
net = net.cuda()
**net.eval()**
classList = ckpt['classes']

labels, preds = get_labels_and_pres(net, testLoader)
accTotal = eval_total(labels, preds)
accPerClass = eval_per_class(labels, preds, classList)
```