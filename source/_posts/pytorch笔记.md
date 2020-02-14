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
import torch
from PIL import Image
import numpy as np

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

# --------------------------------------------------------------------------------------

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

class myLoss(nn.Module):
	def __init__(self):
		super(myLoss, self).__init__()
		
	def forward(self, pred, truth):
		pred = F.log_softmax(pred, 1)
		loss = -torch.sum(pred * truth, 1)
		return loss.mean()

# ------------------------------------------------------------------------

criterion = myLoss()
```