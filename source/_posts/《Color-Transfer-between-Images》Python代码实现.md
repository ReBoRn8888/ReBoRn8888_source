---
title: 《Color_Transfer_between_Images》Python代码实现
date: 2020-03-21 18:46:33
tags: [python, opencv]
categories: 学习笔记
top_img:
cover: https://i.loli.net/2020/03/21/goGBmrjybwk1nOv.png
---

{% meting "28160871" "netease" "song" "autoplay" "mutex:false" "listmaxheight:340px" "preload:auto" "theme:#ad7a86"%}

《Color Transfer between Images》提出了一个非常经典的色彩迁移算法，用来将目标图像的色彩空间迁移到原图像上。
Python的实现代码见：[https://github.com/ReBoRn8888/colorTransfer-python](https://github.com/ReBoRn8888/colorTransfer-python)

# 出发点
- RGB三通道具有很强的相关性，改一个通道就得跟着改另外两个通道，否则图像会失真。
- 在RGB空间改变色彩有困难，因此考虑寻找一个各通道互不相关的色彩空间，$l\alpha\beta$空间就是个不错的选择。
- 在$l\alpha\beta$空间下，改变任何一个通道都不会影响其他通道，从而避免了像改变RGB通道时会导致的失真情况。
- $l$：亮度，$\alpha$：黄蓝通道，$\beta$：红绿通道。

# 算法流程
将RGB空间转换为$l\alpha\beta$空间需要经过一个中间转换过程，即LMS空间，算法的整体流程如下：
$\text{RGB} \Rightarrow \text{LMS} \Rightarrow log_{10}(\text{LMS}) \Rightarrow l\alpha\beta \Rightarrow$ 在 $l\alpha\beta$ 空间下进行色彩变换 $\Rightarrow \text{LMS} \Rightarrow 10^{\text{LMS}} \Rightarrow \text{RGB}$

## 1、RGB $\Rightarrow$ LMS
通过以下矩阵运算将RGB色彩空间转换为LMS色彩空间
![](https://i.loli.net/2020/03/21/eWKXcwqDVps78Lg.png)

```python
def RGB2LMS(RGB):
    RGB2LMSMatrix = np.array([[0.3811, 0.5783, 0.0402], [0.1967, 0.7244, 0.0782], [0.0241, 0.1288, 0.8444]]).astype('float32')
    L = (RGB2LMSMatrix[0]*RGB).sum(2)
    M = (RGB2LMSMatrix[1]*RGB).sum(2)
    S = (RGB2LMSMatrix[2]*RGB).sum(2)
    LMS = cv2.merge([L, M, S])
    LMS[np.where(LMS == 0)] = 1 # 为了防止下一步取对数时0的对数无意义的情况
    return LMS
```

## 2、对LMS空间取以10为底的对数
```python
LMS = np.log10(LMS)
```
## 3、LMS $\Rightarrow$ $l\alpha\beta$
通过以下矩阵运算将LMS色彩空间转换为$l\alpha\beta$色彩空间
![](https://i.loli.net/2020/03/21/VshvNRjzmLwJp4O.png)

```python
def LMS2lab(LMS):
    a = [[1/np.sqrt(3), 0, 0], [0, 1/np.sqrt(6), 0], [0, 0, 1/np.sqrt(2)]]
    b = [[1, 1, 1], [1, 1, -2], [1, -1, 0]]
    c = np.matmul(a, b)
    ll = (c[0]*LMS).sum(2)
    aa = (c[1]*LMS).sum(2)
    bb = (c[2]*LMS).sum(2)
    lab = cv2.merge([ll, aa, bb])
    return lab
```

## 4、在$l\alpha\beta$空间下进行色彩变换
- 求出原图像$src$在$l\alpha\beta$空间下三通道各自的均值$srcMean$和标准差$srcStd$
- 求出目标图像$tar$在$l\alpha\beta$空间下三通道各自的均值$tarMean$和标准差$tarStd$
- 将原图像减去它本身的均值，再除以它本身的标准差：$res = \frac{src - srcMean}{srcStd}$
- 乘以目标图像的标准差，再加上目标图像的均值即可：$res = res * tarStd + tarMean$

```python
def transfer(src, tar):
    srcMean = np.mean(src, axis=(0, 1))
    srcStd = np.std(src, axis=(0, 1))
    tarMean = np.mean(tar, axis=(0, 1))
    tarStd = np.std(tar, axis=(0, 1))

    res = src - srcMean
    res /= srcStd
    res *= tarStd
    res += tarMean
    return res
```

## 5、将经过色彩变换后的图像转回到LMS空间
通过以下矩阵运算将 $l\alpha\beta$ 色彩空间转换为 LMS 色彩空间
![](https://i.loli.net/2020/03/21/Qg28Fo6pDsI3wSJ.png)

```python
def lab2LMS(lab):
    a = [[1, 1, 1], [1, 1, -1], [1, -2, 0]]
    b = [[1/np.sqrt(3), 0, 0], [0, 1/np.sqrt(6), 0], [0, 0, 1/np.sqrt(2)]]
    c = np.matmul(a, b)
    L = (c[0]*lab).sum(2)
    M = (c[1]*lab).sum(2)
    S = (c[2]*lab).sum(2)
    LMS = cv2.merge([L, M, S])
    return LMS
```

## 6、对LMS空间计算10的指数（即还原第二步的操作）
```python
LMS = np.power(10, LMS)
```

## 7、LMS $\Rightarrow$ RGB
通过以下矩阵运算将 LMS 色彩空间转换为 RGB 色彩空间
![](https://i.loli.net/2020/03/21/qbnZmUQ7je6XJ5I.png)

```python
def LMS2RGB(LMS):
    LMS2RGBMatrix = np.array([[4.4679, -3.5873, 0.1193], [-1.2186, 2.3809, -0.1624], [0.0497, -0.2439, 1.2045]]).astype('float32')
    R = (LMS2RGBMatrix[0]*LMS).sum(2)
    G = (LMS2RGBMatrix[1]*LMS).sum(2)
    B = (LMS2RGBMatrix[2]*LMS).sum(2)
    RGB = cv2.merge([R, G, B])
    RGB = np.clip(RGB, 0, 1) # 将像素值截断到0和1之间，避免显示的时候失真
    return RGB
```

# 算法结果演示
![](https://i.loli.net/2020/03/21/kufWEpjV1yRchdw.png)

![](https://i.loli.net/2020/03/21/4KVxS57ZQes6kih.png)

