---
title: sklearn 笔记
date: 2020-02-14 13:50:15
tags: [python, 机器学习]
categories: 学习笔记
top_img:
cover: https://i.loli.net/2020/02/14/8e36MDvRhumIkZO.jpg
---

{% aplayer '夢に形はないけれど' '花たん' 'http://music.163.com/song/media/outer/url?id=27548281.mp3' 'https://i.loli.net/2020/02/14/8e36MDvRhumIkZO.jpg' autoplay %}

# 聚类
## K-Means

## DBSCAN

## T-SNE降维 + 聚类结果可视化

# 分类
## 评估指标

### Confusion Matrix（混淆矩阵）

### Precision、Recall、F1-score（P、R、F1）

## SVM

## 保存模型
```python
from sklearn.externals import joblib

model = xxxxx
model.fit(X)
joblib.dump(model, 'path/to/save')
```

## 加载模型
```python
from sklearn.externals import joblib

model = joblib.load('path/to/model')
```

