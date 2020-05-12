---
title: Azure 使用笔记① —— 初见 Azure
date: 2020-05-10 12:03:52
tags: [Microsoft, Azure, CognitiveService]
categories: 学习笔记
top_img:
cover: https://i.loli.net/2020/05/10/y9H2VhYWEjJdNXr.jpg
---

{% meting "1440622059" "netease" "song" "autoplay" "mutex:false" "listmaxheight:340px" "preload:auto" "theme:#ad7a86"%}

# Microsoft Azure
Microsoft Azure 是微软提供的一整套云计算服务，它可以被用来创建云中运行的应用或者通过基于云的特性来加强现有应用。

# 创建 Azure 账号
进入如下网址创建Azure账号即可开始使用： [https://azure.microsoft.com/en-us/free/](https://azure.microsoft.com/en-us/free/)
![](https://i.loli.net/2020/05/10/M7RAp6NuSxly5FE.png)
可以选择“Start free”试用12个月的免费账号，体验25个永久免费的服务，并在30天内拥有$200的Azure免费额度。

# Azure portal 首页导览
可以通过[https://ms.portal.azure.com](https://ms.portal.azure.com)进入Azure首页
![](https://i.loli.net/2020/05/10/x9bojerSU8COdRw.png)

# Azure 尝鲜 —— CustomVision
Microsoft Azure 上的内容非常丰富，这里我们先介绍其中一个服务 —— **Custom Vision**。
## 什么是CustomVision
[Custom Vision](https://azure.microsoft.com/en-us/services/cognitive-services/custom-vision-service/)：
- 是一个用于计算机视觉任务的端到端平台
- 支持图像分类以及目标检测任务
- 支持训练、测试、部署
- 支持通过SDK在自己的项目中调用已训练完的模型

官方说明文档：[https://docs.microsoft.com/en-us/azure/cognitive-services/custom-vision-service/](https://docs.microsoft.com/en-us/azure/cognitive-services/custom-vision-service/)
Quickstarts：[https://docs.microsoft.com/en-us/azure/cognitive-services/custom-vision-service/getting-started-build-a-classifier](https://docs.microsoft.com/en-us/azure/cognitive-services/custom-vision-service/getting-started-build-a-classifier)

## 为什么要用CustomVision
- 无需自己搭建繁杂的深度学习后端算法
- 可直接用于项目的快速验证
- 齐全的模型评估指标，便于有效评估模型
- 深度学习小白也能使用，非常友好

## 开始使用CustomVision
### 1、创建Custom Vision resource
- 在上方搜索栏输入`custom vision`，找到并点击即可进入资源创建界面（[传送门](https://portal.azure.com/#create/Microsoft.CognitiveServicesCustomVision)）
<img src="https://i.loli.net/2020/05/10/NcGlqJ2OgWARHDV.png" style="zoom: 75%;" />
- 填写相应信息后点击`Review + create`进行创建
![](https://i.loli.net/2020/05/10/HA5YdI8oPTbFENM.png)
- 创建完毕后显示如下界面，即表示创建成功
![](https://i.loli.net/2020/05/10/3DoAKPMHFy9TOiB.png)

### 2、创建新项目
进入[https://customvision.ai/](https://customvision.ai/)，使用相同账号登录后，即可进入Custom Vision首页
- 点击`NEW PROJECT`
![](https://i.loli.net/2020/05/10/aEr8FWUAXwV1Bzc.png)
- 输入Project相关信息后，点击`Create project`
<img src="https://i.loli.net/2020/05/10/62jDFBT9yPRbSJQ.png" style="zoom: 65%;" />
- 进入Project页面，接下来自由发挥即可，详细教程查看[官方文档](https://docs.microsoft.com/en-us/azure/cognitive-services/custom-vision-service/getting-started-build-a-classifier)
![](https://i.loli.net/2020/05/10/tBf4cH2yDwKALXn.png)

### 3、SDK调用
> 考虑到AI算法多是用Python编写，因此本文主要介绍Python SDK

- 训练部分不再赘述
- Publish已训练好的模型
![](https://i.loli.net/2020/05/10/VmYaNOkZbrdBI4P.png)
- Publish完成后，显示如下界面，记住图中红框处的`Published as`，之后调用SDK要用
![](https://i.loli.net/2020/05/10/tZ3wRTalFcQGCIe.png)
- 点击“小齿轮”，找到并记录图中红框处的`Project id`，之后调用SDK要用
![](https://i.loli.net/2020/05/10/YCrRyPnDK1FWpOg.png)
- 在Azure portal中找到之前创建的“Prediction resource”，进入“Keys and Endpoint”，里面可以找到`Key`和`Endpoint`，记下来，之后调用SDK要用
![](https://i.loli.net/2020/05/10/JkL7nFDd6N5ZPVI.png)
- 同理，在“Properties”里可以找到`Resource ID`，记下来，之后调用SDK要用
![](https://i.loli.net/2020/05/10/1zMCE9eYU3GWfH5.png)
- 下面，我们使用Python SDK调用已训练好并Publish的模型对本地数据进行预测
	- 首先pip安装azure-cognitiveservices-vision-customvision包：
	```bash
pip install azure-cognitiveservices-vision-customvision
	```
	> 若下载超时或无法下载，可以去Pypi下载离线安装包：[https://pypi.org/project/azure-cognitiveservices-vision-customvision/#files](https://pypi.org/project/azure-cognitiveservices-vision-customvision/#files)，如图：
	![](https://i.loli.net/2020/05/10/D69BukqZQax4SRb.png)
	```bash
pip install azure_cognitiveservices_vision_customvision-1.0.0-py2.py3-none-any.whl
	```

	- 新建predict.py文件，输入如下代码，填入对应的信息
		- publish_iteration_name -- `Published as`
		- projectID -- `Project id`
		- ENDPOINT -- `Endpoint`
		- prediction_key -- `Key`
		- prediction_resource_id -- `Resource ID`
	```python
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient

# Replace with a valid key
publish_iteration_name = "your iteration name"
projectID = "your project id"
ENDPOINT = "your API endpoint"
prediction_key = "your prediction key"
prediction_resource_id = "your prediction resource id"

predictor = CustomVisionPredictionClient(prediction_key, endpoint=ENDPOINT)

with open("sample.jpg", "rb") as image_contents:
    results = predictor.classify_image(
        projectID, publish_iteration_name, image_contents.read())

    # Display the results.
    for prediction in results.predictions:
        print("\t" + prediction.tag_name +
              ": {0:.2f}%".format(prediction.probability * 100))
	```
	- 运行结果如下
	```bash
python predict.py

>>> 	lying: 100.00%
>>> 	standing: 0.00%
	```

	- 这里只介绍Prediction部分的SDK调用，其他包括Training、Testing等调用方法请参照[官方SDK说明文档](https://docs.microsoft.com/en-gb/azure/cognitive-services/Custom-Vision-Service/quickstarts/image-classification?pivots=programming-language-python)