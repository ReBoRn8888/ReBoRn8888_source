---
title: Azure 学习笔记 VI —— Speech to Text
date: 2020-12-28 15:52:10
tags: [Microsoft, Azure, Speech]
categories: 学习笔记
top_img:
cover: https://rebornas.blob.core.windows.net/rebornhome/AzureSTT/STT.png
---

{% meting "64574" "netease" "song" "autoplay" "mutex:false" "listmaxheight:340px" "preload:auto" "theme:#ad7a86"%}

# 什么是 Azure Speech to Text
**Azure Speech to Text** 官网：[https://azure.microsoft.com/en-us/services/cognitive-services/speech-to-text/](https://azure.microsoft.com/en-us/services/cognitive-services/speech-to-text/)
![](https://rebornas.blob.core.windows.net/rebornhome/AzureSTT/STTHomePage.png)

**Azure Speech to Text**，简称**STT**，是 [Azure Cognitive Service](https://azure.microsoft.com/en-us/services/cognitive-services/) 中的一项 Speech Service，可以用 Azure 中已经预训练好的语言模型或者自定义的模型来将语音转化为文本，常用于 ChatBot 中的数据预处理阶段，将音频信号翻译为计算机可以理解的文本，便于后续的处理，比如后续用 [LUIS](https://www.luis.ai/) 提取意图（Intent）和实体（Entity），进而控制 Bot 进行相应的操作。

**Azure Speech to Text** 具有以下特点：
- 支持超过85种语言：[Language support](https://docs.microsoft.com/en-gb/azure/cognitive-services/speech-service/language-support#speech-to-text)
- 高精度的翻译：使用业界主流的高精度算法，达到 state-of-the-art 效果
- 可定制化语言模型：[Custom Speech](#custom-speech)
- 灵活的部署：可部署在云端或者通过容器部署在边缘设备中

# 支持的编程语言
- C#
- C++
- Go
- Java
- JavaScript
- Objective-C / Swift
- Python
- REST API

# 快速入门
> 此处我们以 C# 为例，其他语言可从对应的 Github Repo 中找到相应的代码

- [Official Tutorial](https://docs.microsoft.com/en-gb/azure/cognitive-services/speech-service/get-started-speech-to-text?tabs=script%2Cbrowser%2Cwindowsinstall&pivots=programming-language-csharp)
- [Source Code](https://github.com/Azure-Samples/cognitive-services-speech-sdk/tree/master/quickstart/csharp/dotnet)

**Azure Speech to Text** 支持两种数据输入的方式：
1. 麦克风：`AudioConfig.FromDefaultMicrophoneInput()`
1. wav 音频文件（必须且只能是 wav 文件，其他格式需要通过工具转换为 wav 后方可识别）：`AudioConfig.FromWavFileInput("PathToFile.wav")`

> 注：以上两种方法只支持单个语句的识别，即末尾沉默时认定为是一个语句的结束，且最长单句时间限定为15秒。若要连续识别，可通过调用`await recognizer.StartContinuousRecognitionAsync();`来实现。（[Tutorial](https://docs.microsoft.com/en-gb/azure/cognitive-services/speech-service/get-started-speech-to-text?tabs=script%2Cbrowser%2Cwindowsinstall&pivots=programming-language-csharp#continuous-recognition)）

# Custom Speech
![](https://rebornas.blob.core.windows.net/rebornhome/AzureSTT/CustomSpeechHomePage.png)

[Custom Speech](https://speech.microsoft.com/customspeech) 是 Azure 提供的用于训练自定义语言模型的一个平台。具有用户友好型UI，对于没有 AI 相关背景的用户也能流畅使用。

虽然 Azure 上已经自带了很多通用的语言模型，对于大部分常见场景都有覆盖，效果也都还不错，但对于某些特定场景，比如某些专业领域的词汇就会有所欠缺。因此我们需要针对这些特定的场景进行定制化的优化，这就需要用我们自己的数据去 finetune 已有的模型，来达到特定场景下的高性能要求。

Custom Speech 提供了如下功能模块：
![](https://rebornas.blob.core.windows.net/rebornhome/AzureSTT/CustomSpeechModules.png)
- **[Data](#data)**：用于数据集的上传，支持单音频用于测试，或者音频+人工标注的文本用于训练
- **[Testing](#testing)**：可直接用 Azure 自带的模型，或者 customized 的模型进行测试
- **Training**：通过音频和对应的标注文本进行 finetune 训练
- **[Deployment](#deployment)**：将训练好的模型进行部署，以便调用
- **Editor**：用于对已上传的数据进行标注

## Data
- 支持的数据类型：[Link](https://docs.microsoft.com/en-gb/azure/cognitive-services/speech-service/how-to-custom-speech-test-and-train#data-types)

- 训练数据存放格式：
![](https://rebornas.blob.core.windows.net/rebornhome/AzureSTT/TrainingSetDirStructure.png)

- 训练数据标注格式：
![](https://rebornas.blob.core.windows.net/rebornhome/AzureSTT/DataLabelDetails.png)

- 一些开源数据集推荐：[OpenSLR](http://www.openslr.org/resources.php)

> 我们将下载下来的开源数据集按照如上格式整理完成后就可以上传到 Custom Speech 进行训练了（当然也可以直接上传纯音频数据，然后使用 Custom Speech 自带的 **Editor** 进行标注）。

## Testing
Word Error Rate（WER）是业界用于评估语言模型的标准指标，公式如下：
$$
WER = \frac{I+D+S}{N} * 100%,
$$
其中：
- **Insertion (I)**：Prediction 相对于 Ground Truth 多出来的单词数
- **Deletion (D)**：Prediction 相对于 Ground Truth 缺少的单词数
- **Substitution (S)**：Prediction 相对于 Ground Truth 错误的单词数
- **N**：总单词数

一个形象的例子如下：
![](https://rebornas.blob.core.windows.net/rebornhome/AzureSTT/IDSdetails.png)

## Deployment
将训练完的模型部署后，可以得到三个 Endpoints 信息：
- **Subscription key**
- **Service Region**
- **Endpoint ID**

在 C# 代码中如下调用即可：
```csharp
var config = SpeechConfig.FromSubscription("YourSubscriptionKey", "YourServiceRegion");
config.EndpointId = "YourEndpointId";
var reco = new SpeechRecognizer(config);
```