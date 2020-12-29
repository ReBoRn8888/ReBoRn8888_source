---
title: Azure 学习笔记 VII —— Text to Speech
date: 2020-12-29 11:51:17
tags: [Microsoft, Azure, Speech]
categories: 学习笔记
top_img:
cover: https://rebornas.blob.core.windows.net/rebornhome/AzureTTS/TTSLogo.png
---

{% meting "28613731" "netease" "song" "autoplay" "mutex:false" "listmaxheight:340px" "preload:auto" "theme:#ad7a86"%}

# 什么是 Azure Text to Speech
**Azure Text to Speech** 官网：[https://azure.microsoft.com/en-gb/services/cognitive-services/text-to-speech/](https://azure.microsoft.com/en-gb/services/cognitive-services/text-to-speech/)
![](https://rebornas.blob.core.windows.net/rebornhome/AzureTTS/TTSHomePage.png)

**Azure Text to Speech** 简称 **TTS**，是 [Azure Cognitive Service](https://azure.microsoft.com/en-us/services/cognitive-services/) 中的一项 Speech Service，可以用 Azure 中已经预训练好的语言模型或者自定义的模型来将文本转化为独具特色的语音，常用于 ChatBot 或者智能语音客服中的 response 阶段，将用户所寻求的回答用语音的方式呈现出来。

**Azure Text to Speech** 具有以下特点：
- 流利自然的语音输出，符合人类听觉习惯
- 支持200多种人物语音和超过50多种语言：[Language support](https://docs.microsoft.com/en-gb/azure/cognitive-services/speech-service/language-support#text-to-speech)
- 可自定义输出语音：[Custom Voice](#custom-voice)
- 方便调整输出格式：语速、音调、口音等

# 支持的编程语言
与 STT 一样，TTS 同样支持以下几种语言：
- C#
- C++
- Go
- Java
- JavaScript
- Objective-C / Swift
- Python
- REST API

# 快速入门
## 相关资源
> 此处我们以 C# 为例，其他语言可从对应的官方 Github Repo 中找到相应的代码

- Official Tutorial：[Microsoft-Doc](https://docs.microsoft.com/en-gb/azure/cognitive-services/speech-service/get-started-text-to-speech?tabs=script%2Cwindowsinstall&pivots=programming-language-csharp)
- Source Code：[Github-Repo](https://github.com/Azure-Samples/cognitive-services-speech-sdk/tree/master/quickstart/csharp/dotnet/text-to-speech)
- 根据 Tutorial 自己实现的 Sample Code：[DevOps-Repo](https://dev.azure.com/RebornAzureLearning/AzureSpeechService/_git/TTS)

## 输出方式
**Azure Text to Speech** 支持两种数据输出的方式：
1. 输出到麦克风：不指定`AudioConfig`即可从麦克风直接播放
1. 输出到音频文件（可以是 .wav/.mp3）：`AudioConfig.FromWavFileOutput("PathToFile.wav")`，格式可通过`config.SetSpeechSynthesisOutputFormat()`方法来进行修改，支持的格式类型可查看：[Audio Format](https://docs.microsoft.com/en-us/dotnet/api/microsoft.cognitiveservices.speech.speechsynthesisoutputformat?preserve-view=true&view=azure-dotnet)

## 通过SSML自定义语音特征
Speech Synthesis Markup Language (**SSML**) 语音合成标记语言，允许我们微调TTS输出语音的音节、发音、语速、音量等特征。

详细 SSML 的配置选项可参考：[SSML-Doc](https://docs.microsoft.com/en-gb/azure/cognitive-services/speech-service/speech-synthesis-markup?tabs=csharp)

一个简单的 SSML 示例如下：
> 复制的话记得将“< /”中间的空格删掉，这里加上空格是为了防止浏览器将其误渲染。
```xml
<speak version="1.0" xmlns="https://www.w3.org/2001/10/synthesis" xml:lang="en-US">
  <voice name="en-GB-George-Apollo">
    When you're on the motorway, it's a good idea to use a sat-nav.
  < /voice>
< /speak>
```
其中`<voice>`标签用来更改输出的语音类型，上例中用的是一个名叫**George**的男性英语口音，可从[Support Voice](https://docs.microsoft.com/en-gb/azure/cognitive-services/speech-service/language-support#standard-voices)中查看所有支持的选项，包含涉及45种语言的75种语音可供选择。

我们运行程序，试听一下输出的音频，会感觉到语速过快，我们可以通过添加一个`<prosody>`标签来设置语速，并且在句子中的逗号处停顿过短，我们也可以通过添加`<break>`标签来设置延迟，修改后的 SSML 文件如下，我们设置语速为原先的 0.9 倍，在逗号处停顿 0.2 秒：
> 复制的话记得将“< /”中间的空格删掉，这里加上空格是为了防止浏览器将其误渲染。
```xml
<speak version="1.0" xmlns="https://www.w3.org/2001/10/synthesis" xml:lang="en-US">
  <voice name="en-GB-George-Apollo">
    <prosody rate="0.9">
      When you're on the motorway,<break time="200ms"/> it's a good idea to use a sat-nav.
    < /prosody>
  < /voice>
< /speak>
```

## 神经语言(Neural Voices)
神经语言(Neural Voices) 是通过深度神经网络来生成的语言，和真人说话的声音和语调更加相似，随着类人的自然韵律和字词的清晰发音，用户在与 AI 系统交互时，神经语音显著减轻了听力疲劳。

我们通过修改 SSML 文件的 `<voice>` 标签中的 `name` 属性来选择一种支持的神经语言（[Support List](https://docs.microsoft.com/en-gb/azure/cognitive-services/speech-service/language-support#neural-voices)），并通过添加 `<mstts:express-as>` 来改变说话的语气，以下例子中只用了 `cheerful` 语气，我们也可以试试 `customerservice` 或者 `chat` 来听听有何不同。
> 复制的话记得将“< /”中间的空格删掉，这里加上空格是为了防止浏览器将其误渲染。
```xml
<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">
  <voice name="en-US-AriaNeural">
    <mstts:express-as style="cheerful">
      This is awesome!
    < /mstts:express-as>
  < /voice>
< /speak>
```

# Custom Voice
![](https://rebornas.blob.core.windows.net/rebornhome/AzureTTS/CustomVoiceHomePage.png)

[Custom Voice](https://speech.microsoft.com/customvoice) 是 Azure 提供的用于自定义语音的一个平台。使用 Custom Voice 可以为客户的自有品牌创建可识别的独一无二的语音，我们只需准备好几个音频文件和关联的转录文本即可实现，通过用户友好型 UI，对于没有 AI 相关背景的用户也能流畅使用。

下图展示了 **Custom Voice** 的 Workflow：
![](https://rebornas.blob.core.windows.net/rebornhome/AzureTTS/CustomVoiceDiagram.png)

**Custom Voice** 提供了如下功能模块：
![](https://rebornas.blob.core.windows.net/rebornhome/AzureTTS/CustomVoiceModules.png)
- **[Data](#data)**：用于数据集的上传，支持单音频用于测试，或者音频+人工标注的文本用于训练
- **[Model](#model)**：可直接用 Azure 自带的模型，或者 customized 的模型进行测试
- **[Endpoint](#endpoint)**：将训练好的模型进行部署，以便调用

## Data
支持的数据类型：

**Custom Voice** 支持三种类型的数据，详情可查看[（Link）](https://docs.microsoft.com/en-gb/azure/cognitive-services/speech-service/how-to-custom-voice-prepare-data)：
1. 短音频（≤ 15s）和对应的文本
1. 长音频（≥ 20s）和对应的文本
1. 单独的音频文件
> - 对于第二种数据类型，**Custom Voice** 会在后台自动对其进行切分，将其变成第一种类型
> - 对于第三种数据类型，**Custom Voice** 会在后台自动进行转录生成对应的文本，若音频太长，还会自动进行分割，将其变成第一种类型
> - 以上几种类型都要求将所有的音频打包成一个zip文件，所有的转录文本打包成一个zip文件，通过文件名来进行音频和文本的配对。

如下图所示，我们将一个 44s 长度的音频按照第二种方式上传后，**Custom Voice** 会将其自动分成合适长度的 4 段短音频，并会计算相应的指标，用作对于该数据集质量的评估：
![](https://rebornas.blob.core.windows.net/rebornhome/AzureTTS/CustomVoiceDataSample.png)

各指标如下：
- **Pronunciation score**：发音得分，用来评判发音是否标准，越高越好，尽量需要保证 ≥ 70
- **Signal-noise ratio (SNR)**：信噪比，声音信号与噪声的比值，越大越好，尽量需要保证 ≥ 20
- **Duration**：音频时长，尽量需要保证 ≤ 15s


## Model
Model 模块包含如下三个阶段：
1. **Training**
	对于 en-US 和 zh-CN 语言，可以使用任意数量的样本来训练模型；对于其他语言，至少需要 2000 个样本才能训练。
1. **Testing**
	训练完后，系统会自动生成100个随机样本进行测试，可以点进去听一下效果。
1. **deployment**
	如果效果合适，则可将模型部署，便于代码中进行调用

## Endpoint
模型部署完后会生成一个 endpoint 节点，参照如下代码即可使用该模型生成语音。

![](https://rebornas.blob.core.windows.net/rebornhome/AzureTTS/CustomVoiceEndpoint.png)