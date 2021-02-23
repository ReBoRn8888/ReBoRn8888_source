---
title: Azure 学习笔记 VIII —— Speech Translation
date: 2020-12-30 15:10:27
tags: [Microsoft, Azure, Speech]
categories: 学习笔记
top_img:
cover: https://rebornas.blob.core.windows.net/rebornhome/AzureSpeechTranslation/SpeechTranslationLogo.png
---

{% meting "454035505" "netease" "song" "autoplay" "mutex:false" "listmaxheight:340px" "preload:auto" "theme:#ad7a86"%}

# 什么是 Azure Speech Translation
**Azure Speech Translation** 官网：[https://azure.microsoft.com/en-gb/services/cognitive-services/speech-translation/](https://azure.microsoft.com/en-gb/services/cognitive-services/speech-translation/)
![](https://rebornas.blob.core.windows.net/rebornhome/AzureSpeechTranslation/SpeechTranslationHomePage.png)

**Azure Speech Translation**，是 [Azure Cognitive Service](https://azure.microsoft.com/en-us/services/cognitive-services/) 中的一项 Speech Service，可以用 Azure 中已经预训练好的语言模型或者自定义的模型，来将**源语言的音频**转换为**目标语言的音频**，常用于同声翻译等场景，实时翻译语音。

## Workflow
**Azure Speech Translation** 的工作流如下图所示，可以看到它主要由4个模块组成，包括前两篇文章提到过的 [**Speech to Text**](http://www.reborn8888.com/2020/12/28/Azure-6-STT/)、[**Text to Speech**](http://www.reborn8888.com/2020/12/29/Azure-7-TTS/) 这两个模块用于输入和输出，**Speech Correction** 模块用于规范化文本（比如删除口头禅）并自动更正某些明显的语病，**Machine Translation** 模块进行文本翻译。
![](https://rebornas.blob.core.windows.net/rebornhome/AzureSpeechTranslation/SpeechTranslationWorkflow.png)

## 特点
**Azure Speech Translation** 具有以下特点：
- 支持翻译 **30** 多种语言的音频
- 神经网络机器翻译技术提供**快速**、**可靠**的翻译
- **自定义**翻译：可定制模型以识别特定域的术语和独特的说话风格
- **规范化**文本：通过经过训练的引擎，自动规范化语音输出
- 数据**安全性**：在处理期间不会记录任何语音数据


# Custom Translator
**Custom Translator** 有利于我们训练自己的机器翻译模型，比如对于特定场景，使用通用翻译模型，在特定词句或语序上会有略微欠缺，这时候就需要我们自己去搜集并构造双语语料，并使用 **Custom Translator** 进行训练，以期取得更好的翻译效果。

![](https://rebornas.blob.core.windows.net/rebornhome/AzureSpeechTranslation/CustomTranslatorHomePage.png)
首页：[Custom Translator](https://portal.customtranslator.azure.ai/)
官方文档：[Official Doc](https://docs.microsoft.com/en-gb/azure/cognitive-services/translator/custom-translator/overview)

## 数据集
Custom Translator 的数据集为**双语对照**的样本对，每一句话对应一个样本，以空行分开。

- 本地准备的双语样本分别放在两个文本文件中，如下图所示：
![](https://rebornas.blob.core.windows.net/rebornhome/AzureSpeechTranslation/CustomTranslatorDatasetPreparation.png)

- 上传到 Custom Translator 后，会自动检测并分割语句，结果如下：
![](https://rebornas.blob.core.windows.net/rebornhome/AzureSpeechTranslation/CustomTranslatorDatasetDemo.png)

> 值得注意的是，训练 Custom Translator 需要的训练样本至少为10,000，也就是10,000对双语对照的样本对，否则无法训练。