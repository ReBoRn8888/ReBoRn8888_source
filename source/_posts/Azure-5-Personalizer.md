---
title: Azure 学习笔记 V —— Personalizer
date: 2020-09-25 11:40:55
tags: [Microsoft, Azure]
categories: 学习笔记
top_img:
cover: https://rebornas.blob.core.windows.net/rebornhome/AzurePersonalizer/AzurePersonalizerLogo.png
---

{% meting "1449782341" "netease" "song" "autoplay" "mutex:false" "listmaxheight:340px" "preload:auto" "theme:#ad7a86"%}

# 什么是 Azure Personalizer
**Azure Personalizer** 官网 ： [https://azure.microsoft.com/en-us/services/cognitive-services/personalizer/](https://azure.microsoft.com/en-us/services/cognitive-services/personalizer/)
![](https://rebornas.blob.core.windows.net/rebornhome/AzurePersonalizer/AzurePersonalizerHomepage.png)

> 简单来说，Azure Personalizer 是一个基于**强化学习**的个性化体验创建服务，应用程序通过不断学习用户的行为，有助于向用户展示最佳匹配的内容，比如可以使用 Azure Personalizer 向特定购物者推荐产品，或使用 Azure Personalizer 来确定广告的最佳投放位置。

DEMO Site: [https://personalizationdemo.azurewebsites.net/](https://personalizationdemo.azurewebsites.net/)

# 使用场景
- **个性化推荐**：通过提供个性化选项，帮助用户在意图不明确时获得更好的体验。
- **提供默认建议**：让机器人在一开始以个性化的方式建议最可能的项目，而不是提供非个人的菜单或替代列表。
- **机器人特质和语气**：对于可以改变**语气和写作风格**的机器人，可进行个性化学习。
- **通知和警报内容**：确定用于警报的**内容**，以吸引更多用户。
- **通知和警报时间**：个性化学习**何时**向用户发送通知以使他们更多地参与其中。

# 支持的编程语言
- C#
- JavaScript
- Python

# 工作流程
- 1、创建 **PersonalizerClient** 用以和 Azure Personalizer 进行**通信**
- 2、创建 **RankRequest** 向 Rank API 发送特征数据，并返回模型认为的**最佳结果**
- 3、对于返回结果，人为评分，并创建 **RewardRequest** 向 Reward API 发送对于返回结果的**得分情况**
- 4、循环2、3两步，使得基于**强化学习**（Reinforcement Learning）的 Personalizer 通过得分情况不断优化模型
![](https://rebornas.blob.core.windows.net/rebornhome/AzurePersonalizer/AzurePersonalizerArchitecture.png)

# 模型参数配置
主要包含以下参数，详情可参考 [Link](https://docs.microsoft.com/en-us/azure/cognitive-services/personalizer/how-to-settings)：
- **Reward**：“奖励机制”的相关配置，包括**等待 Reward 反馈的时间**、**默认 Reward 值**以及收到多个 Reward 值时的 **Reward 聚合方式**。
- **Exploration**：通过浏览备选方式，而不是使用定型模型的预测结果，来发现新模式并适应用户行为的改变。Exploration 值决定了**用于探索备选方法的排名调用的百分比**。
- **Model update frequency**：模型更新频率。
- **Data retention**：数据保留时间，用于设置 Personalizer 要将数据日志保留多少天。

# 快速入门
## C\# Quickstart
- [Official Tutorial](https://docs.microsoft.com/en-us/azure/cognitive-services/personalizer/quickstart-personalizer-sdk?pivots=programming-language-csharp)
- [Source Code](https://github.com/Azure-Samples/cognitive-services-quickstart-code/tree/master/dotnet/Personalizer)

## Personalizer in Web App (C#)
- [Official Tutorial](https://docs.microsoft.com/en-us/azure/cognitive-services/personalizer/tutorial-use-personalizer-web-app)
- [Source Code](https://github.com/Azure-Samples/cognitive-services-personalizer-samples)

## Personalizer in Chat Bot (C#)
- [Official Tutorial](https://docs.microsoft.com/en-us/azure/cognitive-services/personalizer/tutorial-use-personalizer-chat-bot)
- [Source Code](https://github.com/Azure-Samples/cognitive-services-personalizer-samples.git)

## Azure Notebook (Python)
- [Official Tutorial](https://docs.microsoft.com/en-us/azure/cognitive-services/personalizer/tutorial-use-azure-notebook-generate-loop-data)
- [Source Code](https://github.com/Azure-Samples/cognitive-services-personalizer-samples/tree/master/samples/azurenotebook)
