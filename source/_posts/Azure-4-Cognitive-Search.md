---
title: Azure 学习笔记 IV —— Cognitive-Search
date: 2020-09-10 11:40:30
tags: [Microsoft, Azure]
categories: 学习笔记
top_img:
cover: https://rebornas.blob.core.windows.net/rebornhome/AzureCognitiveSearch/AzureCognitiveSearchLogo.png
---

{% meting "407450223" "netease" "song" "autoplay" "mutex:false" "listmaxheight:340px" "preload:auto" "theme:#ad7a86"%}


# 什么是 Azure Cognitive Search
**Azure Cognitive Search** 官网 ： [https://azure.microsoft.com/en-us/services/search/](https://azure.microsoft.com/en-us/services/search/)
![](https://rebornas.blob.core.windows.net/rebornhome/AzureCognitiveSearch/AzureCognitiveSearchHomePage.png)

> 简单来说，Azure Cognitive Search 是一个云上的**搜索服务**，它支持使用内置的 **AI** 来进行内容的挖掘（比如使用 Azure OCR 来提取图像中的文本内容，或使用 NLP 来进行文本语义信息的抽取），以扩展其搜索能力。

DEMO Site: [https://wolterskluwereap.azurewebsites.net/](https://wolterskluwereap.azurewebsites.net/)

# 为什么要使用 Azure Cognitive Search
Azure Cognitive Search 主要有以下特点：
- 是一项**完全托管**的搜索服务，降低了复杂性并可轻松实现扩展
- 内置 **AI**（包括 OCR、关键词提取、命名体识别等）以挖掘更多潜在信息（[More details](https://docs.microsoft.com/en-us/azure/search/cognitive-search-concept-intro)）
- 灵活集成了**自定义**模型、分类器和排名器，可以满足特定领域的需求
- 搜索索引支持来自**任何源**的数据（包括Azure SQL Database, Azure Cosmos DB 或者 Azure Blob Storage），其中Azure Blob 索引器可以从主要文件格式提取文本，包括 Microsoft Office、PDF、Image 和 HTML 文档等。
- **可编程性**：REST API, .Net SDK, JAVA SDK, Python SDK, JS SDK
- 可以创建**可视化界面**，便于用户使用

# 相关概念
- **Index**：存储可用来搜索的内容，可理解为数据库中的表结构，包含键值对以及相应的选项
![](https://rebornas.blob.core.windows.net/rebornhome/AzureCognitiveSearch/AzureCognitiveSearchIndex.png)
- **Indexer**：可理解为**爬虫**，用来从数据源中提取可搜索的数据和元数据（包括使用 AI 提取出的数据），以进一步生成Index（[Details](https://docs.microsoft.com/en-us/azure/search/search-indexer-overview)）
- **AI Enrichment**：是Indexer的扩展功能，可用来从图像、Blob和其他非结构化数据源中提取数据（[Details](https://docs.microsoft.com/en-us/azure/search/cognitive-search-concept-intro)）
![](https://rebornas.blob.core.windows.net/rebornhome/AzureCognitiveSearch/AzureCognitiveSearchAIEnrichment.png)

# 快速入门
Azure Cognitive Search 可以直接在 Azure Portal 中使用，包括引入数据源、创建索引、创建简单的 Demo App 等。当然也支持其他许多的 SDK ，这里主要重点提一下 .Net 和 Postman。

## Azure Portal
- 1、创建 Search Service（[Tutorial Link](https://docs.microsoft.com/en-us/azure/search/search-create-service-portal)）
- 2、创建索引`Index`（[Tutorial Link](https://docs.microsoft.com/en-us/azure/search/search-get-started-portal)）
- 3、添加AI支持（[Tutorial Link](https://docs.microsoft.com/en-us/azure/search/cognitive-search-quickstart-blob)）
- 4、创建 Demo App（[Tutorial Link](https://docs.microsoft.com/en-us/azure/search/search-create-app-portal)）
![](https://rebornas.blob.core.windows.net/rebornhome/AzureCognitiveSearch/AzureCognitiveSearchDemoApp.png)

## .Net
> 使用C#操作 Azure Cognitive Search
> 附官方示例代码： [Official Github Repo](https://github.com/Azure-Samples/search-dotnet-getting-started)

详情可参考 [Tutorial Link](https://docs.microsoft.com/en-us/azure/search/search-get-started-dotnet)

## Postman
> 通过发送 HTTP 请求来进行 Azure Cognitive Search 的相关操作

详情可参考 [Tutorial Link](https://docs.microsoft.com/en-us/azure/search/search-get-started-postman)

## 常用搜索语句
|语句|说明|
|--|--|
|`search=xxx`|返回 xxx 的搜索结果|
|`search=*`|返回全样本|
|`facet=xxx`|按照 xxx 进行聚合，并统计每一类的个数|
|`search=xxx~&queryType=full`|使用模糊搜索|
|`search=seatle~&queryType=full`|表示对 seatle 进行自动纠错(seattle)，返回模糊查找的结果|
|`highlight=xxx`|在 xxx 属性中高亮显示搜索到的目标|
|`$count=true`|统计搜索到的样本数|
|`$top=N`|返回 topN 的结果|
|`$filter=xxxxx`|返回满足以 xxxxx 为条件的结果|
|`$filter=Rating gt 4`|返回Rating大于4的结果|
|`$filter=geo.distance(Location,geography'POINT(-122.12 47.67)') le 5`|返回在地理空间上距离点(-122.12 47.67)的距离小于5公里的结果|
|`$filter=Rooms/any(r: r/BaseRate lt 100)`|返回Rooms中有任一BaseRate小于100的结果|
|`$orderby=xxx [asc/desc]`|返回根据xxx升序或降序排列的结果|

# 写在最后
**需要注意的是，每次更新数据之后，都要更新 Index 索引**