---
title: Azure 使用笔记② —— Azure Storage
date: 2020-05-10 16:26:00
tags: [Microsoft, Azure]
categories: 学习笔记
top_img:
cover: https://i.loli.net/2020/05/10/y9H2VhYWEjJdNXr.jpg
---

{% meting "94639" "netease" "song" "autoplay" "mutex:false" "listmaxheight:340px" "preload:auto" "theme:#ad7a86"%}


Azure Storage 官网：[https://azure.microsoft.com/en-gb/services/storage/](https://azure.microsoft.com/en-gb/services/storage/)
![](https://i.loli.net/2020/05/10/AG6iEMgYNIeZo7s.png)

# 什么是Azure Storage
> **一句话概括**：按需支付的安全的云存储服务，帮你解决繁重的数据中心管理任务。

Azure Storage包括以下存储服务：
- File：简单的分布式跨平台文件系统
- Blob：高度可伸缩的非结构化数据存储
- Disk：适用于每种工作负荷的持久高性能磁盘存储
- Data Lake Storage：适用于大数据分析的、安全、高度可伸缩的存储
- Archive：在存储不常使用的数据方面具有行业领先的价格优势
- Azure vFXT for Azure：在边缘设备上的高性能计算

# 为什么要用Azure Storage
Azure Storage可以简单理解为“**微软云盘**”，若你正为“百度网盘”的限速所苦恼，来拥抱Azure Storage吧！
个人将其优势概括如下：
- 使用简单
- 可用于云计算，满足云服务的存储需求
- 按需支付

# Azure Storage 定价标准
[https://azure.microsoft.com/en-us/pricing/details/storage/](https://azure.microsoft.com/en-us/pricing/details/storage/)
![](https://i.loli.net/2020/05/10/hClZQHA59a8ERbo.png)

# 开始使用Azure Storage
## 创建Azure Storage账户
- 在Azure portal中搜索`storage accounts`，点击进入“Storage accounts”界面
<img src="https://i.loli.net/2020/05/10/ADoFWPrdIZRk7uf.png" style="zoom: 65%;" />
- 点击红框处`Add`，即可新建存储账户
![](https://i.loli.net/2020/05/10/nJAOyTDFgiNUQP1.png)
- 填写必要信息后，创建存储账户（详细填写说明参照[官方文档](https://docs.microsoft.com/en-us/azure/storage/common/storage-account-overview?toc=/azure/storage/blobs/toc.json)）
![](https://i.loli.net/2020/05/10/VmlprY3gbeNWTXU.png)

## Azure Storage Explorer
下载[Azure Storage Explorer](https://azure.microsoft.com/en-us/features/storage-explorer/)，使用同一账号登录，找到对应的订阅即可看到刚刚创建的Storage account。
<img src="https://i.loli.net/2020/05/10/FKkSBuqGQspl83t.png" style="zoom: 80%;" />
我们可以直接使用Azure Storage Explorer进行文件的上传和下载操作，可以看到一个Storage Account下有四种类型的服务：
- [Blob Containers](https://docs.microsoft.com/en-us/azure/storage/blobs/storage-blobs-introduction)
    - 可以新建多个Containers，每个Container下均可以存储文件，Azure中将存储于Container中的文件称为Blob
    - 存储**可通过浏览器直接访问的文档、图片**
    - 存储**视频、音频流**数据
    - 适合存储**大文件**
    - 价格**便宜**
- [File Shares](https://docs.microsoft.com/en-us/azure/storage/files/storage-files-introduction)
    - 同Windows文件夹性质相同，可创建多级子文件夹
    - 存储需要被**多方**（多台虚拟机或多个程序）**访问**的文件
    - 价格**高昂**
- [Queues](https://docs.microsoft.com/en-us/azure/storage/queues/storage-queues-introduction)：不常用，不做介绍
- [Tables](https://docs.microsoft.com/en-us/azure/storage/tables/table-storage-overview)：不常用，不做介绍

## Azure Storage SDK
除了使用桌面端App：Azure Storage Explorer操作Azure Storage之外，若我们需要在代码中对其进行操作，就需要使用对应编程语言的SDK了，此处我们主要介绍[.Net SDK](https://docs.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-dotnet)，其他支持的语言还有：Java、Python、JavaScript、Go、PHP、Ruby等。
### 新建项目
- 打开Visual Studio，新建“**Console App**”项目
<img src="https://i.loli.net/2020/05/10/BHUec5OzugY6sx3.png" style="zoom: 80%;" />

### 添加配置文件
- 右击Project，新建“**App.config**”配置文件，用来配置Azure Storage Account
<img src="https://i.loli.net/2020/05/10/udU1FiphlOj5aHQ.png" style="zoom: 60%;" />
<img src="https://i.loli.net/2020/05/10/Vw1EquLbJjO5MeK.png" style="zoom: 80%;" />
- 在“**App.config**”中添加如下代码：
```xml
< ?xml version="1.0" encoding="utf-8" ?>
<configuration>
    <startup>
        <supportedRuntime version="v4.0" sku=".NETFramework,Version=v4.5" />
    < /startup>
    <appSettings>
        <add key="StorageConnectionString" value="YOUR STORAGECONNECTIONSTRING" />
    < /appSettings>
< /configuration>
```
> 复制的话记得将“< ?”和“< /”中间的空格删掉，这里加上空格是为了防止浏览器将其误渲染。

其中的`YOUR STORAGECONNECTIONSTRING`可从Azure portal对应的Storage account中找到：
![](https://i.loli.net/2020/05/10/b29oLP4lZi1dsuK.png)

### 上传/删除/下载 by C#
```csharp
using Microsoft.Azure.Storage;      // Namespace for Storage Client Library
using Microsoft.Azure.Storage.Blob; // Namespace for Azure Blobs
using Microsoft.Azure.Storage.File; // Namespace for Azure Files
using System;
using System.IO;
using System.Threading.Tasks;

namespace ConsoleApp1
{
    class Program
    {
        static async Task Main()
        {
            // 获取StorageConnectionString
            var appSettings = System.Configuration.ConfigurationManager.AppSettings;
            // 创建与storageAccount的连接
            CloudStorageAccount storageAccount = CloudStorageAccount.Parse(appSettings["StorageConnectionString"]);
            
            // ===========================================================================================

            CloudBlobClient blobClient = storageAccount.CreateCloudBlobClient();    // 创建blob client

            String containerName    = "myContainer";                                // Container名称
            String blobName         = "xxx.ext";                                    // Blob名称
            String localFilePath    = "localFile.ext";                              // 待上传的本地文件路径
            var fileStream = File.OpenRead(localFilePath);                          // 获取fileStream

            // 上传文件到Blob
            upload_to_blob(blobClient, containerName, blobName, fileStream);
            // 从Blob删除文件
            delete_from_blob(blobClient, containerName, blobName);
            // 从Blob读取Bytes
            byte[] bb = readbytes_from_blob(blobClient, containerName, blobName);
            // 从Blob读取文本
            String text = readtext_from_blob(blobClient, containerName, blobName);

            // ===========================================================================================
            
            CloudFileClient fileClient = storageAccount.CreateCloudFileClient();    // 创建file client

            String shareName        = "myShare";                                    // FileShare名称
            String dirName          = "subDir";                                     // 子目录名称
            String cloudFileName    = "xxx.ext";                                    // 云端文件名称
            String localFileName    = "localFile.ext";                              // 待上传的本地文件路径
            var fileStream = System.IO.File.OpenRead(localFileName);                // 获取fileStream

            // 上传文件到fileShare
            upload_to_fileshare(fileClient, shareName, dirName, cloudFileName, fileStream);
            // 从fileShare删除文件
            delete_from_fileshare(fileClient, shareName, dirName, cloudFileName);
            // 从fileShare读取Bytes
            byte[] bb = readbytes_from_fileshare(fileClient, shareName, dirName, cloudFileName);
            // 从fileShare读取文本
            String text = readtext_from_fileshare(fileClient, shareName, dirName, cloudFileName);

        }

        public static void upload_to_blob(CloudBlobClient blobClient, String containerName, String blobName, Stream stream)
        {
            CloudBlobContainer container = blobClient.GetContainerReference(containerName); // 从blobClient获取container的引用
            container.CreateIfNotExists();                                                  // 如果不存在就创建 FileShare
            CloudBlockBlob blockBlob = container.GetBlockBlobReference(blobName);           // 从container获取blob的引用

            blockBlob.UploadFromStream(stream);                                             // 上传文件
        }

        public static void delete_from_blob(CloudBlobClient blobClient, String containerName, String blobName)
        {
            CloudBlobContainer container = blobClient.GetContainerReference(containerName); // 从blobClient获取container的引用
            CloudBlockBlob blockBlob = container.GetBlockBlobReference(blobName);           // 从container获取blob的引用

            blockBlob.DeleteIfExists();                                                     // 删除文件
        }

        public static byte[] readbytes_from_blob(CloudBlobClient blobClient, String containerName, String blobName)
        {
            CloudBlobContainer container = blobClient.GetContainerReference(containerName); // 从blobClient获取container的引用
            CloudBlockBlob blockBlob = container.GetBlockBlobReference(blobName);           // 从container获取blob的引用

            if (!blockBlob.Exists())                                                        // Blob不存在
                return null;

            byte[] content = new byte[blockBlob.Properties.Length];
            blockBlob.DownloadToByteArray(content, 0);                                      // 读取Bytes
            return content;
        }

        public static String readtext_from_blob(CloudBlobClient blobClient, String containerName, String blobName)
        {
            CloudBlobContainer container = blobClient.GetContainerReference(containerName); // 从blobClient获取container的引用
            CloudBlockBlob blockBlob = container.GetBlockBlobReference(blobName);           // 从container获取blob的引用

            if (!blockBlob.Exists())                                                        // Blob不存在
                return null;

            return blockBlob.DownloadText();                                                // 读取文本
        }

        public static void upload_to_fileshare(CloudFileClient fileClient, String shareName, String dirName, String cloudFileName, Stream stream)
        {
            CloudFileShare share = fileClient.GetShareReference(shareName);     // 获取 FileShare 对象
            share.CreateIfNotExists();                                          // 如果不存在就创建 FileShare

            CloudFileDirectory rootDir = share.GetRootDirectoryReference();     // 获取根目录的引用
            CloudFileDirectory webDir = rootDir.GetDirectoryReference(dirName); // 获取子目录的引用
            webDir.CreateIfNotExists();                                         // 获取子目录

            CloudFile cloudFile = webDir.GetFileReference(cloudFileName);       // 获取文件的引用
            cloudFile.UploadFromStream(stream);                                 // 上传文件
        }

        public static void delete_from_fileshare(CloudFileClient fileClient, String shareName, String dirName, String cloudFileName)
        {
            CloudFileShare share = fileClient.GetShareReference(shareName);     // 获取 FileShare 对象

            CloudFileDirectory rootDir = share.GetRootDirectoryReference();     // 获取根目录的引用
            CloudFileDirectory webDir = rootDir.GetDirectoryReference(dirName); // 获取子目录的引用

            CloudFile cloudFile = webDir.GetFileReference(cloudFileName);       // 获取文件的引用
            cloudFile.DeleteIfExists();                                         // 删除文件
        }

        public static byte[] readbytes_from_fileshare(CloudFileClient fileClient, String shareName, String dirName, String cloudFileName)
        {
            CloudFileShare share = fileClient.GetShareReference(shareName);         // 获取 FileShare 对象
            CloudFileDirectory rootDir = share.GetRootDirectoryReference();         // 获取根目录的引用

            CloudFile cloudFile;
            if (String.IsNullOrEmpty(dirName))                                      // 文件在根目录
            {
                cloudFile = rootDir.GetFileReference(cloudFileName);                // 获取文件的引用
            }
            else                                                                    // 文件在子目录
            {
                CloudFileDirectory webDir = rootDir.GetDirectoryReference(dirName); // 获取子目录的引用
                cloudFile = webDir.GetFileReference(cloudFileName);                 // 获取文件的引用
            }

            if (!cloudFile.Exists())                                                // 文件不存在
                return null;

            byte[] content = new byte[cloudFile.Properties.Length];
            cloudFile.DownloadToByteArray(content, 0);                              // 读取Bytes
            return content;
        }

        public static String readtext_from_fileshare(CloudFileClient fileClient, String shareName, String dirName, String cloudFileName)
        {
            CloudFileShare share = fileClient.GetShareReference(shareName);         // 获取 FileShare 对象
            CloudFileDirectory rootDir = share.GetRootDirectoryReference();         // 获取根目录的引用

            CloudFile cloudFile;
            if (String.IsNullOrEmpty(dirName))                                      // 文件在根目录
            {
                cloudFile = rootDir.GetFileReference(cloudFileName);                // 获取文件的引用
            }
            else                                                                    // 文件在子目录
            {
                CloudFileDirectory webDir = rootDir.GetDirectoryReference(dirName); // 获取子目录的引用
                cloudFile = webDir.GetFileReference(cloudFileName);                 // 获取文件的引用
            }

            if (!cloudFile.Exists())                                                // 文件不存在
                return null;

            return cloudFile.DownloadText();                                        // 读取文本
        }

    }
}
```