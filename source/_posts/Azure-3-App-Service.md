---
title: Azure 使用笔记③ —— Azure App Service
date: 2020-05-12 13:55:59
tags: [Microsoft, Azure]
categories: 学习笔记
top_img:
cover: https://i.loli.net/2020/05/10/y9H2VhYWEjJdNXr.jpg
---

{% meting "760173" "netease" "song" "autoplay" "mutex:false" "listmaxheight:340px" "preload:auto" "theme:#ad7a86"%}

# 什么是 Azure App Service
**Azure App Service** 官网 ： [https://azure.microsoft.com/en-us/services/app-service/](https://azure.microsoft.com/en-us/services/app-service/)

**Azure App Service**：是Azure提供的一种用于在**完全托管**的平台上，**构建**和**部署Web应用**的服务。其非常适合大多数的网站，特别是我们不需要对部署的基础架构进行大量控制时。
> 考虑如下场景：当我们写完一个网站（个人主页、Blog、项目演示Demo等），为了能让其他人通过域名或IP地址来访问该站点，我们就需要将其部署到云端，比如Azure，而Azure App Service就提供了这样一种服务。

**Azure App Service** 主要包括三种类型的Apps：
![](https://rebornas.blob.core.windows.net/rebornhome/AzureAppService%2FAzureAppServiceTypes.png)
- **Web Apps**：用于在云端部署网站应用
- **Web App for Containers**：用于在云端部署容器化的网站应用
- **API Apps**：用于在云端部署API
> 其中最常用的是 **Web Apps**，网页端应用，也就是网站，下面将详细介绍 **Azure Web App Service**。


# Azure Web App Service
**Azure Web App Service** 官网： [https://azure.microsoft.com/en-us/services/app-service/web/](https://azure.microsoft.com/en-us/services/app-service/web/)
![](https://rebornas.blob.core.windows.net/rebornhome/AzureAppService%2FAzureWebAppService.png)

## 支持的编程语言
![](https://rebornas.blob.core.windows.net/rebornhome/AzureAppService%2FAzureWebAppStack.png)

## 开始使用
**Web App Service** 视频教程：[https://channel9.msdn.com/Shows/Azure-Friday/App-Service-Domains](https://channel9.msdn.com/Shows/Azure-Friday/App-Service-Domains)

用到的 Azure 服务：
- **App Service**：用于创建App服务
- **App Service plan**：用于设定App的计算资源
- **Application Insights**：用于实时监测App使用情况
- **App Service Domain**：用于购买域名
- **DNS Zone**：用于进行域名相关设定
- **App Service Certificate**：用于进行站点安全认证
- **Azure Key Vault**：用于管理数字证书

整体流程可以概括如下：
- 1、**搭建**本地网站（Required）
- 2、**创建** Web App resource（Required）
- 3、**部署**网站（Required）
- 4、购买**域名**、添加**域名**（Optional）
- 5、添加**SSL认证**（Optional）

### 搭建本地网站
> 此处以ASP. NET为例搭建网站。
我们打开 Visual Studio，新建项目，搜索 ASP. NET，会发现有两种类型：
- 1、**ASP. NET Web App**：是**Windows**上用来构建企业级、基于**服务器**的Web App的成熟的框架。
- 2、**ASP. NET Core Web App**：是对 ASP. NET 的重新设计，是一个开源的、**跨平台**的用来构建基于**云**的现代Web App的框架。
![](https://rebornas.blob.core.windows.net/rebornhome/AzureAppService%2Faspdotnet.png)
二者的区别可参考[官方文档](https://docs.microsoft.com/en-us/aspnet/core/fundamentals/choose-aspnet-framework?view=aspnetcore-3.1)，此处作一归纳：

| ASP. NET Core                  | ASP. NET                                   |
| :------:                      | :----------------:                        |
| 支持Windows、macOS、Linux       |      支持Windows                          |
| 推荐使用Razor Pages创建网页    |      使用Web Forms、Web Pages等创建网页     |
| 支持单机多版本                 |      单机单版本                            |
| 性能强于 ASP. NET               |     正常性能                               |
| C#、F#                         |    C#、VB、F#                               |

创建完后会生成如下的项目结构：
- **ASP. NET Core -- Web Application**
![](https://rebornas.blob.core.windows.net/rebornhome/AzureAppService%2FDotNetCoreWebApp.png)
    - **wwwroot** : 放置css、js和lib文件
    - **Pages** : 放置所有的 Razor 页面文件，后缀为**.cshtml**，可以在一个文件中同时使用C#和HTML语言进行编写
    - **appsettings.json** : 配置文件
    - **Program.cs** : Web App 主函数文件，用来开启Web服务
    - **Startup.cs** : 用来注册当前Web App用到的所有服务项

- **ASP. NET -- Web Form**
![](https://rebornas.blob.core.windows.net/rebornhome/AzureAppService%2FDotNetWebForm.png)
    - **xxxx.aspx** : ASP. NET 页面文件，后缀为**.aspx**
    - **Default.aspx** : Web 主页文件
    - **Site.Master** : 用于为页面创建一致的布局和使用标准
    - **Global.asax** : （可选）用于相应由 ASP. NET 或 HTTP 模块引发的应用程序级事件和会话级事件的代码
    - **Web.config** : 配置文件

直接上传到Github或者DevOps即可。

### 创建 Web App resource
![](https://rebornas.blob.core.windows.net/rebornhome/AzureAppService%2FAzureWebApp.png)
![](https://rebornas.blob.core.windows.net/rebornhome/AzureAppService%2FAzureWebAppCreate.png)
创建完后，进入这个resource
![](https://rebornas.blob.core.windows.net/rebornhome/AzureAppService%2FAzureWebAppOverview.png)
需要关注的部分已用红框标出：
- **URL**：用于直接访问该Web App的初始URL
- **Deployment Center**：部署中心，用于将已写完的网站部署到该Web App Service上
- **Custom domains**：用于给该网站添加域名，之后便可通过这些域名进行访问
- **TLS/SSL settings**：给网站添加SSL认证

### Deployment Center
![](https://rebornas.blob.core.windows.net/rebornhome/AzureAppService%2FAzureWebAppDeploymentCenter.png)
可通过八种方式进行部署，一般选择**Azure Repos**或者**Github**即可，登录自己的账号，选择对应的Repo，一路往下即可开始部署，部署完成后通过**URL**即可访问该网站。

### Custom domains
![](https://rebornas.blob.core.windows.net/rebornhome/AzureAppService%2FAzureWebAppCustomDomains.png)
- 首先我们需要在Azure上购买一个域名：`Buy Domain`。[Learn more](https://docs.microsoft.com/en-gb/azure/app-service/manage-custom-dns-buy-domain)
- 接着在**DNS Zone**中添加“**Record set**”，**Type**选择"CNAME"。

> 这里介绍一下“**A Record**”和“**CNAME Record**”：
>   - **A Record**：直接指向网站IP地址，若直接使用**A Record**，容易遭到DDOS攻击。
>   - **CNAME Record**：指向网站的域名，也叫“别名”，**CNAME Record**指向的域名最终也指向**A Record**，也就是说，在按需更换IP地址的过程中，无需变更**CNAME Record**的值。
>   - 域名解析建议使用**CNAME Record**。

- 之后就可以`Add Custom Domains`了，用于从新的域名访问当前网站。[Learn more](https://docs.microsoft.com/en-gb/azure/app-service/app-service-web-tutorial-custom-domain)

### TLS/SSL settings
![](https://rebornas.blob.core.windows.net/rebornhome/AzureAppService%2FAzureWebAppSSLCertificate.png)
> - **SSL(Secure Socket Layer)**：是一种电子数字证书，用于对服务器的标识进行身份验证，该证书使浏览器能够在允许SSL会话开始前验证服务器的真实性（通过https访问）。
> - **TLS(Transport Layer Security)**：可以简单理解为**SSL**加强版。
添加TLS/SSL认证，我们需要：
- 首先购买证书（`Buy Certificate`）。[Learn more](https://docs.microsoft.com/en-gb/azure/app-service/configure-ssl-certificate)
- 接着前往证书页面，将该证书加入“**Azure Key Vault**”中，并进行认证。
> [**Azure Key Vault**](https://azure.microsoft.com/en-us/services/key-vault/)：是Azure提供的一项密钥管理服务，用于集中管理和存储云端应用和服务所使用到的密钥、安全证书等需加密的东西。
- 然后添加证书（`Import App Service Certificate`）。[Learn more](https://docs.microsoft.com/en-us/azure/app-service/app-service-web-ssl-cert-load)
- 最后将已认证的证书绑定到之前添加的域名上（`TLS/SSL bindings`）。[Learn more](https://go.microsoft.com/fwlink/?linkid=849480)
![](https://rebornas.blob.core.windows.net/rebornhome/AzureAppService%2FAzureWebAppSSLBinding.png)
- 至此，即可使用https对所添加的新域名进行安全访问。e.g.,
    - 本站点旧域名：https://reborn8888.github.io 
    - 本站点新域名：https://www.reborn8888.com 或 https://reborn8888.com

以上所有详细操作可见[视频教程](https://channel9.msdn.com/Shows/Azure-Friday/App-Service-Domains)