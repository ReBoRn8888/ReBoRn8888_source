---
title: Docker & Kubernetes
date: 2020-06-04 11:26:56
tags:
categories: 学习笔记
top_img:
cover: https://rebornas.blob.core.windows.net/rebornhome/Docker%2Fdocker.png
---

{% meting "569200213" "netease" "song" "autoplay" "mutex:false" "listmaxheight:340px" "preload:auto" "theme:#ad7a86"%}

# Docker 是什么
Docker 是一个开源的平台，它允许用户将开发的**应用**（**Application**）以及运行该应用程序所需的所有**依赖项**（**Dependencies**）打包成一个标准的单元，便于在其他系统上运行。
> 简单来说，我们无需配置任何环境，即可直接通过 Docker 运行 Docker Apps。

# 为什么要用 Docker
- **运行环境的一致性**：Docker 容器将应用和所有依赖项一起打包
- **持续交付和部署**：一次性的创建和配置，即可在任意地方运行
- **轻松迁移**：Docker 确保了运行环境的一致性，可以在多平台上迁移并运行
- **高效的系统资源利用率**：一台主机同时运行几千个 Docker 容器
- **快速启动**：毫秒 ~ 秒级别

# Docker vs. VMs
|  | Docker | VMs |
| :------: | :------: | :--------:|
| 隔离级别 | 隔离不同的应用（前后端、数据库） | 彻底隔离整个运行环境（云服务商用于隔离不同用户） |
| 操作系统 | 多个 Docker 容器间共享同一个 OS | 每个 VM 有各自不同的 OS |
| 占用空间 | 小（MB） | 大（GB） |
| 启动速度 | 快（秒级） | 慢（分钟级） |
| 资源利用率 | 较高 | 较低 |

# Docker 基本概念
- **Docker Images**：可以类比理解为 Github 中的 Repository。通过`docker pull`指令从 [Docker Hub](https://hub.docker.com/) 中下载对应的 Docker Image。
- **Docker Containers**：用于从 Docker Images 创建并运行应用，一个 Docker Container 运行一个 Docker Image 的应用。通过`docker run`指令运行。
- **Docker Daemon**：用于管理 Docker Containers 的构建、运行和分发，是运行于主机操作系统的后台服务，与 Docker Client 进行通信。
- **Docker Client**：一个命令行工具，允许用户通过指令与 Docker Daemon 进行交互。
- **Docker Hub**：类比于 Github。

# Docker 常用指令
- `docker pull [IMAGE_NAME]/[IMAGE_ID]`：从 Docker Hub 中下载Docker Image
- `docker run [IMAGE_NAME]/[IMAGE_ID]`：用于运行已下载的 Docker Image，使用`-d`参数可以挂载到后台
- `docker images`：列出所有已下载的 Images
- `docker rmi [IMAGE_NAME]/[IMAGE_ID]`：删除某个 Docker Image
- `docker search [IMAGE_NAME]`：从 Docker Hub 中搜索指定的 Docker Image

- `docker ps`：列出所有运行中的 Docker Containers，效果同`docker container ls`，若要列出已停止的 Docker Container，加上`-a`参数
- `docker stop [CONTAINER_NAME]/[CONTAINER_ID]`：停止某个 Docker Container
- `docker rm [CONTAINER_NAME]/[CONTAINER_ID]`：删除已停止运行的 Docker Container
- `docker container logs [CONTAINER_NAME]/[CONTAINER_ID]`：查看指定 Docker Container 的运行日志

- `docker login`：登录 Docker Hub 的账号，用于发布自己的 Docker Image
- `docker push [IMAGE_NAME]`：push 自己的 Docker Image 到 Docker Hub 上

- `docker network ls`：列出所有的 Docker Network
- `docker network inspect [NETWORK_NAME]`：查看特定 Docker Network 的信息
- `docker network create [NETWORK_NAME]`：创建自己的 [bridge] Network

# 其他 Docker 工具
在 Docker 生态中，有许多其他的开源工具能够与 Docker 相辅相成：
- **Docker Machine**：在主机、云平台或数据中心中创建 Docker hosts
- **Docker Compose**：用于定义和运行多容器（multi-container）的 Docker 应用
- **Docker Swarm**：一个本地集群解决方案
- **Kubernetes**：一个开源系统，用于对容器化应用的自动化部署、扩展和管理

## Docker Compose
当我们的应用变得越来越庞大时，比如应用本身（App）+数据库（Database）+网络配置（Network）等，为了便于后期的维护，我们会选择将这些组件拆分开，一个组件用一个 Docker Container 来运行，因此，在运行这个 App 时，我们就需要使用多次 `docker run` 指令，这就未免有点麻烦。

这时，Docker Compose 就派上用场了，当配置完之后，可以通过一条语句运行和停止整个应用程序。

因为 Docker Compose 是用 Python 编写的，所以我们可以通过 `pip` 指令来进行安装。
```bash
# 首先安装 docker-compose
pip install docker-compose

# 查看 docker-compose 版本
docker-compose --version
```

在使用 Docker Compose 前，我们需要先对我们自己的应用编写一个`docker-compose.yml`文件，接着在同一目录下执行以下指令即可：
- `docker-compose up`：运行整个应用，使用`-d`参数可以挂载到后台
- `docker-compose down`：停止整个应用，包括了`docker stop` + `docker rm`，非常方便

## Kubernetes

# 实践
参考官方 Tutorial：[https://docker-curriculum.com/#docker-network](https://docker-curriculum.com/#docker-network)