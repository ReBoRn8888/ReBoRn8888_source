---
title: git Issues
date: 2020-03-02 14:47:11
tags: [git]
categories: 学习笔记
top_img:
cover: https://i.loli.net/2020/03/02/vO8bI5LnMUjJwlH.png
---

{% meting "1421191830" "netease" "song" "autoplay" "mutex:false" "listmaxheight:340px" "preload:auto" "theme:#ad7a86"%}

# git clone时文件太大导致clone失败
## Issue
> fatal: early EOF 
> fatal: index-pack failed

## Solution
编辑home目录下的`.gitconfig`，添加以下配置，以扩大单个文件最大容量限制：
```
	[core] 
		packedGitLimit = 1024m 
		packedGitWindowSize = 1024m 
	[pack] 
		deltaCacheSize = 2047m 
		packSizeLimit = 2047m 
		windowMemory = 2047m
```