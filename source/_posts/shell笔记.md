---
title: shell 笔记
date: 2020-03-13 11:52:57
tags: [Linux, shell, script]
categories: 学习笔记
top_img:
cover: https://i.loli.net/2020/03/13/Lsl38t9FzoH4Cmr.jpg
---

{% aplayer '小王' '毛不易' 'http://music.163.com/song/media/outer/url?id=1417856017.mp3' 'http://p2.music.126.net/XPPeIZu7wgcGXZ0666mfFg==/109951164640697307.jpg' autoplay %}

# shell 读取 ini 配置文件
```bash
function __readINI() {
	INIFILE=$1;	SECTION=$2;	ITEM=$3
	_readIni=`awk -F '=' '/\['$SECTION'\]/{a=1}a==1&&$1~/'$ITEM'/{print $2;exit}' $INIFILE`
	echo ${_readIni}
}

arg=( $( __readINI filename.ini Section Item ) ) 
```

# $'\r': command not found
## Issue
> $'\r': command not found
> syntax error near unexpected token $'\r'

## Solution
这是由于在Windows下换行符是`\r\n`，而Linux下换行符为`\n`而导致的报错。
使用dos2unix将Windows下的编码改为Unix下的编码即可。
```bash
# 安装dos2unix
sudo apt install dos2UNIX 
# 对文件'filename.sh'，使用dos2unix将Windows下的编码改为Unix下的编码
dos2unix filename.sh

# 反之类似，将Unix下的编码改为Windows下的编码
unix2dos filename.sh
```