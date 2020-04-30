---
title: pip 笔记
date: 2020-03-16 10:40:18
tags: [pip, python]
categories: 学习笔记
top_img:
cover: https://i.loli.net/2020/03/16/Dpn9cQAdxwrtg37.jpg
---

{% meting "31010776" "netease" "song" "autoplay" "mutex:false" "listmaxheight:340px" "preload:auto" "theme:#ad7a86"%}

# python项目的requirements.txt
## 生成requirements.txt
```bash
# 安装pipreqs
pip install pipreqs
# 在当前目录下生成requirements.txt
pipreqs . --encoding=utf8 --force
```
> `--encoding=utf8`表示使用utf-8编码，不然可能会报错：UnicodeDecodeError: 'gbk' codec can't decode byte 0xae in position 406: illegal multibyte sequence
> `--force`表示强制执行，当生成目录下的requirements.txt存在时将其覆盖。

## 使用requirements.txt安装依赖
```bash
pip install -r requirements.txt
```
