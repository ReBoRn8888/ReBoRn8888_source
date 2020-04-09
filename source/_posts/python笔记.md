---
title: python 笔记
date: 2020-04-09 19:29:56
tags: [python]
categories: 学习笔记
top_img:
cover: https://i.loli.net/2020/04/09/kbxQSLTBHNRJfsr.jpg
---

{% aplayer '一荤一素 (Live)' '毛不易' 'http://music.163.com/song/media/outer/url?id=1422992245.mp3' 'http://p2.music.126.net/OBY8dfbP-Q002e4OECrqJA==/109951164703135281.jpg' autoplay %}

# time
```python
import time

# 获取系统当前时间，并编码成：年-月-日_时.分'秒''
test_time = str(time.strftime("%Y-%m-%d_%H.%M'%S\'\'"))
```

# logging
```python
import logging

def get_logging(log_path):
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s %(levelname)s\t%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_path,
        filemode='w')
    logger = logging.getLogger(__name__)
    return logger

def print_log(print_string, logger, log_type):
    print("{}".format(print_string))
    if(log_type == 'info'):
        logger.info("{}".format(print_string))

logger = get_logging('path/to/log.txt')
print_log(opt, logger, 'info')
```
将会在`'path/to/log.txt'`所设路径下生成日志文件，`log.txt`格式如下：
```python
2020-04-08 11:43:30 xxx.py INFO  xxxxxx
2020-04-08 11:43:31 xxx.py INFO  xxxxxxx
......
```

# os.path
python是默认以当前Terminal的运行路径作为基路径的，若要获取某一个py文件所在的路径，os.path相关使用技巧如下：
```python
import os

# 获取当前py文件的绝对路径
abspath = os.path.abspath(__file__)
print(abspath)
>>> /home/reborn/xxx/test.py

# 获取当前路径中的最深层目录路径
pwd = os.path.dirname(abspath)
print(pwd)
>>> /home/reborn/xxx

# 获取当前路径中最后一个'/'(或'\')后的名称
basename1 = os.path.basename(abspath)
basename2 = os.path.basename(pwd)
print(basename1)
print(basename2)
>>> test.py
>>> xxx
```