---
title: python 笔记
date: 2020-04-09 19:29:56
tags: [python]
categories: 学习笔记
top_img:
cover: https://i.loli.net/2020/04/09/kbxQSLTBHNRJfsr.jpg
---

{% meting "1422992245" "netease" "song" "autoplay" "mutex:false" "listmaxheight:340px" "preload:auto" "theme:#ad7a86"%}

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

# 对list型矩阵的变换技巧
```python
def print_matrix(mat):
    for x in mat:
        print(' '.join([str(n) for n in x]))

a = [[1,2,3], [4,5,6],[7,8,9]]
print_matrix(a)
>>> 1 2 3
>>> 4 5 6
>>> 7 8 9

print_matrix(zip(*a))
>>> 1 4 7
>>> 2 5 8
>>> 3 6 9

print_matrix(a[::-1]) # 等价于print_matrix(reversed(a))
>>> 7 8 9
>>> 4 5 6
>>> 1 2 3

print_matrix(zip(*a[::-1])) # 等价于print_matrix(zip(*reversed(a)))
>>> 7 4 1
>>> 8 5 2
>>> 9 6 3

print_matrix(map(reversed, a[::-1])) # 等价于print_matrix(map(reversed, reversed(a)))
>>> 9 8 7
>>> 6 5 4
>>> 3 2 1
```

# thread —— 捕获子线程异常
> 当子线程中发生异常时，子线程生命周期结束，可在主进程中通过判断子线程状态来返回异常

```python
from threading import Thread

class thread1:
    def __init__(self):
        pass
    def start(self):
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True
        self.t.start()
        return self
    def update(self):
        pass

if(__name__ == '__main__'):
    tt = thread1()
    tt.start()
    while(1):
        if(not tt.t.is_alive()):
            raise Exception
```

# 打印异常信息
- traceback.print_exc(): 直接打印异常信息
- traceback.format_exc(): 返回异常信息的字符串，便于记录到日志中

```python
import traceback

try:
    # Some Exception
except:
    print(traceback.format_exc())
    # or
    traceback.print_exc()
```