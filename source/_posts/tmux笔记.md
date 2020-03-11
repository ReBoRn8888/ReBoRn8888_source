---
title: tmux笔记
date: 2020-03-11 10:52:07
tags: [tmux, Linux, terminal]
categories: 学习笔记
top_img:
cover: https://i.loli.net/2020/03/11/1gUhD7z3ciXwYJC.jpg
---

{% aplayer '二零三' '毛不易' 'http://music.163.com/song/media/outer/url?id=1407214788.mp3' 'http://p2.music.126.net/XPPeIZu7wgcGXZ0666mfFg==/109951164640697307.jpg' autoplay %}

# tmux 简介
tmux是一个终端复用（Terminal Multiplexer）工具。

- 在一个**Terminal**中可以新建多个**会话（Session）**
- 在每个**会话**中可以新建多个**窗口（Window）**
- 在每个**窗口**中可以切分多个**窗格（Panel）**

**优势**：tmux解绑了会话和终端窗口，意味着，关闭终端窗口并不会使会话终止，这为我们使用ssh连接服务器跑代码提供了巨大的便利。

# tmux 结果演示
![](https://i.loli.net/2020/03/11/1gUhD7z3ciXwYJC.jpg)

# tmux 安装
```bash
sudo apt-get install tmux
```

# tmux 使用
## 会话管理
```bash
# 新建一个名为newsession的会话
tmux new -s newsession

# 进入名为newsession的会话
tmux a -t newsession

# 将会话s1重命名为s2
tmux rename -t s1 s2

# 关闭名为s1的会话
tmux kill-session -t s1

# 显示所有会话的列表
tmux ls
```

## 会话内快捷键
prefix 为前置按键，默认为：Ctrl+B
```bash
# 列出所有会话，可进行切换
prefix s

# 重命名会话
prefix $

# 离开当前会话
prefix d
```

## 窗口管理
```bash
# 新建一个窗口
prefix o

# 重命名当前窗口
prefix ,

# 列出所有窗口，可进行切换
prefix w

# 进入下一个窗口
prefix n

# 进入上一个窗口
prefix p

# 关闭当前窗口
prefix &
```

## 窗格管理
在会话内可以通过创建窗格的方式进行工作
```bash
# 水平方向创建窗格
prefix %

# 垂直方向创建窗格
prefix "

# 显示窗格编号
prefix q

# 使用方向键切换窗格
prefix Up|Down|Left|Right (方向键)

# 关闭当前窗格
prefix x

# 放大当前窗格（再次按下将还原）
prefix z

# 在当前窗格显示时间
prefix t

# 显示当前窗格信息
prefix i

# 调整窗格大小
prefix :
# 然后输入以下内容，并回车即可
resize-pane -U num # 向上移动num个像素
resize-pane -D num # 向下移动num个像素
resize-pane -L num # 向左移动num个像素
resize-pane -R num # 向右移动num个像素
```

## 其他命令
```bash
# 列出所有绑定的键
tmux list-key
# 等同于
prefix ?
```

## 修改tmux配置文件
- 首先在shell中打开tmux配置文件
```bash
vim ~/.tmux.conf
```
- 然后在其中添加对应的配置信息
```bash
# 将Ctrl+a设为prefix
setw -g prefix C-a
# 并将原先的Ctrl+b与prefix解绑
unbind-key C-b

# 支持鼠标选择和复制。按住Shift然后拖动鼠标选择，后使用对应的复制和粘贴键即可。
# 同时还可支持鼠标拖拽调整窗格大小
setw -g mode-mouse on 

# 不用按prefix，直接用Ctrl+方向键在窗格之间切换，其中的C可以换成S(Shift)等
bind -n C-Left select-pane -L
bind -n C-Right select-pane -R
bind -n C-Up select-pane -U
bind -n C-Down select-pane -D

# 修改原来切分窗格的快捷键，更直观，
bind-key h split-window -h # prefix+h 水平切分
bind-key v split-window -v # prefix+v 竖直切分
```

- 保存并激活配置文件
```bash
# 保存并退出编辑模式
ESC : wq ENTER
# 然后进入tmux，激活刚才的设置即可
prefix : source-file ~/.tmux.conf
```
