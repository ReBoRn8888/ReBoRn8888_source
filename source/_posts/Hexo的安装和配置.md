---
title: Hexo的安装和配置
date: 2019-12-07 19:50:56
tags: hexo
categories: 教程
top_img: 
cover: https://i.loli.net/2020/01/29/JL5EKM2IbAao4h7.png
---
Welcome to [Hexo](https://hexo.io/)! This is your very first post. Check [documentation](https://hexo.io/docs/) for more info. If you get any problems when using Hexo, you can find the answer in [troubleshooting](https://hexo.io/docs/troubleshooting.html) or you can ask me on [GitHub](https://github.com/hexojs/hexo/issues).

## Quick Start

### Installation
```bash
# Install node.js (https://nodejs.org/en/download/)
# Because npm is too slow in China, we use cnpm(http://npm.taobao.org/) for installation
npm install -g cnpm --registry=https://registry.npm.taobao.org
# Install hexo
cnpm install -g hexo-cli
# View version information
hexo -v
```

### Create a new post
``` bash
hexo new "My New Post"
```

More info: [Writing](https://hexo.io/docs/writing.html)

### Generate static files

``` bash
hexo generate
# or 
hexo g
```

More info: [Generating](https://hexo.io/docs/generating.html)

### Run server

``` bash
# Since hexo-server is a seperated plugin now, so we need to install hexo-server manually
npm install --save hexo-server
# Then start the server
hexo server
```

More info: [Server](https://hexo.io/docs/server.html)

### Deploy to remote sites

``` bash
hexo deploy
```

### Add pin-top support

``` bash
# replace `hexo-generator-index` with `hexo-generator-index-pin-top`
npm uninstall hexo-generator-index
npm install hexo-generator-index-pin-top --save
```
Then add `top` attribute to the article you want to pin-top, e.g.,
```markdown
---
title: xxxxxx
date: xxxx
tags: xxx
categories: xxx
top: 100
---
```
> The larger the value of `top` is, the top the article is.

### Add music player `meting` support
When you use MetingJS, your blog can play musics from Tencent, Netease, Xiami, Kugou, Baidu and more. More details [here](https://github.com/MoePlayer/hexo-tag-aplayer).

If you want to use MetingJS in hexo-tag-aplayer, you need to enable it in `_config.yml`:
```yml
aplayer:
  meting: true
```

Now you can use `meting` in your post:
```markdown
<!-- Simple example (id, server, type)  -->
{% meting "60198" "netease" "playlist" %}

<!-- Advanced example -->
{% meting "60198" "netease" "playlist" "autoplay" "mutex:false" "listmaxheight:340px" "preload:none" "theme:#ad7a86"%}
```

The options are shown below:

| Option        | Default      | Description                                                  |
| ------------- | ------------ | ------------------------------------------------------------ |
| id            | **required** | song id / playlist id / album id / search keyword            |
| server        | **required** | Music platform: `netease`, `tencent`, `kugou`, `xiami`, `baidu` |
| type          | **required** | `song`, `playlist`, `album`, `search`, `artist`              |
| fixed         | `false`      | Enable fixed mode                                            |
| mini          | `false`      | Enable mini mode                                             |
| loop          | `all`        | Player loop play, values: 'all', 'one', 'none'               |
| order         | `list`       | Player play order, values: 'list', 'random'                  |
| volume        | 0.7          | Default volume, notice that player will remember user setting, default volume will not work after user set volume themselves |
| lrctype       | 0            | Lyric type                                                   |
| listfolded    | `false`      | Indicate whether list should folded at first                 |
| autoplay      | `false`      | Autoplay song(s), not supported by mobile browsers           |
| mutex         | `true`       | Pause other players when this player playing                 |
| listmaxheight | `340px`      | Max height of play list                                      |
| preload       | `auto`       | The way to load music, can be `none`, `metadata`, `auto`     |
| storagename   | `metingjs`   | LocalStorage key that store player setting                   |
| theme         | `#ad7a86`    | Theme color                                                  |

More info: [Deployment](https://hexo.io/docs/deployment.html)
