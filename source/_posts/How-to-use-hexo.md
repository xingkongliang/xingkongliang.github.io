---
title: How to use hexo?
date: 2018-08-14 15:21:22
categories: 
tags: 
description: 使用Hexo的基本方法。
---


## Hexo基本指令

### init

```
$ hexo init [folder]
```
新建一个网站，如果没有设置｀folder｀，Hexo默认在目前的文件夹建立网站。


### new 

```
$ hexo new [layout] <title>
```

新建一片文章。如果没有设置`layout`的话，默认使用_config.yml中的default_layout参数代替。如果标题包含空格的话，请使用引号括起来。

### generate

```
$ hexo generate
$ hexo g  # 简写
```
<!--more-->

### publish
```
$ hexo publish [layout] <filename>
```
发表草稿。

### server

```
$ hexo server
```
启动服务器。默认情况下，访问网址为：`http://localhost:4000/`。

|选项|描述|
|:--:|:--:|
|`-p`,`--port`|重设端口|
|`-s`, `--static`|只使用静态文本|
|`-l`,`--log`|启动日记记录，使用覆盖记录格式|


### deploy
```
$ hexo deploy
$ hexo d  # 简写
```


### render
```
$ hexo render <file1> [file2] ...
```
渲染文件。

### clean
```
$ hexo clean
```
清楚缓存文件(`db.json`)好已经生成的静态文件(`public`)。

在某些情况(尤其是更换主题后)，如果发现您对站点的更改没有生效，可以使用此命令。

### list

```
$ hexo list <type>
```
列出网站资料。

### version
```
$ hexo version
```
显示Hexo版本。


### 显示草稿

```
$ hexo --draft
```
显示`source/_drafts`文件夹中的草稿文件。

### 自定义CWD

```
$ hexo --cwd /path/to/cwd
```
自定义当前工作目录(Current working dirctory)的路径。

### 在主页截断

- 方法1:
在文中插入以下代码
```
<!--more-->
```
- 方法2:
在文章中的`front-matter`中添加description，
```
---
title: 
date: 2018-08-14 15:21:22
categories: 
tags: 
description: 描述。。
---

```

### 分类和标签

只有文章支持分类和标签，您可以在 Front-matter 中设置。在其他系统中，分类和标签听起来很接近，但是在 Hexo 中两者有着明显的差别：分类具有顺序性和层次性，也就是说 `Foo`, `Bar` 不等于 `Bar`, `Foo`；而标签没有顺序和层次。

```
categories:
- Diary
tags:
- PS3
- Games
```

### 定义一段代码。
```Python
import os
import numpy as np

for i in range(10):
    print('now is ', i)

```

显示一幅图像。
```
{% asset_img 003.jpg This is an example image %}
```
{% asset_img 003.jpg This is an example image %}



## Hexo 资源
[hexo官网](https://hexo.io/zh-cn/)

[theme-next使用说明](http://theme-next.iissnan.com/)

[使用hexo，如果换了电脑怎么更新博客？](https://www.zhihu.com/question/21193762)

其他参考博客：
[我的个人博客之旅：从jekyll到hexo](https://blog.csdn.net/u011475210/article/details/79023429)
[HEXO+NEXT主题个性化配置](http://mashirosorata.vicp.io/HEXO-NEXT%E4%B8%BB%E9%A2%98%E4%B8%AA%E6%80%A7%E5%8C%96%E9%85%8D%E7%BD%AE.html)

[hexo的next主题个性化配置教程](https://segmentfault.com/a/1190000009544924#articleHeader21)

[基于Hexo+Node.js+github+coding搭建个人博客——进阶篇(从入门到入土)](http://blog.csdn.net/MasterAnt_D/article/details/56839222#t50)
