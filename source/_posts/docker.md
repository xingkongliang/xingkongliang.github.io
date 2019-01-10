---
title: Docker 安装与使用
date: 2018-10-10 16:20:12
description: Docker 安装与使用
categories:
tags: 
- docker
---

# Docker 安装与使用
@(工具学习记录)[Docker]

## 1. Docker安装
参考[官网教程](https://docs.docker.com/install/linux/docker-ce/ubuntu/)

### 卸载旧的版本

```
$ sudo apt-get remove docker docker-engine docker.iSET UP THE REPOSITORY
```
SET UP THE REPOSITORY

```
$ sudo apt-get update
```

```
$ sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common
```

```
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
```

```
$ sudo apt-key fingerprint 0EBFCD88

pub   4096R/0EBFCD88 2017-02-22
      Key fingerprint = 9DC8 5822 9FC7 DD38 854A  E2D8 8D81 803C 0EBF CD88
uid                  Docker Release (CE deb) <docker@docker.com>
sub   4096R/F273FCD8 2017-02-22
```

x86_64/amd64
```
$ sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
```

### 安装 DOCKER CE

更新包的索引
```
$ sudo apt-get update
```

安装最新版本的Docker CE
```
$ sudo apt-get install docker-ce
```
安装特定版本Docker CE
```
$ apt-cache madison docker-ce

docker-ce | 18.03.0~ce-0~ubuntu | https://download.docker.com/linux/ubuntu xenial/stable amd64 Packages
```

验证是否安装正确
```
$ sudo docker run hello-world
```
## 2.nvidia-docker 安装
参考[nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
nstalling version 2.0

Debian-based distributions
```
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
```

```
sudo apt-get install nvidia-docker2
sudo pkill -SIGHUP dockerd
```

## 3. detectron 配置
参考[detectron install](https://github.com/facebookresearch/Detectron/blob/master/INSTALL.md)
```
cd $DETECTRON/docker
docker build -t detectron:c2-cuda9-cudnn7 .
```
运行这个镜像
```
nvidia-docker run --rm -it detectron:c2-cuda9-cudnn7 python detectron/tests/test_batch_permutation_op.py
```

## 4. docker 基本命令

### 对容器生命周期管理

#### run
docker run ：创建一个新的容器并运行一个命令
```
使用docker镜像nginx:latest以后台模式启动一个容器,并将容器命名为mynginx。

docker run --name mynginx -d nginx:latest
使用镜像nginx:latest以后台模式启动一个容器,并将容器的80端口映射到主机随机端口。

docker run -P -d nginx:latest
使用镜像 nginx:latest，以后台模式启动一个容器,将容器的 80 端口映射到主机的 80 端口,主机的目录 /data 映射到容器的 /data。

docker run -p 80:80 -v /data:/data -d nginx:latest
绑定容器的 8080 端口，并将其映射到本地主机 127.0.0.1 的 80 端口上。

$ docker run -p 127.0.0.1:80:8080/tcp ubuntu bash
使用镜像nginx:latest以交互模式启动一个容器,在容器内执行/bin/bash命令。

runoob@runoob:~$ docker run -it nginx:latest /bin/bash
root@b8573233d675:/# 
```
#### start/stop/restart

#### kill

#### rm 
docker rm ：删除一个或多少容器

语法
```
docker rm [OPTIONS] CONTAINER [CONTAINER...]
```

**OPTIONS说明：**

- -f :通过SIGKILL信号强制删除一个运行中的容器

- -l :移除容器间的网络连接，而非容器本身

- -v :-v 删除与容器关联的卷docker rm ：删除一个或多少容器

语法
docker rm [OPTIONS] CONTAINER [CONTAINER...]
OPTIONS说明：

-f :通过SIGKILL信号强制删除一个运行中的容器

-l :移除容器间的网络连接，而非容器本身

-v :-v 删除与容器关联的卷
```
强制删除容器db01、db02

docker rm -f db01 db02
移除容器nginx01对容器db01的连接，连接名db

docker rm -l db 
删除容器nginx01,并删除容器挂载的数据卷

docker rm -v nginx01
```
#### pause/unpause

#### create

#### exec
docker exec ：在运行的容器中执行命令
```
在容器mynginx中以交互模式执行容器内/root/runoob.sh脚本

runoob@runoob:~$ docker exec -it mynginx /bin/sh /root/runoob.sh
http://www.runoob.com/
在容器mynginx中开启一个交互模式的终端

runoob@runoob:~$ docker exec -i -t  mynginx /bin/bash
root@b1a0703e41e7:/#
```

### commit 命令
docker commit :从容器创建一个新的镜像。
- -a :提交的镜像作者；
- -c :使用Dockerfile指令来创建镜像；
- -m :提交时的说明文字；
- -p :在commit时，将容器暂停。

将容器a404c6c174a2 保存为新的镜像,并添加提交人信息和说明信息。
```
runoob@runoob:~$ docker commit -a "runoob.com" -m "my apache" a404c6c174a2  mymysql:v1 
sha256:37af1236adef1544e8886be23010b66577647a40bc02c0885a6600b33ee28057
runoob@runoob:~$ docker images mymysql:v1
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
mymysql             v1                  37af1236adef        15 seconds ago      329 MB

```
### 容器与本地之间拷贝文件

```
将主机./RS-MapReduce目录拷贝到容器30026605dcfe的/home/cloudera目录下。
docker cp RS-MapReduce 30026605dcfe:/home/cloudera

将容器30026605dcfe的/home/cloudera/RS-MapReduce目录拷贝到主机的/tmp目录中。
docker cp  30026605dcfe:/home/cloudera/RS-MapReduce /tmp/
```

## 5. 学习资源

- [runoob docker](http://www.runoob.com/docker/docker-tutorial.html)
- [只要一小时，零基础入门Docker](https://zhuanlan.zhihu.com/p/23599229)

