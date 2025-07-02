---
aliases:
  - Git 常用指令
title: Git 常用指令
date: 2025-06-28 09:14:55
excerpt: Git 学习
tags:
  - GIt
---

## Git 常用指令

---

### 1. GIt 安装

#### 1.1 包安装

```shell
# Linux 系统: Ubuntu 10.10(maverick)或更新版本，Debian(squeeze)或更新版本
sudo aptitude install git
sudo aptitude install git-doc git-svn git-email gitk

# Linux 系统:RHEL、Fedora、CentOs 等版本:

yum install git
yum install git-svn git-email gitk
```

#### 1.2 源代码安装

```shell
tar -jxvf git-2.19.0.tar.bz2
cd git-2.19.0

make prefix=/usr/local a11
sudo make prefix=/usr/local install

# 安装Git文档
make prefix=/usr/local doc info
sudo make prefix=/usr/local instal1-doc insta1l-html instal1-info
```

#### 1.3 命令补齐

```shell
# 将Git源码包中的命令补产脚本复制到bash-completion对应的目录中:
cp contrib/completion/git-completion.bash /etc/bash completion.d/

# 重新加载自动补齐脚本，使之在当前shell中生效:
. /etc/bash completion

# 为了能够在终端开启时自动加载bash_completion脚本，在本地配置文件 ~/.bash_profile 或全局文件 /etc/bashrc 文件中添加下面的内容:
if [-f /etc/bash completion ]; then
. /etc/bash completion
fi
```

### 2. Git 初始配置

#### 2.1 Git 环境配置

```shell
# 系统配置(对所有用户都适用):%Git%/etc/gitconfig
git config --system core.autocrlf

# 用户配置(只适用于该用户):~/.gitconfig
git config --global user.name "LinXi"  
git config --global user.email "LinXi198913@163.com"

# 仓库配置(只对当前项目有效)
git config --local remote.origin.url

# 查看配置信息:.git/config
git config --list  
git config user.name
```

#### 2.2 Git 文本换行符配置

假如你正在Windows上写程序，又或者你正在和其他人合作，他们在Windows上编程，而你却在其他系统上，在这些情况下，你可能会遇到行尾 结束符问题。这是因为Windows使用回车和换行两个字符来结束一行，而Mac和Linux只使用换行一个字符。虽然这是小问题，但它会极大地扰乱跨平台协作。

Git可以在你提交时自动地把行结束符CRLF转换成LF，而在签出代码时把LF转换成CRLF。用core.autocrlf来打开此项功能，如果是在Windows系统上，把它设置成true，这样当签出代码时，LF会被转换成CRLF：

```shell
git config --global core.autocrlf true
```

Linux或Mac系统使用LF作为行结束符，因此你不想Git在签出文件时进行自动的转换;当一个以CRLF为行结束符的文件不小心被引入时你肯定想进行修正，把core.autocrlf设置成input来告诉Git在提交时把CRLF转换成LF，签出时不转换：

```shell
git config --global core.autocrlf input
```

这样会在Windows系统上的签出文件中保留CRLF，会在Mac和Linux系统上，包括仓库中保留LF。

如果你是Windows程序员，且正在开发仅运行在Windows上的项目，可以设置false取消此功能，把回车符记录在库中：

```shell
git config --global core.autocrlf false
```

#### 2.3 Git 文本编码配置

- **i18n.commitEncoding 选项**：用来让git commit log存储时，**采用的编码**，默认UTF-8.
- **i18n.logOutputEncoding 选项**：查看git log时，**显示采用的编码**，建议设置为UTF-8.

```shell
# 中文编码支持
git config --global gui.encoding utf-8
git config --global i18n.commitencodinding utf-8
git config --global i18n.logoutputencoding utf-8

# 显示路径中的中文:
git config --global core.quotepath false
```

#### 2.4 Git 认证方式

**http / https 协议认证**

```shell
# 设置口令缓存:
git config --global credential.heler store

# 添加 HTTPS 证书信任:
git config http.sslverify false
```

**ssh 协议认证**

SSH协议是一种非常常用的Git仓库访问协议，使用公钥认证、无需输入密码，加密传输，操作便利又保证安全性

```shell
ssh-keygen -t rsa -C LinXi198913@163.com
```

### 3. Git 常用命令

#### 3.1 工程准备

```shell
# 新建git项目仓库
git init 

# 克隆远端工程
git clone [url]
git lfs clone [url]  # 专用于二进制文件
```

#### 3.2 新增/删除/移动文件到暂存区

```shell
git add code.cpp     # 将未跟踪的文件加入暂存区
git add .            # 将所有文件添加到 Git 暂存区
git rm               # 删除文件
git mv               # 移动或重命名文件
```

#### 3.3 查看工作区

```shell
git diff                              # 查看工作区与暂存区之间的差异（未 add 的修改）

git diff --cached                     # 查看暂存区与最近一次提交之间的差异（已 add 但未 commit 的修改）

git diff --staged                     # 功能同上，与 git diff --cached 等价

git diff HEAD                         # 查看工作区与最近一次提交之间的所有差异（包括已暂存和未暂存的）

git diff <commit1> <commit2>          # 查看两次提交之间的差异

git diff <branch1> <branch2>          # 查看两个分支最新提交之间的差异

git diff --name-only                  # 只显示发生变化的文件名列表

git diff --name-status                # 显示发生变化的文件列表及其状态（A: 新增, M: 修改, D: 删除）

git diff <file-path>                  # 查看某个文件在工作区与暂存区之间的差异

git diff <commit1>..<commit2> -- <file-path>   # 查看某文件在两个提交之间的差异

git diff branch1..branch2 -- <file-path>       # 查看某文件在两个分支之间的差异

git show <commit-hash>                # 查看某次提交的具体改动内容及提交信息

git difftool                            # 使用配置的图形化对比工具查看差异（需提前配置）
```

```shell
# 查看文件状态
git status      
git status -s
```

#### 3.3 提交更改的文件

```shell
git commit -m "subscribe" code.cpp  # 输入提交信息
git commit -v                       # 在提交时显示具体的变化
git commit -amend                   # 补充提交说明
```

#### 3.4 查看日志

```shell
git log 
```

#### 3.5 推送至远端仓库

```shell
git push [remote-name] [branch-name] # 推送到远程仓库
```

#### 3.6 分支管理

```shell
git branch                       # 列出所有本地分支  
git branch -r                    # 列出所有远程分支 
git branch -a                    # 列出所有本地分支和远程分支 

git branch name                  # 创建分支，不切换到新分支
git checkout -b name             # 创建分支并切换到新分支

git checkout name                # 切换分支

git merge name                   # 合并分支

git branch -d name               # 删除本地分支
git branch -d -r name            # 删除远端分支

git pull                         # 从远程仓库获取最新版本并merge到本地仓库
git fetch                        # 从远程仓库获取最新版本到本地仓库，不会自动merge
```

#### 3.7 分支合并

```shell
git merge                 # 从指定分支合并到当前分支
git rebase                # 从指定分支合并到当前分支，不建议在版本主干上使用
```

#### 3.8 撤销

```shell
git reset code.cpp        # 将暂存区的文件取消暂存

git checkout .            # 回退本地所有修改未提交文件内容
git checkout -filename    # 回退本地某个修改未提交文件内容
git checkout commit_id    # 回退某个提交版本
```

#### 3.9 远程仓库操作

```shell
git remote # 查看远程仓库
git remote add <shortname> < url> # 添加一个新的远程 Git 仓库
git remote rm # 从远程仓库克隆

git branch --set-upstream-to origin/master master # 将本地分支与远程分支关联
```

### 参考引用

#### 网页链接

- [超详细的Git使用教程(图文)-CSDN博客](https://blog.csdn.net/qq_37883866/article/details/105349257)
- [添加远程库 - 廖雪峰的官方网站 (liaoxuefeng.com)](https://www.liaoxuefeng.com/wiki/896043488029600/898732864121440)

---