##  Linux基本命令使用

---

### 1. Linux 目录结构

Linux 文件系统采用**层级式**的树状目录结构，最顶层是根目录“/”，所有其他目录和文件都从根目录开始分支。

**结构示意图**：

![](../../../images/Linux/Pasted%20image%2020220207221456.png)

![](../../../images/Linux/Pasted%20image%2020220207223209.png)

**主要目录及其功能分析**：

- `/bin`：`bin`  是 Binaries (二进制文件) 的缩写，存放**最常用的命令**，这些命令是系统运行所**必需的二进制文件**，例如 `ls`、`cp`、`mv`、`bash` 等
- `/boot`：包含启动 `Linux` 时所需的**核心文件**，如**内核镜像**和**引导加载程序**。
- `/dev`： `dev`  是 Device(设备) 的缩写, 存放 `Linux` 的外部设备，在 `Linux` 中通过文件接口访问硬件设备。
- `/etc`：`etc` 是 Etcetera(等等) 的缩写，存放系统**配置文件**和子目录，用于系统管理和配置。
- `/home`：用户的主目录，在 `Linux` 中，每个用户都有一个以用户名命名的子目录，如上图中的 alice、bob 和 eve。
- `/lib`： `lib` 是 Library(库) 的缩写，存放系统基本的**动态连接共享库**，其作用类似于 Windows 里的 `DLL` 文件。大多数应用程序都需要用到这些共享库，一般以 .so 和 .a **动静态链接库**为主。
- `/lost+found`：用于存放系统非正常关机后恢复的文件。
- `/media`： `Linux` 系统会自动识别一些设备，例如**U盘**、**光驱**等等。当识别后，`Linux` 会把识别的设备挂载到这个目录下。
- `/mnt`：用于临时挂载其他文件系统，如网络共享或外部存储设备。可以将光驱挂载在 `/mnt` 上，进入该目录就可以查看光驱里的内容了。
- `/opt`：`opt` 是 optional(可选) 的缩写，用于安装**第三方软件**，通常每个软件包有自己的子目录。默认是空的。
- `/proc`：`proc` 是 Processes(进程) 的缩写，是一种**虚拟文件系统**，提供**内核和进程的实时信息**，可以通过直接访问这些文件来**获取或修改系统状态**。可以通过下面的命令来屏蔽主机的 ping 命令，使别人无法 ping 你的机器：

	```
	echo 1 > /proc/sys/net/ipv4/icmp_echo_ignore_all
	```
	
- `/root`：系统管理员（超级用户）的用户主目录。
- `/sbin`：Superuser Binaries (超级用户的二进制文件) 的缩写，存放**系统管理员使用的基本系统管理命令**，通常用于系统维护和修复。例如 `ifconfig`、`fdisk`、`reboot`。
- `/selinux`：存放与 **SELinux** 安全机制相关的文件（主要在 Redhat/CentOS 系统中）。
- `/srv`：存放一些服务启动之后需要提取的数据，例如 Web 服务器的网站文件。
- `/sys`：虚拟文件系统，提供**内核设备树的信息**，用于管理和配置硬件设备。
- `/tmp`：存放**临时文件**，系统重启后通常会清空。
- `/usr`：unix shared resources(共享资源) 的缩写，存放**用户应用程序和共享资源**，类似于 Windows 的 `Program Files` 目录
    - `/usr/bin`：普通用户使用的**应用程序和非核心命令**，区别于 `/bin` 。例如 `gcc`、`python`、`git` 等
    - `/usr/sbin`：超级用户更高级的**管理命令**和**系统守护程序**。例如 `useradd`、`httpd`、`cron`。
    - `/usr/src`：内核源代码默认存放位置。
- `/var`：variable(变量) 的缩写，存放**经常变化的文件**，如日志文件、缓存和数据库。
- `/run`：是一个临时文件系统，存放系统启动以来的运行时信息，重启后会被清除。

### 2. vim 常用指令

**Vim**（Vi IMproved）是一个高度可配置的、功能强大的文本编辑器，是 Unix 系统上经典编辑器 `Vi` 的增强版。Vim 以其高效性和灵活性著称，广泛用于编程、系统管理和文本编辑任务。

`Vim` 包括以下三种工作模式：

1. **交互模式**

```shell
ESC                   # 进入交互模式
0/Home                # 将光标定位到一行的开始位置
$/End                 # 将光标定位到一行的结束位置
x/delete              # 删除字符，x键之前输入数字可以一次性删除多个字符
hjkl                  # 向上下左右移动一个字符，也可以使用方向键
w                     # 一个单词一个单词地移动
gg                    # 跳转到文件的第一行，先输入数字会跳转到指定行
dd                    # 删除所在一行，先输入数字会删除从光标所在行开始的行数
ggdG                  # 删除整个文件内容
dw                    # 删除一个单词：将光标置于一个单词的首字母处
d0                    # 从光标所在位置删除至行首
d$                    # 从光标所在位置删除至行末
yy                    # 复制所在一行，先输入数字会复制从光标所在行开始的行数
yw                    # 复制一个单词
y0                    # 从光标所在位置复制至行首
y$                    # 从光标所在位置复制至行末
p                     # 在当前行后粘贴
r                     # 替换字符
u                     # 撤销，先输入数字会撤销最近几次修改
/                     # 查找
set nu                # 设置行号
set nonu              # 不设置行号
```

2. **插入模式**

```shell
i                     # 进入插入模式，从光标前插入
I                     # 进入插入模式，从光标所在行第一个非空格符处插入
a                     # 进入插入模式，从光标后插入
A                     # 进入插入模式，从光标所在行最后一个非空格符处插入
o                     # 进入插入模式，从光标所在的下一行处输入新的一行
O                     # 进入插入模式，从光标所在的上一行处输入新的一行
```

3. **命令模式**

```shell
:q                    # 退出
:w                    # 写入
```

三种模式切换示意图：

![](../../../images/Linux/Pasted%20image%2020220216223346.png)

**vim 键盘图**：

![](../../../images/Linux/Pasted%20image%2020250224115523.png)

### 3. Linux 系统管理

#### 3.1. 关机命令

```shell
shutdown -h now                       # 立即关机
shutdown -h +1                        # 1分钟后关机
poweroff                              # 立即关机
halt                                  # 立即关机（不保存任何数据）
```

#### 3.2. 重启命令

```shell
shutdown -r now                       # 立即重启
reboot                                # 立即重启
```

#### 3.3. 数据同步

```shell
sync                                   # 将内存数据同步到硬盘
```

#### 3.4. 用户登录

```shell
su [username]                         # 切换到指定用户（需要权限）
logout                                # 退出当前用户会话
ctrl + D                              # 按下 Ctrl+D 也可退出当前用户会话
```

#### 3.5. 常用快捷指令

```shell
Ctrl + L                              # 清除屏幕并将当前行移到页面顶部
Ctrl + C                              # 中止当前正在执行的命令
Ctrl + D                              # 关闭 Shell 会话
Ctrl + U                              # 从光标位置剪切到行首
Ctrl + K                              # 从光标位置剪切到行尾
Ctrl + W                              # 剪切光标左侧的一个单词
Ctrl + Y                              # 粘贴 Ctrl + U | K | W 剪切的内容
Ctrl + A                              # 光标跳到命令行的开头或Home键
Ctrl + E                              # 光标跳到命令行的结尾或End键
```

### 4.用户管理

#### 4.1. 添加用户

```shell
useradd [username]                    # 添加指定用户（需要权限）   
useradd -d [dir]                      # 添加用户并指定主目录
useradd -s [shell] [username]         # 添加用户并指定登录Shell
useradd -m [username]                 # 添加用户并创建主目录
useradd –g [group] [username]         # 添加用户并指定用户组
useradd -e [expiry_date] [username]   # 添加用户并设置过期日期
useradd -u [UID] [username]           # 添加用户并指定用户ID
adduser                               # 交互式添加用户
```

#### 4.2. 修改密码

```shell
passwd [username]                     # 修改指定用户的密码（需要权限）
passwd -e [username]                  # 强制指定用户在下次登录时修改密码
```

#### 4.3. 删除用户

```shell
userdel [username]                    # 删除指定用户（需要权限）   
userdel -r [username]                 # 删除用户及其主目录
deluser [username]                    # 删除指定用户
deluser [username] [groupname]        # 从指定用户组中删除用户
```

#### 4.4. 查询用户

```shell
id [username]                         # 查询指定用户的UID和GID（需要权限）   
finger [username]                     # 显示用户的详细信息
getent passwd [username]              # 查询系统用户数据库中指定用户的信息   
```
#### 4.5. 查看当前用户

```shell
whoami                                # 查看当前登录的用户名
who am I                              # 查看当前登录的用户信息
w                                     # 显示当前登录用户及其活动
last                                  # 查看用户登录记录
```

#### 4.6. 添加用户组

```shell
groupadd [groupname]                  # 添加新的用户组
groupadd -g [GID] [groupname]         # 添加新的用户组并指定组ID
```

#### 4.7. 修改用户组

```shell
usermod -g [groupname] [username]     # 修改指定用户的主用户组
usermod -aG [groupname] [username]    # 将用户添加到附加用户组（不移除现有组）
usermod -L [username]                 # 锁定指定用户账户
usermod -U [username]                 # 解锁指定用户账户
```

#### 4.8. 修改用户属性

```shell
usermod -c "[new_comment]" [username] # 修改用户的注释信息
usermod -d [new_dir] [username]       # 修改用户的主目录
usermod -s [new_shell] [username]     # 修改用户的登录Shell
```

### 5. 

### 6. 

---

## 参考引用

### 书籍出处

- [韩顺平_2021图解Linux全面升级](../../../asset/Linux/韩顺平_2021图解Linux全面升级.pdf)
- [鸟哥的Linux私房菜-基础篇](../../../asset/Linux/鸟哥的Linux私房菜-基础篇.pdf)
- [鸟哥的Linux私房菜-服务器架设篇](../../../asset/Linux/鸟哥的Linux私房菜-服务器架设篇.pdf)

### 网页链接

- [鳥哥私房菜 - 鳥哥的首頁](https://linux.vbird.org/)
- [Linux最强总结来啦！](https://mp.weixin.qq.com/s/ymKnDXRW06BOFqMiExRZYQ)
- [Linux命令行万能解压命令 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/529338576#)
- [史上最全Vim快捷键大全-CSDN博客](https://blog.csdn.net/ZYC88888/article/details/82947961)










### 1 文件管理

#### find

- [Linux find 命令 (qq.com)](https://mp.weixin.qq.com/s/XUgkOy8Rbwil1ZOHQ6KpUQ)

### 2 文档编辑

#### grep

- [Linux下高效实用的grep命令 (qq.com)](https://mp.weixin.qq.com/s/Hp7IzYVZaJVorlTt0WtNEA)



### 9 备份压缩

#### tar

- Linux命令行万能解压命令


-  `Shell` 种类
```shell
# 1.查看到当前正在使用的 Shell
echo $SHELL

# 2.查看当前系统安装的所有 Shell 种类
cat /etc/shells
```


- `Xshell` 连接 `Ubuntu`
```shell
# 1.更新资料列表
sudo apt-get update

# 2.安装openssh-server
sudo apt-get install openssh-server

# 3.查看ssh服务是否启动
sudo ps -e | grep ssh

# 4.如果没有启动，启动ssh服务
sudo service ssh start

# 5.查看IP地址
sudo ifconfig
inet addr:192.168.252.128
```

- 复制、移动和剪切操作
```shell
# 1.复制操作
sudo cp -r 要复制的文件的路径 复制的目标文件夹
sudo cp -r /xxx/xxx/桌面/a /xxx/xxx/xxx


```

- 权限不够
```shell
# 1.以root账户进入图形文件夹界面
sudo nautilus


```



- 文件操作指令
```shell
1）pwd            #显示当前目录的路径；
2）ls             #列出文件和目录
	-a            #显示所有文件和目录包括隐藏的
	-l            #显示详细列表 
	-h            #适合人类阅读的
	-t            #按文件最近一次修改时间排序
	-i            #显示文件的 inode （inode是文件内容的标识）
3）cat            #一次性显示文件所有内容，更适合查看小的文件
	-n            #显示行号
4）less           #分页显示文件内容，更适合查看大的文件
5）mkdir          #创建一个目录
```

- 网络操作指令
```shell
1）ifconfig       #查看IP网络相关信息；
	eth0          #对应有线连接,Ethernet的缩写
	lo            #表示本地回环（Local Loopback的缩写，对应一个虚拟网卡）
	wlan0         #表示无线局域网
2)host            #IP地址和主机名的互相转换
3)ssh             #通过非对称加密以及对称加密的方式,连接到远端服务器
```

- 截屏
```shell
Alt＋PrintScreen
```
