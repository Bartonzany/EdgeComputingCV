# CANN 环境配置

## 1 CANN 软件包升级

下载安装包,将安装包移动到/home/HwHiAiUser/Downloads这个目录，添加权限

![CANN软件包升级](/images/Model_Deployment/CANN软件包升级.png)

```shell
# 切换root用户，root用户密码：Mind@123
(base) HwHiAiUser@orangepiaipro:~$ su
Password: 
(base) root@orangepiaipro:/home/HwHiAiUser#
# 删除镜像中安装CANN软件包释放磁盘空间，防止安装新的CANN软件包磁盘空间不足
(base) root@orangepiaipro:/home/HwHiAiUser# cd /usr/local/Ascend/ascend-toolkit/ 
(base) root@orangepiaipro:/usr/local/Ascend/ascend-toolkit# rm -rf *
(base) root@orangepiaipro: /usr/local/Ascend/ascend-toolkit# cd /home/HwHiAiUser/Downloads
(base) root@orangepiaipro: /home/HwHiAiUser/Downloads# chmod 777 Ascend-cann-toolkit_8.0.RC1.alpha003_linux-aarch64.run
(base) root@orangepiaipro: /home/HwHiAiUser/Downloads# ./Ascend-cann-toolkit_8.0.RC1.alpha003_linux-aarch64.run --install
(base) root@orangepiaipro: /home/HwHiAiUser/Downloads# source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 2 安装必要软件

```shell
apt-get install -y gcc g++ make cmake zlib1g zlib1g-dev openssl libsqlite3-dev libssl-dev libffi-dev unzip pciutils net-tools libblas-dev gfortran libblas3
```


## 3 系统配置

```shell
# 防止休眠
sudo systemctl status sleep.target
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target

npu-smi info # 查看芯片的硬件信息和当前驱动版本信息
npu-smi info watch # 命令可循环打印芯片的温度、AI Core占用率、AI Cpu占用率、Ctrl Cpu占用率、内存使用率、内存带宽占用率
npu-smi info watch -l # 设备id
npu-smi info watch -chip_id # 芯片id

#CPU 性能测试
stress-ng -c 1 --cpu-ops 5000
stress-ng -c 4 --cpu-ops 5000

sysbench --test=cpu --cpu-max-prime=20000 run
sysbench --test=cpu --cpu-max-prime=20000 --threads=4 run

#内存测试
sysbench memory --memory-block-size=2G run
```


## 参考引用 Reference

### 博客 Blogs

- [如何在香橙派AIpro开发板升级CANN软件包](https://bbs.huaweicloud.com/blogs/425346)