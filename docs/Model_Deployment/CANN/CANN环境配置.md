# CANN ��������

## 1 CANN ���������

���ذ�װ��,����װ���ƶ���/home/HwHiAiUser/Downloads���Ŀ¼�����Ȩ��

![CANN���������](/images/Model_Deployment/CANN���������.png)

```shell
# �л�root�û���root�û����룺Mind@123
(base) HwHiAiUser@orangepiaipro:~$ su
Password: 
(base) root@orangepiaipro:/home/HwHiAiUser#
# ɾ�������а�װCANN������ͷŴ��̿ռ䣬��ֹ��װ�µ�CANN��������̿ռ䲻��
(base) root@orangepiaipro:/home/HwHiAiUser# cd /usr/local/Ascend/ascend-toolkit/ 
(base) root@orangepiaipro:/usr/local/Ascend/ascend-toolkit# rm -rf *
(base) root@orangepiaipro: /usr/local/Ascend/ascend-toolkit# cd /home/HwHiAiUser/Downloads
(base) root@orangepiaipro: /home/HwHiAiUser/Downloads# chmod 777 Ascend-cann-toolkit_8.0.RC1.alpha003_linux-aarch64.run
(base) root@orangepiaipro: /home/HwHiAiUser/Downloads# ./Ascend-cann-toolkit_8.0.RC1.alpha003_linux-aarch64.run --install
(base) root@orangepiaipro: /home/HwHiAiUser/Downloads# source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 2 ��װ��Ҫ���

```shell
apt-get install -y gcc g++ make cmake zlib1g zlib1g-dev openssl libsqlite3-dev libssl-dev libffi-dev unzip pciutils net-tools libblas-dev gfortran libblas3
```


## 3 ϵͳ����

```shell
# ��ֹ����
sudo systemctl status sleep.target
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target

npu-smi info # �鿴оƬ��Ӳ����Ϣ�͵�ǰ�����汾��Ϣ
npu-smi info watch # �����ѭ����ӡоƬ���¶ȡ�AI Coreռ���ʡ�AI Cpuռ���ʡ�Ctrl Cpuռ���ʡ��ڴ�ʹ���ʡ��ڴ����ռ����
npu-smi info watch -l # �豸id
npu-smi info watch -chip_id # оƬid

#CPU ���ܲ���
stress-ng -c 1 --cpu-ops 5000
stress-ng -c 4 --cpu-ops 5000

sysbench --test=cpu --cpu-max-prime=20000 run
sysbench --test=cpu --cpu-max-prime=20000 --threads=4 run

#�ڴ����
sysbench memory --memory-block-size=2G run
```


## �ο����� Reference

### ���� Blogs

- [����������AIpro����������CANN�����](https://bbs.huaweicloud.com/blogs/425346)