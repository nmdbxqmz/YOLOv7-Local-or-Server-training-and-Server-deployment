# YOLOv7-Local-or-Server-training-and-Server-deployment
* 因为大部分文件都是开源的，所以本仓库不附带文件，但是会在相应位置给出下载链接

# REANDME目录
* 配置环境
* 标注数据
* yolov7参数修改
* 本地训练
* 服务器训练
* 服务器部署

# 配置环境
## anaconda新建环境
* 因为yolo对环境安装的第三包的版本要求比较严格，所以建议使用anaconda来创建一个yolo专用的环境
* 安装anaconda下面的链接已经讲的很详细了，不过多赘述

  anaconda安装教程：
  >https://blog.csdn.net/qq_44000789/article/details/142214660
* 打开anaconda自带的anaconda prompt，输入以下指令来创建一个python版本为3.7.4的新环境，其中的yolov7为该虚拟环境的名称，你也可以改为其他的，不过需要注意之后的部分指令也需要做相应的更改
  ```
  conda create -n yolov7 python==3.7.4 
  ```
  执行完毕后会在anaconda安装目录下的envs文件夹中生成一个叫yolov7的文件夹，如下图所示：
  ![](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/envs_environment.png)

## 下载yolov7源码
* yolov7源码下载地址如下：
  >https://github.com/WongKinYiu/yolov7
* 把源码下到本地比较快的方法是下载zip，然后本地解压，操作如下图所示：
  ![](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/yolov7_download_zip.png)
* 建议把yolov7的源码解压到conda的envs目录下，我的文件目录如下图所示，之后会以此路径来执行指令，路径不同的指令做相应变化即可
  ![](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/total_environment_path.png)

## 安装第三方包
### CPU版
* 在yolov7源码里有一个叫requirements.txt的文件，该文件的内容为yolov7所需要的第三包及其对应的版本，如下图所示：
  ![]()
* 打开anaconda prompt，输入以下指令来激活yolov7这个虚拟环境，否则第三包会安装到base这个虚拟环境中而非我们期望的地方
  ```
  conda activate yolov7
  ```
* 输入以下指令来安装requirements.txt中的第三方包
  ```
  cd /d D:\software\conda\envs\yolov7-main  //跳转至yolov7源码的文件夹中，后面的这个地址请根据自己的实际路径去修改
  pip install -r requirements.txt           // 开始安装第三方包
  ```
### GPU版
#### 读取自己电脑的GPU配置
* 参考链接：
  >https://blog.csdn.net/bruce_zhao1407/article/details/109580835
* 需要读取的配置参数为驱动程序版本和最高支持的CUDA版本
* 第一种方法：打开电脑上的NVIDIA Control Panel，点击的系统信息，在显示和组件一栏分别读取驱动程序版本和最高支持的CUDA版本，如下图所示：
  ![]()
* 第二种方法：打开cmd，输入以下指令：
  ```
  nvidia-smi
  ```
  需要的读取的参数如下图所示：
  ![]()
#### 安装CUDA
*
#### 安装第三方包
*
### 安装wandb
