# YOLOv7-Local-or-Server-training-and-Server-deployment
* 因为大部分文件都是开源的，所以本仓库不附带文件，但是会在相应位置给出下载链接

## REANDME目录
* 配置环境
* 标注数据
* yolov7参数修改
* 本地训练
* 服务器训练
* 服务器部署

## 配置环境
### anaconda新建环境
* 因为yolo对环境安装的第三包的版本要求比较严格，所以建议使用anaconda来创建一个yolo专用的环境
* 安装anaconda下面的链接已经讲的很详细了，不过多赘述

  anaconda安装教程：
  >https://blog.csdn.net/qq_44000789/article/details/142214660
* 打开anaconda自带的anaconda prompt，输入以下指令来创建一个python版本为3.7.4的新环境
  ```
  conda create -n yolov7 python==3.7.4 //yolov7为该环境的名称，你也可以改为其他的，不过需要注意之后的部分指令也需要做相应的更改
  ```
  此时会在anaconda安装目录下的envs生成一个叫yolov7的文件，如下图所示：
  ![](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/envs_environment.png)

### 下载yolov7源码
* yolov7源码下载地址如下：
  >https://github.com/WongKinYiu/yolov7
* 把源码下到本地比较快的方法是下载zip，然后本地解压，操作如下图所示：
  ![](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/yolov7_download_zip.png)
* 建议把yolov7的源码解压到conda的envs目录下，我的文件目录如下图所示，之后会以此路径来执行指令，路径不同的指令做相应变化即可
  ![](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/total_environment_path.png)
  
