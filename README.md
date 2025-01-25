# YOLOv7-Local-or-Server-training-and-Server-deployment
* 因为大部分文件都是开源的，所以本仓库只附带了一点文件，但是会在相应位置给出下载链接

# REANDME目录
* [配置环境](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/tree/main?tab=readme-ov-file#%E9%85%8D%E7%BD%AE%E7%8E%AF%E5%A2%83)
* [准备数据集]()
* [yolov7参数修改]()
* [本地训练]()
* [服务器训练]()
* [服务器部署]()

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
* 下载训练好的yolov7参数文件yolov7.pt，进入yolov7源码下载地址，往下翻，点击下图所示的地方进行下载，下载完后将yolov7.pt放在yolov7源码的文件夹中
  ![][(https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/yolov7pt_download.png)

## 安装第三方包
### CPU版
* 在yolov7源码里有一个叫requirements.txt的文件，该文件的内容为yolov7所需要的第三包及其对应的版本，如下图所示：
  ![](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/requirements_example.png)
* 打开anaconda prompt，输入以下指令来激活yolov7这个虚拟环境，否则第三包会安装到base这个虚拟环境中而非我们期望的地方
  ```
  conda activate yolov7
  ```
* 然后在anaconda prompt中输入以下指令来安装requirements.txt中的第三方包
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
  ![](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/nvidai%20control%20panel.png)
* 第二种方法：打开cmd，输入以下指令：
  ```
  nvidia-smi
  ```
  需要的读取的参数如下图所示：
  ![](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/nvidai%20smi.png)
  
#### 安装CUDA
* 去下面链接中的table3寻找适合自己电脑的版本，即自己的驱动程序版本大于CUDA版本最右边对应的所需要最低驱动程序版本
  >https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
  
  我的驱动程序版本为517，大于516，所以我选择CUDA 11.7 Update 1这个版本，如下图所示：
  ![](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/cuda_driver_match.png)
* 去下面的链接中找到合适的版本，下载并安装
  >https://developer.nvidia.com/cuda-toolkit-archive
  
  在上一步中我选择的版本为CUDA 11.7 Update 1，所以这里我下载11.7.1这个版本，选择相应的参数并下载，如下图所示：
  ![](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/cuda_download.png)
  
#### 安装第三方包
* 在anaconda prompt中输入以下指令进行第三方包的安装
  ```
  conda activate yolov7                      //激活环境
  cd /d D:\software\conda\envs\yolov7-main  //跳转至yolov7源码的文件夹中，后面的这个地址请根据自己的实际路径去修改
  pip install -r requirements.txt           // 开始安装第三方包
  ```
* 因为上述指令下载的为CPU版的torch，所以需要先卸载torch，在环境已激活的条件下在anaconda prompt中执行以下指令：
  ```
  pip uninstall torch torchvision torchaudio
  ```
* 根据下面的链接选择合适的GPU版的torch版本
  >https://blog.csdn.net/weixin_44842318/article/details/127492491
  
  因为我的CUDA版本为11.7，所以能用的torch版本为1.13.1，1.13.0，1.13.1 ，2.0.0，2.0.1
* 根据上一步选择的torch版本去下面给的官网中找到对应的指令去执行：
  >https://pytorch.org/get-started/previous-versions/
  
  这里我选择1.13.1版本，翻到对应的版本，执行CUDA 11.7版本的指令，如下图所示：
  ![](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/pytorch_download.png)
  在环境已经激活的条件下在anaconda prompt中执行的指令如下，一个是用conda安装，另一个是用pip安装，两个任选一个执行即可，版本不同的请根据自己的版本找到对应的指令
  ```
  conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia 
  pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
  ```
  
#### 验证环境是否安装成功
* 在yolov7的环境中运行cuda_gpu_test.py，如果如下图所示输出为True，则表示torch可以识别到GPU，GPU环境搭建成功
  ![](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/cuda_gpu_test.png)
### 安装wandb
* yolov7训练的时候会使用wandb来图形化显示训练结果，所以也需要安装，在anaconda prompt中输入如下指令进行安装：
  ```
  conda activate yolov7  //激活环境
  pip install wandb      //安装wandb
  ```
* 安装完后，去下面给的官网中注册一个账号
  >https://wandb.ai/home
* 在环境已激活的条件下在anaconda prompt中输入
  ```
  wandb login
  ```
  接着会出现如下提示，去该[网址](Https://wandb.ai/authorize)中复制api到anaconda prompt中，即可完成登录
  ![](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/wandb_login.png)
# 准备数据集
* yolov7数据集要求的格式为txt，github有开源一个叫yolo mark的标注工具，标注后直接就是txt格式，在以下链接中下载：
  >https://github.com/AlexeyAB/Yolo_mark
* 打开yolo_mark_master->x64->Release->data，其中的obj.names为你的标签名称，用记事本编辑，改为自己需要的标签即可，如下图所示：
  ![](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/obj_names.png)
* 将自己用于训练的图片全部放在yolo_mark_master->x64->Release->data->img文件夹中
* 打开yolo_mark_master->x64->Release->yolo_mark.cmd文件，即可对数据进行标注了（最坐牢的一步）
* 标注完成后会在yolo_mark_master->x64->Release->data->img中产生与图片对应的txt文件
* 在yolov7源码的文件夹中新建dataset文件夹专门用来存放数据集，考虑到可能会用yolov7训练几个不同的数据集得到不同的模型，所以在dataset文件夹中再新建一个文件夹来存放此次的数据集，我的是疲劳驾 驶检测，所以文件夹名为Fatigue_driving_detection，同时再在这个文件夹中新建images和labels文件夹用来存放图片和txt，如下图所示：
  ![](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/obj_names.png)
* 最后在Fatigue_driving_detection这个文件夹（即上一步你在dataset中为此次数据集新建的文件夹）中新建train、test、val这3个txt文件，文件中分别写训练集、测试集、验证集对应的图片地址，如下图所示：
  ![](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/train_test_val_txt.png)
# yolov7参数修改

# 本地训练
## CPU版

## GPU版

# 服务器训练

# 服务器部署
