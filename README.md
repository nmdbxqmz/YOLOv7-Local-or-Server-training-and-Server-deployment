# YOLOv7-Local-or-Server-training-and-Server-deployment
* 因为大部分文件都是开源的，所以本仓库只附了少量文件，但是开源文件会在相应位置给出下载链接

# REANDME目录
* [配置环境](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/tree/main?tab=readme-ov-file#%E9%85%8D%E7%BD%AE%E7%8E%AF%E5%A2%83)：anaconda新建环境、yolov7源码下载、安装第三方包（CPU/GPU版）、安装wandb
* [准备数据集](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment?tab=readme-ov-file#%E5%87%86%E5%A4%87%E6%95%B0%E6%8D%AE%E9%9B%86)：下载yolo mark、修改配置、进行标注、移动图片和标签至指定文件夹中
* [yolov7参数修改](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment?tab=readme-ov-file#yolov7%E5%8F%82%E6%95%B0%E4%BF%AE%E6%94%B9)：yolov7.yaml修改、coco.yaml修改、train.py修改
* [本地训练](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment?tab=readme-ov-file#yolov7%E5%8F%82%E6%95%B0%E4%BF%AE%E6%94%B9)：CPU训练、GPU训练
* [服务器训练](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment?tab=readme-ov-file#yolov7%E5%8F%82%E6%95%B0%E4%BF%AE%E6%94%B9)：mobaxterm工具，autodl服务器租借，mobaxterm连接服务器，服务器环境配置、开始训练
* [服务器部署](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment?tab=readme-ov-file#yolov7%E5%8F%82%E6%95%B0%E4%BF%AE%E6%94%B9)：全部程序的都在服务器上运行，部分程序在服务器上运行
* [debug](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/debug_torchaudio.png)：服务器安装torch失败

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
  ![](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/yolov7pt_download.png)

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
  
  在上一步中我选择的版本为CUDA 11.7 Update 1，所以这里我下载11.7.1这个版本，选择相应的参数下载并安装，如下图所示：
  ![](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/cuda_download.png)
* 安装完成后，打开cmd，输入以下指令，若出现如下图所示的内容则说明CUDA安装成功
  ```
  nvcc -V
  ```
  ![](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/nvcc_V.png)
#### 安装第三方包
* 参考链接
  >https://blog.csdn.net/Stromboli/article/details/142705892
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
  在环境已经激活的条件下在anaconda prompt中执行的指令如下，conda和pip都有2条指令，其中第二条指定了具体版本，兼容性更好，在conda/pip中的两个指令任选一个执行即可，版本不同的请根据自己的版本找到对应的指令
  ```
  # 使用conda安装
  conda install pytorch torchvision torchaudio cudatoolkit=11.7 -c pytorch
  conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
  # 使用pip安装
  pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
  pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
  ```
  
#### 验证环境是否安装成功
* 在yolov7的环境中运行cuda_gpu_test.py，如果如下图所示输出为True，则表示torch可以识别到GPU，GPU环境搭建成功
  ![](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/cuda_gpu_test.png)
  
### 安装wandb
* 参考文档
  >https://blog.csdn.net/SF199853/article/details/132723055
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
* 打开yolo_mark_master->x64->Release->data，其中的obj.names为你的类别名称，用记事本编辑，改为自己需要的类别名称即可，如下图所示：
  ![](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/obj_names.png)
* 将自己用于训练的图片全部放在yolo_mark_master->x64->Release->data->img文件夹中
* 打开yolo_mark_master->x64->Release->yolo_mark.cmd文件，即可对数据进行标注了（最坐牢的一集），如下图所示：
  ![](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/yolo_mark.png)
* 标注完成后会在yolo_mark_master->x64->Release->data->img中产生与图片对应的txt文件
* 在yolov7源码的文件夹中新建dataset文件夹专门用来存放数据集，考虑到可能会用yolov7训练几个不同的数据集得到不同的模型，所以在dataset文件夹中再新建一个文件夹来存放此次的数据集，我的是疲劳驾 驶检测，所以文件夹名为Fatigue_driving_detection，同时再在这个文件夹中新建images和labels文件夹用来存放图片和txt，如下图所示：
  ![](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/images_labels.png)
* 最后在Fatigue_driving_detection这个文件夹（即上一步你在dataset中为此次数据集新建的文件夹）中新建train、test、val这3个txt文件，文件中分别写训练集、测试集、验证集对应的图片地址，如下图所示：
  ![](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/train_test_val_txt.png)
* 觉得上一步手动写txt麻烦？这里有可以一键完成上一步操作的case2.py文件，下载该文件放到yolov7源码文件夹中，其中需要修改的参数有xml_path、验证集、测试集的大小，xml_path修改为自己labels文件夹的路径，验证集、测试集大小根据自己数据集大小进行修改，如下图所示：
  ![](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/case2.png)
  参考文档
  >https://blog.csdn.net/m0_62899814/article/details/129934760
  
# yolov7参数修改
## yolov7源码文件夹->cfg->training->yolov7.yaml修改
* 将yolov7源码文件夹->cfg->training->yolov7.yaml这个文件复制一份后重命名为my_yolov7.yaml，然后将nc后的数字改为自己训练集中的类别总数（我的训练集类别总数为5，所以这里填5），其余都不动，如下图所示：
  ![](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/my_yolov7_yaml.png)
  
## yolov7源码文件夹->data->coco.yaml修改
* 将 yolov7源码文件夹->data->coco.yaml这个文件复制一份后重命名为mydata.yaml
* 首先修改数据集地址，将train、val、test后的txt文件地址改为自己此次数据集中train、val、test.txt文件对应的地址，然后是nc后的数字改为自己训练集中的类别总数（即与my_yolov7.yaml中的一致），最后将names后list中的名称改为自己训练集的类别名称，记得与obj.names中的顺序一致，如下图所示：
  ![](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/mydata_yaml.png)
  
## yolov7源码文件夹->train.py修改
* 打开yolov7源码文件夹->train.py文件，往下翻到‘if __name__ == '__main__':’这里，修改weights、cfg、data、epochs后的参数，分别为yolov7.pt文件地址、my_yolov7.yaml文件地址，mydata.yaml文件地址、训练的轮数，如下图所示：
  ![](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/train_py.png)
* 至此yolov7基本参数修改完毕

# 本地训练
## CPU版
* 将yolov7源码文件夹->train.py文件中的device->default后的参数修改为cpu，如下图所示：
  ![](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/device_cpu.png)
* 打开anaconda prompt，依次输入以下指令即可开始训练
  ```
  conda activate yolov7                      //激活环境
  cd /d D:\software\conda\envs\yolov7-main  //跳转至yolov7源码的文件夹中，后面的这个地址请根据自己的实际路径去修改
  python train.py                           //执行train.py文件，即开始训练
  ```
  
## GPU版
* 将yolov7源码文件夹->train.py文件中的device->default后的参数修改为0（0为GPU训练），batch-size->default后的参数为占用GPU内存的大小，请确保这个值小于你的GPU内存大小（我的存储为2GB，所以这里填1），如下图所示：
  ![](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/device_gpu.png)
* 打开anaconda prompt，依次输入以下指令即可开始训练
  ```
  conda activate yolov7                      //激活环境
  cd /d D:\software\conda\envs\yolov7-main  //跳转至yolov7源码的文件夹中，后面的这个地址请根据自己的实际路径去修改
  python train.py                           //执行train.py文件，即开始训练
  ```
* 训练时anaconda prompt显示如下：
  ![](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/local_gpu_train.png)

# 服务器训练
* 参考文档
>https://blog.csdn.net/weixin_43409991/article/details/134801718
## mobaxterm工具
* mobaxterm功能比较强大，服务器训练将以该工具为例进行演示，下载链接如下，下载完后无脑安装即可
  >https://mobaxterm.mobatek.net/
* 因为会涉及到许多从电脑到mobaxterm与从mobaxterm到电脑的复制粘贴，Crtl + c、Crtl + v大法有的时候没用，所以下面给出一部分复制粘贴快捷键
  ```
  Crtl + c、Crtl + v  //电脑端复制、粘贴快捷键
  选中内容+ Ctrl + Insert(Ins) //mobaxterm复制
  Ctrl + Insert(Ins)  //mobaxterm粘贴
  ```
  参考文档
  >https://blog.csdn.net/Sure_Lee/article/details/115065262
* 电脑端与服务器间的文件传输：mobaxterm连接上服务器后在左侧一栏会显示服务器的目录，电脑端与服务器可以直接拖拽文件来进行传输，如下图所示：
  ![](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/pc_server_translate.png)
  
## AutoDL租借服务器
* 去AutodDL官网注册一个账号，地址如下：
  >https://www.autodl.com/home
* 点击上方的算力市场，然后选择地区和GPU型号，有可用的会显示x卡可租，点击x卡可租即进入具体界面，在镜像一栏选择基础镜像，然后在下面的下拉框中选择Miniconda / conda3 / 3.10(ubuntu22.04) / 11.8，最后点击立刻创建即可完成服务器的租借，流程如下图所示：
  ![](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/autodl_rent.png)

## 使用mobaxterm连接服务器
* AutoDL租借完服务器后，在容器实例一栏可以看到自己刚刚租的服务器，如下图所示：
  ![](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/autodl_ssh_key.png)
* 我们需要复制登录指令和命名，复制完信息如下：
  ```
  ssh -p 33019 root@connect.nmb1.seetacloud.com
  tsZRdXT43EmJ
  ```
  其中33019为端口号，root为用户名，connect.nmb1.seetacloud.com为服务器地址，tsZRdXT43EmJ为登录服务器的密码
* 打开mobaxterm，点击上方的session，在弹出的界面中选择SSH，然后在相应的位置填入服务器地址、用户名、端口号，最后点击ok即可开始连接服务器，如下图所示：
  ![](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/mobaxterm_link.png)
* 随后会让你输入密码，把刚刚的密码复制过来，按下回车即可，注意你输入的密码是不会在上面显示出来的，连接成功后的界面如下图所示：
  ![](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/mobaxterm_key.png)

## 传输文件到服务器
* 将你需要用到的文件拖拽到服务器的目录里（如果按我上述的操作只需要上传整个yolov7源码文件夹即可），传输比较慢，需要耐心等待

## 服务器端配置环境
* 输入以下指令来进行服务器环境配置，其实与本地配置环境差不多，因为我们选择服务器环境时cuda版本选择为11.8，所以torch指令需要做相应的变换
  ```
  conda create -n yolov7 python==3.7.4          //创建虚拟环境
  conda activate yolov7                         //激活环境
  cd /d ./yolov7-main                          //移动至yolov7源码文件夹中
  pip install -r requirements.txt              //安装第三方包
  pip uninstall torch torchvision torchaudio  //卸载cpu版本包
  pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118  //安装cuda 11.8版本对应的torch套件
  ```
  最后一个指令执行失败的点击[这里](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/debug_torchaudio.png)
* 下载并登录wandb，与本地一样，输入以下指令：
  ```
  conda activate yolov7  //激活环境
  pip install wandb      //安装wandb
  wandb login            //登录wandb
  ```
* 将cuda_gpu_test,py也一同传输到yolov7源文件夹中，执行以下指令，输出为True即为服务器环境配置成功
  ```
  conda activate yolov7    //激活环境
  python cuda_gpu_test.py //运行测试文件
  ```
## 开始训练
* 输入以下指令开始训练
  ```
  conda activate yolov7    //激活环境
  python train.py          //开始训练
  ```
* 训练完成后的服务器输出如下图所示：
  ![](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/server_train_over.png)

# 服务器部署
## 全部程序都在服务器上运行
* 先用mobaxterm连接上服务器，配置完yolov7的环境，并将yolov7源文件一并上传，操作与上面一样，这里不过多赘述（其实上传完后，可以用mobaxterm的指令窗口直接开始运行自己的程序，下面使用pycharm连接服务器属于是连接服务器的其他方法，没有mobaxterm的方便，不想用pycharm的可以直接跳过）
* 需要使用pycharm专业版，普通版没有ssh功能
* 打开pycharm，点击上方的文件，选择设置，在弹窗中点击项目：xxx->python解释器，点击右上角的小齿轮，选择添加，然后新弹窗中选择SSH解释器，输入服务器地址、端口、用户名，最后点击下一步，操作如下图所示：
  ![](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/pycharm_setting.png)
* 之后会跳弹窗询问是否连接，选择ok，然后服务器输入密码，如下图所示：
  ![](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/pycharm_key.png)
* 修改第一栏的解释器映射位置，我们创建的yolov7虚拟环境位置为miniconda3/envs/yolo，python解释器位置在yolo/bin/python，选择完成后，把自动更新取消（我测试时自动更新很容易上传文件失败），我们自己通过mobaxterm手动上传即可，最后点击完成即可，操作如下图所示：
  ![](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/pycharm_map.png)
* 等待pycharm更新完框架后，点击上方的工具，选择启动ssh会话。在弹窗中选择第一个，操作如下图所示:
  ![](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/deployment_start.png)
* 之后pycharm就会启动服务器的终端，在终端上去运行自己的程序即可（操作指令与mobaxterm一样），这里我执行my_test.py，该文件会对一张图片进行识别，并把识别后的图片保存为test.png，从下图为运行后的结果：
  ![](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/deployment_ok.png)
  可以看到test.png已生成，部署成功
## 部分程序在服务器上运行
* Q：主包主包，你的服务器运行全程序确实很强，但是还是太死板了，有没有更加灵活又生草的程序推荐一下？
  A：有的兄弟有的，这么强的程序当然是不止一个了，一共又9个，都是当前版本t0.5的强势程序，下面我介绍其中的一个：paramiko大法（bushi）
* 程序为主机开启摄像头将一帧图像保存为png发送给服务器，服务器接收后进行图像识别并将识别的结果存为json，当主机发现json文件生成后就会从服务器上下载下来，读取其中的内容并对发送的那张图片进行画框，最后通过cv2的imshow函数显示出来，程序为死循环程序，当你把cv2.imshow()显示的图片关闭时就会进行下一轮的识别，主机直接关闭进程即可停止，服务器端输入ctrl+c即可终止
* 下载本仓库提供的server_ssh_test.py和my_detect.py文件，放入yolov7源码文件夹中，然后在yolov7源码文件夹创建一个叫temp的文件夹，最后一并上传至服务器
* 主机端安装paramiko第三方包并下载ssh_t.py文件，修改以下指令中的hostname、port、username、password参数为自己租借服务器的实际参数：
  ```
  ssh.connect(hostname='connect.nmb1.seetacloud.com', port='30201', username='root', password='BGR+vtWFsLiN')
  ```
* 在主机端ssh_t.py同目录下创建一个叫temp的文件夹，运行ssh_t.py，同时启动服务器端的server_ssh_test.py即可

# debug
## 服务器torch安装失败
* 我第一次安装环境安装得很快，但是第二次死活都报Could not find a version that satisfies the requirement torch (from versions: none)这个错，好久都没配置好环境，有同样问题的这里给出第二种配置方法
* 在服务器配置基础镜像那里，选择PyTorch / 2.0.0 / 3.8(ubuntu20.04) / 11.8这个镜像，如下图所示：
  ![](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/debug_autodl_setting.png)
* 用mobaxterm连接上服务器后，我们直接在系统自带的虚拟环境下配置环境，即base这个虚拟环境，输入以下指令激活base：
  ```
  conda activite
  ```
* 因为系统自带torch，所以需要如下图所示将requirements.txt文件中torch和torchvision这两行注释掉，再输入以下指令开始安装第三方包
  ```
  pip install -r requirements.txt
  ```
  ![](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/debug_requirements.png)
* 输入pip list，可以看到torch和torchvision都是GPU版本的，此时去[官网](https://pytorch.org/get-started/previous-versions/#wheel)找到与之对应的torchaudio版本安装，这里直接给出相应的安装指令，如果版本不一样的需要自行找到的对应版本的指令并删去torch和torchvision的部分，如下图所示：
  ```
  pip install torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
  ```
  ![](https://github.com/nmdbxqmz/YOLOv7-Local-or-Server-training-and-Server-deployment/blob/main/images/debug_torchaudio.png)
* 最后执行cuda_gpu_test.py，如果输出为True，则环境安装成功
