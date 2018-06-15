# deeplab_v3


### 数据集：
CCF卫星影像的AI分类与识别提供的数据集初赛复赛训练集，一共五张卫星遥感影像



* 百度云盘：[点击这里](https://pan.baidu.com/s/1LWBMklOr39yI7fYRQ185Og)  
* 密码：3ih2
* 预训练模型：[点击这里下载](http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz)


### 主要采用方法：
* 采样方法为随机采样，划窗采样，数据增强
* 使用模型为deeplabv_3，使用预训练好的resnet_v2_50 fine-tuning
* 这里放的预测图是单模型跑出来的结果，多模型融合肯定会使得结果变得更好
* 没有针对道路类别进行过采样，导致道路类别分的不好





### 数据处理
* 旋转90°，180°，270°
* 左右翻转，上下翻转

### 如何训练

1. 将百度云中的数据集文件夹dataset下载并存放到项目主目录下
2. 运行 proprecess.py 生成训练集
3. 运行 train.py 开始训练


### 测试图片：
<div align=center><img width="600" height="600" src="/sample_image/test1.jpg"/></div>  

### 预测图片：
<div align=center><img width="600" height="600" src="/sample_image/predict_color.png"/></div>  

### 损失曲线：
<center>
    <img src="/sample_image/loss&acc.png  " width="90%" height="90%">
</center>
有问题可加QQ：1363090802一起交流
