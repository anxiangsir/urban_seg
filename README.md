# 基于deeplabv3对遥感图像的语义分割

### 数据集：
CCF卫星影像的AI分类与识别提供的数据集初赛复赛训练集，一共五张卫星遥感影像
* 百度云盘：[点击这里](https://pan.baidu.com/s/1LWBMklOr39yI7fYRQ185Og)  
* 密码：3ih2
* 预训练模型：[点击这里下载](http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz)  

```
dataset
├── origin //5张遥感图片，有标签，用来做训练
├── test   //3张遥感图片，无标签，用来做测试
└── train  //为空，通过`python preprocess.py`随机采样生成
    ├── images       
    └── labels
```                    

### 采用方法：
* 使用模型为Deeplab_v3，使用预训练好的resnet_v2_50 fine-tuning
* 将原始的遥感图像裁成大小为(256x256)的图片块，裁剪的方法为随机采样，并进行数据增强
### 依赖：
* GPU Nvidia Tesla V100 (16G)
* tensorflow 
* opencv-python 
* python3

单卡跑一天就可以收敛~
### How To Train?
1. 将百度云中的数据集文件夹dataset下载并存放到项目主目录下
2. python proprecess.py 生成训练集 时间稍长，需要等待
3. python main.py 开始训练


### TODO:  
Pull requests are welcome.  
- [x] Resnet_v2_50 as Network Backbone using tensorflow API.
- [x] Ready for slim pretrained parameters.
- [ ] Xception as Network Backbone
- [ ] Multi-GPU support
- [ ] TFRecords
- [ ] 后处理优化，比如消除拼接痕迹，CRF后处理
### 测试图片：
<table border=0>
<tr>
    <td>
        <img src="/sample_image/test1.png" border=0 margin=1 width=512>
    </td>
    <td>
        <img src="/sample_image/color_1.png" border=0 margin=1 width=512>
    </td>
</tr>
</table>

### 训练曲线：
<div align=center><img src="/sample_image/acc.png"/></div>  
<div align=center><img src="/sample_image/loss.png"/></div>  


