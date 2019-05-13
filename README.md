# 基于Deeplab-v3对遥感图像的语义分割
QQ:2812728382

### 数据集：
CCF卫星影像的AI分类与识别提供的数据集初赛复赛训练集，一共五张卫星遥感影像
* 百度云盘：[点击这里](https://pan.baidu.com/s/1LWBMklOr39yI7fYRQ185Og)  
* 密码：3ih2
* 预训练模型：[点击这里下载](http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz)  

```
dataset
├── origin //5张遥感图片，有标签
├── test   //3张遥感图片，无标签，在这个任务中没有用到
└── train  //为空，通过`python preprocess.py`随机采样生成
    ├── images       
    └── labels
```     
其中我们使用前四张用来做训练，最后一张用来做测试

### 主要策略：
- [x] 将原始的遥感图像裁成大小为(256x256)的图片块，裁剪的方法为随机采样，并进行数据扩增
- [x] 搭建Deeplab-v3模型，使用预训练的 resnet-v2-50 迁移学习
- [x] 完整的训练测试程序，使用 tensorboard 监控模型训练
- [x] 多尺度拼接预测，提升模型
- [ ] 后处理优化，比如消除预测图片拼接痕迹
- [ ] 使用更好的骨干网络，如 Xception

### 最终结果：
评价方法为 mean-IoU，在数据集极少的情况下，测试集评价结果得到了 **77.3** 的分数

| 方法 | mean-IoU | accuracy |
| :-----| :----: |  :----: |
| baseline(deeplabv3) | 71.2 | - |
| resnet-v2-50 pretrain | 77.1 | - |
| 旋转四次预测取平均 | 77.6 | 85.5 |

    
### 如何训练
```
将百度云中的数据集文件夹dataset下载并存放到项目主目录下
python proprecess.py 时间稍长，需要等待
python train.py 时间稍长，可以更改args.test_display 多久查看一次测试结果
```

### 如何可视化训练过程
```
cd 到主目录下
tensorboard --logdir=./
```

### 测试结果： 
<table border=0>
<tr>
    <td><img src="/images/step_10000.png" border=0 margin=1 width=512></td>
    <td><img src="/images/step_50000.png" border=0 margin=1 width=512></td>
</tr>
<tr>
    <td>模型预测step = 10000</td>
    <td>模型预测step = 50000</td>
</tr>
<tr>
    <td><img src="images/5_view.png" border=0 margin=1 width=512></td>
    <td><img src="images/5_color.png" border=0 margin=1 width=512></td>
</tr>
<tr>
    <td>测试图片缩略图</td>
    <td>测试图片标签上色图</td>
</tr>
</table>
<tr>
    <td><img src="images/metric.png" border=0 margin=1></td>
</tr>
<tr>
    <td>tensorboard</td>
</tr>    
