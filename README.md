 # deeplab_v3


### 数据集：
CCF卫星影像的AI分类与识别提供的数据集
初赛复赛训练集，一共五张卫星遥感影像



百度云盘：[点击这里](https://pan.baidu.com/s/1LWBMklOr39yI7fYRQ185Og)  

密码：3ih2


### 主要采用方法：
1. 采样方法为随机采样，划窗采样
2. 使用模型为deeplabv_3，使用预训练好的resnet_v2_50 fine-tuning
3. 不同采样训练多个模型融合




### 数据处理
使用随机采样训练不同的模型  
运行 preprocess.py




### 测试图片：
<center class="half">
    <img src="/sample_image/test1.jpg " width="60%" height="60%">
</center>

### 预测图片：
<center>
    <img src="/sample_image/predict_color.png  " width="60%" height="60%">
</center>

### 损失曲线：
<center>
    <img src="/sample_image/loss&acc.png  " width="90%" height="90%">
</center>
