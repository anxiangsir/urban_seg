 # deeplab_v3


### 数据集：
CCF卫星影像的AI分类与识别提供的数据集

1. 训练集为：初赛复赛训练集，一共五张卫星遥感影像
2. 测试集为： 初赛测试集，一共三张


### 主要采用方法：
1. 采样方法为随机采样，划窗采样
2. 使用模型为deeplabv_3，使用预训练好的resnet_v2_50 fine-tuning
3. 模型融合


### 测试图片：
<center class="half">
    <img src="/sample_image/test1.jpg " width="70%" height="60%">
</center>

### 预测图片：
<center>
    <img src="/sample_image/predict_color.png  " width="70%" height="60%">
</center>
