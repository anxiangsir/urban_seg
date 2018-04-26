import cv2,os
import numpy as np
import tensorflow as tf



class Init_DataSet:
    def __init__(self,
                 image_path=list(map(lambda x:'dataset/train/sample_image/'+x,os.listdir('dataset/train/sample_image/'))),
                 label_path=list(map(lambda x:'dataset/train/labels/'+x,os.listdir('dataset/train/sample_image/'))),):

        image_path = np.array(image_path)
        label_path = np.array(label_path)

        dataset_num = image_path.shape[0]
        random_index = np.random.permutation(dataset_num)

        # 打乱
        image_path = image_path[random_index]
        label_path = label_path[random_index]
        # 划分训练集
        image_path_tr = image_path[:int(dataset_num*0.8)]
        label_path_tr = label_path[:int(dataset_num*0.8)]
        # 划分验证集
        image_path_val = image_path[int(dataset_num*0.8):]
        label_path_val = label_path[int(dataset_num*0.8):]



        # 训练集
        self.train_DataSet = DataSet(image_path_tr,label_path_tr)
        # 验证集
        self.val_DataSet = DataSet(image_path_val,label_path_val)

    def get_DataSet(self):
        return self.train_DataSet, self.val_DataSet






class DataSet():
    '''
    通过读入文件生成数据集
    '''

    def __init__(self, image_path, label_path):

        self.image_path = np.array(image_path)
        self.label_path = np.array(label_path)
        self.batch_count = 0
        self.epoch_count = 0

    def num_examples(self):

        return self.image_path.shape[0]


    def next_batch(self, batch_size):

        start = self.batch_count * batch_size
        end = start + batch_size
        self.batch_count += 1

        if end > self.image_path.shape[0]:
            self.batch_count = 0
            random_index = np.random.permutation(self.image_path.shape[0])
            self.image_path = self.image_path[random_index]
            self.label_path = self.label_path[random_index]
            self.epoch_count += 1
            start = self.batch_count * batch_size
            end = start + batch_size
            self.batch_count += 1

        image_batch, label_batch = self.read_path(self.image_path[start:end],
                                                  self.label_path[start:end])
        return image_batch, label_batch

    def read_path(self, x_path, y_path):
        '''
        将路径读为图片
        :param x_path:
        :param y_path:
        :return:
        '''
        x = []
        y = []
        for i in range(x_path.shape[0]):
            x.append(self.transform(cv2.imread(x_path[i])))
            y.append(cv2.imread(y_path[i], cv2.CAP_MODE_GRAY))

        return np.array(x),np.array(y)

    def transform(self,img):
        return img