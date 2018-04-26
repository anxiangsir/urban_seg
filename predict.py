from net.deeplab_v3 import deeplab_v3
from net import model
from utils.color import color_annotation
import tensorflow as tf
import numpy as np
import cv2,os,shutil,sys

class test_config:
    batch_norm_epsilon = 1e-5
    size = 256
    batch_norm_decay = 0.9997
    number_of_classes = 5
    l2_regularizer = 0.01
    starting_learning_rate = 0.0001
    multi_grid = [1,2,4]
    output_stride = 16
    resnet_model = "resnet_v2_50"
    batch_size = 1
    is_training = False


class Predict:
    def __init__(self):
        self.model = model.Model(test_config)
        # 将影像采样为小图片，并进行保存，这个集合记录小图片的地址
        self.sample_image_path = []
        self.size = test_config.size


    def fit(self, image_path, predict_path, color_path, sess):

        # 获得原始的图像
        self.ori_image = cv2.imread(image_path)
        # 开始切图
        self.split_to_net()

        # 开始遍历所有的小图片，这里是有序的
        print('开始预测图片...')
        for path in self.sample_image_path:
            print(path)
            # 使得x shape 为 [1, size, size, channels]
            x = np.expand_dims(cv2.imread(path),axis=0)
            y = np.ones([test_config.batch_size, self.size, self.size])
            feed_dict = {self.model.x:x,self.model.y:y}
            # 这里的predict为 [1, size, size]
            predict = sess.run(self.model.predicts,feed_dict=feed_dict)
            # 保存覆盖小图片
            cv2.imwrite(path,np.squeeze(predict))

        sys.stdout.flush()
        self.combin_image(predict_path,color_path)

    def split_to_net(self):
        '''
        将你要预测的图片切成小图，保存到文件夹内
        以便于输入给网络
        :return:
        '''
        print('开始划窗采样...')
        # 删除原有的文件夹并新建
        path = 'sample_image_for_predict/'
        if os.path.exists(path):
            shutil.rmtree(path)
        else:
            os.mkdir(path)

        # 开始切图
        self.h_step = self.ori_image.shape[0] // test_config.size
        self.w_step = self.ori_image.shape[1] // test_config.size

        count = 1

        # 循环切图
        for h in range(self.h_step):
            for w in range(self.w_step):
                # 划窗采样
                image_sample = self.ori_image[(h * self.size):(h * self.size + self.size),
                                              (w * self.size):(w * self.size + self.size), :]
                image_path = path + str(count) + '.png'
                cv2.imwrite(image_path, image_sample)
                self.sample_image_path.append(image_path)
                count += 1


    def combin_image(self,predict_path,color_path):
        tmp = np.ones([self.h_step*self.size,self.w_step*self.size])
        for h in range(self.h_step):
            for w in range(self.w_step):
                tmp[h * self.size:(h + 1) * self.size,
                    w * self.size:(w + 1) * self.size] = cv2.imread(self.sample_image_path[h*self.w_step+w],cv2.CAP_MODE_GRAY)
        cv2.imwrite(predict_path,tmp)
        color_annotation(predict_path,color_path)



if __name__ == '__main__':

    predict = Predict()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        restorer = tf.train.Saver()
        restorer.restore(sess,'models/3/model.ckpt')
        args = {
            'image_path':'dataset/test/1_8bits.png',
            'sess':sess,
            'predict_path':'predict.png',
            'color_path':'predict_color.png'
        }
        predict.fit(**args)
