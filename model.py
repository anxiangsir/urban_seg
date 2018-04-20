from net.deeplab_v3 import deeplab_v3
#
import numpy as np
import tensorflow as tf
import sklearn
class Model:
    def __init__(self, args):
        self.x = tf.placeholder(tf.float32, [None, args.size, args.size, 3])
        self.y = tf.placeholder(tf.int32, [None, args.size, args.size])

        # 预测结果 [batch_size, size, size, n_class]
        self.logits = deeplab_v3(inputs=self.x,args=args,reuse=False,is_training=True)
        # 交叉熵损失
        self.loss = self.get_loss()
        # 预测结果 [batch_size, size, size]
        self.predicts = tf.argmax(self.logits,axis=3)

        self.global_step = tf.train.get_or_create_global_step()

    def get_loss(self):
        # 交叉熵损失函数
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y,logits=self.logits))


    @staticmethod
    def accuracy_score(y_true,y_pred):
        return sklearn.metrics.accuracy_score(y_true=y_true,y_pred=y_pred)

