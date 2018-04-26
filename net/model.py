from net.deeplab_v3 import deeplab_v3
from global_config import Config
import numpy as np
import tensorflow as tf



class Model:
    def __init__(self, config):

        self.size = config.size
        self.batch_size = config.batch_size
        self.x = tf.placeholder(tf.float32, [None, self.size, self.size, 3])
        self.y = tf.placeholder(tf.int64, [None, self.size, self.size])
        # 预测结果 [batch_size, size, size, n_class]
        self.logits = deeplab_v3(inputs=self.x,args=config,reuse=False,is_training=config.is_training)
        # 交叉熵损失
        self.loss = self.get_loss()
        # 预测结果 [batch_size, size, size]
        self.predicts = tf.argmax(self.logits,axis=3)
        self.global_step = tf.train.get_or_create_global_step()
        self.pixel_acc = self.get_pixel_accuracy()
        self.summary_op = tf.summary.merge_all()

    def get_loss(self):
        # 交叉熵损失函数
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y,logits=self.logits))
        tf.summary.scalar('loss',loss)
        return loss


    def get_pixel_accuracy(self):
        y_true = tf.reshape(self.y,[self.batch_size*self.size*self.size])
        y_pred = tf.reshape(self.predicts,[self.batch_size*self.size*self.size])
        acc = tf.reduce_mean(tf.cast(tf.equal(y_true,y_pred),'float'))
        tf.summary.scalar('pixel_accuracy',acc)
        return acc



    @staticmethod
    def get_variables_to_train(include=None, exclude=None):
        if include is None:
            # 包括所有的变量
            vars_to_include = tf.trainable_variables()
        else:
            if not isinstance(include, (list, tuple)):
                raise TypeError('include 必须是一个list或者tuple')
            vars_to_include = []
            for scope in include:
                vars_to_include += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

        vars_to_exclude = set()
        if exclude is not None:
            if not isinstance(exclude, (list, tuple)):
                raise TypeError('exclude 必须是一个list或者tuple')
            for scope in exclude:
                vars_to_exclude |= set(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope))

        return [v for v in vars_to_include if v not in vars_to_exclude]