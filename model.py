from net.deeplab_v3 import deeplab_v3
from tensorflow.contrib import slim
#
import numpy as np
import tensorflow as tf
import sklearn
class Model:
    def __init__(self, args):
        self.args = args
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



    def accuracy_score(self,y_true,y_pred):
        y_true = np.reshape(y_true,(self.args.batch_size*self.args.size*self.args.size))
        y_pred = np.reshape(y_pred,(self.args.batch_size*self.args.size*self.args.size))
        right = np.sum(y_true == y_pred)
        all = self.args.batch_size*self.args.size*self.args.size
        return right/all


    def restore(self, sess, path):
        variables_to_restore = slim.get_variables_to_restore(exclude=[self.args.resnet_model + "/logits", "optimizer_vars",
                                                                      "DeepLab_v3/ASPP_layer", "DeepLab_v3/logits"])
        restorer = tf.train.Saver(variables_to_restore)
        restorer.restore(sess, path)


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