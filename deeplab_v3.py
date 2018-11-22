"""ResNet model.
Related papers:
https://arxiv.org/pdf/1603.05027v2.pdf
https://arxiv.org/pdf/1512.03385v1.pdf
https://arxiv.org/pdf/1605.07146v1.pdf
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.training import moving_averages

import tensorflow as tf


# 为了finetune resnet_v2_50 对数据每个通道中心化
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94


class Deeplab_v3():
    def __init__(self,
                 batch_norm_decay=0.99,
                 batch_norm_epsilon=1e-3,):

        self._batch_norm_decay = batch_norm_decay
        self._batch_norm_epsilon = batch_norm_epsilon
        self._is_training = tf.placeholder(tf.bool, name='is_training')
        self.num_class = 5
        self.filters = [64, 256, 512, 1024, 2048]
        self.strides = [2, 2, 1, 1]
        self.n = [3, 4, 6, 3]

    def forward_pass(self, x):
        """Build the core model within the graph"""
        with tf.variable_scope('resnet_v2_50'):
            size = tf.shape(x)[1:3]

            x = x - [_R_MEAN, _G_MEAN, _B_MEAN]

            x = self._conv(x, 7, 64, 2, 'conv1', False, False)
            x = self._max_pool(x, 3, 2, 'max')

            res_func = self._bottleneck_residual_v2

            for i in range(4):
                with tf.variable_scope('block%d' % (i + 1)):
                    for j in range(self.n[i]):
                        with tf.variable_scope('unit_%d' % (j + 1)):
                            if j == 0:
                                x = res_func(x, self.filters[i], self.filters[i+1], 1)
                            elif j == self.n[i] - 1:
                                x = res_func(x, self.filters[i+1], self.filters[i+1], self.strides[i])
                            else:
                                x = res_func(x, self.filters[i+1], self.filters[i+1], 1)
                tf.logging.info('the shape of features after block%d is %s' % (i+1, x.get_shape()))

        # DeepLab_v3的部分
        with tf.variable_scope('DeepLab_v3'):
            x = self._atrous_spatial_pyramid_pooling(x)
            x = self._conv(x, 1, 5, 1, 'logits', False, False)
            x = tf.image.resize_bilinear(x, size)
            return x

    def _atrous_spatial_pyramid_pooling(self, x):
        """空洞空间金字塔池化
        """
        with tf.variable_scope('ASSP_layers'):

            feature_map_size = tf.shape(x)

            image_level_features = tf.reduce_mean(x, [1, 2], keep_dims=True)
            image_level_features = self._conv(image_level_features, 1, 256, 1, 'global_avg_pool', True)
            image_level_features = tf.image.resize_bilinear(image_level_features, (feature_map_size[1],
                                                                                   feature_map_size[2]))

            at_pool1x1   = self._conv(x, kernel_size=1, filters=256, strides=1, scope='assp1', batch_norm=True)
            at_pool3x3_1 = self._conv(x, kernel_size=3, filters=256, strides=1, scope='assp2', batch_norm=True, rate=6)
            at_pool3x3_2 = self._conv(x, kernel_size=3, filters=256, strides=1, scope='assp3', batch_norm=True, rate=12)
            at_pool3x3_3 = self._conv(x, kernel_size=3, filters=256, strides=1, scope='assp4', batch_norm=True, rate=18)

            net = tf.concat((image_level_features, at_pool1x1, at_pool3x3_1, at_pool3x3_2, at_pool3x3_3), axis=3)

            net = self._conv(net, kernel_size=1, filters=256, strides=1, scope='concat', batch_norm=True)

            return net

    def _bottleneck_residual_v2(self,
                                x,
                                in_filter,
                                out_filter,
                                stride,):

        """Bottleneck residual unit with 3 sub layers, plan B shortcut."""

        with tf.variable_scope('bottleneck_v2'):
            origin_x = x
            with tf.variable_scope('preact'):
                preact = self._batch_norm(x)
                preact = self._relu(preact)

            residual = self._conv(preact, 1, out_filter // 4, stride, 'conv1', True, True)
            residual = self._conv(residual, 3, out_filter // 4, 1, 'conv2', True, True)
            residual = self._conv(residual, 1, out_filter, 1, 'conv3', False, False)

            if in_filter != out_filter:
                short_cut = self._conv(preact, 1, out_filter, stride, 'shortcut', False, False)
            else:
                short_cut = self._subsample(origin_x, stride, 'shortcut')
            x = tf.add(residual, short_cut)
            return x

    def _conv(self,
              x,
              kernel_size,
              filters,
              strides,
              scope,
              batch_norm=False,
              activation=False,
              rate=None
              ):
        """Convolution."""
        with tf.variable_scope(scope):
            x_shape = x.get_shape().as_list()
            w = tf.get_variable(name='weights',
                                shape=[kernel_size, kernel_size, x_shape[3], filters])
            if rate == None:
                x = tf.nn.conv2d(input=x,
                                 filter=w,
                                 padding='SAME',
                                 strides=[1, strides, strides, 1],
                                 name='conv', )
            else:
                x = tf.nn.atrous_conv2d(value=x,
                                        filters=w,
                                        padding='SAME',
                                        name='conv',
                                        rate=rate)

            if batch_norm:
                with tf.variable_scope('BatchNorm'):
                    x = self._batch_norm(x)
            else:
                b = tf.get_variable(name='biases', shape=[filters])
                x = x + b
            if activation:
                x = tf.nn.relu(x)
            return x

    def _batch_norm(self, x):
        x_shape = x.get_shape()
        params_shape = x_shape[-1:]

        axis = list(range(len(x_shape) - 1))
        beta = tf.get_variable(name='beta',
                               shape=params_shape,
                               initializer=tf.zeros_initializer)

        gamma = tf.get_variable(name='gamma',
                                shape=params_shape,
                                initializer=tf.ones_initializer)

        moving_mean = tf.get_variable(name='moving_mean',
                                      shape=params_shape,
                                      initializer=tf.zeros_initializer,
                                      trainable=False)

        moving_variance = tf.get_variable(name='moving_variance',
                                          shape=params_shape,
                                          initializer=tf.ones_initializer,
                                          trainable=False)

        tf.add_to_collection('BN_MEAN_VARIANCE', moving_mean)
        tf.add_to_collection('BN_MEAN_VARIANCE', moving_variance)

        # These ops will only be preformed when training.
        mean, variance = tf.nn.moments(x, axis)
        update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                                   mean,
                                                                   self._batch_norm_decay,
                                                                   name='MovingAvgMean')
        update_moving_variance = moving_averages.assign_moving_average(moving_variance,
                                                                       variance,
                                                                       self._batch_norm_decay,
                                                                       name='MovingAvgVariance')

        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_mean)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_variance)

        mean, variance = tf.cond(
            pred=self._is_training,
            true_fn=lambda: (mean, variance),
            false_fn=lambda: (moving_mean, moving_variance)
        )
        x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, self._batch_norm_epsilon)
        return x

    def _relu(self, x):
        return tf.nn.relu(x)

    def _max_pool(self, x, pool_size, stride, scope):
        with tf.name_scope('max_pool') as name_scope:
            x = tf.layers.max_pooling2d(
                x, pool_size, stride, 'SAME', name=scope
            )
        return x

    def _avg_pool(self, x, pool_size, stride):
        with tf.name_scope('avg_pool') as name_scope:
            x = tf.layers.average_pooling2d(
                x, pool_size, stride, 'SAME')
        tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
        return x

    def _global_avg_pool(self, x):
        with tf.name_scope('global_avg_pool') as name_scope:
            assert x.get_shape().ndims == 4

            x = tf.reduce_mean(x, [1, 2])
        tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
        return x

    def _concat(self, x, y):
        with tf.name_scope('concat') as name_scope:
            assert x.get_shape().ndims == 4
            assert y.get_shape().ndims == 4

            x = tf.concat([x, y], 3)
        tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
        return x

    def _subsample(self, inputs, stride, scope=None):
        """Subsamples the input along the spatial dimensions."""
        if stride == 1:
            return inputs
        else:
            return self._max_pool(inputs, 3, stride, scope)