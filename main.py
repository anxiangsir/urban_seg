from deeplab_v3 import Deeplab_v3
from utils.data_utils import DataSet
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import tensorflow as tf
import pandas as pd
import numpy as np
import os


class args:
    batch_size = 32
    lr = 0.002
    display = 500
    weight_decay = 0.00001
    model_name = 'deeplab_v3'
    batch_norm_decay = 0.95
    momentum = 0.9



os.environ["CUDA_VISIBLE_DEVICES"] = "7"
data_path_df = pd.read_csv('dataset/path_list.csv')

tf.logging.set_verbosity(tf.logging.INFO)
train_path, val_path = train_test_split(data_path_df, test_size=0.25, shuffle=True)
dataset_tr = DataSet(image_path=train_path['image'].values, label_path=train_path['label'].values)
dataset_val = DataSet(image_path=val_path['image'].values, label_path=val_path['label'].values)

model = Deeplab_v3(batch_norm_decay=args.batch_norm_decay)

image = tf.placeholder(tf.float32, [None, 256, 256, 3], name='input_x')
label = tf.placeholder(tf.int32, [None, 256, 256])
lr = tf.placeholder(tf.float32,)


logits = model.forward_pass(image)
predicts = tf.argmax(logits, axis=-1, name='predicts')

variables_to_restore = tf.trainable_variables(scope='resnet_v2_50')

# 打印所有需要finetune的参数
# 你可以打印出来看一看
# tf.logging.info('variables_to_restore:')
# for var in variables_to_restore:
#     tf.logging.info(var.name)

# finetune resnet_v2_50的参数(block1到block4)
restorer = tf.train.Saver(variables_to_restore)
# cross_entropy
cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label))
# l2_norm l2正则化
l2_loss = args.weight_decay * tf.add_n(
     [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])

optimizer = tf.train.AdamOptimizer(learning_rate=lr)


loss = cross_entropy + l2_loss

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

# 打印batchnorm所有需要更新的均值和方差的op
# 你可以打印出来看一看
# tf.logging.info('update_ops:')
# for ops in update_ops:
#     tf.logging.info(ops.name)

with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss=loss)

total_acc_tr, total_acc_val, total_loss_tr, total_loss_val, total_l2_loss = 0, 0, 0, 0, 0
# 我们要保存所有的参数
saver = tf.train.Saver(tf.all_variables())

# print(model._is_training.name)
# print(image.name)
# print(predicts.name)
best_val_acc = 0.8

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    # finetune resnet_v2_50参数
    restorer.restore(sess, 'ckpts/resnet_v2_50/resnet_v2_50.ckpt')

    log_path = 'logs/%s/' % args.model_name
    model_path = 'ckpts/%s/' % args.model_name
    if not os.path.exists(model_path): os.mkdir(model_path)
    if not os.path.exists('./logs'): os.mkdir('./logs')
    if not os.path.exists(log_path): os.mkdir(log_path)
    train_summary_writer = tf.summary.FileWriter('%s/train' % log_path, sess.graph)
    val_summary_writer = tf.summary.FileWriter('%s/val' % log_path, sess.graph)

    learning_rate = args.lr
    for step in range(1, 100000):
        if step == 40000 or step == 80000:
            learning_rate = learning_rate / 10
        x_tr, y_tr = dataset_tr.next_batch(args.batch_size)
        x_val, y_val = dataset_val.next_batch(args.batch_size)

        loss_tr, l2_loss_tr, predicts_tr, _ = sess.run(
            fetches=[cross_entropy, l2_loss, predicts, train_op],
            feed_dict={
                image: x_tr,
                label: y_tr,
                model._is_training: True,
                lr: learning_rate})
        loss_val, predicts_val = sess.run(
            fetches=[cross_entropy, predicts],
            feed_dict={
                image: x_val,
                label: y_val,
                model._is_training: False})
        # 计数
        total_loss_tr += loss_tr
        total_loss_val += loss_val
        total_l2_loss += l2_loss_tr

        acc_tr = accuracy_score(np.reshape(y_tr, [-1]), np.reshape(predicts_tr, [-1]))
        acc_val = accuracy_score(np.reshape(y_val, [-1]), np.reshape(predicts_val, [-1]))
        total_acc_tr += acc_tr
        total_acc_val += acc_val

        # 每隔多少步打印日志
        if step % args.display == 0:
            tf.logging.info("Iter:%d, lr:%.6f, loss_tr:%.4f, acc_tr:%.4f, loss_val:%.4f, acc_val:%.4f" %
                            (step,
                             learning_rate,
                             total_loss_tr / args.display,
                             total_acc_tr / args.display,
                             total_loss_val / args.display,
                             total_acc_val / args.display))

            train_summary = tf.Summary(
                value=[tf.Summary.Value(tag='loss', simple_value=total_loss_tr / args.display),
                       tf.Summary.Value(tag='accuracy', simple_value=total_acc_tr / args.display),
                       tf.Summary.Value(tag='l2_loss', simple_value=total_l2_loss / args.display)])

            val_summary = tf.Summary(
                value=[tf.Summary.Value(tag='loss', simple_value=total_loss_val / args.display),
                       tf.Summary.Value(tag='accuracy', simple_value=total_acc_val / args.display)])
            # 记录summary
            train_summary_writer.add_summary(train_summary, step)
            train_summary_writer.flush()
            val_summary_writer.add_summary(val_summary, step)
            val_summary_writer.flush()


            # 保存模型
            if (total_acc_val / args.display) > best_val_acc:
                # 保存acc最高的模型
                saver.save(sess, model_path + 'model.ckpt')
                best_val_acc = total_acc_val / args.display
                tf.logging.info('Save model successfully to "%s"!' % (model_path + 'model.ckpt'))
            # 累计归零
            total_acc_tr, total_acc_val, total_loss_tr, total_loss_val, total_l2_loss = 0, 0, 0, 0, 0

