from deeplab_v3 import Deeplab_v3
from data_utils import DataSet


import cv2
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from color_utils import color_predicts
from predicts_utils import total_image_predict

from metric_utils import iou


class args:
    batch_size = 16
    lr = 2e-4
    test_display = 2000
    weight_decay = 5e-4
    model_name = 'baseline'
    batch_norm_decay = 0.95
    test_image_path = 'dataset/origin/5.png'
    test_label_path = 'dataset/origin/5_class.png'
    multi_scale = True # 是否多尺度预测
    gpu_num = 0
    pretraining = False

# 打印以下超参数
for key in args.__dict__:
    if key.find('__') == -1:
        offset = 20 - key.__len__()
        print(key + ' ' * offset, args.__dict__[key])

# 使用那一块显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.gpu_num

data_path_df = pd.read_csv('dataset/path_list.csv')
data_path_df = data_path_df.sample(frac=1) # 第一次打乱

dataset = DataSet(image_path=data_path_df['image'].values, label_path=data_path_df['label'].values)

model = Deeplab_v3(batch_norm_decay=args.batch_norm_decay)

image = tf.placeholder(tf.float32, [None, 256, 256, 3], name='input_x')
label = tf.placeholder(tf.int32, [None, 256, 256])
lr = tf.placeholder(tf.float32, )

logits = model.forward_pass(image)
logits_prob = tf.nn.softmax(logits=logits, name='logits_prob')
predicts = tf.argmax(logits, axis=-1, name='predicts')

variables_to_restore = tf.trainable_variables(scope='resnet_v2_50')

# finetune resnet_v2_50的参数(block1到block4)
restorer = tf.train.Saver(variables_to_restore)
# cross_entropy
cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label))


# 只将weight加入到weight_decay中
# https://arxiv.org/pdf/1807.11205.pdf
# weight_for_weightdecay = []
# for var in tf.trainable_variables():
#     if var.name.__contains__('weight'):
#         weight_for_weightdecay.append(var)
#         print(var.op.name)
#     else:
#         continue
# l2_norm l2正则化
with tf.name_scope('weight_decay'):
    l2_loss = args.weight_decay * tf.add_n(
         [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])


optimizer = tf.train.AdamOptimizer(learning_rate=lr)
loss = cross_entropy + l2_loss

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
# 计算梯度
grads = optimizer.compute_gradients(loss=loss, var_list=tf.trainable_variables())
# for grad, var in grads:
#     if grad is not None:
#         tf.summary.histogram(name='%s_gradients' % var.op.name, values=grad)
#         tf.summary.histogram(name='%s' % var.op.name, values=var)
# 梯度裁剪
# gradients, variables = zip(*grads)
# gradients, global_norm = tf.clip_by_global_norm(gradients, 5)

#更新梯度
apply_gradient_op = optimizer.apply_gradients(grads_and_vars=grads, global_step=tf.train.get_or_create_global_step())
batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
train_op = tf.group(apply_gradient_op, batch_norm_updates_op)

saver = tf.train.Saver(tf.all_variables())

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# summary_op = tf.summary.merge_all()

with tf.Session(config=config) as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    graph = tf.get_default_graph()

    if args.pretraining:
        # finetune resnet_v2_50参数，需要下载权重文件
        restorer.restore(sess, 'ckpts/resnet_v2_50/resnet_v2_50.ckpt')

    log_path = 'logs/%s/' % args.model_name
    model_path = 'ckpts/%s/' % args.model_name
    
    if not os.path.exists(model_path): os.makedirs(model_path)
    if not os.path.exists('./logs'): os.makedirs('./logs')
    if not os.path.exists(log_path): os.makedirs(log_path)
        

    summary_writer = tf.summary.FileWriter('%s/' % log_path, sess.graph)

    learning_rate = args.lr
    for step in range(1, 50001):
        if step == 30000 or step == 40000:
            learning_rate = learning_rate / 10
        x_tr, y_tr = dataset.next_batch(args.batch_size)


        loss_tr, l2_loss_tr, predicts_tr, _ = sess.run(
            fetches=[cross_entropy, l2_loss, predicts, train_op],
            feed_dict={
                image: x_tr,
                label: y_tr,
                model._is_training: True,
                lr: learning_rate})

        # 前50, 100, 200 看一下是否搞错了
        if (step in [50, 100, 200]) or (step > 0 and step % args.test_display == 0):

            test_predict = total_image_predict(
                ori_image_path=args.test_image_path,
                input_placeholder=image,
                logits_prob_node=logits_prob,
                is_training_placeholder=model._is_training,
                sess=sess,
                multi_scale=args.multi_scale
            )

            test_label = cv2.imread(args.test_label_path, cv2.IMREAD_GRAYSCALE)

            # 保存图片
            cv2.imwrite(filename='%spredict_color_%d.png' % (log_path, step),
                        img=color_predicts(img=test_predict))

            result = iou(y_pre=np.reshape(test_predict, -1),
                         y_true=np.reshape(test_label, -1))

            print("======================%d======================" % step)
            for key in result.keys():
                offset = 40 - key.__len__()
                print(key + ' ' * offset + '%.4f' % result[key])


            test_summary = tf.Summary(
                value=[tf.Summary.Value(tag=key, simple_value=result[key]) for key in result.keys()]
            )


            # 记录summary
            summary_writer.add_summary(test_summary, step)
            summary_writer.flush()
