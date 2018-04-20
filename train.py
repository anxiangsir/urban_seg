from data_provider.batch import Init_DataSet
from model import Model
import tensorflow as tf
import numpy as np
import logging
import os,args

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
#
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#
# 设置显存大小
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.88
#
# 得到数据生成器
dateset = Init_DataSet()
train_dataset = dateset.train_DataSet
test_dataset = dateset.val_DataSet

#
args = args.args
# 初始化模型，生成Graph
model = Model(args)
#
saver = tf.train.Saver()
#
#
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
# 获得需要训练的权重
vars_to_train = model.get_variables_to_train(exclude=['resnet_v2_50/conv1','resnet_v2_50/block1','resnet_v2_50/block2','resnet_v2_50/block3'])



with tf.control_dependencies(extra_update_ops):

    with tf.variable_scope("optimizer_vars"):
        # 优化器
        train_op = tf.train.AdamOptimizer(
            learning_rate=0.0001,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8
        ).minimize(
            loss=model.loss,
            global_step=model.global_step,
            var_list=vars_to_train)


#
with tf.Session(config= config) as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # 载入残差网络预训练好的权重
    model.restore(sess,'net/resnet_model/resnet_v2_50.ckpt')

    # 用来计算总数
    total_acc_tr,total_acc_test,total_loss_tr,total_loss_test = 0,0,0,0
    best_acc = 0.5


    for step in range(1,100000):
        # 生成batch
        x_tr, y_tr = train_dataset.next_batch(args.batch_size)
        x_test, y_test = test_dataset.next_batch(args.batch_size)

        loss_tr, predict_tr, _ = sess.run([model.loss, model.predicts, train_op],feed_dict={model.x:x_test,model.y:y_test})
        loss_test, predict_test = sess.run([model.loss, model.predicts],feed_dict={model.x:x_tr,model.y:y_tr})

        # 计数
        total_loss_tr += loss_tr
        total_loss_test += loss_test


        total_acc_tr += model.accuracy_score(y_true=y_tr,y_pred=predict_tr)
        total_acc_test += model.accuracy_score(y_true=y_test,y_pred=predict_test)

        # w = tf.get_default_graph().get_tensor_by_name('resnet_v2_50/block4/unit_1/bottleneck_v2/conv1/weights:0')
        #  查看权重是否更新
        # print(np.sum(sess.run(w)))

        # step for display
        display = 50

        if (step % display == 0):
            logging.info(
                "Iter {:}, 训练损失= {:.4f}, 训练精度= {:.4f}, 测试损失= {:.4f}, 测试精度= {:.4f} ".
                    format(step, total_loss_tr/display,total_acc_tr/display,total_loss_test/display,total_acc_test/display))
            # 保存效果最好的模型
            if total_acc_test/display > best_acc:
                best_acc = total_acc_test/display
                saver.save(sess, save_path='model/model.ckpt')
                logging.info("保存成功！")


            total_acc_tr, total_acc_test, total_loss_tr, total_loss_test = 0, 0, 0, 0
            # 归零



