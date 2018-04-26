from utils.batch import Init_DataSet
from tensorflow.contrib import slim
from net.model import Model
import tensorflow as tf
import logging, shutil, os
import global_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
# 配置文件
config = global_config.Config
# 设置显存大小
gpu_config = tf.ConfigProto()
gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.88
# 得到训练集与验证集
dateset = Init_DataSet()
train_dataset = dateset.train_DataSet
val_dataset = dateset.val_DataSet


# 初始化模型，生成Graph
model = Model(config)
#
saver = tf.train.Saver()
# 得到需要载入预训练参数的集合
variables_to_restore = slim.get_variables_to_restore(
    exclude=[config.resnet_model + "/logits","DeepLab_v3/ASPP_layer", "DeepLab_v3/logits"])
restorer = tf.train.Saver(variables_to_restore)
#
optimizer = tf.train.AdamOptimizer(learning_rate=config.starting_learning_rate)
# 优化器
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    # https://blog.csdn.net/sinat_30372583/article/details/79943743
    train_op = optimizer.minimize(loss=model.loss)


slim.batch_norm()

with tf.Session(config= gpu_config) as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # 载入残差网络预训练好的权重
    restorer.restore(sess,'net/resnet_model/resnet_v2_50.ckpt')

    # 用来计算总数
    total_acc_tr,total_acc_val,total_loss_tr,total_loss_val = 0,0,0,0
    best_loss = 1.8


    for step in range(1,100000):
        # 生成batch
        x_tr, y_tr = train_dataset.next_batch(config.batch_size)
        x_val, y_val = val_dataset.next_batch(config.batch_size)
        #
        loss_tr, predict_tr, _ = sess.run([model.loss, model.predicts, train_op],feed_dict={model.x:x_tr,model.y:y_tr})
        loss_val, predict_val = sess.run([model.loss, model.predicts],feed_dict={model.x:x_val,model.y:y_val})
        # 计数
        total_loss_tr += loss_tr
        total_loss_val += loss_val
        #
        total_acc_tr += model.accuracy_score(y_true=y_tr,y_pred=predict_tr)
        total_acc_val += model.accuracy_score(y_true=y_val,y_pred=predict_val)




        # step for display
        display = 50

        if (step % display == 0):

            logging.info(
                "Iter {:}, loss_tr= {:.4f}, acc_tr= {:.4f}, loss_val= {:.4f}, acc_val= {:.4f} ".
                    format(step, total_loss_tr/display,total_acc_tr/display,total_loss_val/display,total_acc_val/display))

            # 保存效果最好的模型
            if  total_loss_val/display < best_loss:
                shutil.rmtree('single_model/')
                os.mkdir('single_model')
                best_loss = total_loss_val/display
                saver.save(sess, save_path='single_model/model.ckpt')
                logging.info("保存成功！")


            total_acc_tr, total_acc_val, total_loss_tr, total_loss_val = 0, 0, 0, 0
            # 归零



