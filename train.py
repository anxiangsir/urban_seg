from data_provider.batch import Init_DataSet
from model import Model
import tensorflow as tf
import logging
import os,args

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
#
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#
# 设置显存大小
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.88
#
# 得到数据生成器
train_DataSet, test_DataSet = Init_DataSet().get_DataSet()
#
args = args.args
# 初始化模型，生成Graph
model = Model(args)
#
saver = tf.train.Saver()
#
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#
with tf.control_dependencies(extra_update_ops):
    learning_rate = tf.train.exponential_decay(
        learning_rate=args.starting_learning_rate,
        global_step=model.global_step,
        decay_steps=5000,
        decay_rate=0.9,
        staircase=True)

    # 优化器
    train_op = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8
    ).minimize(
        loss=model.loss,
        global_step=model.global_step)


#
with tf.Session(config= config) as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # 用来计算总数
    total_acc_tr,total_acc_test,total_loss_tr,total_loss_test = 0,0,0,0
    best_loss =100


    for step in range(1,100000):
        # 生成batch
        x_tr, y_tr = train_DataSet.next_batch(args.batch_size)
        x_test, y_test = test_DataSet.next_batch(args.batch_size)

        loss_tr, acc_tr, lr, _ = sess.run([model.loss, model.acc_score, learning_rate, train_op],feed_dict={model.x:x_tr,model.y:y_tr})
        loss_test, acc_test = sess.run([model.loss, model.acc_score],feed_dict={model.x:x_test,model.y:y_test})

        # 计数
        total_loss_tr += loss_tr
        total_loss_test += loss_test
        total_acc_tr += acc_tr[1]
        total_acc_test += acc_test[1]





        # step for display
        display = 50

        if (step % display == 0):
            logging.info(
                "Iter {:},学习率= {:g}, 训练损失= {:.4f}, 训练精度= {:.4f}, 测试损失= {:.4f}, 测试精度= {:.4f} ".
                    format(step,lr, total_loss_tr/display,total_acc_tr/display,total_loss_test/display,total_acc_test/display))

            if total_loss_test/display < best_loss:
                best_loss = total_loss_test/display
                saver.save(sess, save_path='model/model.ckpt')
                # logging.info("保存成功！")



            total_acc_tr, total_acc_test, total_loss_tr, total_loss_test = 0, 0, 0, 0
            # 归零



