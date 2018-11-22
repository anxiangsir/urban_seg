from utils.color_utils import color_annotation
import tensorflow as tf
import numpy as np
import cv2
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"


def cut_inference_combin_color(ori_image_path,
                               input_node,
                               is_training_node,
                               predict_node,
                               predict_path,
                               color_path):


    ori_image = cv2.imread(ori_image_path, cv2.CAP_MODE_RGB)
    # 开始切图
    h_step = ori_image.shape[0] // 256
    w_step = ori_image.shape[1] // 256


    image_list = []
    predict_list = []
    # 循环切图
    for h in range(h_step):
        for w in range(w_step):
            # 划窗采样
            image_sample = ori_image[(h * 256):(h * 256 + 256),
                                          (w * 256):(w * 256 + 256), :]
            image_list.append(image_sample)

    # 对每个图像块预测
    for image in image_list:

        feed_dict = {input_node: np.expand_dims(image, 0),
                     is_training_node: False}
        # 这里的predict为 [1, size, size]
        predict = sess.run(predict_node, feed_dict=feed_dict)
        # 保存覆盖小图片
        predict_list.append(np.squeeze(predict))

    # 将预测后的图像块再拼接起来
    tmp = np.ones([h_step * 256, w_step * 256])
    for h in range(h_step):
        for w in range(w_step):
            tmp[
            h * 256:(h + 1) * 256,
            w * 256:(w + 1) * 256
            ] = predict_list[h * w_step + w]
    cv2.imwrite(predict_path, tmp)
    color_annotation(predict_path, color_path)


if __name__ == '__main__':

    checkpoint_file = tf.train.latest_checkpoint('ckpts/batch_norm_decay=0.99')
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        # 加载图
        saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_file))
        # 恢复图
        saver.restore(sess, checkpoint_file)
        # 根据节点名在图中找到节点
        is_training = graph.get_tensor_by_name('Placeholder: 0')
        input_node = graph.get_tensor_by_name('Placeholder_1: 0')
        predicts = graph.get_tensor_by_name('ArgMax: 0')

        # 切图->预测->拼接
        param = {
            'ori_image_path': 'dataset/test/2_8bits.png',
            'input_node': input_node,
            'is_training_node': is_training,
            'predict_node': predicts,
            'predict_path': './annotation.png',
            'color_path': './color.png'
        }
        cut_inference_combin_color(**param)






