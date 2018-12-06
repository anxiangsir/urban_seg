from utils.color_utils import color_annotation
import tensorflow as tf
import numpy as np
import cv2
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def cut_inference_combin_color(ori_image_path,
                               input_node,
                               is_training_node,
                               predict_node,
                               predict_path,
                               color_path):
    '''
    the function to pridict picture.
    cut -> predict -> combine -> color
    :param ori_image_path: the origin image path which you want to predict
    :param input_node: the input_node which in the graph you design
    :param is_training_node: the is_training_node which make batch_norm to test
    :param predict_node: run this node to get the prediction
    :param predict_path: the predict image path you want to save
    :param color_path: the colored image path you want to save
    :return:
    '''
    '''
    :param ori_image_path: 原始图片地址
    :param input_node: input 节点
    :param is_training_node: 需要训练的开关，这个是batch_norm的开关
    :param predict_node: predict的节点
    :param predict_path: 预测图片保存的节点，这是个黑乎乎的annotation图
    :param color_path: 将annotation上色后的图保存的地址
    :return: 
    '''

    ori_image = cv2.imread(ori_image_path, cv2.CAP_MODE_RGB)
    # 开始切图 cut
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
    # predict
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

    # restore the graph from .ckpt files
    checkpoint_file = tf.train.latest_checkpoint('ckpts/deeplab_v3')
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        # 加载图
        saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_file))
        # 恢复图
        # restore graph
        saver.restore(sess, checkpoint_file)
        # 根据节点名在图中找到节点

        is_training = graph.get_tensor_by_name('is_training: 0')
        input_node = graph.get_tensor_by_name('input_x: 0')
        predicts = graph.get_tensor_by_name('predicts: 0')

        # 切图->预测->拼接
        param = {
            # 只有第一张图的测试结果还行
            'ori_image_path': 'dataset/test/1_8bits.png', # the image you want to predict
            'input_node': input_node,
            'is_training_node': is_training,
            'predict_node': predicts,
            'predict_path': './annotation.png', # 预测annotation图的保存地址
            'color_path': './color_1.png'       # 上色之后图的保存地址
        }
        cut_inference_combin_color(**param)






