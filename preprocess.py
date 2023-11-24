import argparse
import os
import random

import cv2
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, default=224)
parser.add_argument('--image_num', type=int, default=10000)
args = parser.parse_args()


# 随机窗口采样
def generate_train_dataset(size, image_num,
                           train_image_path='dataset/train/images/',
                           train_label_path='dataset/train/labels/'):
    '''
    该函数用来生成训练集，切图方法为随机切图采样
    :param image_num: 生成样本的个数
    :param train_image_path: 切图保存样本的地址
    :param train_label_path: 切图保存标签的地址
    :return:
    '''

    # 用来记录所有的子图的数目
    g_count = 1
    images_path = ['dataset/origin/1.png','dataset/origin/2.png',
                   'dataset/origin/3.png','dataset/origin/4.png']
    labels_path = ['dataset/origin/1_class.png','dataset/origin/2_class.png',
                   'dataset/origin/3_class.png','dataset/origin/4_class.png']

    # 每张图片生成子图的个数
    image_each = image_num // len(images_path)
    image_path, label_path = [], []
    for i in tqdm(range(len(images_path))):
        count = 0
        image = cv2.imread(images_path[i])
        label = cv2.imread(labels_path[i], cv2.IMREAD_GRAYSCALE)
        X_height, X_width = image.shape[0], image.shape[1]
        while count < image_each:
            random_width = random.randint(0, X_width - size - 1)
            random_height = random.randint(0, X_height - size - 1)
            image_ogi = image[random_height: random_height + size, random_width: random_width + size,:]
            label_ogi = label[random_height: random_height + size, random_width: random_width + size]
            image_path.append(train_image_path + '%05d.png' % g_count)
            label_path.append(train_label_path + '%05d.png' % g_count)
            cv2.imwrite((train_image_path + '%05d.png' % g_count), image_ogi)
            cv2.imwrite((train_label_path + '%05d.png' % g_count), label_ogi)
            count += 1
            g_count += 1

            if g_count % 100 == 0:
                print(f'finish {g_count}/{image_num} images')

    df = pd.DataFrame({'image':image_path, 'label':label_path})
    df.to_csv('dataset/path_list.csv', index=False)


if __name__ == '__main__':
    os.makedirs('dataset/train/images', exist_ok=True)
    os.makedirs('dataset/train/labels', exist_ok=True)
    generate_train_dataset(args.size, args.image_num)
