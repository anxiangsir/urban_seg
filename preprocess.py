import cv2
import numpy as np
import args
import random

from tqdm import tqdm



size = args.args.size


def color_annotation(label_path, output_path):

    '''

    给class图上色

    '''

    img = cv2.imread(label_path,cv2.CAP_MODE_GRAY)

    color = np.ones([img.shape[0], img.shape[1], 3])



    color[img==0] = [255, 255, 255] #其他，白色，0

    color[img==1] = [0, 255, 0]     #植被，绿色，1

    color[img==2] = [0, 0, 0]       #道路，黑色，2

    color[img==3] = [131, 139, 139] #建筑，黄色，3

    color[img==4] = [139, 69, 19]   #水体，蓝色，4



    cv2.imwrite(output_path,color)

def generate_train_dataset(image_num = 40000,
                           train_image_path='dataset/train/images/',
                           train_label_path='dataset/train/labels/'):



    # 用来记录所有的子图的数目
    g_count = 1


    images_path = ['dataset/origin/1.png',
                   'dataset/origin/2.png', 'dataset/origin/3.png',
                   'dataset/origin/4.png', 'dataset/origin/5.png',
                   'dataset/origin/6.png', 'dataset/origin/7.png',
                   'dataset/origin/8.png', 'dataset/origin/9.png']
    labels_path = ['dataset/origin/1_class.png',
                   'dataset/origin/2_class.png', 'dataset/origin/3_class.png',
                   'dataset/origin/4_class.png', 'dataset/origin/5_class.png',
                   'dataset/origin/6_class.png', 'dataset/origin/7_class.png',
                   'dataset/origin/8_class.png', 'dataset/origin/9_class.png']

    # 每张图片生成子图的个数
    image_each = image_num // len(images_path)

    for i in tqdm(range(len(images_path))):
        count = 0
        image = cv2.imread(images_path[i])


        label = cv2.imread(labels_path[i], cv2.CAP_MODE_GRAY)
        X_height, X_width= image.shape[0], image.shape[1]
        while count < image_each:
            random_width = random.randint(0, X_width - size - 1)
            random_height = random.randint(0, X_height - size - 1)
            image_ogi = image[random_height: random_height + size, random_width: random_width + size,:]
            label_ogi = label[random_height: random_height + size, random_width: random_width + size]

            image_d,label_d = data_augment(image_ogi,label_ogi)

            cv2.imwrite((train_image_path+'%05d.png' % g_count), image_d)
            cv2.imwrite((train_label_path+'%05d.png' % g_count), label_d)

            count += 1
            g_count += 1




def generate_test_dataset(size=size, stride=size,
                           train_image_path='dataset/test/images/',
                           train_label_path='dataset/test/labels/'):
    '''
    这个函数用来生成测试数据集
    :return:
    '''
    count = 1

    images_path = ['dataset/origin/10.png']
    labels_path = ['dataset/origin/10_class.png']

    for i in range(len(images_path)):
        image = cv2.imread(images_path[i])
        label = cv2.imread(labels_path[i], cv2.CAP_MODE_GRAY)

        # 根据划窗步长切图
        for h in tqdm(range((image.shape[0]-size)//stride)):
            for w in range((image.shape[1]-size)//stride):
                image_ogi = image[h*stride:h*stride+size,w*stride:w*stride+size,:]
                label_ogi = label[h*stride:h*stride+size,w*stride:w*stride+size]
                # 保存原图
                cv2.imwrite((train_image_path+'%05d.png' % count), image_ogi)
                cv2.imwrite((train_label_path+'%05d.png' % count), label_ogi)
                count += 1







def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(size)]

    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)

    return cv2.LUT(img, gamma_table)


def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)

    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)

    gamma = np.exp(alpha)

    return gamma_transform(img, gamma)


def rotate(xb, yb, angle):
    M_rotate = cv2.getRotationMatrix2D((size /2, size / 2), angle, 1)

    xb = cv2.warpAffine(xb, M_rotate, (size, size))

    yb = cv2.warpAffine(yb, M_rotate, (size, size))

    return xb, yb


def blur(img):
    img = cv2.blur(img, (3, 3))

    return img


def add_noise(img):
    for i in range(size):  # 添加点噪声

        temp_x = np.random.randint(0, img.shape[0])

        temp_y = np.random.randint(0, img.shape[1])

        img[temp_x][temp_y] = 255

    return img


def data_augment(xb, yb):
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 90)

    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 180)

    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 270)

    if np.random.random() < 0.25:
        xb = cv2.flip(xb, 1)  # flipcode > 0：沿y轴翻转

        yb = cv2.flip(yb, 1)

    if np.random.random() < 0.25:
        xb = random_gamma_transform(xb, 1.0)

    if np.random.random() < 0.25:
        xb = blur(xb)

    # 双边过滤
    if np.random.random() < 0.25:
        xb =cv2.bilateralFilter(xb,9,75,75)

    #  高斯滤波
    if np.random.random() < 0.25:
        xb = cv2.GaussianBlur(xb,(5,5),1.5)

    # #   腐蚀
    # if np.random.random() < 0.25:
    #     kernel = np.ones((5, 5), np.uint8)
    #     xb = cv2.erode(xb, kernel, iterations=1)

    # 自适应阈值
    # if np.random.random() < 0.25:
    #     xb = cv2.adaptiveThreshold(xb, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    if np.random.random() < 0.2:
        xb = add_noise(xb)

    return xb, yb

if __name__ == '__main__':
    generate_test_dataset()
    generate_train_dataset()