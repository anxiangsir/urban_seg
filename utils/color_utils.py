import numpy as np
import cv2

# 给标签图上色

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