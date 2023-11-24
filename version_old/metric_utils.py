import numpy as np
from sklearn.metrics import confusion_matrix
# 其他，白色，0
# 植被，绿色，1
# 道路，黑色，2
# 建筑，黄色，3
# 水体，蓝色，4

def iou(y_pre: np.ndarray, y_true: np.ndarray) -> 'dict':
    # cm是混淆矩阵
    cm = confusion_matrix(
        y_true=y_true,
        y_pred=y_pre,
        labels=[0, 1, 2, 3, 4])

    result_iou = [
        cm[i][i] / (sum(cm[i, :]) + sum(cm[:, i]) - cm[i, i]) for i in range(len(cm))
    ]

    metric_dict = {}
    metric_dict['IOU_其他/other'] = result_iou[0]
    metric_dict['IOU_植物/plant'] = result_iou[1]
    metric_dict['IOU_道路/road']  = result_iou[2]
    metric_dict['IOU_建筑/arch']  = result_iou[3]
    metric_dict['IOU_水体/water'] = result_iou[4]

    metric_dict['iou'] = np.mean(result_iou)
    metric_dict['accuracy'] = sum(np.diag(cm)) / sum(np.reshape(cm, -1))

    return metric_dict

