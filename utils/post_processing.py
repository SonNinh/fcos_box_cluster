from networkx.algorithms import cluster
import numpy as np
import networkx as nx
import cv2
from numpy.lib.type_check import imag
from torch import dtype


def cal_iou(bboxes1, bboxes2):
    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))

    # compute the area of intersection rectangle
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)

    ret = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    return ret

def draw_bboxs(image, bboxs, color):
    for box in bboxs:
        cv2.rectangle(
            image, 
            (box[0], box[1]),
            (box[2], box[3]),
            color, 1
        )

def box_clutering(bboxs):
    iou = cal_iou(bboxs, bboxs)
    mask = iou > 0.95
    G = nx.from_numpy_array(mask)

    
    clusters = sorted(nx.connected_components(G), key = len, reverse=True)
    ret_box = []
    for i, c in enumerate(clusters):
        if len(c) > 2:
            box = np.mean(bboxs[list(c)], axis=0, dtype=np.int)
            ret_box.append(box)
    return ret_box
    # clusters = np.matmul(mask, mask.T)
    # print(clusters)

    # print(
    #     iou.shape[0], 
    #     ((iou>0.95).sum()+iou.shape[0])/2
    # )

if __name__ == "__main__":

    for i in range(8):
        print('image:', i)
        bboxs = np.load(f'outputs/margin_exp_seg/box_data_{i}.npy')
        image = cv2.imread(f'outputs/margin_exp_seg/box_{i}.png')
        bboxs = box_clutering(bboxs)
        draw_bboxs(image, bboxs, (255, 0, 0))
        cv2.imwrite(f'outputs/margin_exp_seg/final_box_{i}.png', image)

