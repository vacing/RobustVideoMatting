# encoding=utf-8

import cv2
import numpy as np
from . import util_container

def img_phash(imgfile, img_resize=64, dct_w=16):
    '''
    使用phash算法，获取图像指纹
    :param imgfile:
    :param img_resize:
    :param dct_w:
    :return: 二进制和十六进制的图像hash表示
    '''
    # 加载并调整图片大小
    img = cv2.imdecode(np.fromfile(imgfile, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread(imgfile, 0) # can't support chinese character
    img = cv2.resize(img, (img_resize, img_resize), interpolation=cv2.INTER_CUBIC)

    # 创建二维列表
    vis0 = np.array(img, dtype=np.float32)

    # 二维Dct变换
    vis1 = cv2.dct(vis0)
    # 拿到左上角的8 * 8
    vis1 = vis1[0:dct_w, 0:dct_w]

    # 把二维list变成一维list
    img_list = util_container.flatten(vis1.tolist())

    # 计算均值
    avg = sum(img_list) * 1. / len(img_list)
    hash_b = ['0' if i < avg else '1' for i in img_list]

    # 得到哈希值
    hash_x = ''.join(['%x' % int(''.join(hash_b[x:x + 4]), 2) for x in range(0, dct_w * dct_w, 4)])

    return hash_b, hash_x


