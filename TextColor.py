import os
import numpy as np
import cv2

# Text Direction
def get_hor(binary):
    rows, cols = binary.shape
    hor_list = [0] * rows
    for i in range(rows):
        for j in range(cols):
            # 统计每一行的黑色像素总数
            if binary.item(i, j) == 0:
                hor_list[i] = hor_list[i] + 1

    # 对hor_list中的元素进行筛选，可以去除一些噪点
    hor_arr = np.array(hor_list)
    hor_arr[np.where(hor_arr < 5)] = 0
    hor_list = hor_arr.tolist()

    # 计算均方差
    mean = np.mean(hor_list)
    sum = 0
    for i in range(rows):
        sum = sum + pow(hor_list[i] - mean, 2)
    sum = sum / rows
    return hor_list, sum


def get_ver(binary):
    rows, cols = binary.shape
    ver_list = [0] * cols
    for j in range(cols):
        for i in range(rows):
            # 统计每一列的黑色像素总数
            if binary.item(i, j) == 0:
                ver_list[j] = ver_list[j] + 1

    # 对ver_list中的元素进行筛选，可以去除一些噪点
    ver_arr = np.array(ver_list)
    ver_arr[np.where(ver_arr < 5)] = 0
    ver_list = ver_arr.tolist()

    # 计算均方差
    mean = np.mean(ver_list)
    sum = 0
    for i in range(cols):
        sum = sum + pow(ver_list[i] - mean, 2)
    sum = sum / cols
    return ver_list, sum

def get_List(list_data):
    # 取出list中像素存在的区间
    vv_list = list()
    v_list = list()
    for index, i in enumerate(list_data):
        if i > 0:
            v_list.append(index)
        else:
            if v_list:
                vv_list.append(v_list)
                # list的clear与[]有区别
                v_list = []
    return vv_list


# Text Color
def findColor(img):
    pixel = []
    rows, cols, _ = img.shape
    for j in range(cols):
        for i in range(rows):
            if (img[i, j, 0] == img[i, j, 1]) & (img[i, j, 1] == img[i, j, 2]):  # 灰色一律排除
                continue
            if (img[i, j, 0] > 240) & (img[i, j, 1] > 240) & (img[i, j, 2] > 240):
                continue
            else:
                pixel.append([i, j, img[i, j]])  # i,j以及像素值  i是行坐标、j是列坐标
    return pixel


def get_hor_char(hor_list, img_bgr):
    hh_list = get_List(hor_list)  # 列表类型，存储的每一行对应的一系列像素的行号，比如12，13，...，27
    if hh_list == []:  # 针对空白情况
        exit("No information detected ")
    res = []
    i = 0
    for hh in hh_list:  # 每一行是从x行像素到y行像素
        split = img_bgr[hh[0]:(hh[-1] + 1), :]  # 把对应的几行分割出来
        res.append(split)
        # cv2.imwrite('split\split_' + str(i) + '.jpg', img_bgr[hh[0] - 5:hh[-1] + 5, :])
        i = i + 1

    return res


def get_ver_char(ver_list, img_bgr):
    vv_list = get_List(ver_list)  # 列表类型，存储的每一行对应的一系列像素的行号，比如12，13，...，27
    if vv_list == []:  # 针对空白情况
        exit("No information detected ")
    i = 0
    res = []

    for vv in vv_list:  # 每一行是从x行像素到y行像素
        split = img_bgr[:, vv[0]:(vv[-1] + 1)]  # 把对应的几行分割出来
        res.append(split)
        # cv2.imwrite('split\split' + str(i) + '-' + str(len(vv)) + '.jpg', img_bgr[vv[0] - 5:vv[-1] + 5, :])
        i = i + 1
    return res


def Solve_Image(img_bgr):
    # img_bgr = cv2.imread(img_bgr, 1)
    img = img_bgr.copy()

    # 待检测属性
    horizon = True
    font_size = 0
    indent = False
    alignment = 'left'

    # 二值化
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    # 水平 & 竖直投影
    hor_list, hor_sum = get_hor(binary)
    ver_list, ver_sum = get_ver(binary)

    # 选择文本方向和排列方式
    if hor_sum >= ver_sum:
        # 获取单个字符位置信息
        res = get_hor_char(hor_list, img_bgr)  # 这一步把行方向的统计变成了方格角度的统计


    if hor_sum < ver_sum:
        res= get_ver_char(ver_list, img_bgr)  # 这一步把列方向的统计变成了方格角度的统计

    return res
