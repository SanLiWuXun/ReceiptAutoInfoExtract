#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from cnocr import CnOcr
import pandas as pd
from pandas import DataFrame
import os

#后续生成票据图像时的大小，按照标准增值税发票版式240mmX140mm来设定
height_resize = 1400
width_resize = 2400

# 调整原始图片尺寸
def resizeImg(image, height=height_resize):
    h, w = image.shape[:2]
    pro = height / h
    size = (int(w * pro), int(height))
    img = cv2.resize(image, size)
    return img

# 边缘检测
def getCanny(image):
    # 高斯模糊
    binary = cv2.GaussianBlur(image, (3, 3), 2, 2)
    # 边缘检测
    binary = cv2.Canny(binary, 60, 240, apertureSize=3)
    # 膨胀操作，尽量使边缘闭合
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    return binary

# 求出面积最大的轮廓
def findMaxContour(image):
    # 寻找边缘
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # 计算面积
    max_area = 0.0
    max_contour = []
    for contour in contours:
        currentArea = cv2.contourArea(contour)
        if currentArea > max_area:
            max_area = currentArea
            max_contour = contour
    return max_contour, max_area

# 多边形拟合凸包的四个顶点
def getBoxPoint(contour):
    # 多边形拟合凸包
    hull = cv2.convexHull(contour)
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    approx = approx.reshape((len(approx), 2))
    return approx

# 适配原四边形点集
def adapPoint(box, pro):
    box_pro = box
    if pro != 1.0:
        box_pro = box/pro
    box_pro = np.trunc(box_pro)
    return box_pro

# 四边形顶点排序，[top-left, top-right, bottom-right, bottom-left]
def orderPoints(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# 计算长宽
def pointDistance(a, b):
    return int(np.sqrt(np.sum(np.square(a - b))))

# 透视变换
def warpImage(image, box):
    w, h = pointDistance(box[0], box[1]), \
           pointDistance(box[1], box[2])
    dst_rect = np.array([[0, 0],
                         [w - 1, 0],
                         [w - 1, h - 1],
                         [0, h - 1]], dtype='float32')
    M = cv2.getPerspectiveTransform(box, dst_rect)
    warped = cv2.warpPerspective(image, M, (w, h))
    return warped

# 统合图片预处理
def imagePreProcessing(path):
    image = cv2.imread(path)
    # 转灰度、降噪
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #image = cv2.GaussianBlur(image, (3,3), 0)
    # 边缘检测、寻找轮廓、确定顶点
    ratio = height_resize / image.shape[0]
    img = resizeImg(image)
    binary_img = getCanny(img)
    max_contour, max_area = findMaxContour(binary_img)
    boxes = getBoxPoint(max_contour)
    boxes = adapPoint(boxes, ratio)
    boxes = orderPoints(boxes)
    # 透视变化
    warped = warpImage(image, boxes)
    # 调整最终图片大小
    height, width = warped.shape[:2]
    #size = (int(width*height_resize/height), height_resize)
    size = (width_resize, height_resize)
    warped = cv2.resize(warped, size, interpolation=cv2.INTER_CUBIC)
    return warped

# 截取图片中部分区域图像，测试阶段使用，包括显示与保存图片，实际使用时不使用这个函数，使用下面的正式版函数
def cropImage_test(img, crop_range, filename='Undefined'):
    xpos, ypos, width, height = crop_range
    crop = img[ypos:ypos+height, xpos:xpos+width]
    if filename=='Undefined': #如果未指定文件名，采用坐标来指定文件名
        filename = 'crop-'+str(xpos)+'-'+str(ypos)+'-'+str(width)+'-'+str(height)+'.jpg'
    cv2.imshow(filename, crop) #展示截取区域图片---测试用
    #cv2.imwrite(filename, crop) #imwrite在文件名含有中文时会有乱码，应该采用下方imencode---测试用
    # 保存截取区域图片---测试用
    cv2.imencode('.jpg', crop)[1].tofile(filename)
    return crop

# 截取图片中部分区域图像
def cropImage(img, crop_range):
    xpos, ypos, width, height = crop_range
    crop = img[ypos:ypos+height, xpos:xpos+width]
    return crop

# 从截取图片中识别文字
def cropOCR(crop, ocrType):
    if ocrType==0:
        text_crop_list = ocr.ocr_for_single_line(crop)
    elif ocrType==1:
        text_crop_list = ocr_numbers.ocr_for_single_line(crop)
    elif ocrType==2:
        text_crop_list = ocr_UpperSerial.ocr_for_single_line(crop)
    text_crop = ''.join(text_crop_list)
    return text_crop


if __name__ == '__main__':
    # 实例化不同用途CnOcr对象
    ocr = CnOcr(name='') #混合字符
    ocr_numbers = CnOcr(name='numbers', cand_alphabet='0123456789') #纯数字
    ocr_UpperSerial = CnOcr(name='UpperSerial', cand_alphabet='0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ') #编号，只包括大写字母(没有O)与数字

    # 截取图片中部分区域图像-名称
    crop_range_list_name = ['发票代码', '发票号码', '开票日期',
                            '校验码', '销售方名称', '销售方纳税人识别号',
                            '销售方地址电话', '销售方开户行及账号', '价税合计',
                            '备注']

    # 截取图片中部分区域图像-坐标
    crop_range_list_data = [[1870, 40, 380, 38], [1867, 104, 380,38], [1866, 166, 380, 50],
                            [1867, 230, 450, 50], [421, 1046, 933, 46], [419, 1091, 933, 48],
                            [420, 1145, 933, 47], [421, 1193, 933, 40], [1892, 976, 414, 48],
                            [1455, 1045, 325, 38]]

    # 截取图片中部分区域图像-使用ocr的类型，0：混合字符，1：纯数字，2：编号
    crop_range_list_type = [1, 1, 0,
                            1, 0, 2,
                            0, 0, 0,
                            0]
    
    # 预处理图像
    path = 'test.jpg'
    warped = imagePreProcessing(path)

    # 展示与保存预处理的图片---测试用
    #cv2.imshow('warpImage', warped)
    cv2.imwrite('result.jpg',warped)

    # 处理预处理图像并将结果保存到text_ocr列表中
    text_ocr = []
    for i in range(len(crop_range_list_data)):
        #filename = crop_range_list_name[i]+'.jpg' #测试阶段保存截取图片时使用的文件名，实际使用时不需要
        crop = cropImage(warped, crop_range_list_data[i])
        crop_text = cropOCR(crop, crop_range_list_type[i])
        crop_text = crop_text.replace('o','0') #发票中不会有小写字母o，凡是出现o的都使用0替代
        print(crop_range_list_name[i],':',crop_text)
        text_ocr.append(crop_text)
    
    # 按年月来保存结果到xlsx文件中，计算文件名
    date_temp = text_ocr[2].split('年')
    year_num = date_temp[0]
    month_num = date_temp[1].split('月')[0]
    filename = year_num+'-'+month_num+'.xlsx'

    # 如果文件还没建立，新建文件
    if not os.path.exists(filename):
        dic = {}
        for i in range(len(crop_range_list_name)):
            dic[crop_range_list_name[i]] = []
        df = pd.DataFrame(dic)
        df.to_excel(filename, index=False)

    data = pd.read_excel(filename)
    if not int(text_ocr[1]) in data['发票号码'].values.tolist():
        new_line_num = data.shape[0]
        data.loc[new_line_num] = text_ocr
        DataFrame(data).to_excel(filename, index=False, header=True)
    else:
        print(path,'is already in',filename,'!')

    cv2.waitKey(0)