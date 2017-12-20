#!/usr/bin/python
# -*- coding: utf-8

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def file2matrix(filename):
	"""
	Desc:
		读取knn训练集
	parameters:
		filename: 训练数据文件路径
	return:
		数据矩阵 returnMat 对应分类级别 classLabelVector
	"""

	file = open(filename)
	# 获得文件总行数
	numberOfLines = len(file.readlines())
	#生成对应的空数据矩阵
	returnMat = np.zeros((numberOfLines, 3)) 
	classLabelVector = []
	index = 0
	file = open(filename)
	for line in file.readlines():
		line = line.strip()
		listFromLine = line.split('\t')
		returnMat[index, :] = listFromLine[0:3]
		classLabelVector.append(int(listFromLine[-1]))
		index += 1

	return returnMat, classLabelVector

datingDataMat, datingLabels=file2matrix("../../data/knn/train/datingTestSet2.txt")
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0*np.array(datingLabels), 15.0*np.array(datingLabels))
plt.show()