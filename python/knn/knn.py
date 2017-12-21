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

def autoNorm(dataSet):
	"""
	Desc:
		归一化特征值，消除特征之间量级不同导致的影响
	parameter:
		dataSet: 数据集
	return:
		归一化后的数据集normDataSet. ranges和minVals即最小值与范围，并没有用到

	归一化公式:
		Y = (X-Xmin)/(Xmax-Xmin)
		其中的min和max分别是数据集中的最小特征值和最大特征值。该函数可以自动将数据特征值转化为0到1的区间
	"""
	# 计算每种属性的最大值、最小值、范围
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	# 极差
	ranges = maxVals - minVals
	normDataSet = zeros(shape(dataSet))
	m = dataSet.shape[0]
	print m
	normDataSet = dataSet - tile(minVals, (m, 1))
	normDataSet = normDataSet/tile(ranges, (m, 1))
	return normDataSet, ranges, minVals

def datingClassTest():
	"""
	Desc:
		测试方法
	parameters:
		none
	return：
		错误数
	"""
	# 设置测试数据的一个比例(训练数据集比例=1-hoRatio)
	hoRatio = 0.1 # 测试范围,一部分测试，一部分作为样本
	# 从文件中加载数据
	datingDataMat, datingLabels=file2matrix("../../data/knn/train/datingTestSet2.txt")
	#归一化数据
	print datingDataMat
	normMat, ranges, minVals = autoNorm(datingDataMat)
	print normMat
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0*np.array(datingLabels), 15.0*np.array(datingLabels))
plt.show()