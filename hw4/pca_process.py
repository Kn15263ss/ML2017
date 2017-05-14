import sys
import glob
import Image as img
import numpy as np 
import matplotlib.pyplot as plt
from scipy import linalg
import os

class PCA_process:
	"""docstring for PCA"""
	def __init__(self,pathname,count_num,char_count_num):
		trainData = []
		idx = "A"
		count = 0
		char_count = 1
		for file in sorted(os.listdir(pathname)):
			if file.endswith(".bmp"):
				if (file.startswith(idx) and (count < count_num)):
					filepath = pathname + file
					imgArr = np.asarray(img.open(filepath))
					imgList = np.reshape(imgArr, (np.product(imgArr.shape), )).astype('int')
					trainData.append(imgList)
					count += 1
				elif ((char_count < char_count_num) and (count == count_num)):
					count = 0
					char_count += 1
					idx = chr(ord(idx)+1)
		trainData = np.asarray(trainData)
		self.trainData = trainData

	def column(self,matrix, i):
		return [row[i] for row in matrix]
	
	def Mean(self):
		Mean =[]
		trainData = self.trainData
		for idx in np.arange(len(trainData[0])):
			meanValue = np.mean(self.column(trainData,idx))
			Mean.append(meanValue)
		Mean = np.asarray(Mean)
		self.Mean = Mean
		return Mean

	def setCenter(self):
		Mean = self.Mean
		trainData = self.trainData
		for idx in np.arange(len(self.column(trainData,0))):
			trainData[idx] = trainData[idx] - Mean
		dataAdjust = np.asarray(trainData)
		return dataAdjust

	def SVD(self,matrix):
		[u,s,v] = linalg.svd(matrix)
		self.s = s
		return u,s,v

	def reduceDim(self,EigenNum):
		s_red = []
		s = self.s
		for idx in np.arange(len(s)):
			if (idx < EigenNum ):
				s_red.append(s[idx])
			else:
				s_red.append(0)
		s_red = np.asarray(s_red)
		S_red = linalg.diagsvd(s_red, 4096, 100)
		return S_red

	def reconData(self,u,s,v,mean):
		reconSet =[]
		recon = np.dot(np.dot(u,s),v)
		recon_t = recon.T
		for idx in np.arange(len(recon_t)):
			reconSet.append(recon_t[idx]+mean)
		reconSet = np.asarray(reconSet)
		return reconSet

	def uncenterData(self,matrix,mean):
		reconSet =[]
		for idx in np.arange(len(matrix)):
			reconSet.append(matrix[idx]+mean)
		reconSet = np.asarray(reconSet)
		return reconSet

	def saveImg(self,inputArr,imgNum):
		num = int(np.sqrt(imgNum))
		imgArr = np.asarray(inputArr).reshape(len(inputArr),64,64)
		for idx in np.arange(imgNum):
			plt.subplot(num,num,idx+1)
			plt.imshow(imgArr[idx],cmap='gray')
			plt.axis('off')
		plt.savefig("test.jpg")
		plt.show()

	def averageFace(self,meanList):
		averageFace =np.asarray(meanList).reshape(64,64)
		plt.imshow(averageFace, cmap='gray')
		plt.savefig("averageFace.jpg")
		plt.show()

	def findEigenFace(self,u,s,vectorNum,mean):
		eigenFace = []
		temp = np.dot(u,s).T
		for idx in np.arange(vectorNum):
			eigenFace.append(temp[idx]+mean)
		eigenFace = np.asarray(eigenFace)
		return eigenFace