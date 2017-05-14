import sys
import glob
import Image as img
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.misc as misc 
from scipy import linalg
from pca_process import PCA_process
import PIL
import os


pca = PCA_process(pathname = sys.argv[1], count_num = 10, char_count_num = 10)


mean_value = pca.Mean()

#---- Plot the average face ----#
#pca.averageFace(mean_value)
#---- --------------------- ----#

dataAdjust = pca.setCenter()
data = pca.uncenterData(dataAdjust,mean_value)

#---- plot the origin dataset ----#
#pca.saveImg(data,100)
#---- ----------------------- ----#

dataAdjust_t = dataAdjust.T
u,s,v = pca.SVD(dataAdjust_t)


s_red = pca.reduceDim(9)
eigenFace = pca.findEigenFace(u,s_red,9,mean_value)
#---- Plot the eigen face ----#
#pca.saveImg(eigenFace,9)
#---- ------------------- ----#

s_red2 = pca.reduceDim(5)
recon = pca.reconData(u,s_red2,v,mean_value)
#---- Plot the reconstruct face ----#
pca.saveImg(recon,100)
#---- ------------------------- ----#
'''
errorList =[]

for idx in np.arange(100):
	s_red = pca.reduceDim(idx+1)
	eigenFace = pca.findEigenFace(u,s_red,idx+1,mean_value)
	recon = pca.reconData(u,s_red,v,mean_value)

	recon = recon.astype(np.float64)

	error = ((np.sqrt(((data-recon)**2).mean()))/256)*100
	
	if error < 1:
		print "top eigenfaces is: ", idx

	errorList.append(error)


errorList = np.asarray(errorList).astype(np.float64)

plt.plot(errorList)
plt.xlabel('dim')
plt.ylabel('error')
plt.axis([50,60,0,2])
plt.savefig("error.jpg")
plt.show()
'''




