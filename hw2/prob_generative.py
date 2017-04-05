import numpy as np
import sys
import matplotlib.pyplot as plt


class ProbGen:
	def __init__(self, x_train_data, y_train_data, x_test_data, w_model, b_model):
		
		self.x_train_set = x_train_data#(32561,106)
		self.y_train_set = y_train_data#(32561,1)
		
		#create train data set and validation data set
		self.train_data_set = self.create_train_data_set()#(32561,107)
		np.random.shuffle(self.train_data_set)#(let random the train data)

		#create test data set
		self.test_data_set = x_test_data

		#create weighting function and b parameter
		if w_model is None or b_model is None:
			self.w, self.b = self.classification()
		else:
			print("=== Model Detected ===")
			self.w = w_model.astype(np.float)
			self.b = b_model.astype(np.float)

	def create_train_data_set(self):
		
		train_data = np.append(self.x_train_set,self.y_train_set,1)
		return train_data

	def classification(self):

		class_1_numbers = np.where(self.train_data_set[:,-1] == 0)
		class_2_numbers = np.where(self.train_data_set[:,-1] == 1)

		N1 = class_1_numbers[0].size#(19807)
		N2 = class_2_numbers[0].size#(6241)

		class_1_data = np.array([]).reshape(0,int(self.train_data_set[:,:-1].shape[1]))
		class_2_data = np.array([]).reshape(0,int(self.train_data_set[:,:-1].shape[1]))
		
		for idx in class_1_numbers[0]:
			class_1_data = np.vstack((class_1_data,self.train_data_set[idx,:-1]))#(19807,106)

		for idx2 in class_2_numbers[0]:
			class_2_data = np.vstack((class_2_data,self.train_data_set[idx2,:-1]))#(6241,106)
		
		mean_u1 = np.array([]).reshape(0,class_1_data.shape[1])
		mean_u2 = np.array([]).reshape(0,class_2_data.shape[1])

		for idx in range(class_1_data.shape[1]):
			m = np.mean(class_1_data[:,idx])
			mean_u1 = np.append(mean_u1,m)

		for idx2 in range(class_2_data.shape[1]):
			m = np.mean(class_2_data[:,idx2])
			mean_u2 = np.append(mean_u2,m)

		mean_u1 = mean_u1.reshape(class_1_data.shape[1],1)#(106,1)
		mean_u2 = mean_u2.reshape(class_2_data.shape[1],1)#(106,1)

		cov1 = np.cov(class_1_data, rowvar = False, bias = True)#(106,106)
		cov2 = np.cov(class_2_data, rowvar = False, bias = True)#(106,106)

		cov_total = ((N1*1.0)/(N1+N2))*cov1+((N2*1.0)/(N1+N2))*cov2#(106,106)

		inverse_cov_total = np.linalg.inv(cov_total)

		w = np.dot((mean_u1-mean_u2).T, inverse_cov_total).T#(106,1)

		b1 = (-1.0/2)*np.dot(np.dot((mean_u1.T), inverse_cov_total), mean_u1)
		b2 = (1.0/2)*np.dot(np.dot((mean_u2.T), inverse_cov_total), mean_u2)
		b = b1+b2+np.log((N1*1.0)/N2)

		return w,b
		
	def sigmoid(self, z):
		res = 1.0/(1.0+np.exp(-z))
		return np.clip(res, 0.00000000000001, 0.99999999999999)

	def test_function(self):

		z = self.test_data_set.dot(self.w)+self.b
		predict_result = self.sigmoid(z)

		for i in range(len(predict_result)):
			if predict_result[i] > 0.5 :
				predict_result[i] = 0

			else:
				predict_result[i] = 1

		predict_result_output = np.zeros((16281+1, 1+1), dtype='|S6')
		predict_result_output[0,0] = "id"
		predict_result_output[0,1] = "label"

		for idx in range (16281):
		    predict_result_output[idx+1,0] = str(idx+1)
		    predict_result_output[idx+1,1] = int(predict_result[idx,0])

		np.savetxt("./model/w_prob.csv", self.w, delimiter = ",", fmt = "%s")
		np.savetxt("./model/b_prob.csv", self.b, delimiter = ",", fmt = "%s")
		np.savetxt(sys.argv[6], predict_result_output, delimiter=",", fmt = "%s")

		

