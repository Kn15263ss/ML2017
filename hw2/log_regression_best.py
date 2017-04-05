import numpy as np
import sys
import matplotlib.pyplot as plt



class LogReg:
	def __init__(self, x_train_data, y_train_data, x_test_data, valid_percent, w_model, w2_model, w3_model, b_model):

		self.x_train_set = x_train_data#(32561,106)
		self.y_train_set = y_train_data#(32561,1)
		self.test_data_set = x_test_data#(16281,106)
		self.valid_percent = valid_percent
		
		#create train data set and validation data set
		self.train_data_set = self.create_train_data_set()#(32561,107)
		self.train_data_set = self.shuffle(self.train_data_set)#(let random the train data)

		#(select the 80 percent data be a training data)
		self.training_data_set = self.train_data_set[:int(self.train_data_set.shape[0]*(1-self.valid_percent)), :]
		#(26048,107)
		
		#(select the 20 percent data be a training data)
		self.valid_data_set = self.train_data_set[int(self.train_data_set.shape[0]*(1-self.valid_percent)):, :]
		#(6513,107)

		#create weighting function and b parameter
		if w_model is None or w2_model is None or w3_model is None or b_model is None:
			self.w = np.random.rand(self.x_train_set.shape[1],1)#(106,1)
			self.w2 = np.random.rand(self.x_train_set.shape[1],1)#(106,1)
			self.w3 = np.random.rand(self.x_train_set.shape[1],1)#(106,1)
			self.b = np.random.rand(1,1)
		else:
			print("=== Model Detected ===")
			self.w = w_model.astype(np.float)
			self.w2 = w2_model.astype(np.float)
			self.w3 = w3_model.astype(np.float)
			self.b = b_model.astype(np.float)		

	def shuffle(self, X):
		randomize = np.arange(len(X))
		np.random.shuffle(randomize)
		return X[randomize]

	def create_train_data_set(self):
		
		train_data = np.append(self.x_train_set,self.y_train_set,1)
		return train_data

	def logistic_Regression_ada(self, interation, lr_rate, lamda):

		self.lr_rate = lr_rate
		self.interation = interation

		w_diff_total = 1
		w2_diff_total = 1
		w3_diff_total = 1
		b_diff_total = 1
		
		cross_entropy_valid_error = np.zeros(shape = (self.interation,1))
		cross_entropy_train_error = np.zeros(shape = (self.interation,1))

		for idx in range(1,self.interation+1):

			w_diff = 0
			w2_diff = 0
			w3_diff = 0
			b_diff = 0

			X = self.training_data_set[:,:-1].reshape(self.training_data_set.shape[0],self.training_data_set.shape[1]-1)#(26048,106)

			z = X.dot(self.w)+(X**2).dot(self.w2)+(X**3).dot(self.w3)+self.b#(26048,1)
			y = self.sigmoid(z)

			Y = self.training_data_set[:,-1].reshape(self.training_data_set.shape[0],1)#(26048,1)

			w_diff = -1*np.dot(X.T, (Y-y))+(lamda/y.size)*(self.w.T.dot(self.w)+self.w2.T.dot(self.w2)+self.w3.T.dot(self.w3))#(106,1)
			w2_diff = -1*np.dot((X**2).T, (Y-y))+(lamda/y.size)*(self.w.T.dot(self.w)+self.w2.T.dot(self.w2)+self.w3.T.dot(self.w3))#(106,1)
			w3_diff = -1*np.dot((X**2).T, (Y-y))+(lamda/y.size)*(self.w.T.dot(self.w)+self.w2.T.dot(self.w2)+self.w3.T.dot(self.w3))#(106,1)
			b_diff = np.sum(-1*(Y-y))

	     	#use Adagrad to improve learning rate
			w_diff_total += w_diff**2
			w2_diff_total += w2_diff**2
			w3_diff_total += w3_diff**2
			b_diff_total += b_diff**2

			ada_w = np.sqrt(w_diff_total)
			ada_w2 = np.sqrt(w2_diff_total)
			ada_w3 = np.sqrt(w3_diff_total)
			ada_b = np.sqrt(b_diff_total)

			ada_w[ada_w == 0] = 1
			ada_w2[ada_w2 == 0] = 1
			ada_w3[ada_w3 == 0] = 1
			if ada_b == 0: ada_b = 1

			self.w -= (self.lr_rate*w_diff)/ada_w
			self.w2 -= (self.lr_rate*w2_diff)/ada_w2
			self.w3 -= (self.lr_rate*w3_diff)/ada_w3
			self.b -= (self.lr_rate*b_diff)/ada_b

			cross_entropy_valid_error[idx-1], cross_entropy_train_error[idx-1] = self.loss_function()
			
			if idx%10 == 0 or idx == 1:
				print "The iterations times: %d\nThe valid error value is %f\nThe train error value is %f\n"\
				 %(idx,cross_entropy_valid_error[idx-1],cross_entropy_train_error[idx-1])
			
	def sigmoid(self, z):
		res = 1.0/(1.0+np.exp(-z))
		return np.clip(res, 0.00000000000001, 0.99999999999999)

	def loss_function(self):
		'''
    	Comput error for logistic regression
		'''
		x_valid = self.valid_data_set[:,:-1]#(6513,106)
		y_valid = self.valid_data_set[:,-1]
		y_valid = y_valid.reshape(y_valid.shape[0],1)#(6513,1)

		x_train = self.training_data_set[:,:-1]#(26048,106)
		y_train = self.training_data_set[:,-1]
		y_train = y_train.reshape(y_train.shape[0],1)#(26048,1)

		z_valid = x_valid.dot(self.w)+(x_valid**2).dot(self.w2)+(x_valid**3).dot(self.w3)+self.b
		z_train = x_train.dot(self.w)+(x_train**2).dot(self.w2)+(x_train**3).dot(self.w3)+self.b

		predict_valid = self.sigmoid(z_valid)
		predict_train = self.sigmoid(z_train)

		predict_valid = np.around(predict_valid)
		predict_train = np.around(predict_train)

		cross_entropy_valid_error = np.sum(np.absolute(y_valid-predict_valid))
		cross_entropy_valid_error = 1-(1.0*cross_entropy_valid_error)/y_valid.size

		cross_entropy_train_error = np.sum(np.absolute(y_train-predict_train))
		cross_entropy_train_error = 1-(1.0*cross_entropy_train_error)/y_train.size

		return cross_entropy_valid_error, cross_entropy_train_error

	def test_function(self):	

		predict_test = self.test_data_set.dot(self.w)+(self.test_data_set**2).dot(self.w2)+(self.test_data_set**3).dot(self.w3)+self.b
		predict_result = self.sigmoid(predict_test)
		
		predict_result = np.around(predict_result)

		predict_result_output = np.zeros((16281+1, 1+1), dtype='|S6')
		predict_result_output[0,0] = "id"
		predict_result_output[0,1] = "label"

		for idx in range (16281):
		    predict_result_output[idx+1,0] = str(idx+1)
		    predict_result_output[idx+1,1] = int(predict_result[idx,0])

		np.savetxt("./model/w_log_best.csv", self.w, delimiter = ",", fmt = "%s")
		np.savetxt("./model/w2_log_best.csv", self.w2, delimiter = ",", fmt = "%s")
		np.savetxt("./model/w3_log_best.csv", self.w3, delimiter = ",", fmt = "%s")
		np.savetxt("./model/b_log_best.csv", self.b, delimiter = ",", fmt = "%s")
		np.savetxt(sys.argv[6], predict_result_output, delimiter=",", fmt = "%s")
