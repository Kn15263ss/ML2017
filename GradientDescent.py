import numpy as np
import sys
import matplotlib.pyplot as plt


class GradDesc:
	def __init__(self, train_set, test_set, valid_percent, w_model, b_model):
		'''
		The feature have ['AMB_TEMP' 'CH4' 'CO' 'NMHC' 'NO' 'NO2' 'NOx' 'O3' 'PM10' 'PM2.5'
 		'RAINFALL' 'RH' 'SO2' 'THC' 'WD_HR' 'WIND_DIREC' 'WIND_SPEED' 'WS_HR'] and it represented 
 		the [0,1,....,17]
		'''
		#self.feature = np.arange(18)#(select all feature(0~17))
		self.feature = np.array([9])#(u can select feature in here)
		self.hours_range = np.arange(1,9)#(The hours of the feature, u can select a range from here)
		#example self.hours_range = np.arange(1,9), the range is 1 to 8

		self.train_data = train_set
		self.test_data = test_set
		self.valid_percent = valid_percent

		#create train data set and validation data set
		self.train_data_set = self.create_train_data_set()

		#Scale features and set them to zero mean
		self.train_data_set, self.mean_value, self.SD_value = self.feature_scaling()
		np.random.shuffle(self.train_data_set)#(let random the train data)

		#(select the 80 percent data be a training data)
		self.training_data_set = self.train_data_set[:int(self.train_data_set.shape[0]*(1-self.valid_percent)), :]
		#(4521,163)

		#(select the 20 percent data be a training data)
		self.valid_data_set = self.train_data_set[:int(self.train_data_set.shape[0]*self.valid_percent), :]
		#(1130,163)
		
		#create test data set
		self.test_data_set = self.creat_test_data()

		#create weighting function and b parameter

		if w_model is None or b_model is None:
			self.w = np.random.rand(len(self.feature)*len(self.hours_range),1)#(162,1)
			self.b = 0.1#np.random.rand(1,1)
		else:
			print("=== Model Detected ===")
			self.w = w_model.astype(np.float)
			self.b = b_model.astype(np.float)

	def create_train_data_set(self):		
		# This is an array which will be the shape of (5652, 163) and be returned.
		train_x_set = np.array([]).reshape(0,len(self.feature)*len(self.hours_range))
		pm25 = np.array([]).reshape(0,(24*20-9)*12)
		'''
		Here will create a 5652,163 training array
		the temp = train_data[feature[:,None],number] it represened if feature is 0 to 18
		and number is 9, the train data will be return train_data[:18,9], the row is 0 to 18, the col is 0 to 9
		or if feature is 0,2,4, the train data will be return train_data[[0,2,4],9], the row is 0,2,4, the col is 0 to 9
		'''
		for months in range(12):
		    for hours in range(self.hours_range[0],24*20-len(self.hours_range)):
		        temp = self.train_data[self.feature[:,None],hours+months*480:hours+months*480+len(self.hours_range)].\
		        flatten().reshape(1,len(self.feature)*len(self.hours_range))
		        train_x_set = np.vstack((train_x_set,temp))#(5652,162)

		#Append the correct pm25 value to train_x_set
		for months in range(12):
			pm25 = np.append(pm25, self.train_data[9,9+months*480:480+months*480])
		pm25 = pm25.reshape(train_x_set.shape[0],1)#(5652,1)

		train_total_data_set = np.append(train_x_set,pm25,axis = 1)#(5652,163)

		return train_total_data_set
		
	def creat_test_data(self):
		# This is an array which will be the shape of (240, 162) and be returned.
		test_set = self.test_data[self.feature[:,None],self.hours_range].flatten().reshape(1,len(self.feature)*len(self.hours_range))
		for days in range(1,12*20):
			test_set = np.vstack((test_set,self.test_data[self.feature[:,None],self.hours_range+days*9].flatten()))

		return test_set
	def train_grad_ada(self, interation, lr_rate):
		'''
	    Performs gradient descent to learn w
	    by taking iterations gradient steps with learning
	    rate lr_rate
	    '''
		self.lr_rate = lr_rate
		self.interation = interation

		w_diff_total = 1
		b_diff_total = 1

		for idx in range(1,self.interation+1):

			w_diff = 0
			b_diff = 0

			## Do the gradient descent
	      	# Make the feature X an numpy array of shape (4521,162).
	      	# 162 = 18 * 9 is the range of the hours from feature, and there are
	      	# 4521 features in total in the training set.
			X = self.training_data_set[:,:-1].reshape(self.training_data_set.shape[0],self.training_data_set.shape[1]-1)#(4521,162)
			
			# The shape of the predict array is (4521,1)
			predict = X.dot(self.w)#(4521,1)
			
			# Create an array of correct pm2.5 value
			pm25 = self.training_data_set[:,-1].reshape(self.training_data_set.shape[0],1)#(4521,1)
			#(y = wx+b, y^ - y = y^ - (wx+b))
			delta = (pm25 - (predict + self.b))#(4521,1)

			w_diff_temp = (2*delta*(-X)).T#(162,4521)
			b_diff_temp = 2*delta*(-1)#(4521,1)

			w_diff = np.sum(w_diff_temp,1).reshape(w_diff_temp.shape[0],1)#(162,1)
			b_diff =   np.sum(b_diff_temp,0).reshape(1,1)
			
	     	#use Adagrad to improve learning rate
			w_diff_total += w_diff**2
			b_diff_total += b_diff**2

			self.w -= (self.lr_rate*w_diff)/np.sqrt(w_diff_total)
			self.b -= (self.lr_rate*b_diff)/np.sqrt(b_diff_total)
			'''
			self.w -= self.lr_rate*w_diff
			self.b -= self.lr_rate*b_diff
			'''
		first = 'train ' + str((1-self.valid_percent)*100) + '%'
		second = 'validation ' + str(self.valid_percent*100) + '%'
		plt.title('Prediction of PM2.5')
		plt.xlabel('Iterations')
		plt.ylabel('error')
		plt.plot(np.arange(self.interation), train_loss_error, 'r-x')
		plt.plot(np.arange(self.interation), valid_loss_error, 'b--')
		plt.legend([first , second], loc='upper left')
		plt.axis([0, self.interation, 5, 8])
		plt.show()

	def feature_scaling(self):
		'''
		Returns a normalized X where the mean value of
		each feature is 0 and the standard deviation is 1
		'''
		mean_value = []
		SD_value = []
		x_set = self.train_data_set[:,:-1]
		x_norm = self.train_data_set[:,:-1]

		for idx in range(x_norm.shape[1]):
			m = np.mean(x_set[:,idx])
	        s = np.std(x_set[:,idx])
	        mean_value.append(m)
	        SD_value.append(s)
	        x_norm[:,idx] = (x_norm[:,idx]-m)/s

		x_norm = np.append(x_norm,self.train_data_set[:,-1].\
	    	reshape(self.train_data_set.shape[0],1),1)

		return x_norm, mean_value, SD_value

	def test_function(self):	

		self.test_data_set = (self.test_data_set-self.mean_value)/self.SD_value
		predict_result = self.test_data_set.dot(self.w)+self.b

		predict_result_output = np.zeros((12*20+1, 1+1), dtype='|S6')
		predict_result_output[0,0] = "id"
		predict_result_output[0,1] = "value"

		for idx in range (12*20):
		    predict_result_output[idx+1,0] = "id_" + str(idx)
		    predict_result_output[idx+1,1] = float(predict_result[idx,0])

		np.savetxt("./model/w_pm2.5.csv", self.w, delimiter = ",", fmt = "%s")
		np.savetxt("./model/b_pm2.5.csv", self.b, delimiter = ",", fmt = "%s")
		np.savetxt(sys.argv[3], predict_result_output, delimiter=",", fmt = "%s")
