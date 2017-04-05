import numpy as np
import sys
from prob_generative import ProbGen

#read train_data file
train_data = np.genfromtxt(sys.argv[1], dtype = "S", skip_header=False ,delimiter = ",")
x_train_data = np.genfromtxt(sys.argv[3], dtype = "int", skip_header=True ,delimiter = ",")
y_train_data = np.genfromtxt(sys.argv[4], dtype = "int", skip_header=False ,delimiter = ",")

feature = np.arange(31,38)
feature = feature.reshape(1,len(feature))

#x_train_data = np.delete(x_train_data,feature,1)
y_train_data = y_train_data.reshape(y_train_data.shape[0],1)

#read test_data file
test_data = np.genfromtxt(sys.argv[2], dtype = "S", skip_header=False ,delimiter = ",")
x_test_data = np.genfromtxt(sys.argv[5], dtype = "int", skip_header=True ,delimiter = ",")

#x_test_data = np.delete(x_test_data,feature,1)

# Read model
try:
  	with open("./model/w_prob.csv") as model_file:
  		model_list = model_file.read().splitlines()
  		w_model = np.array([[str(item) for item in line.split(",")] for line in model_list])

	with open("./model/b_prob.csv") as model_file:
		model_list = model_file.read().splitlines()
		b_model = np.array([[str(item) for item in line.split(",")] for line in model_list])

except IOError:
	w_model = None
	b_model = None

predict = ProbGen(x_train_data, y_train_data, x_test_data, w_model = w_model, b_model = b_model)
predict.test_function()
