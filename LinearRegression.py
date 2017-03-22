import numpy as np
import sys
from GradientDescent import GradDesc

#read train_data file
train_data = np.genfromtxt(sys.argv[1], dtype = "S", skip_header=True ,delimiter = ",")
train_data = train_data[:,3:] 

#read test_data file
test_data = np.genfromtxt(sys.argv[2], dtype = "S", skip_header=False ,delimiter = ",")
test_data = test_data[:,2:]

#create train data set
'''
create a shape = 18,5760 array
'''
train_set = train_data[:18,:]

for days in range(1,12*20):
    train_set = np.append(train_set, train_data[days*18:days*18+18,:],1)

train_set[train_set == "NR"] = 0#(if array have "NR" string, let it convert to float)
train_set = train_set.astype(np.float)#(18,5760)


#create test data set
'''
create a shape = 18,5760 array
'''
test_set = test_data[:18,:]

for days in range(1,12*20):
    test_set = np.append(test_set, test_data[days*18:days*18+18,:],1)#(18,2160)

test_set[test_set == "NR"] = 0
test_set = test_set.astype(np.float)

# Read model
try:
  	with open("./model/w_pm2.5_5.751.csv") as model_file:
  		model_list = model_file.read().splitlines()
  		w_model = np.array([[str(item) for item in line.split(",")] for line in model_list])

	with open("./model/b_pm2.5_5.751.csv") as model_file:
		model_list = model_file.read().splitlines()
		b_model = np.array([[str(item) for item in line.split(",")] for line in model_list])

except IOError:
	w_model = None
	b_model = None
#train and test
predict = GradDesc(train_set, test_set, valid_percent = 0.2, w_model = w_model, b_model = b_model)
if w_model is None or b_model is None:
	predict.train_grad_ada(interation = 50000, lr_rate = 0.2)
	predict.test_function()
else:
	predict.test_function()