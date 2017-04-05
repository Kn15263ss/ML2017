import numpy as np
import sys
from log_regression import LogReg
np.set_printoptions(suppress=True)


def feature_normalize(x_train_data, x_test_data):
	# feature normalization with all X
	X_train = x_train_data#(32561,106)
	X_test = x_test_data#(16281,106)

	X_all = np.concatenate((X_train, X_test))#(48842,106)
	mean_value = np.mean(X_all, axis = 0)#(106,)
	SD_value = np.std(X_all, axis = 0)#(106,)
	
	# only apply normalization on continuos attribute
	feature = [0, 1, 3, 4, 5]
	m = np.zeros(shape = X_all.shape[1])
	s = np.ones(shape = X_all.shape[1])

	m[feature] = mean_value[feature]
	s[feature] = SD_value[feature]

	X_all_normalize = (X_all-m)/s#(48842,106)

	X_train_normalize = X_all_normalize[:X_train.shape[0], :]#(32561,106)
	X_test_normalize = X_all_normalize[X_train.shape[0]:, :]#(16281,106)

	return X_train_normalize, X_test_normalize

def sort_ranges(inputCsv):
	arrayAge = inputCsv[:,0]
	arrayHours = inputCsv[:,5]

	for i in range(inputCsv.shape[0]):
		age = arrayAge[i]
		pay = arrayHours[i]
		if age >= 0 and age <= 25:
			arrayAge[i] = 0
		elif age>=26 and age <= 45:
			arrayAge[i] = 1
		elif age>=46 and age <= 65:
			arrayAge[i] = 2
		elif age>65:
			arrayAge[i] = 3 

		if pay >= 0 and pay <= 25:
			arrayHours[i] = 0
		elif pay>=26and pay <= 40:
			arrayHours[i] = 1
		elif pay>=41 and pay <= 60:
			arrayHours[i] = 2
		elif pay>61:
			arrayHours[i] = 3 

	inputCsv[:,0] = arrayAge
	inputCsv[:,5] = arrayHours
	return inputCsv

#read train_data file
train_data = np.genfromtxt(sys.argv[1], dtype = "S", skip_header=False ,delimiter = ",")
x_train_data = np.genfromtxt(sys.argv[3], dtype = "int", skip_header=True ,delimiter = ",")
y_train_data = np.genfromtxt(sys.argv[4], dtype = "int", skip_header=False ,delimiter = ",")

#read test_data file
test_data = np.genfromtxt(sys.argv[2], dtype = "S", skip_header=False ,delimiter = ",")
x_test_data = np.genfromtxt(sys.argv[5], dtype = "int", skip_header=True ,delimiter = ",")

#feature normalize
x_train_data, x_test_data = feature_normalize(x_train_data, x_test_data)
x_train_data = sort_ranges(x_train_data)
x_test_data = sort_ranges(x_test_data)

#1 age
#2 fnlwgt
#3 sex
#4 capital_gain
#5 capital_loss
#6 hours_per_week
#7-14 Employer 			8
#15-30 Education 		15
#31-37 Marital 			6
#38-52 Occupation 		14
#53-58 Family Role  	5
#59-63 Race 			4
#64-106 Origin of Countries 42

#1 4 15~52
age = np.array([0])
fnlwgt = np.array([1])
sex = np.array([2])
capital_gain = np.array([3])
capital_loss = np.array([4])
hours_per_week = np.array([5])
Employer = np.arange(6,15)
Education = np.arange(15,31)
Marital = np.arange(31,38)
Occupation = np.arange(38,53)
Family_Role = np.arange(53,59)
Race = np.arange(59,64)
Origin_of_Countries = np.arange(64,106)

idxCountry = np.array([64,88,93,100,103,65,72,82,84,97,66,80,99,67,70,71,92,68,83,87\
					,69,76,77,79,86,89,90,91,96,101,73,74,78,85,75,81,94,95,98,104,102,105])
idxCountry = np.sort(idxCountry, axis=None)

#feature = np.array([age,capital_gain,idxEmployer,idxEducation,idxMarital,idxOccupation,idxFamRole])
feature = np.array([age,fnlwgt,sex,capital_gain,capital_loss,hours_per_week\
	,Employer,Education,Occupation,Family_Role,Race,idxCountry])
feature = np.concatenate(feature)

x_train_data = x_train_data[:,feature[:,None]].reshape(x_train_data.shape[0],len(feature))
x_test_data = x_test_data[:,feature[:,None]].reshape(x_test_data.shape[0],len(feature))
y_train_data = y_train_data.reshape(y_train_data.shape[0],1)

# Read model
try:
  	with open("./model/w_log.csv") as model_file:
  		model_list = model_file.read().splitlines()
  		w_model = np.array([[str(item) for item in line.split(",")] for line in model_list])

  	with open("./model/w2_log.csv") as model_file:
  		model_list = model_file.read().splitlines()
  		w2_model = np.array([[str(item) for item in line.split(",")] for line in model_list])

	with open("./model/b_log.csv") as model_file:
		model_list = model_file.read().splitlines()
		b_model = np.array([[str(item) for item in line.split(",")] for line in model_list])

except IOError:
	w_model = None
	w2_model = None
	b_model = None

predict = LogReg(x_train_data, y_train_data, x_test_data, valid_percent = 0.2, w_model = w_model, w2_model = w2_model, b_model = b_model)
if w_model is None or w2_model is None or b_model is None:
	predict.logistic_Regression_ada(interation = 10000, lr_rate = 0.03, lamda = 10)
	predict.test_function()
else:
	predict.test_function()

