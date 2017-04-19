import numpy as np
import sys
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, adam
from keras.utils import np_utils
import tensorflow as tf
np.set_printoptions(suppress=True)


batch_size = 128
nb_epoch = 30

#load data
with open(sys.argv[1]) as trainFile:
	trainList = trainFile.read().splitlines()
	train_arr = np.array([line.split(",") for line in trainList])
	x_arr = train_arr[1:,1]
	y_arr = train_arr[1:,0]
	x_arr = np.array([str(line).split() for line in x_arr])#(28709,2304)
	y_arr = np.array([str(line).split() for line in y_arr])

	x_train_data = x_arr.reshape(x_arr.shape[0], 48, 48, 1).astype(np.float32)#(28709,48,48,1)
	y_train_data = y_arr.astype(np.int)#(28709,1)

	'''
	x_arr2 = x_arr.reshape(x_arr.shape[0], 48, 48)

	for idx in range(x_arr.shape[0]):
		temp = np.flipud(x_arr2[idx].transpose()).transpose()
		x_arr2[idx] = temp

	x_train_set = x_arr.reshape(x_arr.shape[0], 48, 48, 1).astype(np.float32)#(28709,48,48,1)
	x_train_set2 = x_arr2.reshape(x_arr2.shape[0], 48, 48, 1).astype(np.float32)#(28709,48,48,1)

	x_train_data = np.vstack((x_train_set,x_train_set2))#(57418,48,48,1)
	y_train_data = y_arr.astype(np.int)#(28709,1)
	y_train_data = np.vstack((y_train_data,y_train_data))#(57418,1)
	'''

#rescale
x_train_data /= 255

# convert class vectors to binary class matrices (one hot vectors)
original, idx = np.unique(y_train_data, return_inverse = True)
y_train_data = np_utils.to_categorical(y_train_data, 7)

model = Sequential()

model.add(Conv2D(32,3,3 ,border_mode="valid", input_shape = (48,48,1)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))

model.add(Conv2D(64,3,3))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))

model.add(Conv2D(128,3,3))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))

model.add(Conv2D(256,3,3))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(64, init='normal'))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(128, init='normal'))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(256, init='normal'))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(7, init='normal'))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

history = model.fit(x_train_data, y_train_data, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, shuffle=True, validation_split=0.2)

a = model.predict(x_train_data)
b = np.around(a)

print original[b.argmax(1)]
print original[y_train_data.argmax(1)]

with open(sys.argv[2]) as testFile:
  testList = testFile.read().splitlines()
  test_arr = np.array([line.split(",") for line in testList])
  test_x_arr = test_arr[1:,1]
  test_x_arr = np.array([str(line).split() for line in test_x_arr])

  x_test_data = test_x_arr.reshape(test_x_arr.shape[0], 48, 48, 1).astype(np.float32)#(28709,48,48,1)

#rescale
x_test_data /= 255

test_y_probability = model.predict(x_test_data)
test_y_int = np.around(test_y_probability)
test_y = original[test_y_int.argmax(1)]

test_output = np.zeros((len(test_y)+1, 2), dtype='|S5')
test_output[0,0] = "id"
test_output[0,1] = "label"

for i in range (test_output.shape[0]-1):
    test_output[i+1,0] = str(i)
    test_output[i+1,1] = str(test_y[i])

np.savetxt("original", original, delimiter=",", fmt="%s")
np.savetxt(sys.argv[3], test_output, delimiter=",", fmt = "%s")
