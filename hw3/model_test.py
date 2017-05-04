import os
import sys
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.utils.vis_utils import plot_model

with open(sys.argv[1]) as testFile:
  testList = testFile.read().splitlines()
  test_arr = np.array([line.split(",") for line in testList])
  test_x_arr = test_arr[1:,1]
  test_x_arr = np.array([str(line).split() for line in test_x_arr])

  x_test_data = test_x_arr.reshape(test_x_arr.shape[0], 48, 48, 1).astype(np.float32)#(28709,48,48,1)

#rescale
x_test_data /= 255

model = load_model("./model/model.h5")

model.summary()

plot_model(model, to_file="./model/model.png")

test_y = model.predict_classes(x_test_data)

test_output = np.zeros((len(test_y)+1, 2), dtype='|S5')
test_output[0,0] = "id"
test_output[0,1] = "label"

for i in range (test_output.shape[0]-1):
    test_output[i+1,0] = str(i)
    test_output[i+1,1] = str(test_y[i])


np.savetxt(sys.argv[2], test_output, delimiter=",", fmt = "%s")