import numpy as np
import csv
import sys
from sklearn.svm import LinearSVR as SVR
from data_process import process

# Train a linear SVR

npzfile = np.load('train_data.npz')
X = npzfile['X']
y = npzfile['y']

svr = SVR(C=1)
svr.fit(X, y)


# svr.get_params() to save the parameters
# svr.set_params() to restore the parameters

# predict 
testdata = np.load(sys.argv[1])
test_X = []
process_data = process()

for i in range(200):
	data = testdata[str(i)]
	data = np.asarray(data)
	w = process_data.get_eigenvalues(data)
	test_X.append(w)

test_X = np.array(test_X)

pred_y = svr.predict(test_X)

ans = np.log(pred_y)


file = open(sys.argv[2], 'w') 
f_write = csv.writer(file,delimiter =',')
f_write.writerow(['Setid','LogDim'])
for i in range(200):
	f_write.writerow([str(i),str(ans[i])])
file.close()
