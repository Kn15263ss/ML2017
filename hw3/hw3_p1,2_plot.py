import sys
import matplotlib.pyplot as plt
import numpy as np

print (sys.argv[1])
print (sys.argv[2])



title = "Accuracy Plot of CNN Model"
figure_title = "CNN Training Procedure.png"

#accTrain= np.genfromtxt (sys.argv[1], delimiter=",")
#accVal = np.genfromtxt(sys,argv[2],delimiter=",")

accTrain = np.loadtxt(sys.argv[1])
accVal = np.loadtxt(sys.argv[2])

y1 = len(accTrain)
y2 = len(accVal)

if (y1 != y2): print("Incorrect Accuracy Files Given")
else: y = np.arange(y1)+1

print(y.shape)

fig = plt.figure()

labelAccTrain = plt.plot(y,accTrain, label = 'Train', color = 'r')
labelAccValid = plt.plot(y,accVal, label = 'Validation', color = 'b')


plt.xlabel('# of Epoch', fontsize = 18)
plt.ylabel('Accuracy', fontsize = 18)

plt.title(title)
plt.legend(loc='lower right')
plt.savefig(figure_title)
plt.show()





'''
plt.figure()
plt.imshow(heatmap, cmap=plt.cm.jet)
plt.colorbar()
plt.tight_layout()
fig = plt.gcf()
plt.draw()
fig.savefig(os.path.join(cmap_dir, '{}.png'.format(idx)),
'''