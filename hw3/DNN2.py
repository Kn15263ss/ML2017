import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, adam, Nadam
from keras.utils import np_utils
from keras.models import load_model
from keras.callbacks import Callback
from keras import regularizers
from keras.utils.vis_utils import plot_model


np.set_printoptions(suppress=True)

def dump_history(store_path,logs):
    with open(os.path.join(store_path,'train_loss'),'a') as f:
        for loss in logs.tr_losses:
            f.write('{}\n'.format(loss))
    with open(os.path.join(store_path,'train_accuracy'),'a') as f:
        for acc in logs.tr_accs:
            f.write('{}\n'.format(acc))
    with open(os.path.join(store_path,'valid_loss'),'a') as f:
        for loss in logs.val_losses:
            f.write('{}\n'.format(loss))
    with open(os.path.join(store_path,'valid_accuracy'),'a') as f:
        for acc in logs.val_accs:
            f.write('{}\n'.format(acc))

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

class History(Callback):
    def on_train_begin(self,logs={}):
        self.tr_losses=[]
        self.val_losses=[]
        self.tr_accs=[]
        self.val_accs=[]

    def on_epoch_end(self,epoch,logs={}):
        self.tr_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.tr_accs.append(logs.get('acc'))
        self.val_accs.append(logs.get('val_acc'))

log_filepath = "log_history"
validation = 0.2 #0.2

first_layer = 64
second_layer = 128
third_layer = 250
fourth_layer = 500

first_dropout = 0.2#0.2
second_dropout = 0.3#0.3
third_dropout = 0.3#0.3
fourth_dropout = 0.6#0.5

batch_size = 64
nb_epoch = 2100

#load data
with open(sys.argv[1]) as trainFile:
  trainList = trainFile.read().splitlines()
  train_arr = np.array([line.split(",") for line in trainList])
  x_arr = train_arr[1:,1]
  y_arr = train_arr[1:,0]
  x_arr = np.array([str(line).split() for line in x_arr])
  y_arr = np.array([str(line).split() for line in y_arr])

  x_arr_valid = x_arr[x_arr.shape[0]*0.9:,:]
  y_arr_valid = y_arr[y_arr.shape[0]*0.9:,:]

  x_arr_train = x_arr[:x_arr.shape[0]*0.9,:]
  y_arr_train = y_arr[:y_arr.shape[0]*0.9,:]

  x_train_data = x_arr_train.reshape(x_arr_train.shape[0], 48, 48, 1).astype(np.float32)
  y_train_data = y_arr_train.astype(np.int)

  x_train_valid = x_arr_valid.reshape(x_arr_valid.shape[0], 48, 48, 1).astype(np.float32)
  y_train_valid = y_arr_valid.astype(np.int)

#rescale
x_train_data /= 255
x_train_valid /= 255

# convert class vectors to binary class matrices (one hot vectors)
y_train_data = np_utils.to_categorical(y_train_data, 7)
y_train_valid = np_utils.to_categorical(y_train_valid, 7)

model = Sequential()
model.add(Conv2D(first_layer,(3,3) ,padding="valid", input_shape = (48,48,1)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(2,2))
model.add(Dropout(first_dropout))

#model.summary()

model.add(Conv2D(second_layer,(3,3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(2,2))
model.add(Dropout(second_dropout))

#model.summary()

model.add(Conv2D(third_layer,(3,3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(2,2))
model.add(Dropout(third_dropout))

#model.summary()

model.add(Conv2D(fourth_layer,(3,3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(2,2))
model.add(Dropout(fourth_dropout))

#model.summary()

model.add(Flatten())

model.add(Dense(64, activation='relu', kernel_initializer='normal'))
#model.add(Dropout(0.2))

#model.summary()

model.add(Dense(128, activation='relu', kernel_initializer='normal'))
model.add(Dropout(0.2))

#model.summary()

model.add(Dense(256, activation='relu', kernel_initializer='normal'))
model.add(Dropout(0.5))

model.add(Dense(7, activation='softmax', kernel_initializer='normal'))

model.summary()

#sgd = SGD(lr=0.001, decay=1e-7, momentum=0.7, nesterov=True)
sgd = Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.003)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


#...............................................................................
base_dir = os.path.dirname(os.path.realpath(__file__))
result_dir = os.path.join(base_dir,'result')

dir_cnt = 0
log_path = "epoch_{}".format(str(nb_epoch))
log_path += '_'
store_path = os.path.join(result_dir,log_path+str(dir_cnt))
while dir_cnt < 30:
    if not os.path.isdir(store_path):
        os.mkdir(store_path)
        break
    else:
        dir_cnt += 1
        store_path = os.path.join(result_dir,log_path+str(dir_cnt))

history_data = History()
#...............................................................................

history = model.fit(x_train_data, y_train_data, batch_size=batch_size, epochs=nb_epoch, verbose=1, shuffle=True, validation_split=validation, callbacks=[history_data])

dump_history(store_path,history_data)
model.save(os.path.join(store_path,'model.h5'))
#plot_model(model,to_file=os.path.join(store_path,'model.png'))