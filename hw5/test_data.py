import os
import sys
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from data_process import process

test_path = sys.argv[1]
result_path = sys.argv[2]

max_article_length = 190
range_value = 0.4

data_process = process()

(_, X_test,_) = data_process.read_data(test_path,False)

tag_list = data_process.load_pickle("./model/label.pickle")
tokenizer = data_process.load_pickle("./model/tokenizer.pickle")

test_sequences = tokenizer.texts_to_sequences(X_test)
test_sequences = pad_sequences(test_sequences,maxlen=max_article_length)

model = load_model("./model/best.hdf5", custom_objects={"f1_score": data_process.f1_score})

Y_pred = model.predict(test_sequences)

linfnorm = np.linalg.norm(Y_pred, axis=1, ord=np.inf)
preds = Y_pred.astype(np.float) / linfnorm[:, None]

preds[preds >= range_value] = 1
preds[preds < range_value] = 0

original_y = data_process.translate_Categorial2label(preds, tag_list)

output_file = []

output_file.append('"id","tags"')
for i in range (len(original_y)):
    temp = '"'+str(i)+'"' + ',' + '"' + str(" ".join(original_y[i])) + '"'
    output_file.append(temp)

    with open(result_path,'w') as f:
        for data in output_file:
            f.write('{}\n'.format(data))