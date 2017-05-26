import os
import sys
import nltk
import string
import cPickle as pickle
import numpy as np
import keras.backend as K 
from keras.callbacks import Callback

class process():

	def get_path(self):
		base_dir = os.path.dirname(os.path.realpath(__file__))
		result_dir = os.path.join(base_dir,'result')
		if not os.path.exists(result_dir):
			os.makedir(result_dir)
		return base_dir, result_dir

	def read_data(self,path,training):
	    with open(path,'r') as f:
	    
	        tags = []
	        articles = []
	        tags_list = []
	        
	        f.readline()
	        for line in f:
	            if training :
	                start = line.find('\"')
	                end = line.find('\"',start+1)
	                tag = line[start+1:end].split(' ')
	                article = line[end+2:]
	                
	                for t in tag :
	                    if t not in tags_list:
	                        tags_list.append(t)
	               
	                tags.append(tag)
	            else:
	                start = line.find(',')
	                article = line[start+1:]
	            
	            articles.append(article)
	            
	        if training :
	            assert len(tags_list) == 38,(len(tags_list))
	            assert len(tags) == len(articles)
	    return (tags,articles,tags_list)

	def get_embedding_dict(self,path):
	    embedding_dict = {}
	    with open(path,'r') as f:
	        for line in f:
	            values = line.split(' ')
	            word = values[0]
	            coefs = np.asarray(values[1:],dtype='float32')
	            embedding_dict[word] = coefs
	    return embedding_dict

	def get_embedding_matrix(self,word_index,embedding_dict,num_words,embedding_dim):
	    embedding_matrix = np.zeros((num_words,embedding_dim))
	    for word, i in word_index.items():
	        if i < num_words:
	            embedding_vector = embedding_dict.get(word)
	            if embedding_vector is not None:
	                embedding_matrix[i] = embedding_vector
	    return embedding_matrix

	def to_multi_categorical(self,tags,tags_list): 
	    tags_num = len(tags)
	    tags_class = len(tags_list)
	    Y_data = np.zeros((tags_num,tags_class),dtype = 'float32')
	    for i in range(tags_num):
	        for tag in tags[i] :
	            Y_data[i][tags_list.index(tag)]=1
	        assert np.sum(Y_data) > 0
	    return Y_data

	def split_data(self,X,Y,split_ratio):
	    indices = np.arange(X.shape[0])  
	    np.random.shuffle(indices) 
	    
	    X_data = X[indices]
	    Y_data = Y[indices]
	    
	    num_validation_sample = int(split_ratio * X_data.shape[0] )
	    
	    X_train = X_data[num_validation_sample:]
	    Y_train = Y_data[num_validation_sample:]

	    X_val = X_data[:num_validation_sample]
	    Y_val = Y_data[:num_validation_sample]

	    return (X_train,Y_train),(X_val,Y_val)

	def translate_Categorial2label(self,output, label):
	    Y = []
	    for row in output:
	        find = [pos for pos,x in enumerate(row) if x==1]
	        temp = []
	        for i in find:
	            temp.append(label[i])
	        temp = [" ".join(temp)]
	        Y.append(temp)
	    return Y

	def f1_score(self,y_true,y_pred):

	    thresh = 0.4
	    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
	    tp = K.sum(y_true * y_pred,axis=-1)
	    
	    precision=tp/(K.sum(y_pred,axis=-1)+K.epsilon())
	    recall=tp/(K.sum(y_true,axis=-1)+K.epsilon())
	    return K.mean(2*((precision*recall)/(precision+recall+K.epsilon())))


	def store_pickle(self, pack_item, path):
		with open(path, 'wb') as handle:
			pickle.dump(pack_item, handle, protocol=pickle.HIGHEST_PROTOCOL)

	def load_pickle(self, path):
		with open(path, 'rb') as handle:
			unpack_item = pickle.load(handle)
		return unpack_item

	def dump_history(self,store_path,logs):
		with open(os.path.join(store_path,'train_loss'),'a') as f:
			for loss in logs.tr_loss:
				f.write('{}\n'.format(loss))
		with open(os.path.join(store_path,'train_accuracy'),'a') as f:
			for acc in logs.tr_f1score:
				f.write('{}\n'.format(acc))
		with open(os.path.join(store_path,'valid_loss'),'a') as f:
			for loss in logs.val_loss:
				f.write('{}\n'.format(loss))
		with open(os.path.join(store_path,'valid_accuracy'),'a') as f:
			for acc in logs.val_f1score:
				f.write('{}\n'.format(acc))

	def history_data(self, exp_dir, nb_epoch):
		dir_cnt = 0
		log_path = "epoch_{}".format(str(nb_epoch))
		log_path += '_'
		store_path = os.path.join(exp_dir,log_path+str(dir_cnt))
		while dir_cnt < 100:
			if not os.path.isdir(store_path):
				os.mkdir(store_path)
				break
			else:
				dir_cnt += 1
				store_path = os.path.join(exp_dir,log_path+str(dir_cnt))

		history_data = History()

		return store_path, history_data

class History(Callback):
    def on_train_begin(self,logs={}):
        self.tr_loss=[]
        self.val_loss=[]
        self.tr_f1score=[]
        self.val_f1score=[]

    def on_epoch_end(self,epoch,logs={}):
        self.tr_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.tr_f1score.append(logs.get('f1score'))
        self.val_f1score.append(logs.get('val_f1score'))