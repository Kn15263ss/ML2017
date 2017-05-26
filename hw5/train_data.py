import os
import sys
import nltk
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam, Nadam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from nltk.corpus import stopwords
from data_process import process
from keras.callbacks import Callback, CSVLogger
from keras.utils.vis_utils import plot_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

train_path = sys.argv[1]
test_path = sys.argv[2]

#####################
###   parameter   ###
#####################
split_ratio = 0.1
embedding_dim = 100
nb_epoch = 500
batch_size = 64
max_article_length = 190
range_value = 0.4


#########################
###   Main function   ###
#########################
def main():

    nltk.download("stopwords")
    data_process = process()

    base_dir, result_dir = data_process.get_path()
    store_path, history_data = data_process.history_data(result_dir, nb_epoch)
    
    ### read training and testing data
    (Y_data,X_data,tag_list) = data_process.read_data(train_path,True)
    (_, X_test,_) = data_process.read_data(test_path,False)
    all_corpus_temp = X_data + X_test

    train_tag = data_process.to_multi_categorical(Y_data,tag_list)

    # #part3
    # train_tag_all = np.sum(train_tag, axis=0)
    # train_tag_all = np.array(train_tag_all, dtype = "int")
    # for i in range(len(tag_list)):
    #     print (str(tag_list[i])+":"+" "+str(train_tag_all[i]))

### RNN
#----------------------------------------------------------------------------------------------------------------------------------------- 
    ### tokenizer for all data
    tokenizer_temp = Tokenizer()
    tokenizer_temp.fit_on_texts(all_corpus_temp)
    word_index_temp = tokenizer_temp.word_index

    all_corpus = [w for w in word_index_temp if not w in stopwords.words("english")]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_corpus)
    word_index = tokenizer.word_index

    tokenizer_file = os.path.join(store_path, "tokenizer.pickle")
    data_process.store_pickle(tokenizer, tokenizer_file)

    label_file = os.path.join(store_path, "label.pickle")
    data_process.store_pickle(tag_list, label_file)


    ### convert word sequences to index sequence
    train_sequences = tokenizer.texts_to_sequences(X_data)    
    test_sequences = tokenizer.texts_to_sequences(X_test)

    ### padding to equal length
    train_sequences = pad_sequences(train_sequences, maxlen=max_article_length)
    test_sequences = pad_sequences(test_sequences,maxlen=max_article_length)
    
    ### split data into training set and validation set
    (X_train,Y_train),(X_val,Y_val) = data_process.split_data(train_sequences,train_tag,split_ratio)

    ### get mebedding matrix from glove
    embedding_dict = data_process.get_embedding_dict('glove.6B.%dd.txt'%embedding_dim)
    num_words = len(word_index) + 1
    embedding_matrix = data_process.get_embedding_matrix(word_index,embedding_dict,num_words,embedding_dim)

    ### build model
    model = Sequential()
    model.add(Embedding(num_words,
                        embedding_dim,
                        weights=[embedding_matrix],
                        input_length=max_article_length,
                        trainable=False))
    
    model.add(GRU(128,activation='tanh', dropout=0.5))
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(len(tag_list),activation='sigmoid'))
    model.summary()

    #rms = RMSprop(lr = 0.0005)
    model.compile(loss='categorical_crossentropy',
                  optimizer="adam",
                  metrics=[data_process.f1_score])

   
    earlystopping = EarlyStopping(monitor='val_f1_score', patience = 10, verbose=1, mode='max')
    checkpoint = ModelCheckpoint(filepath='best.hdf5',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=False,
                                 monitor='val_f1_score',
                                 mode='max')
    #csv_logger = CSVLogger('RNN.log')

    hist = model.fit(X_train, Y_train, 
                     validation_data=(X_val, Y_val),
                     epochs=nb_epoch, 
                     batch_size=batch_size,
                     callbacks=[history_data,earlystopping,checkpoint])
                     #callbacks=[history_data,earlystopping,checkpoint,csv_logger])

    data_process.dump_history(store_path,history_data)
    plot_model(model,to_file=os.path.join(store_path,'model.png'))

#-----------------------------------------------------------------------------------------------------------------------------------------

### bag of word
#-----------------------------------------------------------------------------------------------------------------------------------------
    # vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, min_df=2, max_df  = 0.7)
    # all_corpus_temp = vectorizer.fit_transform(all_corpus_temp)
    # #####normalize
    # all_corpus_temp = normalize(all_corpus_temp, norm='l2')
    # all_corpus_temp = all_corpus_temp.toarray()
    # X_data = all_corpus_temp[0:len(X_data),:]
    # X_test = all_corpus_temp[len(X_data):len(all_corpus_temp),:]

    # (X_train,Y_train),(X_val,Y_val) = data_process.split_data(X_data,train_tag,split_ratio)

    # model = Sequential()
    # model.add(Dense(128,input_shape=X_train.shape[1:],activation='tanh'))
    # model.add(Dropout(0.4))
    # model.add(Dense(256,activation='relu'))
    # model.add(Dropout(0.4))
    # model.add(Dense(128,activation='relu'))
    # model.add(Dropout(0.4))
    # model.add(Dense(64,activation='relu'))
    # model.add(Dropout(0.4))

    # model.add(Dense(len(tag_list),activation='sigmoid'))
    # model.summary()

    # model.compile(loss='categorical_crossentropy',
    #               optimizer="adam",
    #               metrics=[data_process.f1_score])

   
    # earlystopping = EarlyStopping(monitor='val_f1_score', patience = 10, verbose=1, mode='max')
    # checkpoint = ModelCheckpoint(filepath='best.hdf5',
    #                              verbose=1,
    #                              save_best_only=True,
    #                              save_weights_only=False,
    #                              monitor='val_f1_score',
    #                              mode='max')
    # csv_logger = CSVLogger('bag_of_word.log')

    # hist = model.fit(X_train, Y_train, 
    #                  validation_data=(X_val, Y_val),
    #                  epochs=nb_epoch, 
    #                  batch_size=batch_size,
    #                  callbacks=[history_data,earlystopping,checkpoint,csv_logger])

    # data_process.dump_history(store_path,history_data)
    # plot_model(model,to_file=os.path.join(store_path,'model.png'))
    
if __name__=='__main__':
    main()