import os
import sys
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Concatenate, Dot, Merge, Add
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.models import load_model
from keras.utils import np_utils
import keras.backend as K
from movies_class import data_process
#from progress.bar import Bar

epochs = 1000
batch_size = 256
validation_split = 0.1
latent_dim = 256

# def store_pickle(pack_item, path):
#         with open(path, 'wb') as handle:
#             pickle.dump(pack_item, handle, protocol=pickle.HIGHEST_PROTOCOL)

# def load_pickle(path):
#     with open(path, 'rb') as handle:
#         unpack_item = pickle.load(handle)
#     return unpack_item


if __name__ == "__main__":
    movies_class = data_process()
    base_dir, result_dir = movies_class.get_path()
    store_path, history_data = movies_class.history_data(result_dir, epochs)

    ratings_data = pd.read_csv(os.path.join(sys.argv[1],'train.csv'), sep=',', engine='python')
    users_data = pd.read_csv(os.path.join(sys.argv[1],'users.csv'), sep='::', engine='python')
    movies_data = pd.read_csv(os.path.join(sys.argv[1],'movies.csv'), sep='::', engine='python')

    moviesGen = movies_data.Genres.astype("str")
    
    test_data = pd.read_csv(os.path.join(sys.argv[1],'test.csv'), sep=',', engine='python')

    #use user data Gender
    numUserGender = {"M":0, "F":1}
    users_data.Gender = users_data.Gender.replace(numUserGender)
    users_data.Gender = users_data.Gender.astype('int')

    n_movies = ratings_data['MovieID'].drop_duplicates().max()
    n_users = ratings_data['UserID'].drop_duplicates().max()

    movieID = ratings_data.MovieID.values
    userID = ratings_data.UserID.values
    Y_data = ratings_data.Rating.values

    test_users = test_data.UserID.values
    test_movies = test_data.MovieID.values

#part 5 fail
#---------------------------------------------------------------------------------------------------------------------------
    # print("Processing train %d Movies Data" %(len(movieID)))
    # bar = Bar('Processing', max=len(movieID))
    # moviesGenInput = []
    # for i in range(len(movieID)):
    #     temp = np.where(movies_data.movieID == movieID[i])
    #     moviesGenInput.append(moviesGen[temp[0][0]])
    #     bar.next()
    # bar.finish()
    # moviesGenInput = np.asarray(moviesGenInput, dtype='str')
    # moviesGenInput = np.array([str(line).split("|") for line in moviesGenInput]).reshape(moviesGenInput.shape[0],1)
    # store_pickle(moviesGenInput,"moviesGenInput.pickle")

    # moviesGenInput = load_pickle("moviesGenInput.pickle")

    # temp = []

    # for i in range(len(movieID)):
    #     for k in moviesGenInput[i][0]:
    #         if k not in temp:
    #             temp.append(k)
    # print temp
    
    # total=[]
    # for i in range(len(movieID)):
    #     lable1 = [0,0,0,0,0,0,0,0,0,0]
    #     lable2 = []
    #     for k in range(len(moviesGenInput[i][0])):
    #         if (moviesGenInput[i][0][k] == 'Drama') or (moviesGenInput[i][0][k] == 'Musical'):
    #             if 0 not in lable2:
    #                 lable1[0] += 1
    #                 lable2.append(0)
    #         elif (moviesGenInput[i][0][k] == 'Animation') or (moviesGenInput[i][0][k] == "Children's"):
    #             if 1 not in lable2:
    #                 lable1[1] += 1
    #                 lable2.append(1)
    #         elif (moviesGenInput[i][0][k] == 'Comedy'):
    #             if 2 not in lable2:
    #                 lable1[2] += 1
    #                 lable2.append(2)
    #         elif (moviesGenInput[i][0][k] == 'Romance'):
    #             if 3 not in lable2:
    #                 lable1[3] += 1
    #                 lable2.append(3)
    #         elif (moviesGenInput[i][0][k] == 'Action') or (moviesGenInput[i][0][k] == 'Adventure'):
    #             if 4 not in lable2:
    #                 lable1[4] += 1
    #                 lable2.append(4)
    #         elif (moviesGenInput[i][0][k] == 'War'):
    #             if 5 not in lable2:
    #                 lable1[5] += 1
    #                 lable2.append(5)
    #         elif (moviesGenInput[i][0][k] == 'Fantasy') or (moviesGenInput[i][0][k] == 'Sci-Fi') or (moviesGenInput[i][0][k] == 'Mystery'):
    #             if 6 not in lable2:
    #                 lable1[6] += 1
    #                 lable2.append(6)
    #         elif (moviesGenInput[i][0][k] == 'Thriller') or (moviesGenInput[i][0][k] == 'Crime') or (moviesGenInput[i][0][k] == 'Horror'):
    #             if 7 not in lable2:
    #                 lable1[7] += 1
    #                 lable2.append(7)
    #         elif (moviesGenInput[i][0][k] == 'Western'):
    #             if 8 not in lable2:
    #                 lable1[8] += 1
    #                 lable2.append(8)
    #         elif (moviesGenInput[i][0][k] == 'Film-Noir') or (moviesGenInput[i][0][k] == 'Documentary'):
    #             if 9 not in lable2:
    #                 lable1[9] += 1
    #                 lable2.append(9)

    #     total.append(lable1)
    # total = np.asarray(total,dtype="int")
#---------------------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------------------
    # #BONUS
    # feature = np.array([0,3])#[UserID, Gender, Age, Occupation, Zip-code]

    # print("Processing train %d User's Data" %(len(userID)))
    # bar = Bar('Processing', max=len(userID))
    # userDetailsInput = []
    # for i in range(len(userID)):
    #     userDetailsInput.append(users_data[users_data.UserID == userID[i]].values[0][feature])
    #     bar.next()
    # bar.finish()
    # userDetailsInput = np.asarray(userDetailsInput, dtype='int')
    # print('User Input data: ', userDetailsInput.shape)
    # print userDetailsInput

    # print("Processing test %d User's Data" %(len(test_users)))
    # bar = Bar('Processing', max=len(test_users))
    # testUserDetailsInput = []
    # for i in range(len(test_users)):
    #     testUserDetailsInput.append(users_data[users_data.UserID == test_users[i]].values[0][feature])
    #     bar.next()
    # bar.finish()
    # testUserDetailsInput = np.asarray(testUserDetailsInput, dtype='int')
    # print('Test User Input data: ', testUserDetailsInput.shape)
    # print testUserDetailsInput
#---------------------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------------------
    # ##normalize ratings
    # rating_mean = np.mean(Y_data)
    # rating_SD = np.std(Y_data)
    # rationg_norm = (Y_data-rating_mean)/(rating_SD*1.0)
#---------------------------------------------------------------------------------------------------------------------------
    X1_train, X2_train, Y_train, X1_val, X2_val, Y_val = movies_class.split_data(userID, movieID, Y_data, validation_split)
#---------------------------------------------------------------------------------------------------------------------------------------
    # #MODEL-MF
    # user_input = Input(shape=[1])
    # user_vec = Embedding(n_users+1, latent_dim, embeddings_initializer="random_normal")(user_input)
    # user_vec = Flatten()(user_vec)
    # user_bias = Embedding(n_users+1, 1, embeddings_initializer="zeros")(user_input)
    # user_bias = Flatten()(user_bias)

    # movie_input = Input(shape=[1])
    # movie_vec = Embedding(n_movies+1, latent_dim, embeddings_initializer="random_normal")(movie_input)
    # movie_vec = Flatten()(movie_vec)
    # movie_bias = Embedding(n_movies+1, 1, embeddings_initializer="zeros")(movie_input)
    # movie_bias = Flatten()(movie_bias)

    # r_hat = Dot(axes=1)([user_vec,movie_vec])
    # r_hat = Add()([r_hat,user_bias,movie_bias])

    # model = Model([user_input, movie_input], r_hat)
#---------------------------------------------------------------------------------------------------------------------------------------
    # #movie embedding output
    # intermediate_layer_model = Model(input=movie_input,
    #                                  output=movie_vec)
    # intermediate_output = intermediate_layer_model.predict(movieID)
    # print intermediate_output
    # print intermediate_output.shape
#---------------------------------------------------------------------------------------------------------------------------------------
    # #MODEL-DNN
    # user_input = Input(shape=[1])
    # user_vec = Embedding(n_users+1, latent_dim, embeddings_initializer="random_normal")(user_input)
    # user_vec = Flatten()(user_vec)

    # movie_input = Input(shape=[1])
    # movie_vec = Embedding(n_movies+1, latent_dim, embeddings_initializer="random_normal")(movie_input)
    # movie_vec = Flatten()(movie_vec)

    # vec_inputs = Concatenate()([user_vec, movie_vec])
    # model = Dense(1024, activation='relu')(vec_inputs)
    # model = Dropout(0.5)(model)
    # model = Dense(512, activation='relu')(model)
    # model = Dropout(0.5)(model)
    # model = Dense(256, activation='relu')(model)
    # model = Dropout(0.5)(model)
    # model = Dense(128, activation='relu')(model)
    # model = Dropout(0.5)(model)
    # model = Dense(32, activation='relu')(model)
    # model = Dropout(0.5)(model)
    # model = Dense(16, activation='relu')(model)
    # model = Dropout(0.5)(model)
    # model_out = Dense(1, activation='linear')(model)

    # model = Model([user_input, movie_input], model_out)
#---------------------------------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------------------------------
    # model.compile(loss='mse', optimizer='rmsprop', metrics=[movies_class.root_mean_squared_error])

    # checkpoint = ModelCheckpoint(filepath=os.path.join(store_path,'best_model.h5'), 
    #                             verbose=1, 
    #                             save_best_only=True, 
    #                             monitor='val_root_mean_squared_error', 
    #                             mode='min')

    # earlystopping = EarlyStopping(monitor='val_root_mean_squared_error', 
    #                             patience = 10, 
    #                             verbose=1, 
    #                             mode='min')

    # model.fit([X1_train, X2_train], Y_train, validation_data=([X1_val, X2_val], Y_val), 
    #                                         batch_size=batch_size, 
    #                                         epochs=epochs, 
    #                                         callbacks=[earlystopping,checkpoint])

    # plot_model(model,to_file=os.path.join(store_path,'model.png'))
#---------------------------------------------------------------------------------------------------------------------------------------

    #model = load_model(os.path.join(store_path,'best_model.h5'), custom_objects={'root_mean_squared_error': movies_class.root_mean_squared_error})
    model = load_model(('./model/best_model.h5'), custom_objects={'root_mean_squared_error': movies_class.root_mean_squared_error})
    Y_test = model.predict([test_users, test_movies])

    #Y_test = Y_test*rating_SD+rating_mean
    
    test_output = test_data
    test_output['Rating'] = Y_test
    test_output = test_output.drop('MovieID', 1)
    test_output = test_output.drop('UserID', 1)
   
    test_output.to_csv(sys.argv[2], sep=',' , index=False)
