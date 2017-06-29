import numpy as np 
import pandas as pd 
import scipy
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout
from subprocess import check_output

print(check_output(["ls", "./input"]).decode("utf8"))

import warnings
warnings.filterwarnings('ignore')


macro = pd.read_csv('./input/macro.csv')
train = pd.read_csv('./input/train.csv')
test = pd.read_csv('./input/test.csv')

X_list_num = ['full_sq', 'num_room', 'floor', 'timestamp',
              'preschool_education_centers_raion', 'school_education_centers_raion', 
              'children_preschool', 'children_school',
              'shopping_centers_raion', 'healthcare_centers_raion', 
              'office_raion', 'sport_objects_raion',
              'metro_min_walk', 'public_transport_station_min_walk', 
              'railroad_station_walk_min', 'cafe_count_500',
              'kremlin_km', 'workplaces_km', 'ID_metro',
              'public_healthcare_km', 'kindergarten_km', 'school_km', 'university_km', 
              'museum_km', 'fitness_km', 'park_km', 'shopping_centers_km',
              'additional_education_km', 'theater_km', 
              'raion_popul', 'work_all', 'young_all', 'ekder_all']

features_train = train[X_list_num]
features_test = test[X_list_num]
target_train = train['price_doc']

print ("Sberbank Russian Housing Dataset Statistics: \n")
print ("Number of houses = ", len(target_train))
print ("Number of features = ", len(list(features_train.keys())))
print ("Minimum house price = ", np.min(target_train))
print ("Maximum house price = ", np.max(target_train))
print ("Mean house price = ", "%.2f" % np.mean(target_train))
print ("Median house price = ", "%.2f" % np.median(target_train))
print ("Standard deviation of house prices =", "%.2f" % np.std(target_train))

df = pd.DataFrame(features_train, columns=X_list_num)
df['prices'] = target_train

df = df.dropna(subset=['num_room'])

df['metro_min_walk'] = df['metro_min_walk'].interpolate(method='linear')
features_test['metro_min_walk'] = features_test['metro_min_walk'].interpolate(method='linear')

df['railroad_station_walk_min'] = df['railroad_station_walk_min'].interpolate(method='linear')
features_test['railroad_station_walk_min'] = features_test['railroad_station_walk_min'].interpolate(method='linear')

df['floor'] = df['floor'].fillna(df['floor'].median())

ID_metro_cat = pd.factorize(df['ID_metro'])
df['ID_metro'] = ID_metro_cat[0]

ID_metro_pairs = dict(zip(list(ID_metro_cat[1]), list(set(ID_metro_cat[0]))))
ID_metro_pairs[224] = 219
features_test['ID_metro'].replace(ID_metro_pairs,inplace=True)

usdrub_pairs = dict(zip(list(macro['timestamp']), list(macro['usdrub'])))

df['timestamp'].replace(usdrub_pairs,inplace=True)
features_test['timestamp'].replace(usdrub_pairs,inplace=True)

df.rename(columns={'timestamp' : 'usdrub'}, inplace=True)
features_test.rename(columns={'timestamp' : 'usdrub'}, inplace=True)

target_train = df['prices'].as_matrix()
features_train = df.drop('prices', 1).as_matrix()


def mlp_model():
    model = Sequential()
    
    model.add(Dense(128, input_dim=33, kernel_initializer='normal', activation='relu'))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    
    model.add(Dropout(0.1))
    
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    
    model.add(Dropout(0.1))
    
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    model.add(Dense(32, kernel_initializer='normal', activation='relu')) 
 
    model.add(Dense(1, kernel_initializer='normal'))

    model.summary()

    model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])
    return model

mlp_model = mlp_model()

mlp_history = mlp_model.fit(X_train, y_train, validation_data=(X_test, y_test),
                            nb_epoch=80, batch_size=16, verbose=1)

scale = StandardScaler()
features_train = scale.fit_transform(features_train)
features_test = scale.fit_transform(features_test.as_matrix())


reg1 = BaggingRegressor(n_estimators=10)
reg1.fit(features_train, target_train)

target_test_predict1 = reg1.predict(features_test)
target_test_predict2 = mlp_model.predict(features_test)


pd.set_option('display.float_format', lambda x: '%.2f' % x)
target_predict1 = ["{0:.2f}".format(x) for x in target_test_predict1]
target_predict2 = ["{0:.2f}".format(float(x)) for x in target_test_predict2]

submission = pd.DataFrame({"id": test['id'], "price_doc": target_predict1})
submission.to_csv('result_Bagging_1.csv', index=False)

submission = pd.DataFrame({"id": test['id'], "price_doc": target_predict2})
submission.to_csv('result_DNNmodel1_1.csv', index=False)
