import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from keras import backend as K
from keras.wrappers.scikit_learn import KerasRegressor
import re, string
import os
from google.colab import drive



from keras import backend as K
K.tensorflow_backend._get_available_gpus()


df_train = pd.read_csv('./clean_train.csv')
df_test= pd.read_csv('./clean_test.csv')

#df_train.describe()
#df_train.head()
print(df_test.dtypes)

#normal distribution
#histogram and normal probability plot
from scipy import stats 
from scipy.stats import norm
sns.distplot(df_train['adview'],fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['adview'], plot=plt)
plt.show()

# Always standard scale the data before using NN
scale = StandardScaler()

X_train = df_train[['views', 'likes', 'dislikes', 'comment','duration']]
X_train = scale.fit_transform(X_train)

y = df_train['adview'].values
seed = 7
np.random.seed(seed)
# split into 80% for train and 20% for test
X_train, X_test, y_train, y_test = train_test_split(X_train, y, test_size=0.1, random_state=seed)


#trying to see which algorithm is better
"""    
# model = RandomForestRegressor(n_estimators=150, max_features='sqrt', n_jobs=-1) 
models = [LinearRegression(),
              RandomForestRegressor(n_estimators=1500, max_features='sqrt'), 
              SVR(kernel='linear') ]
 
TestModels = pd.DataFrame()
tmp = {}
 
for model in models:
        # get model name
    m = str(model)
    tmp['Model'] = m[:m.index('(')]
        # fit model on training dataset
    model.fit(X_train, y_train)
        # predict prices for test dataset and calculate r^2
    tmp['R2_Price'] = r2_score(y_test, model.predict(X_test))
        # write obtained data
    TestModels = TestModels.append([tmp])
 
TestModels.set_index('Model', inplace=True)
 
fig, axes = plt.subplots(ncols=1, figsize=(10, 4))
TestModels.R2_Price.plot(ax=axes, kind='bar', title='R2_Price')
plt.show()
"""

#start with linear regression
"""from sklearn.linear_model import LinearRegression

reg=LinearRegression()
reg.fit(X_train,y_train)
reg.score(X_train,y_train)

pred=reg.predict(X_test)
X_test
"""

# Neural Network
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras import regularizers
from keras.layers.advanced_activations import LeakyReLU
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

#model
"""
model = Sequential()
model.add(Dense(16,input_dim=X_train.shape[1],activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(32,activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization(axis=-1))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2048,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')

model.summary()

history = model.fit(X_train,y_train, validation_data=(X_test,y_test),verbose=2,epochs=1000,batch_size=32)
"""
#xg boost
import xgboost
xgb = xgboost.XGBRegressor(n_estimators=1000, learning_rate=0.0001, gamma=0, subsample=0.75,colsample_bytree=1, max_depth=7)
xgb.fit(X_train,y_train)
predictions1 = xgb.predict(X_test)

#Finding RMSE
from sklearn.metrics import mean_squared_error
import math
testScore=math.sqrt(mean_squared_error(y_test,predictions1))
print(testScore1)

df_test.drop('category',axis=1 ,inplace=True)
df_test.drop('adview',axis=1 ,inplace=True)

df_test.head()
df_test_x = df_test.drop('vidid',axis=1)
df_test_x = scale.fit_transform(df_test_x)
predictions = xgb.predict(df_test_x)

df_test['adview']= predictions
df_test[df_test['adview']<0]=0
df_test.to_csv('./prediction_final.csv')

#10-fold Cross Validation for counteract overfitting
from sklearn.model_selection import cross_val_score
accuracies1=cross_val_score(estimator=xgb, X=X_train, y=y_train, cv=10)
accuracies1.mean()
accuracies1.std()

print("Cross Validation Accuracy : ",round(accuracies1.mean()* 100 , 2 ),"%")

#tried with random forest but xgboost performed better

 """

from sklearn.tree import DecisionTreeRegressor as dtr
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
 

rfr=RandomForestRegressor(n_estimators=1500, max_depth=20,criterion='mse',n_jobs=-1,min_samples_leaf=3,min_samples_split=5)

rfr.fit(X_train,y_train)
rfr.score(X_test,y_test)
scores_rfr = cross_val_score(rfr,X_train,y_train,cv=10,scoring='explained_variance')

#     print('explained variance scores for k=10 fold validation:',scores_rfr)
print("Est. explained variance: %0.2f (+/- %0.2f)" % (scores_rfr.mean(), scores_rfr.std() * 2))
"""


