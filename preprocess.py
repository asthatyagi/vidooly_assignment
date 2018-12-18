
# coding: utf-8

# In[137]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler # Used for scaling of data
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from keras import backend as K
from keras.wrappers.scikit_learn import KerasRegressor




df_train = pd.read_csv('./train.csv')
df_train.drop(['vidid','published'],axis=1,inplace=True)



df_test=pd.read_csv('./test.csv')
df_test.drop(['vidid','published'],axis=1,inplace=True)




print(df_test.describe())





print(df_train.describe())




print(df_train.dtypes)




dataset= df_train.values
print(df_train.head())



#correlation matrix
corrmat = df_train.corr()
sns.heatmap(corrmat, square=True)
plt.show()


"""
k = 6 #number of variables for heatmap
cols = corrmat.nlargest(k, 'adview')['adview'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 6}, yticklabels=cols.values, xticklabels=cols.values)

"""


#missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head())


#convert category to ascii
#df_train['category'] = df_train['category'].apply(ord)
df_train['category'].dtypes
df_train['category'] = df_train['category'].map(ord)
print(df_train['category'].head())



print(df_train['duration'].head())
df_train['duration'].astype('str')




#convert duration
j = df_train['duration']

dur=[]
for a in df_train.duration:
    dur.append(a[2:])


dur_sec=[]
for t in dur:
    h=t.find('H')
    m=t.find('M')
    s=t.find('S')
    sec=0
    if h>=0:
        sec=sec+int(t[:h])*60*60
        if m>=0:
            sec=sec+int(t[h+1:m])*60
            if s>=0:
                sec=sec+int(t[m+1:s])
    elif m>=0:
        sec=sec+int(t[:m])*60
        if s>=0:
            sec=sec+int(t[m+1:s])
    else:
        sec=sec+int(t[:s])
        
    dur_sec.append(sec)


df_train['duration']=dur_sec
df_train.drop(['duration1'],axis=1,inplace=True)



df_train=df_train[df_train['dislikes']!='F']
df_train=df_train[df_train['likes']!='F']
df_train=df_train[df_train['views']!='F']
df_train=df_train[df_train['comment']!='F']
pd.to_numeric(df_train['dislikes'])
pd.to_numeric(df_train['likes'])
pd.to_numeric(df_train['views'])
pd.to_numeric(df_train['comment'])


print(df_train.dtypes)
df_train.to_csv('./clean_train.csv',index=False)


print(df_train.head())


#correlation matrix
corrmat = df_train.corr()
sns.heatmap(corrmat, square=True)
plt.show()




