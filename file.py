#from itertools import product
import numpy as np
import pandas as pd
#from sklearn.preprocessing import LabelEncoder ,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt
import csv
import pickle
#***************************Data Preparation*****************************
df_names=['sep_len', 'sep_wid', 'pet_len', 'pet_wid', 'class']
df=pd.read_csv('iris_Missing.csv',names=df_names)
imp=SimpleImputer(missing_values=np.nan,strategy='mean')
df.sep_len=imp.fit_transform(df['sep_len'].values.reshape(-1,1))[:,0]
df.sep_wid=imp.fit_transform(df['sep_wid'].values.reshape(-1,1))[:,0]
df.pet_len=imp.fit_transform(df['pet_len'].values.reshape(-1,1))[:,0]
df.pet_wid=imp.fit_transform(df['pet_wid'].values.reshape(-1,1))[:,0]


#***************************Feature Engineering*****************************
Sep_Features=df.iloc[:,0:2].values
Pet_Features=df.iloc[:,2:4].values
labels=df.iloc[:,4].values
S_pca=PCA(n_components=1)
Sep_PCA=S_pca.fit_transform(Sep_Features)
P_pca=PCA(n_components=1)
Pet_PCA=P_pca.fit_transform(Pet_Features)
new_names=['Sepal','Petal','class']
new_data=np.array([Sep_PCA[:,0],Pet_PCA[:,0],labels]).T
new_iris_df=pd.DataFrame(new_data,columns=new_names)
new_iris_df.to_csv('new_DF.csv',index=False)


#***************************Modeling*****************************
new_data_set=pd.read_csv('new_DF.csv')
x=new_data_set.iloc[:,0:2].values
y=new_data_set.iloc[:,2].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
Cls=SVC()
Cls.fit(x_train,y_train)
y_predicted=Cls.predict(x_test)
print(y_predicted)
#***************************Performance Measure*****************************
print('Accuracy : %.2f %%' % (metrics.accuracy_score(y_test,y_predicted)*100))
#pickle.dump(Cls,open('Finalized_model.pickel','wb'))