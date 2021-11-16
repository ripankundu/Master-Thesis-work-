# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 21:10:31 2021

@author: USER
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import IsolationForest
from sklearn.metrics import make_scorer, f1_score
from sklearn import model_selection
from scipy import stats
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score, roc_auc_score
from sklearn.metrics import precision_recall_curve, confusion_matrix, average_precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,roc_curve,auc,recall_score
from sklearn.model_selection import cross_val_score

    
#read the data from CSV file
df1 = pd.read_csv(r"C:\Users\USER\.spyder-py3\anaomaly_new\New folder/Attack_data_01to20.csv")
df2 = pd.read_csv(r"C:\Users\USER\.spyder-py3\anaomaly_new\New folder/Attack_data_19to39.csv")
df3 = pd.read_csv(r"C:\Users\USER\.spyder-py3\anaomaly_new\New folder/Normal data for bus upto 16.csv")
df4 = pd.read_csv(r"C:\Users\USER\.spyder-py3\anaomaly_new\New folder/Normal data for bus upto 39.csv")
df1.describe()
df1.mean()
df_fault=df2.drop('Time', axis=1)# Droping the time axis from datatset
df_normal=df4.drop('Time', axis=1)# Droping the time axis from datatset
#Concaating the noise dataset with the timeseries
fault_data=pd.concat([df1,df_fault], axis = 1)
normal_data=pd.concat([df3,df_normal], axis = 1)

dataf1=fault_data.loc[:,['Bus21']]
dataf2=fault_data.loc[:,['Bus22']]


#only check the faulty data mean and standard deviation
only_f_std_mean=dataf1.loc[6011:6026,:]

only_f_std_mean.describe()

#Describe the data properties
dataf1.describe()
dataf2.describe()


Time_take=fault_data.loc[:,['Time']]
Time_data=Time_take.loc[42010:60015,:]

Time_data.columns = range(len(Time_data.columns))
Time_data=Time_data.reset_index()
Time_data=Time_data.drop(['index'],axis=1)








#######Starting the false data in the dataset#######
df_train1=fault_data


train=df_train1.loc[0:60015,:]




df_time = train.loc[:,['Time']]#Define the time series from the data

df_new=train.drop('Time', axis=1)#New data for false data injection

#This is the value for only FDIA is injected
#mu, sigma = 0.030757, 0.100132#set the mean and variance for the noise

#Calculating the mean and standand deviation from the dataset for false data injection
mu, sigma = 0.282876, 0.0997#set the mean and variance for the noise

noise = np.random.normal(mu, sigma, [17,2]) #Noise generation for the dataset
#injection of false data for the specific colum for the specific position of the dataset
noise_est=df_new.loc[59910:59926, ['Bus21','Bus22']]

signal = noise_est - noise


#Adding the false data to the specific colum for the specific position of the dataset 

df_new.loc[59910:59926, ['Bus21','Bus22']] = signal

all_noise2=df_new

#Concaating the false dataset with the timeseries
all_noise2=pd.concat([all_noise2,df_time], axis = 1)
cols = all_noise2.columns.tolist()
cols = cols[-1:] + cols[:-1]
all_noise2= all_noise2[cols]
#all_noise2.to_csv("output_filename.csv", index=False)


plot_noise_data=all_noise2.loc[42010:60015, ['Time','Bus21','Bus22']]

reduced_plot_noise_data=all_noise2.loc[58010:60015, ['Time','Bus21','Bus22']]

all_noise=all_noise2.loc[:,['Bus21','Bus22']]
# #Now make the labels for the test dataset 

# m = 108000 # zeros
# n = 4 # ones

# label_anomaly = []
# while m + n > 0:
#     if (m > 0 and random.random() < float(m)/float(m + n)):
#         label_anomaly.append(0)
#         m -= 1
#     else:
#        label_anomaly.append(1)
#        n -= 1
listofzeros = [0] * 60015
df_label = pd.DataFrame(listofzeros)
all_noise=pd.concat([all_noise,df_label], axis = 1)
cols = all_noise.columns.tolist()
cols = cols[-1:] + cols[:-1]
all_noise= all_noise[cols]
# all_noise2.to_csv("output_filename.csv", index=False)
all_noise.rename( columns={0 :'labels'}, inplace=True )
all_noise.labels.value_counts()

to_modify=all_noise.loc[:,['labels']]

#To check the value one from the test dataset

# pos_one=all_noise2.loc[all_noise2['labels']==1]
# one_values_index=list(pos_one.index)
# print(one_values_index)
#all_noise2.at[40,'labels'] = 1
#all_noise.iloc[[6011,6012,6013,6014,6015,6016,6017,6018,6019,6020,6021,6022,6023,6024,6025,6026,58013,58014,58015,58016,58017,58018,58019,58020,58021,58022,58023,58024,58025,58026,59910,59911,59912,59913,59914,59915,59916,59917,59918,59919,59920,59921,59922,59923,59924,59925,59926], 0] = 1



all_noise.iloc[[6011,6012,6013,6014,6015,6016,6017,6018,6019,6020,6021,6022,6023,6024,6025,6026,58013,58014,58015,58016,58017,58018,58019,58020,58021,58022,58023,58024,58025,58026], 0] = 1
all_noise.iloc[[59910,59911,59912,59913,59914,59915,59916,59917,59918,59919,59920,59921,59922,59923,59924,59925,59926], 0] =2

dgl = all_noise['labels'].astype(str).values.tolist()

#Encoding the categorial value to integer number for the testing dataset
label_encoder = LabelEncoder()

y_test_new= label_encoder.fit_transform(dgl)#convert string to numreic value

y_test1= all_noise.loc[:,['labels']]




#y_test= all_noise.loc[:,['labels']]

data=all_noise.drop('labels', axis=1)#

all_noise.drop('labels', axis=1)#

train_data=all_noise.drop('labels', axis=1)#

x=data
#X1=train_data.drop('Time', axis=1)#
X=x.to_numpy(dtype='float', na_value=np.nan)
y1=y_test1
y2=y1.to_numpy(dtype='int', na_value=-1)
y=y2.flatten() 


col_names = x.columns

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.45,shuffle = True)

scaler = MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(x_train))
X_test = pd.DataFrame(scaler.transform(x_test))


# scaler = MinMaxScaler()
# x_scaled = scaler.fit_transform(x)


# x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size = 0.3, shuffle = True)


#Define the model for the Ensemble methods 

knn=KNeighborsClassifier()
svc=SVC(kernel='rbf', gamma=0.5, C=0.1)
lr=LogisticRegression()
dt=DecisionTreeClassifier()
gnb=GaussianNB()
rfc=RandomForestClassifier()
xgb=xgb.XGBClassifier()
gbc=GradientBoostingClassifier()
ada=AdaBoostClassifier()


#Combine the model outputs from the ensemble learners
models=[]
models.append(('KNeighborsClassifier',knn))
models.append(('SVC',svc))
models.append(('LogisticRegression',lr))
models.append(('DecisionTreeClassifier',dt))
models.append(('GaussianNB',gnb))
models.append(('RandomForestClassifier',rfc))
models.append(('XGBClassifier',xgb))
models.append(('Adaboostclssifier',ada))
models.append(('GradientBoostingClassifier',gbc))




#Ensemble learners and score generation

Model=[]
score=[]
cv=[]

for name,model in models:
    print('*****************',name,'*******************')
    print('\n')
    Model.append(name)
    model.fit(x_train,y_train)
    print(model)
    pre=model.predict(x_test)
    print('\n')
    AS=accuracy_score(y_test,pre)
    print('Accuracy_score  -',AS)
    score.append(AS*100)
    print('\n')
    sc=cross_val_score(model,x,y,cv=20,scoring='accuracy').mean()
    print('cross_val_score  -',sc)
    cv.append(sc*100)
    print('\n')
        
    print('classification report\n',classification_report
          (y_test,pre))
    print('\n')
    cm=confusion_matrix(y_test,pre)
    print(cm)
    print('\n')
    plt.figure(figsize=(10,40))
    plt.subplot(911)
    # sns.heatmap(cm, columns = ['Actual', 'Predicted']),
    #             xticklabels=['Normal [0]', 'Fault [1]', 'Cyber attack [2]'], 
    #             yticklabels=['Normal [0]', 'Fault [1]', 'Cyber attack [2]'], 
    #             annot=True, fmt="d", linewidths=.5, cmap="YlGnBu",fontsize=18)
    # plt.ylabel('Predicted class',fontsize=18)
    # plt.xlabel('Actual class',fontsize=18)
    # #plt.savefig(r"C:\Users\USER\.spyder-py3\anaomaly_new\Thesis\IF_graph/IF_new_model_performance.eps", format='eps', dpi = 1000)
    # plt.show()

    sns.heatmap(cm, xticklabels=["Normal [0]", "Fault [1]", "Cyber attack [2]"], yticklabels=["Normal [0]", "Fault [1]", "Cyber attack [2]"], annot=True, fmt="d", linewidths=.5, cmap="YlGnBu" );
    plt.title("Confusion matrix of Isolation forest",fontsize=18)
    plt.ylabel('True class',fontsize=18)
    plt.xlabel('Predicted class',fontsize=18)
    plt.title(name)
    # # print(sns.heatmap(cm,annot=True))
    
    

    result=pd.DataFrame({'Model':Model,'Accuracy_score':score,'Cross_val_score':cv})
    result