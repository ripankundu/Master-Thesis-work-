# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 22:22:17 2021

@author: Ripan Kumar Kundu
Master Thesis student at University of Rostock
"""

#Import the Librarry 
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
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

#read the data from CSV file
#First dataset with the Three phase fault
df1 = pd.read_csv(r"C:\Users\USER\.spyder-py3\anaomaly_new\New folder/Attack_data_01to20.csv")
df2 = pd.read_csv(r"C:\Users\USER\.spyder-py3\anaomaly_new\New folder/Attack_data_19to39.csv")


#Second one with no fault
df3 = pd.read_csv(r"C:\Users\USER\.spyder-py3\anaomaly_new\New folder/Normal data for bus upto 16.csv")
df4 = pd.read_csv(r"C:\Users\USER\.spyder-py3\anaomaly_new\New folder/Normal data for bus upto 39.csv")


df_fault=df2.drop('Time', axis=1)# Droping the time axis from datatset
df_normal=df4.drop('Time', axis=1)# Droping the time axis from datatset

#Concaating the dataset to make one dataset for the normal and fault dataset
fault_data=pd.concat([df1,df_fault], axis = 1)
normal_data=pd.concat([df3,df_normal], axis = 1)


#Now check the mean and standard deviation of the fault Bus node
dataf1=fault_data.loc[:,['Bus21']]
dataf2=fault_data.loc[:,['Bus22']]


#only check the faulty data mean and standard deviation
only_f_std_mean=dataf1.loc[6011:6026,:]

only_f_std_mean.describe()

#Describe the data properties of the Bus node 21 and 22
dataf1.describe()
dataf2.describe()


Time_take=fault_data.loc[:,['Time']]
Time_data=Time_take.loc[42010:60015,:]

Time_data.columns = range(len(Time_data.columns))
Time_data=Time_data.reset_index()
Time_data=Time_data.drop(['index'],axis=1)








#######Starting the false data injection attack ( Deductive attack) in low voltage in the dataset#######
df_train1=fault_data


train=df_train1.loc[0:60015,:]




df_time = train.loc[:,['Time']]#Define the time series from the data

df_new=train.drop('Time', axis=1)#New data for false data injection

#This is the value for only FDIA is injected
#mu, sigma = 0.030757, 0.100132#set the mean and variance for the noise

#Calculating the mean and standand deviation from the dataset for false data injection
mu, sigma = 0.282876, 0.0997#set the mean and variance for the false data injection attack

attack_vector = np.random.normal(mu, sigma, [17,2]) #Random attack vector geneartion from the mean and standard deviation 

#injection of false data for the specific colum for the specific position of the dataset
attack_est=df_new.loc[59910:59926, ['Bus21','Bus22']]

signal = attack_est - attack_vector


#Adding the false data to the specific colum for the specific position of the dataset 

df_new.loc[59910:59926, ['Bus21','Bus22']] = signal

all_attack2=df_new

#Concaating the false dataset with the timeseries
all_attack2=pd.concat([all_attack2,df_time], axis = 1)
cols = all_attack2.columns.tolist()
cols = cols[-1:] + cols[:-1]
all_attack2= all_attack2[cols]
#all_attack2.to_csv("output_filename.csv", index=False)


plot_attack_data=all_attack2.loc[42010:60015, ['Time','Bus21','Bus22']]

reduced_plot_attack_data=all_attack2.loc[58010:60015, ['Time','Bus21','Bus22']]

all_attack=all_attack2.loc[:,['Bus21','Bus22']]



# #Now make the labels for the test dataset  and Evaluate the Machine learning model performance 

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
all_attack=pd.concat([all_attack,df_label], axis = 1)
cols = all_attack.columns.tolist()
cols = cols[-1:] + cols[:-1]
all_attack= all_attack[cols]
# all_attack2.to_csv("output_filename.csv", index=False)
all_attack.rename( columns={0 :'labels'}, inplace=True )
all_attack.labels.value_counts()

to_modify=all_attack.loc[:,['labels']]

#To check the value one from the test dataset

# pos_one=all_attack2.loc[all_noise2['labels']==1]
# one_values_index=list(pos_one.index)
# print(one_values_index)
#all_noise2.at[40,'labels'] = 1
all_attack.iloc[[6011,6012,6013,6014,6015,6016,6017,6018,6019,6020,6021,6022,6023,6024,6025,6026,58013,58014,58015,58016,58017,58018,58019,58020,58021,58022,58023,58024,58025,58026,59910,59911,59912,59913,59914,59915,59916,59917,59918,59919,59920,59921,59922,59923,59924,59925,59926], 0] = 1




#all_attack.iloc[[59920,59921,59922], 0] =2

#dgl = all_attack['labels'].astype(str).values.tolist()

# #Encoding the categorial value to integer number for the testing dataset
# label_encoder = LabelEncoder()

# y_test= label_encoder.fit_transform(dgl)#convert string to numreic value

#y_test1= all_noise.loc[:,['labels']]




y_test= all_attack.loc[:,['labels']]

data=all_attack.drop('labels', axis=1)#

all_attack.drop('labels', axis=1)#

train_data=all_attack.drop('labels', axis=1)#

X1=data
#X1=train_data.drop('Time', axis=1)#

#Convert the dataframe to numpy for better calculation
X=X1.to_numpy(dtype='float', na_value=np.nan)

y1=y_test
y2=y1.to_numpy(dtype='int', na_value=-1)
y=y2.flatten() 




#Scaling the data using standard scaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(X)

####Split into train and test 
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size = 0.3, shuffle = False)
# #data=all_attack2.drop('Time', axis=1)#
# fault=all_attack2.loc[:,['Time','Bus21','Bus22']]

# #X_train, X_test, y_train, y_test = train_test_split(, y, test_size = .3, random_state = 123)
# #fault=data
# #Detection based on prediction/classification based on binary values
# #Where -1 indicate anomalous and 1 indicate normal 
# fault.columns
# #specify the 12  column names to be modelled
# to_model_columns=fault.columns





######First of all performing the prediction to calcaulate the anomalous datapoints using Isolation forest######
#Initialize the model 

clf=IsolationForest(n_estimators=400, max_samples='auto', contamination=float(.00207), \
                        max_features=2, bootstrap=False, n_jobs=-1, random_state=42, verbose=0)

#Train the model with the dataset
clf.fit(x_train)


#Now Evaluate the model
pred = clf.predict(x_test)
df = pd.DataFrame(pred)
preeed=df

df['anomaly']=preeed
pred = df['anomaly'].map( {1: 0, -1: 1} )
df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})

pred_roc=df['anomaly']




#Calculate the anomaly score of anomalous data 

# 0.00135 we get 24 and 7 TP  
#Initialize the model 

clf=IsolationForest(n_estimators=400, max_samples='auto', contamination=float(.00207), \
                        max_features=2, bootstrap=False, n_jobs=-1, random_state=42, verbose=0)

#Train the model with the dataset
clf.fit(x_train)


pred_score=clf.decision_function(x_test)

df_score = pd.DataFrame(pred_score)
df_score= df_score.abs()

#Now check the null values####
for column in df_score.columns.values.tolist():
    print(column)
    print (df_score[column].value_counts())
    print("")




# data_test=pd.DataFrame(x_test)
# preeed_score=df_score

# df_score['anomaly_score']=preeed_score
# df_score['level']=data_test

# anomaly['scores']=pred_score
# print(fault['scores'].value_counts())

# fault.head(20)

# #Ploting the score values in the reduced dataset 
# score_anomaly=fault
# df_scores = score_anomaly.loc[:,['Time','scores']]
# df_scores.columns = ['Time in seconds', 'Anomaly Scores']
# #s2=df_scores.loc[100:370,:]
# s2=df_scores


# #Plotting the anomaly scores
# sns.lmplot('Time in seconds', 'Anomaly Scores', s2, fit_reg=False).set(title='Isolation forest based anomaly scores')
# fig = plt.gcf()
# fig.set_size_inches(15, 10)
# plt.show()



# outliers=x_test.loc[df['anomaly']==1]
# outlier_index=list(outliers.index)
# #print(outlier_index)
# #Find the number of anomalies and normal points here points classified -1 are anomalous
# print(fault['anomaly'].value_counts())


# anomaly=fault.loc[fault['anomaly']==1]
# anomaly_index=list(anomaly.index)
# print(anomaly)
# fault.anomaly.value_counts()



####Now calculate the Confusion matrix to check the TP,FP,TN,FN of the isolation forest
from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(y_test,pred_roc)

plt.figure(figsize=(5, 5))
sns.heatmap(cnf_matrix, xticklabels=["Normal","Abnormal"], yticklabels=["Normal","Abnormal"], annot=True, fmt="d");
#plt.title("Confusion matrix of Isolation forest",fontsize=18)
plt.ylabel('True class',fontsize=18)
plt.xlabel('Predicted class',fontsize=18)
#plt.savefig(r"C:\Users\USER\.spyder-py3\anaomaly_new\Thesis\IF_graph/IF_CNF_best_without_title.eps", format='eps', dpi = 1000)
#plt.savefig('confusion_matrix_IF.eps', format='eps')
plt.show()


# #####Plot with prediction#######

pred_roc=df['anomaly']

y=scaler.inverse_transform(x_test)
y_plot = pd.DataFrame(y)
anomalyscore = pd.DataFrame(pred_score)
column_indices = [0,1]
new_names = ['Bus21','Bus22']
old_names = y_plot.columns[column_indices]
y_plot.rename(columns=dict(zip(old_names, new_names)), inplace=True)
bus21_plot=y_plot.loc[:,['Bus21']]
bus22_plot= y_plot.loc[:,['Bus22']]
df['Bus21'] = bus21_plot
df['Bus22'] =bus22_plot
df['Anomaly_score']=anomalyscore
df['Time'] = Time_data


# ###### Plotting data#######

fig, ax = plt.subplots(figsize=(10,6))

a = df.loc[df['anomaly'] == 1, ['Time','Bus21']] #anomaly

ax.plot(plot_attack_data['Time'], plot_attack_data['Bus21'], color='blue', label ='Normal')
ax.scatter(a['Time'],a['Bus21'], color='red', label = 'Anomaly')
plt.xlabel('Time in seconds',fontsize=18)
plt.ylabel('Voltage magnitude (pu)',fontsize=18)
plt.title('Isolation forest identified anomolous dataponts',fontsize=18)
plt.legend()
#plt.savefig(r"C:\Users\USER\.spyder-py3\anaomaly_new\Thesis\IF_graph/IF_identified_all_data_with_title.eps", format='eps', dpi = 1000)
plt.show()

####### Reduced plot to see the better visualization######
reduced_df=df.loc[16000:18005,:]


fig, ax = plt.subplots(figsize=(10,6))

a = reduced_df.loc[df['anomaly'] == 1, ['Time','Bus21']] #anomaly

ax.plot(reduced_plot_attack_data['Time'], reduced_plot_attack_data['Bus21'], color='blue', label ='Normal')
ax.scatter(a['Time'],a['Bus21'], color='red', label = 'Anomaly')
plt.xlabel('Time in seconds',fontsize=18)
plt.ylabel('Voltage magnitude (pu)',fontsize=18)
plt.title('Isolation forest identified anomolous dataponts',fontsize=18)
plt.legend()
#plt.savefig(r"C:\Users\USER\.spyder-py3\anaomaly_new\Thesis\IF_graph/IF_identified_data_reduced_without_title.eps", format='eps', dpi = 1000)
plt.show()


#ROC Curve to check further model performance in the anomalous data points (Cyber attack and three phase fault)

false_pos_rate, true_pos_rate, thresholds = roc_curve(y_test,pred_roc)
roc_auc = auc(false_pos_rate, true_pos_rate,)

plt.plot(false_pos_rate, true_pos_rate, linewidth=5, label='AUC = %0.3f'% roc_auc)
plt.plot([0,1],[0,1], linewidth=5)

plt.xlim([-0.01, 1])
plt.ylim([0, 1.01])
plt.legend(loc='lower right',fontsize=18)
plt.title('Receiver operating characteristic curve (ROC)',fontsize=18)
plt.ylabel('True Positive Rate',fontsize=18)
plt.xlabel('False Positive Rate',fontsize=18)
#plt.savefig(r"C:\Users\USER\.spyder-py3\anaomaly_new\Thesis\IF_graph/IF_AUC_title.eps", format='eps', dpi = 1000)

plt.show()


#####check this section deeply#######

precision_rt, recall_rt, threshold_rt = precision_recall_curve(y_test,pred_roc)
plt.plot(recall_rt, precision_rt, linewidth=5, label='Precision-Recall curve')
plt.title('Recall vs Precision',fontsize=18)
plt.xlabel('Recall',fontsize=18)
plt.ylabel('Precision',fontsize=18)
plt.show()


plt.plot(threshold_rt, precision_rt[1:], label="Precision",linewidth=5)
plt.plot(threshold_rt, recall_rt[1:], label="Recall",linewidth=5)
plt.title('Precision and recall for different threshold values')
plt.xlabel('Threshold')
plt.ylabel('Precision/Recall')
plt.legend()
#plt.savefig('PrecisionRecall.eps', format='eps')
plt.show()

#####################

#Classification report to check the precision, recall, F1 score of the model ####
print(classification_report(y_test,pred_roc))