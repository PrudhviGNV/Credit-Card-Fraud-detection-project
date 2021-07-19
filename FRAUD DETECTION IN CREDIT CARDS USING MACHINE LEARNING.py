#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Statements
import os
import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pylab import rcParams
#The figure size in width, height in inches
rcParams['figure.figsize'] = 14, 8
#The resolution of figure dots-per-inch.
#rcParams['figure.dpi']=110.0
#The Random is used to generate random numbers every time of execution. So seed function is used to save the state of the random function, so that it can generate same random numbers on multiple execution of the code on the same machine or different machines.
RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]


# In[2]:


#importing the Dataset
data = pd.read_csv('C:/Users/venkataramayya/Desktop/Main Project/creditcard.csv',sep=',')
data.head()


# In[3]:


data.info()


# # Exploratory Data Analysis

# In[4]:


data.isnull().values.any()


# In[5]:


data['Class'].unique()


# In[6]:


count_classes = pd.value_counts(data['Class'], sort = True)

count_classes.plot(kind = 'bar', rot=2)

plt.title("Transaction Class Distribution")

plt.xticks(range(2), LABELS)

plt.xlabel("Class")

plt.ylabel("Frequency")


# In[7]:


## Get the Fraud and the normal dataset 

fraud = data[data['Class']==1]

normal = data[data['Class']==0]


# In[8]:


print(fraud.shape,normal.shape)


# In[9]:


## We need to analyze more amount of information from the transaction data
#How different are the amount of money used in different transaction classes?

fraud.Amount.describe()


# In[10]:


normal.Amount.describe()


# In[11]:


#corr() is used to find the pairwise correlation of all columns in the dataframe.
#Any na values are automatically excluded. For any non-numeric data type columns in the dataframe it is ignored.

fraud.corr()


# In[12]:


normal.corr()


# In[13]:


#Covariance provides the a measure of strength of correlation between two variable or more set of variables. The covariance matrix element Cij is the covariance of xi and xj. The element Cii is the variance of xi.

#If COV(xi, xj) = 0 then variables are uncorrelated
#If COV(xi, xj) > 0 then variables positively correlated
#If COV(xi, xj) > < 0 then variables negatively correlated


# In[14]:


fraud.cov()


# In[15]:


normal.cov()


# In[16]:


f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Amount per transaction by class')
bins = 50
ax1.hist(fraud.Amount, bins = bins)
ax1.set_title('Fraud')
ax2.hist(normal.Amount, bins = bins)
ax2.set_title('Normal')
plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.xlim((0, 20000))
plt.yscale('log')
plt.show();


# In[17]:


# We Will check Do fraudulent transactions occur more often during certain time frame ? Let us find out with a visual representation.

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time of transaction vs Amount by class')
ax1.scatter(fraud.Time, fraud.Amount)
ax1.set_title('Fraud')
ax2.scatter(normal.Time, normal.Amount)
ax2.set_title('Normal')
plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show()


# In[18]:


## Take some sample of the data

data1= data.sample(frac = 0.1,random_state=1)

data1.shape


# In[19]:


data.shape


# In[20]:


#Determine the number of fraud and valid transactions in the dataset

Fraud = data1[data1['Class']==1]
print("Fraud Cases : {}".format(len(Fraud)))

Valid = data1[data1['Class']==0]
print("Valid Cases : {}".format(len(Valid)))

outlier_fraction = len(Fraud)/float(len(Valid))
print("Outlier_Fraction : {}".format(outlier_fraction))


# In[21]:


## Correlation
import seaborn as sns
#get correlations of each features in dataset
corrmat = data1.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="YlGn")


# In[22]:


#Create independent and Dependent Features
columns = data1.columns.tolist()
# Filter the columns to remove data we do not want 
columns = [c for c in columns if c not in ["Class"]]
# Store the variable we are predicting 
target = "Class"
# Define a random state 
state = np.random.RandomState(42)
X = data1[columns]
Y = data1[target]
X_outliers = state.uniform(low=0, high=1, size=(X.shape[0], X.shape[1]))
# Print the shapes of X & Y
print(X.shape)
print(Y.shape)


# In[23]:


#Examine the target variable ‘Class’ which is the variable to predict
#Get a count of diagnosis observations by type
data.Class.value_counts()


# 0 ----> Normal
# 
# 1 ----> Fraud

# In[24]:


#creating target series
target=data['Class']
target


# In[25]:


#dropping the target variable from the data set
data.drop('Class',axis=1,inplace=True)
data.shape


# In[26]:


#converting them to numpy arrays
X=np.array(data)
y=np.array(target)
X.shape
y.shape


# In[27]:


#distribution of the target variable
k = len(y[y==1])
print(k)
j = len(y[y==0])
print(j)


# # Normalization

# In[28]:


#Apply normalization to rescale the features to a standard range of values.
#Normalize the numeric variables from column2 to column 31 in the dataframe
from sklearn import preprocessing, neighbors
minmax=preprocessing.MinMaxScaler(feature_range=(0,1))
minmax.fit(X).transform(X)


# # Splitting the Dataset into Train and Test

# In[29]:


#splitting the data set into train and test (75:25)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=1)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)


# In[30]:


X_train


# In[31]:


X_test


# In[32]:


y_train


# In[33]:


y_test


# # Model Prediction
# Now it is time to start building the model .The types of algorithms we are going to use to try to do anomaly detection on this dataset are as follows

# # Isolation Forest Classifier 

# In[34]:


#Build Isolation Forest Classifier

#Importing Isolation Forest Classifier
from sklearn.ensemble import IsolationForest

#Creating a Isolation Forest Classifier object
if_clf = IsolationForest(n_estimators=100, max_samples='auto', contamination='auto',
                         max_features=1.0, bootstrap=False, n_jobs=None, random_state=None, verbose=0, warm_start=False)

#Fitting Classifier For Training Set and Testing Set
if_clf_train = if_clf.fit(X_train, y_train)

if_clf_test = if_clf.fit(X_test, y_test) 


# In[35]:


#Predicting the Test Set results
if_y_test= if_clf.predict(X_test)
print("Test set predictions:\n {}".format(if_y_test))

print("Test set score: {:.2f}".format(np.mean(if_y_test == y_test)))

#Predicting the Train Set results
if_y_train= if_clf.predict(X_train)
print("Test set predictions:\n {}".format(if_y_train))

print("Train set score: {:.2f}".format(np.mean(if_y_train == y_train)))


# In[36]:


#Accuracy score on Test
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix, classification_report

print("\nAccuracy score: %f" %(accuracy_score(y_test, if_y_test) * 100))
#print("Recall score : %f" %(recall_score(y_test, if_y_test) * 100))
print("ROC score : %f\n" %(roc_auc_score(y_test, if_y_test) * 100))
print(confusion_matrix(y_test, if_y_test)) 
print(classification_report(y_test, if_y_test, labels=[1,0]))


# In[37]:


#Accuracy score on Train
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix

print("\nAccuracy score: %f" %(accuracy_score(y_train, if_y_train) * 100))
#print("Recall score : %f" %(recall_score(y_train, if_y_train) * 100))
print("ROC score : %f\n" %(roc_auc_score(y_train, if_y_train) * 100))
print(confusion_matrix(y_train, if_y_train))
print(classification_report(y_train, if_y_train, labels=[1,0]))


# In[38]:


#Precision score on Test and Train
from sklearn.metrics import precision_score

if_precision_test =precision_score(y_test, if_y_test, average='weighted', zero_division=1)  
print(if_precision_test)
if_precision_train =precision_score(y_train, if_y_train, average='weighted', zero_division=1)
print(if_precision_train)


# In[39]:


#Evaluate model
#train 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
def ROC_curve(title, y_train, scores, label=None):
    
    # calculate the ROC score
    fpr, tpr, thresholds = roc_curve(y_train, scores)
    print('AUC Score ({}): {:.2f} '.format(title, roc_auc_score(y_train, scores)))

    # plot the ROC curve
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, linewidth=2, label=label, color='b')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.2, 1.2])
    plt.ylim([-0.2, 1.2])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC Curve: {}'.format(title), fontsize=16)
    plt.show()


# In[40]:


#Evaluating Cross_validation_Scores And Cross_Validation_PredictionProbability

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

from sklearn.model_selection import cross_val_score, cross_val_predict

#for Train
if_acc_train = cross_val_score(if_clf_train, X_train_std, y_train, cv=3, scoring='accuracy', n_jobs=-1)
if_proba_train = cross_val_predict(if_clf_train, X_train_std, y_train, cv=3, method='predict')
if_scores_train = if_proba_train[:, ]

#for Test
if_acc_test = cross_val_score(if_clf_test, X_test_std, y_test, cv=3, scoring='accuracy', n_jobs=-1)
if_proba_test = cross_val_predict(if_clf_test, X_test_std, y_test, cv=3, method='predict')
if_scores_test = if_proba_test[:, ]


# In[41]:


#Plot ROC Curve for Train
ROC_curve('Isolation Forest Classification For Train', y_train, if_scores_train)


# In[42]:


#Plot ROC Curve for Test
ROC_curve('Isolation Forest Classification For Test', y_test, if_scores_test)


# #  Local Outlier Factor

# In[43]:


#Build Local Outlier Factor

#Importing Local Outlier Factor
from sklearn.neighbors import LocalOutlierFactor

#Creating a Local Outlier Factor object
lcf_clf = LocalOutlierFactor(n_neighbors=20, algorithm='auto', leaf_size=30, metric='minkowski',
                             novelty=True, contamination=0.1, n_jobs=None)

#Fitting Classifier For Training Set and Testing Set
lcf_clf_train = lcf_clf.fit(X_train, y_train)

lcf_clf_test = lcf_clf.fit(X_test, y_test) 


# In[44]:


#Predicting the Test Set results
lcf_y_test= lcf_clf.predict(X_test)
print("Test set predictions:\n {}".format(lcf_y_test))

print("Test set score: {:.2f}".format(np.mean(lcf_y_test == y_test)))

#Predicting the Train Set results
lcf_y_train= lcf_clf.predict(X_train)
print("Test set predictions:\n {}".format(lcf_y_train))

print("Train set score: {:.2f}".format(np.mean(lcf_y_train == y_train)))


# In[45]:


#Accuracy score on Test
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix, classification_report

print("\nAccuracy score: %f" %(accuracy_score(y_test, lcf_y_test) * 100))
#print("Recall score : %f" %(recall_score(y_test, lcf_y_test) * 100))
print("ROC score : %f\n" %(roc_auc_score(y_test, lcf_y_test) * 100))
print(confusion_matrix(y_test, lcf_y_test)) 
print(classification_report(y_test, lcf_y_test, labels=[1,0]))


# In[46]:


#Accuracy score on Train
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix, classification_report

print("\nAccuracy score: %f" %(accuracy_score(y_train, lcf_y_train) * 100))
#print("Recall score : %f" %(recall_score(y_train, lcf_y_train) * 100))
print("ROC score : %f\n" %(roc_auc_score(y_train, lcf_y_train) * 100))
print(confusion_matrix(y_train, lcf_y_train)) 
print("\n",classification_report(y_train, lcf_y_train, labels=[1,0]))


# In[47]:


#Precision score on Test and Train
from sklearn.metrics import precision_score

lcf_precision_test =precision_score(y_test, lcf_y_test, average='weighted', zero_division=1)  
print(lcf_precision_test)
lcf_precision_train =precision_score(y_train, lcf_y_train, average='weighted', zero_division=1)
print(lcf_precision_train)


# In[48]:


#Evaluate model
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
def ROC_curve(title, y_train, scores, label=None):
    
    # calculate the ROC score
    fpr, tpr, thresholds = roc_curve(y_train, scores)
    print('AUC Score ({}): {:.2f} '.format(title, roc_auc_score(y_train, scores)))

    # plot the ROC curve
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, linewidth=2, label=label, color='b')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.2, 1.2])
    plt.ylim([-0.2, 1.2])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC Curve: {}'.format(title), fontsize=16)
    plt.show()


# In[ ]:


#Evaluating Cross_validation_Scores And Cross_Validation_PredictionProbability

#n_error_test = y_pred_test[y_pred_test == -1].size
#n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

from sklearn.model_selection import cross_val_score, cross_val_predict

#for Train
lcf_acc_train = cross_val_score(lcf_clf_train, X_train_std, y_train, cv=3, scoring='accuracy', n_jobs=-1)
lcf_proba_train = cross_val_predict(lcf_clf_train, X_train_std, y_train, cv=3, method='predict_proba')
lcf_scores_train = if_proba_train[:, 1]

#for Test
lcf_acc_test = cross_val_score(lcf_clf_test, X_test_std, y_test, cv=3, scoring='accuracy', n_jobs=-1)
lcf_proba_test = cross_val_predict(lcf_clf_test, X_test_std, y_test, cv=3, method='predict_proba')
lcf_scores_test = lcf_proba_test[:, 1]


# In[ ]:


#Plot ROC Curve for Train
ROC_curve('Local Outiler Factor For Train', y_train, lcf_scores_train)


# In[ ]:


#Plot ROC Curve for Train
ROC_curve('Local Outlier Factor For Test', y_test, lcf_scores_test)


# # Support Vector Machine

# In[107]:


#Build Support Vector Machine

#Importing Support Vector Machine
from sklearn import svm

#Creating a Support Vector Machine object
svm_clf = svm.SVC(probability=True)

#Fitting Classifier For Training Set and Testing Set
svm_clf_train = svm_clf.fit(X_train, y_train)

svm_clf_test = svm_clf.fit(X_test, y_test) 


# In[50]:


#Predicting The Test Set Results
svm_y_test = svm_clf.predict(X_test)  #test
print("Test set predictions:\n {}".format(svm_y_test))

print("Test set score: {:.2f}".format(np.mean(svm_y_test == y_test)))

#Predicting The Train Set Results
svm_y_train = svm_clf.predict(X_train)  #train 
print("Train set predictions:\n {}".format(svm_y_train))

print("Train set score: {:.2f}".format(np.mean(svm_y_train == y_train)))


# In[51]:


#Accuracy score on Test
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix, classification_report

print("\nAccuracy score: %f" %(accuracy_score(y_test,svm_y_test) * 100))
print("Recall score : %f" %(recall_score(y_test, svm_y_test) * 100))
print("ROC score : %f\n" %(roc_auc_score(y_test, svm_y_test) * 100))
print(confusion_matrix(y_test, svm_y_test)) 
print("\n",classification_report(y_test, svm_y_test, labels=[1,0]))


# In[52]:


#Accuracy score on  Train
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix, classification_report

print("\nAccuracy score: %f" %(accuracy_score(y_train, svm_y_train) * 100))
print("Recall score : %f" %(recall_score(y_train, svm_y_train) * 100))
print("ROC score : %f\n" %(roc_auc_score(y_train, svm_y_train) * 100))
print(confusion_matrix(y_train, svm_y_train)) 
print("\n",classification_report(y_train, svm_y_train, labels=[1,0]))


# In[53]:


#Precision score on Test and Train
from sklearn.metrics import precision_score

svm_precision_test =precision_score(y_test, svm_y_test, average='weighted', zero_division=1)  
print(svm_precision_test)
svm_precision_train =precision_score(y_train, svm_y_train, average='weighted', zero_division=1)
print(svm_precision_train)


# In[54]:


#Evaluate model
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import sklearn.metrics as metrics
def ROC_curve(title, y_train, scores, label=None):
    
    # calculate the ROC score
    fpr, tpr, thresholds = metrics.roc_curve(y_train, scores)
    print('AUC Score ({}): {:.2f} '.format(title, roc_auc_score(y_train, scores)))

    # plot the ROC curve
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, linewidth=2, label=label, color='b')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.2, 1.2])
    plt.ylim([-0.2, 1.2])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC Curve: {}'.format(title), fontsize=16)
    plt.show()


# In[ ]:


#Evaluating Cross_validation_Scores And Cross_Validation_PredictionProbability

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

from sklearn.model_selection import cross_val_score, cross_val_predict

#for Train
svm_acc_train = cross_val_score(svm_clf_train, X_train_std, y_train, cv=3, scoring='accuracy', n_jobs=-1)
svm_proba_train = cross_val_predict(svm_clf_train, X_train_std, y_train, cv=3, method='predict_proba')
svm_score_train = svm_proba_train[:, 1]

#for Test
svm_acc_test = cross_val_score(svm_clf_test, X_test_std, y_test, cv=3, scoring='accuracy', n_jobs=-1)
svm_proba_test = cross_val_predict(svm_clf_test, X_test_std, y_test, cv=3, method='predict_proba')
svm_score_test = svm_proba_test[:, 1]


# In[ ]:


#Plot ROC Curve for Train
ROC_curve('Support Vector Machine For Train', y_train, svm_score_train)


# In[ ]:


#Plot ROC Curve for Train
ROC_curve('Support Vector Machine For Test', y_test, svm_score_test)


# # kNearest Neighbors

# In[55]:


#Importing kNearest Neighbor
from sklearn.neighbors import KNeighborsClassifier

#Creating Object For kNearest Neighbor
clf = neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform', 
                                     algorithm='auto', leaf_size=30, p=2, metric='minkowski', 
                                     metric_params=None, n_jobs=None)

#Fitting Classifier to the Training set and Testing Set
kn_clf_train = clf.fit(X_train, y_train)

kn_clf_test = clf.fit(X_test, y_test)


# In[56]:


#Predicting the Test Set results
y_pred_test=clf.predict(X_test)
print("Test set predictions:\n {}".format(y_pred_test))

print("Test set score: {:.2f}".format(np.mean(y_pred_test == y_test)))

#Predicting the Train Set results
y_pred_train=clf.predict(X_train)
print("Test set predictions:\n {}".format(y_pred_train))

print("Train set score: {:.2f}".format(np.mean(y_pred_train == y_train)))


# In[57]:


#Accuracy score on Test 
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix, classification_report

print("\nAccuracy score: %f" %(accuracy_score(y_test,y_pred_test) * 100))
print("Recall score : %f" %(recall_score(y_test, y_pred_test) * 100))
print("ROC score : %f\n" %(roc_auc_score(y_test, y_pred_test) * 100))
print(confusion_matrix(y_test, y_pred_test)) 
print("\n",classification_report(y_test, y_pred_test, labels=[1,0]))


# In[58]:


#Accuracy score on Train
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix, classification_report

print("\nAccuracy score: %f" %(accuracy_score(y_train,y_pred_train) * 100))
print("Recall score : %f" %(recall_score(y_train, y_pred_train) * 100))
print("ROC score : %f\n" %(roc_auc_score(y_train, y_pred_train) * 100))
print(confusion_matrix(y_train, y_pred_train)) 
print("\n",classification_report(y_train, y_pred_train, labels=[1,0]))


# In[59]:


#Precision score on Test and Train
from sklearn.metrics import precision_score

kn_precision_test =precision_score(y_test, y_pred_test, average='weighted', zero_division=1)  
print(kn_precision_test)
kn_precision_train =precision_score(y_train, y_pred_train, average='weighted', zero_division=1)
print(kn_precision_train)


# In[60]:


#Evaluate model
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
def ROC_curve(title, y_train, scores, label=None):
    
    # calculate the ROC score
    fpr, tpr, thresholds = roc_curve(y_train, scores)
    print('AUC Score ({}): {:.2f} '.format(title, roc_auc_score(y_train, scores)))

    # plot the ROC curve
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, linewidth=2, label=label, color='b')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.2, 1.2])
    plt.ylim([-0.2, 1.2])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC Curve: {}'.format(title), fontsize=16)
    plt.show()


# In[ ]:


#Evaluating Cross_validation_Scores And Cross_Validation_PredictionProbability

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.fit_transform(X_test)

from sklearn.model_selection import cross_val_score, cross_val_predict

#for Train
kn_acc_train = cross_val_score(kn_clf_train, X_train_std, y_train, cv=3, scoring='accuracy', n_jobs=-1)
kn_proba_train = cross_val_predict(kn_clf_train, X_train_std, y_train, cv=3, method='predict_proba')
kn_scores_train = kn_proba_train[:, 1]

#for Test
kn_acc_test = cross_val_score(kn_clf_test, X_test_std, y_test, cv=3, scoring='accuracy', n_jobs=-1)
kn_proba_test = cross_val_predict(kn_clf_test, X_test_std, y_test, cv=3, method='predict_proba')
kn_scores_test = kn_proba_test[:, 1]


# In[ ]:


#Plot ROC Curve for Train
ROC_curve('kNearest Neighbors of Train', y_train, kn_scores_train)


# In[ ]:


#Plot ROC Curve for Test
ROC_curve('kNearest Neighbors of Test', y_test, kn_scores_test)


# # Logistic Regression

# In[61]:


#Import Logistic Regression
from sklearn.linear_model import LogisticRegression

#Creating object for Logistic Regression
lr_cls = LogisticRegression(random_state =0, solver='lbfgs', intercept_scaling=1, class_weight='balanced',
                            max_iter=10000, multi_class='auto', verbose=0, n_jobs=None, l1_ratio=None)


#Fitting Classifier for Training Set and Testing Set
lr_cls_train = lr_cls.fit(X_train, y_train)

lr_cls_test = lr_cls.fit(X_test, y_test)


# In[62]:


#Predicting The Test Set Results
lr_y_test = lr_cls.predict(X_test)  #test
print("Test set predictions:\n {}".format(lr_y_test))

print("Test set score: {:.2f}".format(np.mean(lr_y_test == y_test)))

#Predicting The Train Set Results
lr_y_train = lr_cls.predict(X_train)  #train 
print("Train set predictions:\n {}".format(lr_y_train))

print("Train set score: {:.2f}".format(np.mean(lr_y_train == y_train)))


# In[63]:


#Accuracy score on Test
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix, classification_report

print("\nAccuracy score: %f" %(accuracy_score(y_test,lr_y_test) * 100))
print("Recall score : %f" %(recall_score(y_test, lr_y_test) * 100))
print("ROC score : %f\n" %(roc_auc_score(y_test, lr_y_test) * 100))
print(confusion_matrix(y_test, lr_y_test)) 
print("\n",classification_report(y_test, lr_y_test, labels=[1,0]))


# In[64]:


#Accuracy score on  Train
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix, classification_report

print("\nAccuracy score: %f" %(accuracy_score(y_train,lr_y_train) * 100))
print("Recall score : %f" %(recall_score(y_train, lr_y_train) * 100))
print("ROC score : %f\n" %(roc_auc_score(y_train, lr_y_train) * 100))
print(confusion_matrix(y_train, lr_y_train)) 
print("\n",classification_report(y_train, lr_y_train, labels=[1,0]))


# In[65]:


#Precision score on Test and Train
from sklearn.metrics import precision_score

lr_precision_test =precision_score(y_test, lr_y_test, average='weighted', zero_division=1)  
print(lr_precision_test)
lr_precision_train =precision_score(y_train, lr_y_train, average='weighted', zero_division=1)
print(lr_precision_train)


# In[66]:


#Evaluate model
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import sklearn.metrics as metrics
def ROC_curve(title, y_train, scores, label=None):
    
    # calculate the ROC score
    fpr, tpr, thresholds = metrics.roc_curve(y_train, scores)
    print('AUC Score ({}): {:.2f} '.format(title, roc_auc_score(y_train, scores)))

    # plot the ROC curve
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, linewidth=2, label=label, color='b')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.2, 1.2])
    plt.ylim([-0.2, 1.2])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC Curve: {}'.format(title), fontsize=16)
    plt.show()


# In[67]:


#Evaluating Cross_validation_Scores And Cross_Validation_PredictionProbability

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

from sklearn.model_selection import cross_val_score, cross_val_predict

#for Train
lr_acc_train = cross_val_score(lr_cls_train, X_train_std, y_train, cv=3, scoring='accuracy', n_jobs=-1)
lr_proba_train = cross_val_predict(lr_cls_train, X_train_std, y_train, cv=3, method='predict_proba')
lr_score_train = lr_proba_train[:, 1]

#for Test
lr_acc_test = cross_val_score(lr_cls_test, X_test_std, y_test, cv=3, scoring='accuracy', n_jobs=-1)
lr_proba_test = cross_val_predict(lr_cls_test, X_test_std, y_test, cv=3, method='predict_proba')
lr_score_test = lr_proba_test[:, 1]


# In[68]:


#Plot ROC Curve for train
ROC_curve('logistic regression for Train', y_train, lr_score_train)


# In[69]:


#Plot ROC Curve for test
ROC_curve('logistic regression for Test', y_test, lr_score_test)


# # Naive Bayes Classification

# In[70]:


#Import Naibe Bayes Classification
from sklearn.naive_bayes import GaussianNB

#creating Navie Bayes Classifier object
nb_classfier =GaussianNB()

#Fitting Classifier For Training Set and Testing Set
nb_cls_train=nb_classfier.fit(X_train, y_train)

nb_cls_test=nb_classfier.fit(X_test,y_test)


# In[71]:


#Predicting The Test Set Results
nb_y_test = nb_classfier.predict(X_test)  #test
print("Test set predictions:\n {}".format(nb_y_test))

print("Test set score: {:.2f}".format(np.mean(nb_y_test == y_test)))

#Predicting The Train Set Results
nb_y_train = nb_classfier.predict(X_train)  #train 
print("Train set predictions:\n {}".format(nb_y_train))

print("Train set score: {:.2f}".format(np.mean(nb_y_train == y_train)))


# In[72]:


#Accuracy score on Test
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix

print("\nAccuracy score: %f" %(accuracy_score(y_test,nb_y_test) * 100))
print("Recall score : %f" %(recall_score(y_test, nb_y_test) * 100))
print("ROC score : %f\n" %(roc_auc_score(y_test, nb_y_test) * 100))
print(confusion_matrix(y_test, nb_y_test)) 
print("\n",classification_report(y_test, nb_y_test, labels=[1,0]))


# In[73]:


#Accuracy score on Train
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix

print("\nAccuracy score: %f" %(accuracy_score(y_train,nb_y_train) * 100))
print("Recall score : %f" %(recall_score(y_train, nb_y_train) * 100))
print("ROC score : %f\n" %(roc_auc_score(y_train, nb_y_train) * 100))
print(confusion_matrix(y_train, nb_y_train)) 
print("\n",classification_report(y_train, nb_y_train, labels=[1,0]))


# In[74]:


#Precision score on Test and Train
from sklearn.metrics import precision_score

nb_precision_test =precision_score(y_test, nb_y_test, average='weighted', zero_division=1)  
print(nb_precision_test)
nb_precision_train =precision_score(y_train, nb_y_train, average='weighted', zero_division=1)
print(nb_precision_train)


# In[75]:


#Evaluate model
#train 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
y_nb_prob=nb_classfier.predict_proba(X_train)[:,1]
def ROC_curve(title, y_train, scores, label=None):
    
    # calculate the ROC score
    fpr, tpr, thresholds = roc_curve(y_train, scores)
    print('AUC Score ({}): {:.2f} '.format(title, roc_auc_score(y_train, scores)))

    # plot the ROC curve
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, linewidth=2, label=label, color='b')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.2, 1.2])
    plt.ylim([-0.2, 1.2])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC Curve: {}'.format(title), fontsize=16)
    plt.show()


# In[76]:


#Evaluating Cross_validation_Scores And Cross_Validation_PredictionProbability

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

from sklearn.model_selection import cross_val_score, cross_val_predict

#for Train
nb_acc_train = cross_val_score(nb_cls_train, X_train_std, y_train, cv=3, scoring='accuracy', n_jobs=-1)
nb_proba_train = cross_val_predict(nb_cls_train, X_train_std, y_train, cv=3, method='predict_proba')
nb_scores_train = nb_proba_train[:, 1]

#for Test
nb_acc_test = cross_val_score(nb_cls_test, X_test_std, y_test, cv=3, scoring='accuracy', n_jobs=-1)
nb_proba_test = cross_val_predict(nb_cls_test, X_test_std, y_test, cv=3, method='predict_proba')
nb_scores_test = nb_proba_test[:, 1]


# In[77]:


#Plot ROC Curve for Train
ROC_curve('Naive Bayes Classification For Train', y_train, nb_scores_train)


# In[78]:


#Plot ROC Curve for Test
ROC_curve('Naive Bayes Classification For Test', y_test, nb_scores_test)


# # K-Means Clustering

# In[79]:


#importing Kmeans

from sklearn.cluster import KMeans

#create a kmeans objects

km = KMeans(n_clusters=2)

#Fitting Classifier For Training Set and Testing Set

km_clu_train = km.fit(X_train,y_train)

km_clu_test = km.fit(X_test,y_test)


# In[80]:


#Predicting The Test Set Results
km_y_test = km.predict(X_test)  #test
print("Test set predictions:\n {}".format(km_y_test))

print("Test set score: {:.2f}".format(np.mean(km_y_test == y_test)))

#Predicting The Train Set Results
km_y_train = km.predict(X_train)  #train 
print("Train set predictions:\n {}".format(km_y_train))

print("Train set score: {:.2f}".format(np.mean(km_y_train == y_train)))


# In[81]:


#Accuracy score on Test
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix

print("\nAccuracy score: %f" %(accuracy_score(y_test,km_y_test) * 100))
#print("Recall score : %f" %(recall_score(y_test, km_y_test) * 100))
print("ROC score : %f\n" %(roc_auc_score(y_test, km_y_test) * 100))
print(confusion_matrix(y_test, km_y_test)) 
print("\n",classification_report(y_test, km_y_test, labels=[1,0]))


# In[82]:


#Accuracy score on Train
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix

print("\nAccuracy score: %f" %(accuracy_score(y_train,km_y_train) * 100))
#print("Recall score : %f" %(recall_score(y_train, km_y_train) * 100))
print("ROC score : %f\n" %(roc_auc_score(y_train, km_y_train) * 100))
print(confusion_matrix(y_train, km_y_train)) 
print("\n",classification_report(y_train, km_y_train, labels=[1,0]))


# In[83]:


#Precision score on Test and Train
from sklearn.metrics import precision_score

km_precision_test =precision_score(y_test, km_y_test, average='weighted', zero_division=1)  
print(km_precision_test)
km_precision_train =precision_score(y_train, km_y_train, average='weighted', zero_division=1)
print(km_precision_train)


# In[84]:


#Evaluate model
#train 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
def ROC_curve(title, y_train, scores, label=None):
    
    # calculate the ROC score
    fpr, tpr, thresholds = roc_curve(y_train, scores)
    print('AUC Score ({}): {:.2f} '.format(title, roc_auc_score(y_train, scores)))

    # plot the ROC curve
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, linewidth=2, label=label, color='b')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.2, 1.2])
    plt.ylim([-0.2, 1.2])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC Curve: {}'.format(title), fontsize=16)
    plt.show()


# In[85]:


#Evaluating Cross_validation_Scores And Cross_Validation_PredictionProbability

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)


from sklearn.model_selection import cross_val_score, cross_val_predict

#for Train
km_acc_train = cross_val_score(km_clu_train, X_train_std, y_train, cv=3, scoring='accuracy', n_jobs=-1)
km_proba_train = cross_val_predict(km_clu_train, X_train_std, y_train, cv=3, method='predict')
km_scores_train = km_proba_train[:, ]


#for Test
km_acc_test = cross_val_score(km_clu_test, X_test_std, y_test, cv=3, scoring='accuracy', n_jobs=-1)
km_proba_test = cross_val_predict(km_clu_test, X_test_std, y_test, cv=3, method='predict')
km_scores_test = km_proba_test[:, ]


# In[86]:


#Plot ROC Curve for Train
ROC_curve('K-Means Clustering for Train', y_train, km_scores_train)


# In[87]:


#Plot ROC Curve for Test
ROC_curve('K-Means Clustering for Test', y_test, km_scores_test)


# # Decision Tree Classifier

# In[88]:


#Importing Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

#Creating a Decision Tree Classifier object
dtc_clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2,
                                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None,
                                 random_state=None, max_leaf_nodes=None)

#Fitting Classifier For Training Set and Testing Set

dtc_clf_train = dtc_clf.fit(X_train, y_train)

dtc_clf_test = dtc_clf.fit(X_test, y_test)


# In[89]:


#Predicting The Test Set Results
dtc_y_test = dtc_clf.predict(X_test)  #test
print("Test set predictions:\n {}".format(dtc_y_test))

print("Test set score: {:.2f}".format(np.mean(dtc_y_test == y_test)))

#Predicting The Train Set Results
dtc_y_train = dtc_clf.predict(X_train)  #train 
print("Train set predictions:\n {}".format(dtc_y_train))

print("Train set score: {:.2f}".format(np.mean(dtc_y_train == y_train)))


# In[90]:


#Accuracy score on Test
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix

print("\nAccuracy score: %f" %(accuracy_score(y_test,dtc_y_test) * 100))
print("Recall score : %f" %(recall_score(y_test, dtc_y_test) * 100))
print("ROC score : %f\n" %(roc_auc_score(y_test, dtc_y_test) * 100))
print(confusion_matrix(y_test, dtc_y_test)) 
print("\n",classification_report(y_test, dtc_y_test, labels=[1,0]))


# In[91]:


#Accuracy score on Train
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix

print("\nAccuracy score: %f" %(accuracy_score(y_train,dtc_y_train) * 100))
print("Recall score : %f" %(recall_score(y_train, dtc_y_train) * 100))
print("ROC score : %f\n" %(roc_auc_score(y_train, dtc_y_train) * 100))
print(confusion_matrix(y_train, dtc_y_train)) 
print("\n",classification_report(y_train, dtc_y_train, labels=[1,0]))


# In[92]:


#Precision score on Test and Train
from sklearn.metrics import precision_score

dtc_precision_test =precision_score(y_test, dtc_y_test, average='weighted', zero_division=1)  
print(dtc_precision_test)
dtc_precision_train =precision_score(y_train, dtc_y_train, average='weighted', zero_division=1)
print(dtc_precision_train)


# In[93]:


#Evaluate model
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
def ROC_curve(title, y_train, scores, label=None):
    
    # calculate the ROC score
    fpr, tpr, thresholds = roc_curve(y_train, scores)
    print('AUC Score ({}): {:.2f} '.format(title, roc_auc_score(y_train, scores)))

    # plot the ROC curve
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, linewidth=2, label=label, color='b')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.2, 1.2])
    plt.ylim([-0.2, 1.2])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC Curve: {}'.format(title), fontsize=16)
    plt.show()


# In[94]:


#Evaluating Cross_validation_Scores And Cross_Validation_PredictionProbability

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

from sklearn.model_selection import cross_val_score, cross_val_predict

#for Train
dtc_acc_train = cross_val_score(dtc_clf_train, X_train_std, y_train, cv=3, scoring='accuracy', n_jobs=-1)
dtc_proba_train = cross_val_predict(dtc_clf_train, X_train_std, y_train, cv=3, method='predict_proba')
dtc_scores_train = dtc_proba_train[:, 1]

#for Test
dtc_acc_test = cross_val_score(dtc_clf_test, X_test_std, y_test, cv=3, scoring='accuracy', n_jobs=-1)
dtc_proba_test = cross_val_predict(dtc_clf_test, X_test_std, y_test, cv=3, method='predict_proba')
dtc_scores_test = dtc_proba_test[:, 1]


# In[95]:


#Plot ROC Curve for Train
ROC_curve('Decision Tree Classification For Train', y_train, dtc_scores_train)


# In[96]:


#Plot ROC Curve for Test
ROC_curve('Decision Tree Classification For Test', y_test, dtc_scores_test)


# # Random Forest Classifier

# In[97]:


#Build Random Forest Calssifier

#Importing Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

#Creating a Random Forest Classifier object
rmf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, max_features='auto',
                             max_leaf_nodes=None, n_jobs=None, random_state=None,
                             verbose=0, class_weight=None, ccp_alpha=0.0, max_samples=None)

#Fitting Classifier For Training Set and Testing Set
rmf_clf_train = rmf.fit(X_train, y_train)

rmf_clf_test = rmf.fit(X_test, y_test) 


# In[98]:


#Predicting The Test Set Results
rmf_y_test = rmf.predict(X_test)  #test
print("Test set predictions:\n {}".format(rmf_y_test))

print("Test set score: {:.2f}".format(np.mean(rmf_y_test == y_test)))

#Predicting The Train Set Results
rmf_y_train = rmf.predict(X_train)  #train 
print("Train set predictions:\n {}".format(rmf_y_train))

print("Train set score: {:.2f}".format(np.mean(rmf_y_train == y_train)))


# In[99]:


#Accuracy score on Test
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix

print("\nAccuracy score: %f" %(accuracy_score(y_test,rmf_y_test) * 100))
print("Recall score : %f" %(recall_score(y_test, rmf_y_test) * 100))
print("ROC score : %f\n" %(roc_auc_score(y_test, rmf_y_test) * 100))
print(confusion_matrix(y_test, rmf_y_test)) 
print("\n", classification_report(y_test, rmf_y_test, labels=[1,0]))


# In[100]:


#Accuracy score on Train
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix

print("\nAccuracy score: %f" %(accuracy_score(y_train,rmf_y_train) * 100))
print("Recall score : %f" %(recall_score(y_train, rmf_y_train) * 100))
print("ROC score : %f\n" %(roc_auc_score(y_train, rmf_y_train) * 100))
print(confusion_matrix(y_train, rmf_y_train)) 
print("\n",classification_report(y_train, rmf_y_train, labels=[1,0]))


# In[101]:


#Precision score on Test and Train
from sklearn.metrics import precision_score

rmf_precision_test =precision_score(y_test, rmf_y_test, average='weighted', zero_division=1)  
print(rmf_precision_test)
rmf_precision_train =precision_score(y_train, rmf_y_train, average='weighted', zero_division=1)
print(rmf_precision_train)


# In[102]:


#Evaluate model
#train 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
def ROC_curve(title, y_train, scores, label=None):
    
    # calculate the ROC score
    fpr, tpr, thresholds = roc_curve(y_train, scores)
    print('AUC Score ({}): {:.2f} '.format(title, roc_auc_score(y_train, scores)))

    # plot the ROC curve
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, linewidth=2, label=label, color='b')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.2, 1.2])
    plt.ylim([-0.2, 1.2])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC Curve: {}'.format(title), fontsize=16)
    plt.show()


# In[103]:


#Evaluating Cross_validation_Scores And Cross_Validation_PredictionProbability

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

from sklearn.model_selection import cross_val_score, cross_val_predict

#for Train
rmf_acc_train = cross_val_score(rmf_clf_train, X_train_std, y_train, cv=3, scoring='accuracy', n_jobs=-1)
rmf_proba_train = cross_val_predict(rmf_clf_train, X_train_std, y_train, cv=3, method='predict_proba')
rmf_scores_train = rmf_proba_train[:, 1]

#for Test
rmf_acc_test = cross_val_score(rmf_clf_test, X_test_std, y_test, cv=3, scoring='accuracy', n_jobs=-1)
rmf_proba_test = cross_val_predict(rmf_clf_test, X_test_std, y_test, cv=3, method='predict_proba')
rmf_scores_test = rmf_proba_test[:, 1]


# In[104]:


#Plot ROC Curve for Train
ROC_curve('Random Forest Classification For Train', y_train, rmf_scores_train)


# In[105]:


#Plot ROC Curve for Test
ROC_curve('Random Forest Classification For Test', y_test, rmf_scores_test)


# In[ ]:




