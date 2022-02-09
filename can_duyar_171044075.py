# CAN DUYAR - 171044075 / DATA MINING PROJECT FOR TUMOR ANALYSIS

import pandas as pd               
import numpy as np              
import matplotlib.pyplot as plt  
import seaborn as sns            
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import matplotlib as ml
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder


########################## ADABOOST CLASSIFIER IMPLEMENTATION ###################
class AdaBoostTumorClassification():
  
    # initializing of AdaBoost
    def __init__(self,param=DecisionTreeClassifier(max_depth=1,max_leaf_nodes=2),n_estimators=20):
        self.n_estimators = n_estimators
        self.param = param
                    
    # model.prediction method
    def predict(self,data1):
        predictions = np.array([stump.predict(data1) for stump in self.stumps])
        return np.sign(np.dot(self.keepWeights,predictions))


    # model.fit method
    def fit(self,data1,data2):
        keepNum = data1.shape[0]
        self.keepWeights = np.zeros(shape=self.n_estimators)
        self.stumps = np.zeros(shape=self.n_estimators,dtype=object)
        self.weights = np.zeros(shape=(self.n_estimators,keepNum))
        self.weights[0] = (np.ones(shape=keepNum))/keepNum
        self.errors = np.zeros(shape=self.n_estimators)
        
        range_control_lower = 0
        range_control_upper = self.n_estimators
        
        for t in range(range_control_lower,range_control_upper):
          iter_weight = self.weights[t]
          stump = self.param
          stump.fit(data1,data2,sample_weight = iter_weight)
          prediction = stump.predict(data1)
          summationOfErrors = iter_weight[(prediction != data2)].sum()
          keepWeight = np.log((1 - summationOfErrors)/summationOfErrors)/2
          #it updates the weights 
          updatedWeights = (iter_weight*np.exp(-keepWeight*data2*prediction))
          updatedWeights /= updatedWeights.sum()

          if t < range_control_upper-1:
            self.weights[t+1] = updatedWeights
          
          self.errors[t] = summationOfErrors
          self.stumps[t] = stump
          self.keepWeights[t] = keepWeight
          
        return self
    

def PCA(data , numberOfComponents):
    #Step- -> Standardize the dataset. 
    meanedData = data - np.mean(data , axis = 0)     
    #Step-2 -> Calculate the covariance matrix for the features in the dataset. 
    covariance_matrix = np.cov(meanedData , rowvar = False)     
    #Step-3 -> Calculate the eigenvalues and eigenvectors for the covariance matrix. 
    eigen_values , eigen_vectors = np.linalg.eigh(covariance_matrix)     
    #Step-4 -> Sort eigenvalues and their corresponding eigenvectors.
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]     
    #Step-5 -> Pick k eigenvalues and form a matrix of eigenvectors.
    eigenvector_subset = sorted_eigenvectors[:,0:numberOfComponents]     
    #Step-6 -> Transform the original matrix.
    reducedData = np.dot(eigenvector_subset.transpose(),meanedData.transpose()).transpose()
     
    return reducedData




# DATA CLEANING 
    
df = pd.read_csv("data.csv")
# data cleaning operation
print(df.isna().sum()) # Unnamed: 32 -> this column has 569 NAN value
print("\n")
#removing data that are not necessary for the project
df = df.drop(columns=['id','Unnamed: 32'],axis =1) 


"""
Diagnosis was represented by letters(M = malignant, B = benign). 
I changed it to binary values

Benign = iyi huylu tümör(0)
Malignant = kötü huylu tümör(1)

"""

df['diagnosis'] = [0 if detect == "B" else 1 for detect in df.diagnosis]


# FILTER BASED FEATURE SELECTION

#Correlation between diagnosis and other features 
plt.figure(figsize = (6, 4), dpi = 80)
df.corr(method ='pearson')['diagnosis'].sort_values().plot(kind  = 'bar')
# It clears data from attributes that have low correlation values.
df = df.drop(columns=['smoothness_se','fractal_dimension_mean','texture_se','symmetry_se'],axis =1)


#correlation values with the help of correlation matrix

ret, coord = plt.subplots(figsize = (25,25))
sns.heatmap(df.corr(), annot=True, fmt='.1f',
            ax=coord, cmap='Spectral', vmin=-1, vmax=1)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title('Correlation Matrix', size=13);
plt.show()

# distribution of diagnosis attribute
sns.countplot(df["diagnosis"])
plt.title('Distribution of diagnosis attribute', size=10);



# OUTLIER DETECTION with LOF (Local Outlier Detection)

data_without_diagnosis = df.drop(['diagnosis'], axis=1) # data without diagnosis
data_only_diagnosis = df.diagnosis # data with only diagnosis

column = data_without_diagnosis.columns.tolist()

LOF = LocalOutlierFactor()
without_diagnosis_predict_LOF = LOF.fit_predict(data_without_diagnosis)
LOFscore = LOF.negative_outlier_factor_

# To save outlier scores
outlier_score = pd.DataFrame()
outlier_score['outlierScores'] = LOFscore



# outlier condition is -2.5 it means that it shows outlier values above 2.5
outlier_condition = -2.5
outlier_control = outlier_score["outlierScores"] < outlier_condition
outlier_index = outlier_score[outlier_control].index.tolist()
print("outlier index = ", outlier_index)


plt.figure()
plt.scatter(data_without_diagnosis.iloc[outlier_index,0]
            ,data_without_diagnosis.iloc[outlier_index,1],color = "black"
            , s = 50, label = "outlier data")
plt.scatter(data_without_diagnosis.iloc[:,0],data_without_diagnosis.iloc[:,1]
            ,color = "k", s = 3, label = "normal data")

# normalization for plotting process to see outliers
size = (LOFscore.max()- LOFscore) / (LOFscore.max() - LOFscore.min())
outlier_score["size"] = size
plt.scatter(data_without_diagnosis.iloc[:,0],data_without_diagnosis.iloc[:,1],
            s = 1000*size, edgecolors = "g", facecolors = "none", label = "Outlier Scores")
plt.legend()
plt.show()

# we drop outliers from the actual data
data_without_diagnosis = data_without_diagnosis.drop(outlier_index)
data_only_diagnosis = data_only_diagnosis.drop(outlier_index).values


# FEATURE EXTRACTION USING PCA(PRINCIPAL COMPONENT ANALYSIS)

data_without_diagnosis = StandardScaler().fit_transform(data_without_diagnosis)
priComp = PCA(data_without_diagnosis,2)

dataframeOfPrincipalComp = pd.DataFrame(data = priComp,columns = ['PCA_data1', 'PCA_data2'])



# CLASSIFICATION

X_train,X_test,y_train,y_test=train_test_split(data_without_diagnosis,
                                               data_only_diagnosis,test_size=0.25,
                                               random_state=42)

scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


key = ['RandomForestClassifier','GradientBoostingClassifier','AdaBoostTumorClassification']
value = [RandomForestClassifier(n_estimators=60, random_state=0),
         GradientBoostingClassifier(random_state=20),
         AdaBoostTumorClassification()]
model = dict(zip(key,value))

print("\n")
print("Classification accuracy results using different techniques\n")

for classificationName,algorithmImp in model.items():
    model=algorithmImp
    model.fit(X_train,y_train)
    predict = model.predict(X_test)
    acc = accuracy_score(y_test, predict)
    print(classificationName, end = ":")
    print(" %",acc*100)









