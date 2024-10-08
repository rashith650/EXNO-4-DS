# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
df = pd.read_csv('/content/bmi.csv')
df.head()
```
![image](https://github.com/user-attachments/assets/879bdda5-081e-4e73-bbd2-e91b55aa3a64)

```
df.dropna()
```
![image](https://github.com/user-attachments/assets/070b7e6f-d744-483b-a77e-163aca442c39)


![image](https://github.com/user-attachments/assets/894f1d97-61e7-4853-acc3-ad98a93d863e)
```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/62b951d7-7ec1-48e5-a8a5-f91f145bfdf9)
```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/eed8d27c-a458-492b-8ae0-84ffd233a71e)

```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/user-attachments/assets/eb4f471f-2306-41b3-be88-5af39464b233)
```
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/user-attachments/assets/9c066155-eaab-41f0-9d8e-392dafc36441)
```
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head()
```
![image](https://github.com/user-attachments/assets/df574662-7fda-4afa-94aa-d26ebf11d60d)

```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
data=pd.read_csv('/content/income(1) (1).csv',na_values=' ?')
data
```
![image](https://github.com/user-attachments/assets/810ea9da-8a25-443b-9a9a-f8a94be61a2a)
```
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/f0d7c047-cca8-4ae8-a17c-5e705206985b)
```
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/user-attachments/assets/1eee4caa-6904-4200-a6da-c9d40530ab97)
```
data2 = data.dropna(axis=0)
data2
```
![image](https://github.com/user-attachments/assets/5f17d2a3-f3e5-48ad-b8e9-70eb6058be78)
```
sal=data['SalStat']
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/user-attachments/assets/9198227d-226b-42fa-85f0-75f7a7f6143a)
```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/user-attachments/assets/0d34d374-4504-4d9b-91d8-a928ffe1eba0)
```
data2
```
![image](https://github.com/user-attachments/assets/174ac9c1-afd3-43fe-a89d-33076aeff55a)
```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![image](https://github.com/user-attachments/assets/3ee29a4e-2436-4ba9-9d78-95b05c9d8c72)
![image](https://github.com/user-attachments/assets/d117bb30-062c-44e1-b1d8-3f998290b29f)
```
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier = KNeighborsClassifier(n_neighbors=5)
KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/user-attachments/assets/62b12a27-47c2-4b49-a003-4288ce973b33)
![image](https://github.com/user-attachments/assets/b853fcd8-eeb8-4566-80c2-2befd6991388)

```
import pandas as pd 
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/user-attachments/assets/fde2f3ee-10fb-4811-a35f-3d79ff44c36c)
```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![image](https://github.com/user-attachments/assets/5eec1d3f-5833-4633-8eac-0a658bbda6ed)

```
chi2,p,dof,expected=chi2_contingency(contingency_table)
print('Chi-square statistic:',chi2)
print('p-value:',p)

```
![image](https://github.com/user-attachments/assets/493b2f44-f67e-4d49-9241-6d4e530e3405)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest,mutual_info_classif
data={
    'Feature1':[1,2,3,4,5],
    'Feature2':['A','B','C','A','B'],
    'Feature3':[0,1,1,0,1],
    'Target':[0,1,1,0,1]
}
df=pd.DataFrame(data)
X=df[['Feature1','Feature3']]
y=df['Target']
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(X,y)
selected_features_indices=selector.get_support(indices=True)
selected_features=X.columns[selected_features_indices]
print('Selected Features:')
print(selected_features)
```
![image](https://github.com/user-attachments/assets/fa1ed080-e6d3-478c-bdae-b03db8868ebf)

# RESULT:

Thus, Feature selection and Feature scaling has been used on thegiven dataset.
