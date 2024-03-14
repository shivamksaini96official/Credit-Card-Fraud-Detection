# Credit Card Fraud Detection

### **AIM :** The challenge is to recognize fraudulent credit card transactions so that the customers of credit card companies are not charged for items that they did not purchase. 

### **Main Challenges:**

1. The challenge is to recognize fraudulent credit card transactions so that the customers of credit card companies are not charged for items that they did not purchase.
2. Imbalanced Data i.e most of the transactions (99.8%) are not fraudulent which makes it really hard for detecting the fraudulent ones
3. Data availability as the data is mostly private.
4. Misclassified Data can be another major issue, as not every fraudulent transaction is caught and reported.
5. Adaptive techniques used against the model by the scammers.

### How to tackle these challenges ?

1. The model used must be simple and fast enough to detect the anomaly and classify it as a fraudulent transaction as quickly as possible.
2. For protecting the privacy of the user the dimensionality of the data can be reduced.
3. A more trustworthy source must be taken which double-check the data, at least for training the model.
4. We can make the model simple and interpretable so that when the scammer adapts to it with just some tweaks we can have a new model up and running to deploy.

### Code : Importing the necessary libraries


```python
# import the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
```

### Code : Loading the Data


```python
# load the dataset from the csv file using pandas
data = pd.read_csv('../DATA/creditcard.csv')
```

### Code : Understanding the Data


```python
# Take a look at the data
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>-1.359807</td>
      <td>-0.072781</td>
      <td>2.536347</td>
      <td>1.378155</td>
      <td>-0.338321</td>
      <td>0.462388</td>
      <td>0.239599</td>
      <td>0.098698</td>
      <td>0.363787</td>
      <td>...</td>
      <td>-0.018307</td>
      <td>0.277838</td>
      <td>-0.110474</td>
      <td>0.066928</td>
      <td>0.128539</td>
      <td>-0.189115</td>
      <td>0.133558</td>
      <td>-0.021053</td>
      <td>149.62</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.191857</td>
      <td>0.266151</td>
      <td>0.166480</td>
      <td>0.448154</td>
      <td>0.060018</td>
      <td>-0.082361</td>
      <td>-0.078803</td>
      <td>0.085102</td>
      <td>-0.255425</td>
      <td>...</td>
      <td>-0.225775</td>
      <td>-0.638672</td>
      <td>0.101288</td>
      <td>-0.339846</td>
      <td>0.167170</td>
      <td>0.125895</td>
      <td>-0.008983</td>
      <td>0.014724</td>
      <td>2.69</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>-1.358354</td>
      <td>-1.340163</td>
      <td>1.773209</td>
      <td>0.379780</td>
      <td>-0.503198</td>
      <td>1.800499</td>
      <td>0.791461</td>
      <td>0.247676</td>
      <td>-1.514654</td>
      <td>...</td>
      <td>0.247998</td>
      <td>0.771679</td>
      <td>0.909412</td>
      <td>-0.689281</td>
      <td>-0.327642</td>
      <td>-0.139097</td>
      <td>-0.055353</td>
      <td>-0.059752</td>
      <td>378.66</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>-0.966272</td>
      <td>-0.185226</td>
      <td>1.792993</td>
      <td>-0.863291</td>
      <td>-0.010309</td>
      <td>1.247203</td>
      <td>0.237609</td>
      <td>0.377436</td>
      <td>-1.387024</td>
      <td>...</td>
      <td>-0.108300</td>
      <td>0.005274</td>
      <td>-0.190321</td>
      <td>-1.175575</td>
      <td>0.647376</td>
      <td>-0.221929</td>
      <td>0.062723</td>
      <td>0.061458</td>
      <td>123.50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>-1.158233</td>
      <td>0.877737</td>
      <td>1.548718</td>
      <td>0.403034</td>
      <td>-0.407193</td>
      <td>0.095921</td>
      <td>0.592941</td>
      <td>-0.270533</td>
      <td>0.817739</td>
      <td>...</td>
      <td>-0.009431</td>
      <td>0.798278</td>
      <td>-0.137458</td>
      <td>0.141267</td>
      <td>-0.206010</td>
      <td>0.502292</td>
      <td>0.219422</td>
      <td>0.215153</td>
      <td>69.99</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>



### Code : Describing the Data


```python
# shape of the data
print(data.shape)

# statistical parameters' information
print(data.describe())
```

    (284807, 31)
                    Time            V1            V2            V3            V4  \
    count  284807.000000  2.848070e+05  2.848070e+05  2.848070e+05  2.848070e+05   
    mean    94813.859575  3.918649e-15  5.682686e-16 -8.761736e-15  2.811118e-15   
    std     47488.145955  1.958696e+00  1.651309e+00  1.516255e+00  1.415869e+00   
    min         0.000000 -5.640751e+01 -7.271573e+01 -4.832559e+01 -5.683171e+00   
    25%     54201.500000 -9.203734e-01 -5.985499e-01 -8.903648e-01 -8.486401e-01   
    50%     84692.000000  1.810880e-02  6.548556e-02  1.798463e-01 -1.984653e-02   
    75%    139320.500000  1.315642e+00  8.037239e-01  1.027196e+00  7.433413e-01   
    max    172792.000000  2.454930e+00  2.205773e+01  9.382558e+00  1.687534e+01   
    
                     V5            V6            V7            V8            V9  \
    count  2.848070e+05  2.848070e+05  2.848070e+05  2.848070e+05  2.848070e+05   
    mean  -1.552103e-15  2.040130e-15 -1.698953e-15 -1.893285e-16 -3.147640e-15   
    std    1.380247e+00  1.332271e+00  1.237094e+00  1.194353e+00  1.098632e+00   
    min   -1.137433e+02 -2.616051e+01 -4.355724e+01 -7.321672e+01 -1.343407e+01   
    25%   -6.915971e-01 -7.682956e-01 -5.540759e-01 -2.086297e-01 -6.430976e-01   
    50%   -5.433583e-02 -2.741871e-01  4.010308e-02  2.235804e-02 -5.142873e-02   
    75%    6.119264e-01  3.985649e-01  5.704361e-01  3.273459e-01  5.971390e-01   
    max    3.480167e+01  7.330163e+01  1.205895e+02  2.000721e+01  1.559499e+01   
    
           ...           V21           V22           V23           V24  \
    count  ...  2.848070e+05  2.848070e+05  2.848070e+05  2.848070e+05   
    mean   ...  1.473120e-16  8.042109e-16  5.282512e-16  4.456271e-15   
    std    ...  7.345240e-01  7.257016e-01  6.244603e-01  6.056471e-01   
    min    ... -3.483038e+01 -1.093314e+01 -4.480774e+01 -2.836627e+00   
    25%    ... -2.283949e-01 -5.423504e-01 -1.618463e-01 -3.545861e-01   
    50%    ... -2.945017e-02  6.781943e-03 -1.119293e-02  4.097606e-02   
    75%    ...  1.863772e-01  5.285536e-01  1.476421e-01  4.395266e-01   
    max    ...  2.720284e+01  1.050309e+01  2.252841e+01  4.584549e+00   
    
                    V25           V26           V27           V28         Amount  \
    count  2.848070e+05  2.848070e+05  2.848070e+05  2.848070e+05  284807.000000   
    mean   1.426896e-15  1.701640e-15 -3.662252e-16 -1.217809e-16      88.349619   
    std    5.212781e-01  4.822270e-01  4.036325e-01  3.300833e-01     250.120109   
    min   -1.029540e+01 -2.604551e+00 -2.256568e+01 -1.543008e+01       0.000000   
    25%   -3.171451e-01 -3.269839e-01 -7.083953e-02 -5.295979e-02       5.600000   
    50%    1.659350e-02 -5.213911e-02  1.342146e-03  1.124383e-02      22.000000   
    75%    3.507156e-01  2.409522e-01  9.104512e-02  7.827995e-02      77.165000   
    max    7.519589e+00  3.517346e+00  3.161220e+01  3.384781e+01   25691.160000   
    
                   Class  
    count  284807.000000  
    mean        0.001727  
    std         0.041527  
    min         0.000000  
    25%         0.000000  
    50%         0.000000  
    75%         0.000000  
    max         1.000000  
    
    [8 rows x 31 columns]
    

### Code : Imbalance in the Data

##### Explaining the data we are dealing with.


```python
# Determine the number of Fraud cases in the dataset
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]

outlier_fraction = len(fraud) / len(valid)
print('Outlier fraction: ',outlier_fraction)
print('Fraud Transactions: ',len(fraud))
print('Valid Transactions: ',len(valid))
```

    Outlier fraction:  0.0017304750013189597
    Fraud Transactions:  492
    Valid Transactions:  284315
    

**Insight:**
>Only *0.17%* fraudulent transactions out of all.
This means the data is highly imbalanced.
Let's first apply our model without balancing it and if we don't get good accuracy, then we will find a way to balance it if needed.

### Code : Amount details for Fraudulent transactions


```python
print('Amount details of the fraudulent transactions:')
fraud.Amount.describe()
```

    Amount details of the fraudulent transactions:
    




    count     492.000000
    mean      122.211321
    std       256.683288
    min         0.000000
    25%         1.000000
    50%         9.250000
    75%       105.890000
    max      2125.870000
    Name: Amount, dtype: float64



### Code : Amount details for Valid transactions 


```python
print('Amount details of the valid transactions:')
valid.Amount.describe()
```

    Amount details of the valid transactions:
    




    count    284315.000000
    mean         88.291022
    std         250.105092
    min           0.000000
    25%           5.650000
    50%          22.000000
    75%          77.050000
    max       25691.160000
    Name: Amount, dtype: float64



**Insight:**
>As we can see from this, the average money transcations in fraudulent ones (*122.211*) is more than the valid ones (*88.29*).
This makes this problem crucial to deal with.

### Code : Plotting the Correlation Matrix

>Correlation matrix graphically gives us an idea of how features correlate with each other and can help us predict which features are more relevant for prediction. 


```python
# Correlation Matrix
corr_mat = data.corr()
fig = plt.figure(figsize=(12,9),dpi=150)
sns.heatmap(corr_mat)
plt.show()
```


    
![png](output_20_0.png)
    


**Insight:**
>In the heatmap, we can clearly see that most of the features are not correlated to other features but there are some features that either have a positive or negative correlation with the others. For example, *V2* and *V5* are highly negatively correlated to the feature *Amount*. 
Some positive correlation between *V20* and *Amount* can also be noticed.

### Code : Seperating X and y values
> Dividing the data into input parameters and output values format.


```python
# Dividing the X and y from the dataset
X = data.drop(['Class'],axis=1)
y = data['Class']
print(X.shape)
print(y.shape)

# getting just the values for the sake of processing
# (numpy array with no columns)
X_data = X.values
y_data = y.values
```

    (284807, 30)
    (284807,)
    

### Training and Test data Bifurcation
> We will be dividing the model into two main groups:
> - One for training the model.
> - And another for testing our trained model performance.


```python
# Using Scikit-Learn to split data into training and test sets.
from sklearn.model_selection import train_test_split

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=101)
```

### Code : Building a Random Forest Model using scikit learn


```python
# Building the Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

# random forest model creation
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)

# predictions
y_pred = rfc.predict(X_test)
```

### Code : Building all kinds of Evaluating Parameters


```python
# Evaluating the classifier
from sklearn.metrics import classification_report,accuracy_score
from sklearn.metrics import precision_score,recall_score
from sklearn.metrics import f1_score,matthews_corrcoef,confusion_matrix

n_outliers = len(fraud)
n_errors   = (y_pred != y_test).sum()

print('The model used is Random Forest Classifier')

acc = accuracy_score(y_test,y_pred)
print('The accuracy is: ',acc)

prec = precision_score(y_test,y_pred)
print('The precision is: ',prec)

rec = recall_score(y_test,y_pred)
print('The recall is: ',rec)

f1 = f1_score(y_test,y_pred)
print('The F1-score is: ',f1)

mcc = matthews_corrcoef(y_test,y_pred)
print('The Matthews correlation coefficient is: ',mcc)
```

    The model used is Random Forest Classifier
    The accuracy is:  0.9996137776061234
    The precision is:  0.9550561797752809
    The recall is:  0.8252427184466019
    The F1-score is:  0.8854166666666667
    The Matthews correlation coefficient is:  0.8875949570286975
    

### Code : Visulaizing the Confusion Matrix 


```python
# printing the confusion matrix
labels = ['Valid','Fraud']
conf_matrix = confusion_matrix(y_test,y_pred)

plt.figure(figsize=(7,7),dpi=150)
sns.heatmap(conf_matrix,xticklabels=labels,yticklabels=labels,annot=True)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.show()
```


    
![png](output_31_0.png)
    


### Conclusion : As evident from the metric scores (especially recall score), our  random forest model is performing really good even with imbalanced data.
