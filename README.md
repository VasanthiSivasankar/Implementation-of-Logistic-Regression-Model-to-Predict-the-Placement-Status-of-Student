# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: VASANTHI SIVASANKAR
RegisterNumber:  212223040234
*/
```
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
df=pd.read_csv('Placement_Data.csv')
df
# drop the sl_no column as it does have any impact in the dataset
df= df.drop("sl_no",axis=1)
df.info
df=df.drop("salary",axis=1)
#changing to categorical code from object type
df["gender"]=df["gender"].astype('category')
df["ssc_b"]=df["ssc_b"].astype('category')
df["hsc_b"]=df["hsc_b"].astype('category')
df["degree_t"]=df["degree_t"].astype('category')
df["workex"]=df["workex"].astype('category')
df["specialisation"]=df["specialisation"].astype('category')
df["status"]=df["status"].astype('category')
df["hsc_s"]=df["hsc_s"].astype('category')
#use cat.code to change the type
df["gender"]=df["gender"].cat.codes
df["ssc_b"]=df["ssc_b"].cat.codes
df["hsc_b"]=df["hsc_b"].cat.codes
df["degree_t"]=df["degree_t"].cat.codes
df["workex"]=df["workex"].cat.codes
df["specialisation"]=df["specialisation"].cat.codes
df["status"]=df["status"].cat.codes
df["hsc_s"]=df["hsc_s"].cat.codes
df
#selecting the feature as lable
X=df.iloc[:,:-1].values #values till the column -1
Y=df.iloc[:,-1].values # value of only one column -1
X.shape
Y.shape
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1)
X_train.shape
X_test.shape
Y_train.shape
Y_test.shape
clf=LogisticRegression()
clf.fit(X_train,Y_train)
Y_pred=clf.predict(X_test)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(Y_test,Y_pred)
confusion=confusion_matrix(Y_test,Y_pred)
cr=classification_report(Y_test,Y_pred)
print("Accuracy score:",accuracy)
print("\nConfusion matrix:\n",confusion)
print("\nClassification Report:\n",cr)
from sklearn import metrics
cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=confusion,display_labels=[True
cm_display.plot()
```

## Output:
![output1](https://github.com/user-attachments/assets/adada00f-0780-4427-80c5-51392504742e)

![output2](https://github.com/user-attachments/assets/25cfdf9e-78c7-47ea-9585-0b77d94656a4)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
