#modules importing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


#train.scv file data is assinged to titanic_data
#test.csv file data is assinged to test_ti
titanic_data = pd.read_csv('train.csv')
test_ti = pd.read_csv('test.csv')


#as there are many values missing for cabin we shall remove it
titanic_data=titanic_data.drop(columns='Cabin',axis=1)
test_ti=test_ti.drop(columns='Cabin',axis=1)


#few age cell are missing as will assign them mean/average of all ages
titanic_data['Age'].fillna(titanic_data['Age'].mean(),inplace=True)
test_ti['Age'].fillna(test_ti['Age'].mean(),inplace=True)

#few fare cell are missing as will assign them mean/average of all ages
test_ti['Fare'].fillna(test_ti['Fare'].mean(),inplace=True)


#Embarked should be c or s or q so cant find the mean for letters so let us find the most value which is mode and assign them for missing values
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0],inplace=True)


#we can view it using the graph of people dead dead and survived of male and female
'''sns.countplot(data=titanic_data,x='Sex',hue='Survived')
plt.show()'''


#replacing the Sex and Embarked valuse by numbers to make it easier
titanic_data.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)
test_ti.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)

#assigning data required to predict for X and predicted data of train.csv to Y
X = titanic_data.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1)
Y = titanic_data['Survived']

#test.csv data to be predicted is assinged to X1
X1=test_ti.drop(columns = ['PassengerId','Name','Ticket'],axis=1)

#model training , logistic regression
model = LogisticRegression()
model.fit(X, Y)

#prediction of train.csv to check accurarcy
'''X_predict = model.predict(X)
print(X_predict)
training_data_accuracy = accuracy_score(Y, X_predict)
print('Accuracy score of training data : ', training_data_accuracy)'''
#so now we are going to load test.csv data for prediction
X_test_prediction = model.predict(X1)
print(X_test_prediction)