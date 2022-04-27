import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#read dataset
df=pd.read_csv('C:/0_MSc_IT_Notes/Big Data Analytics/practicals/decision_tree_practical/social.csv')
print(df)

#choose independent(input) and dependent(ouput) variables
x = df.iloc[:,[2,3]]    #x=df[['Age','EstimatedSalary']]
y = df.iloc[:,4]        #y=df['Purchased']

#split dataset into x_train x_test y_train y_test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

# feature scaling
from sklearn.preprocessing import MinMaxScaler
ss = MinMaxScaler()
ss.fit(x_train)
x_train_scaled = ss.transform(x_train)
ss.fit(x_test)
x_test_scaled = ss.transform(x_test)

#implement decision tree
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()
classifier.fit(x_train_scaled,y_train)
y_predict = classifier.predict(x_test_scaled)

#accuracy score
print(classifier.score(x_test_scaled,y_test))

#plot the graph
plt.scatter(x_test[y_test==0]['Age'],x_test[y_test==0]['EstimatedSalary'],c='cyan',alpha=0.7)
#plotting the scatter plot, c is color alpha is for transparency  y_test==0 indicates product not purchased
plt.scatter(x_test[y_test==1]['Age'],x_test[y_test==1]['EstimatedSalary'],c='magenta',alpha=0.7)
plt.show()
print("__By Mazhar Solkar")

