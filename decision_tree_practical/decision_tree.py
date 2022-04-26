from audioop import minmax
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read dataframe
df = pd.read_csv('C:/0_MSc_IT_Notes/Big Data Analytics/practicals/decision_tree_practical/social.csv')
print(df)

#set independent variable x and dependent variable y
# x= df.iloc[:,[2,3]]
# y= df.iloc[:,4]
x= df[['Age','EstimatedSalary']]
y = df['Purchased']
#split the dataset x and y in x_train x_test y_train y_test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=0)

print("Training data:",x_train)
print('*******************')
print("Testing data:",x_test)

# feature scaling
from sklearn.preprocessing import MinMaxScaler
ss = MinMaxScaler()
ss.fit(x_train)
x_train = ss.transform(x_train)
ss.fit(x_test)
x_test = ss.transform(x_test)

# train the classifier 
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(x_train,y_train)

# predict the output
y_pred = classifier.predict(x_test)

plt.scatter(x_test[y_test==0]['Age'],x_test[y_test==0]['EstimatedSalary'],c='red',alpha=0.7)
#plotting the scatter plot, c is color alpha is for transparency  y_test==0 indicates product not purchased
plt.scatter(x_test[y_test==1]['Age'],x_test[y_test==1]['EstimatedSalary'],c='blue',alpha=0.7)
print(classifier.score(x_test,y_test))
plt.show()



# import pandas as pd
# import matplotlib.pyplot as plt
# df=pd.read_csv('C:/0_MSc_IT_Notes/Big Data Analytics/practicals/decision_tree_practical/social.csv')
# #df.head()
# print(df)
# x=df[['Age','EstimatedSalary']]
# #x=df.iloc[:,[2,3]]
# print(x)
# y=df['Purchased']
# #y=df.iloc[:,4]
# print(y)
# #x.shape
# #y.shape
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
# x_train.shape,y_train.shape,x_test.shape,y_test.shape
# from sklearn.preprocessing import MinMaxScaler
# ss=MinMaxScaler()
# ss.fit(x_train)
# x_train_scaled=ss.transform(x_train)
# ss.fit(x_test)
# x_test_scaled=ss.transform(x_test)
# x_train_scaled
# x_test_scaled
# from sklearn.tree import DecisionTreeClassifier
# model_DT= DecisionTreeClassifier()
# model_DT.fit(x_train_scaled,y_train)
# y_predict=model_DT.predict(x_test_scaled)
# plt.scatter(x_test[y_test==0]['Age'],x_test[y_test==0]['EstimatedSalary'],c='red',alpha=0.7)
# #plotting the scatter plot, c is color alpha is for transparency  y_test==0 indicates product not purchased
# plt.scatter(x_test[y_test==1]['Age'],x_test[y_test==1]['EstimatedSalary'],c='blue',alpha=0.7)
# print(model_DT.score(x_test_scaled,y_test))
# plt.show()
