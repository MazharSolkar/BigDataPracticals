import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#read dataframe data
data = pd.read_csv("C:/0_MSc_IT_Notes/Big Data Analytics/practicals/linear_regression_practical/weightwaist.csv")
print(data)

#data.plot to view to structure of our data
data.plot(kind='scatter',x='waist_cm',y='weight_kg')
plt.show()

#data.corr() correlation
print('\nCorrelation')
print(data.corr())

# Defining dependent and independent variables
waist=pd.DataFrame(data['waist_cm'])
weight=pd.DataFrame(data['weight_kg'])
print('\nwaist')
print(waist)
print('\nweight')
print(weight)

#implementing linear regression
from sklearn import linear_model
lm =linear_model.LinearRegression()
model = lm.fit(waist,weight)

#model.coef_
print('\nCoefficient')
print(model.coef_)
#model.intercept_
print('\nintercept')
print(model.intercept_)
#model.score
print('\nscore')
print(model.score(waist,weight))

#predict value
Waist_new = ([[97]])
Weight_predict = model.predict(Waist_new)
print('\nWeight_predict')
print(Weight_predict)

data.plot(kind='scatter',x='waist_cm',y='weight_kg')
plt.plot(waist,model.predict(waist),color='red', linewidth=2)
plt.scatter(Waist_new,Weight_predict, color='black')
plt.title('__By Abhijeet Maity')
plt.show()
print('__By Abhijeet Maity')

#predict more values
x = [85,90,60]
x = pd.DataFrame(x)
y = model.predict(x)
y = pd.DataFrame(y)
df = pd.concat([x,y],axis=1,keys=['waist_new','weight_new'])
print("\nnew predicted values")
print(df)


