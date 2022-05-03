import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import linear_model

data=pd.read_csv("C:/0_MSc_IT_Notes/Big Data Analytics/practicals/linear_regression_practical/weightwaist.csv")
print(data)
data.plot(kind='scatter',x='waist_cm',y='weight_kg')
plt.title('__By Mazhar Solkar')
plt.show()

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
lm =linear_model.LinearRegression()
model = lm.fit(waist,weight)

print('\nCoefficient')
print(model.coef_)
print('\nintercept')
print(model.intercept_)
print('\nscore')
print(model.score(waist,weight))

Waist_new = np.array([97])
Waist_new = Waist_new.reshape(-1,1)
Weight_predict = model.predict(Waist_new)
print('\nWeight_predict')
print(Weight_predict)

X=([67,78,94])
X=pd.DataFrame(X)
Y=model.predict(X)
Y=pd.DataFrame(Y)
df = pd.concat([X,Y], axis=1, keys=['Waist_new','Weight_predicted'])
print('\ndf')
print(df)

data.plot(kind='scatter',x='waist_cm',y='weight_kg')
plt.plot(waist,model.predict(waist),color='red', linewidth=2)
plt.scatter(Waist_new,Weight_predict, color='black')
plt.title('__By Mazhar Solkar')
plt.show()
print('__By Mazhar Solkar')

