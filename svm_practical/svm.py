import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# inputfilename = 'C:\0_MSc_IT_Notes\Big Data Analytics\practicals\svm_practical\social.csv'
#make sure to give comman slash(forward slash / in path)

inputfilename = 'C:/0_MSc_IT_Notes/Big Data Analytics/practicals/svm_practical/social.csv'
df = pd.read_csv(inputfilename)
print(df)

x = df.iloc[:,[2,3]]
y = df.iloc[:,4]

#Splitting the dataset into the training set and test set

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=0)

print("Training data:",x_train)
print('*******************')
print("Testing data:",x_test)

#Feature scaling
