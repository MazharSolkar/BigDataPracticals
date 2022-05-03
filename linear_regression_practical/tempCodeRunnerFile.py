lm =linear_model.LinearRegression()
# model = lm.fit(waist,weight)
# print(model.coef_)

# print(model.intercept_)

# print(model.score(waist,weight))

# Waist_new = np.array([97])
# Waist_new = Waist_new.reshape(-1,1)
# Weight_predict = model.predict(Waist_new)
# print(Weight_predict)

# X=([67,78,94])
# X=pd.DataFrame(X)
# Y=model.predict(X)
# Y=pd.DataFrame(Y)
# df = pd.concat([X,Y], axis=1, keys=['Waist_new','Weight_predicted'])
# print(df)

# data.plot(kind='scatter',x='waist_cm',y='weight_kg')
# plt.plot(waist,model.predict(waist),color='red', linewidth=2)
# plt.scatter(Waist_new,Weight_predict, color='black')
# plt.show()