import pandas as pd
from sklearn import linear_model
from matplotlib import pyplot as plt 

#read data
df=pd.read_fwf('brain_body.txt')
x_values=df[['Brain']]
y_values=df[['Body']]

#train model on data
body_reg=linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

#visualise results
plt.scatter(x_values,y_values)
plt.plot(x_values,body_reg.predict(x_values))
plt.show()

#make a prediction
prediction=[]#array or a single value
prediction=body_reg.predict(prediction)