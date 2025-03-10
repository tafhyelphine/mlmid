import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression,LogisticRegression 
from sklearn.metrics import accuracy_score,r2_score,mean_squared_error


df=pd.read_csv(r"C:\Users\srihari\Desktop\ML\datasets\cardio.csv",delimiter=";")

print(df.describe())

df['height'].fillna(0)

df['weight'].fillna(0)

#predicting api_hi with height and weight 

x=df[['height','weight']]
y=df[['ap_hi']].fillna(0)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)


mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)


print(f"Mean Squared Error {mse:.2f}")
print(f"R2 value is  {r2:.2f}")


plt.scatter(x_test['height'],y_test,color='red',label='independent variables')

plt.plot(x_test['height'],y_pred,color='blue',label='dependent variable')

plt.xlabel('rings')
plt.ylabel('height')
plt.legend()
plt.show()

