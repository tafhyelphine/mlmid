import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv(r"C:\Users\srihari\Desktop\ML\datasets\abalone.csv")

print(df.head(10))

# Predicting sex with age and height

x=df[['Height','Shell weight']]
y=df['Sex']

encoder=LabelEncoder()
y=encoder.fit_transform(y) # to conver MFI values to 012

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.8,random_state=42)

model=LogisticRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

accuracy=accuracy_score(y_test,y_pred)
print(f"accuracy score {accuracy*100:.2f}%")

cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('actual')
plt.title('confusion matrix')
plt.show()
