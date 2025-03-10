import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

df = pd.read_csv(r"C:\Users\srihari\Desktop\ML\datasets\abalone.csv")

# add an age group column
df['agegroup']=df['Rings'].apply(lambda x: "Young" if x<=10 else "Old")

print(df.head(5))

x=df[['Height','Length','Diameter','Whole weight']]
y=df['agegroup']


#convert age group values to 0 and 1
encoder=LabelEncoder()
y=encoder.fit_transform(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#important in knn
scaler=StandardScaler()
scaler.fit_transform(x_train)
scaler.transform(x_test)


model=KNeighborsClassifier(n_neighbors=5)
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

accuracy=accuracy_score(y_test,y_pred)
print(f"accuracy of model {accuracy*100:.2f}%")


#confusion matrix to confuse teacher 
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=['Old','Young'],yticklabels=['Old','Young'])
plt.show()
