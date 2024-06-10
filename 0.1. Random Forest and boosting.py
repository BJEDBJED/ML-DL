#Rando Forest and boosting
#Bagging (averaging correlated Decision Trees) vs Random Forest (averaging uncorrelated Decision Trees)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#data
iris=load_iris()

X=iris.data
y=iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#train
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train,y_train)

#predict
y_pred=model.predict(X_test)


#validation
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy}")
classification_report=classification_report(y_test,y_pred)
print("Classification report:\n",classification_report)
confusion_matrix=confusion_matrix(y_test,y_pred)
print("Confusion matrix:\n",confusion_matrix)

# Scatter Plot Visualizing Classes
plt.figure(figsize=(10, 6))

for target, marker, color,label in zip([0, 1,2], ['o', 's','D'], ['forestgreen', 'darkred','navy'],iris.target_names):
    plt.scatter(X[y == target, 0],
                X[y == target, 1],
                marker=marker, color=color, label=label, edgecolor='k')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('Scatter Plot of Iris Data by Class')
plt.legend()
plt.tight_layout()
plt.show()


#2 ADABoost

#Cena kakao a temperatura w Zimbabwe i opady deszczu w mm

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error

np.random.seed(42)

samples=365
rain=np.random.randint(0,350,samples)
temp=np.random.randint(18,38,samples)
noise=np.random.normal(0,50,samples)

kakao_price=20*rain+200*temp+noise

data=pd.DataFrame({'rain':rain,'temp':temp,'kakao_price':kakao_price})

plt.scatter(data['rain'],data['kakao_price'],label='Rain vs Kakao price',color='blue')
plt.scatter(data['temp'],data['kakao_price'],label='Temperature vs Kakao price',color='red')
plt.xlabel('Feature value')
plt.ylabel('price')
plt.legend()
plt.title('Features vs price')
plt.show()

X=data[['rain','temp']]
y=data['kakao_price']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model=AdaBoostRegressor(n_estimators=100,random_state=42)
model.fit(X_train,y_train)

predictions=model.predict(X_test)

mse=mean_squared_error(y_test,predictions)
rmse=np.sqrt(mse)

print(f'Mean square error: {mse}')
print(f'Root mean square error: {rmse}')

plt.scatter(y_test,predictions,color='darkred')
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'k--',lw=3)
plt.xlabel('Actual price')
plt.ylabel('predicted price')
plt.title('actual vs predicted price')
plt.show()

# Gradient Boosting Model

from sklearn.ensemble import GradientBoostingRegressor

GBmodel=GradientBoostingRegressor(n_estimators=100,learning_rate=0.2,max_depth=1,random_state=42)
GBmodel.fit(X_train,y_train)

GBpredictions=GBmodel.predict(X_test)

mse=mean_squared_error(y_test,GBpredictions)
rmse=np.sqrt(mse)

print(f'Mean square error: {mse}')
print(f'Root mean square error: {rmse}')

plt.scatter(y_test,GBpredictions,color='green')
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'k--',lw=3)
plt.xlabel('Actual prices')
plt.ylabel('predicted prices')
plt.title('predicted vs actual prices with GBM')
plt.show()

#XGBoost

import xgboost as xgb

model_xgb=xgb.XGBRegressor(objective ='reg:squarederror', n_estimators = 100, seed = 42)
model_xgb.fit(X_train,y_train)

predictions=model_xgb.predict(X_test)

mse=mean_squared_error(y_test,predictions)
rmse=np.sqrt(mse)

print(f'Mean square error: {mse}')
print(f'Root mean square error: {rmse}')

plt.scatter(y_test,predictions,color='blue')
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'k--',lw=3)
plt.xlabel('Actual prices')
plt.ylabel('predicted prices')
plt.title('predicted vs actual prices with XGBoost')
plt.show()


