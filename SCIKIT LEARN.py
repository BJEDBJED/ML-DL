#SCIKIT LEARN

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler  
import matplotlib.pyplot as plt

digits=load_digits()

#for i in range(5):
#    plt.matshow(digits.images[i], cmap='gray')
#    plt.title(f"Obrazek: {digits.target[i]}")
#    plt.show()
#print(digits.data[:5])
#print(digits.target[:5])

X_train, X_test, y_train, y_test=train_test_split(digits.data,digits.target,test_size=0.15)

scaler=StandardScaler()

scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

mlp=MLPClassifier(hidden_layer_sizes=(200,),activation='relu',solver='adam',batch_size=20, random_state=42)

mlp.fit(X_train,y_train)

predictions=mlp.predict(X_test)
accuracy=accuracy_score(y_test,predictions)

print(f'Accuracy to {accuracy}')
