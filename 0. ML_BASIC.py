# MACHINE LEARNING BASICS

#1. Linear Regression
#prognozowanie ciaglych wartosci/zwiazki miedzy zmiennymi/zaleznosci

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#Zakupy to ilosc produktow zakupionych/wydatki to koszt

zakupy=np.array([2,3,4,5,5,7,8]).reshape((-1,1))
wydatki=np.array([10,20,40,50,60,70,90])

model=LinearRegression()

model.fit(zakupy,wydatki)

zakupy_nowy_tydzien=6
przewidywane_wydatki=model.predict([[zakupy_nowy_tydzien]])

plt.scatter(zakupy,wydatki,color='blue')
plt.plot(zakupy,model.predict(zakupy),color='green')
plt.title("Zakupy vs wydatki")
plt.xlabel("Zakupione produkty")
plt.ylabel("Wysatki/koszt")
plt.grid(True)
plt.show()

print(f"jesli kupie {zakupy_nowy_tydzien} produktow to zaplace {przewidywane_wydatki}")

#2. Logistic Regression
#wyniki ktore maja 2 mozliwosci/prawdopodobienstow wystapienia zdarzen(tak/nie)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#cena produktu vs decyzja zakupu

cena=np.array([12,13,14,16,18,20,21,22]).reshape(-1,1)
zakup=np.array([0,1,1,1,0,0,0,1]) #0 - brak zakupu / 1 - zakup

model=LogisticRegression()
model.fit(cena,zakup)

predict_cena=17.1
predict_zakup=model.predict([[predict_cena]])

plt.scatter(cena,zakup,color='green')
plt.plot(cena,model.predict_proba(cena)[:,1],color='red')
plt.title('Cena piwa vs decyzja zakupu')
plt.xlabel('cena')
plt.ylabel('zakup')
plt.axvline(x=predict_cena,color='lightgreen',linestyle='--')
plt.axhline(y=0.5, color='grey',linestyle='--')
plt.show()

print(f'Klient {"kupi piwo" if predict_zakup==1 else "nie kupi piwa"} w cenie {predict_cena}.')

#3. Linear Discriminant Analysis
#rozdielanie na pare klas/redukcja wymiarow

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#czy polubisz ta osobe po oczach i wlosach?
#([oczy 1-zielone,2-brazowe,3-czarne,4-niebieskie][wlosy 1-blond,2-czarne,3-szatynowe,4-rude])
wyglad=np.array([[1,3],[2,4],[4,3],[3,2],[3,1],[4,4],[3,3],[1,2]])
ocena=np.array([1,0,0,1,1,1,0,1]) #1-lubie 0-nie lubie

model=LinearDiscriminantAnalysis()

model.fit(wyglad,ocena)

nowa_osoba=np.array([[1,1]])
moja_ocena=model.predict(nowa_osoba)

plt.scatter(wyglad[:,0],wyglad[:,1],c=ocena,cmap='viridis',marker='o')
plt.scatter(nowa_osoba[:,0],nowa_osoba[:,1],color='darkred',marker='x')
plt.title('Czy polubisz osobe na podstawie jej wlosow i oczu')
plt.xlabel('oczy')
plt.ylabel('wlosy')
plt.show()

print(f"{'polubie' if moja_ocena==1 else 'nie polubie'} osobe z oczami {nowa_osoba[0,0]} i wlosami {nowa_osoba[0,1]}")

#4. Naive Bayes
#klasyfikacja tekstow/ szukanie wzorcow

from sklearn.naive_bayes import GaussianNB

#dlugosc wystapienia polityka w min i typ parti[1-PO,2-PIS,3-Konf,4-Lewica], a poparcie

wystep=np.array([[23,1],[12,4],[2,4],[15,3],[31,1],[20,2],[43,3],[13,2]])
poparcie=np.array([1,1,1,0,0,0,1,1])

model=GaussianNB()
model.fit(wystep,poparcie)

nowy_wystep=np.array([[15,1]])
przew_poparcie=model.predict(nowy_wystep)

plt.scatter(wystep[:,0],wystep[:,1],c=poparcie,cmap="viridis",marker='o')
plt.scatter(nowy_wystep[:,0],nowy_wystep[:,1],color='darkred',marker='x')
plt.title('Poparcie polityka, a dlugosc przemowienia i typ partii')
plt.xlabel('DLugos przemowienia')
plt.ylabel('Typ partii')
plt.show()

print(f"Przemowienie o dlugosci {nowy_wystep[0,0]} min polityka z partii {nowy_wystep[0,1]} {'zyska poparcie' if przew_poparcie==1 else 'nie zyska poparcia'}")

#5. Decision Tree

#wyslane maile a ilosc bledow

from sklearn.tree import DecisionTreeRegressor, plot_tree

sent_emails=np.array([21,23,25,31,15,32,41,47]).reshape(-1,1)
errors=np.array([0,1,0,1,0,2,3,4])

model=DecisionTreeRegressor(max_depth=3)
model.fit(sent_emails,errors)

new_emails=np.array([[30]])
predicted_errors=model.predict(new_emails)

plt.figure(figsize=(12,8))
plot_tree(model,filled=True,rounded=True,feature_names=["sent emails"])
plt.title("Decision Tree Regresor Tree")
plt.show()

plt.scatter(sent_emails,errors,color='darkred')
plt.plot(np.sort(sent_emails,axis=0),model.predict(np.sort(sent_emails,axis=0)),color="orange")
plt.scatter(new_emails,predicted_errors,color="green")
plt.title("wyslane maile a bledy")
plt.xlabel("wyslane maile")
plt.ylabel("bledy")
plt.grid(True)
plt.show()

print(f"Przy wysylce {new_emails[0]} maili, mozemy spodziewac sie : {predicted_errors[0]:.2f} bledow.")

#6. BAGGING

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree  # Ensure plot_tree is imported
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Sample Data
clients_data = np.array([[2000, 60], [2500, 45], [1800, 75], [2200, 50], [2100, 62], [2300, 70], [1900, 55], [2000, 65]])
weight_loss = np.array([3, 2, 4, 3, 3.5, 4.5, 3.7, 4.2])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(clients_data, weight_loss, test_size=0.25, random_state=42)

# Creating a Bagging Model
base_estimator = DecisionTreeRegressor(max_depth=4)
model = BaggingRegressor(base_estimator=base_estimator, n_estimators=10, random_state=42)

# Training the Model
model.fit(X_train, y_train)

# Prediction & Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Displaying Prediction and Evaluation
print(f"True weight loss: {y_test}")
print(f"Predicted weight loss: {y_pred}")
print(f"Mean Squared Error: {mse:.2f}")

# Visualizing One of the Base Estimators (if desired)
plt.figure(figsize=(12, 8))
tree = model.estimators_[0]
plt.title('One of the Base Decision Trees from Bagging')
plot_tree(tree, filled=True, rounded=True, feature_names=["Calorie Intake", "Workout Duration"])
plt.show()