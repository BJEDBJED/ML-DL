#SCIKIT GridSearch

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

digits = load_digits()

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

pipeline = make_pipeline(StandardScaler(), MLPClassifier(max_iter=300, random_state=42))

param_grid = {
    'mlpclassifier__hidden_layer_sizes': [(50,), (100,), (50,50)],
    'mlpclassifier__activation': ['tanh', 'relu'],
    'mlpclassifier__solver': ['sgd', 'adam'],
    'mlpclassifier__alpha': [0.0001, 0.05],
    'mlpclassifier__learning_rate': ['constant', 'adaptive'],
}

grid_search = GridSearchCV(pipeline, param_grid, n_jobs=-1, cv=5)

grid_search.fit(X_train, y_train)

print("Najlepsze parametry:", grid_search.best_params_)

predictions = grid_search.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy najlepszego modelu: {accuracy:.2f}')
