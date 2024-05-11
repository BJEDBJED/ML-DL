import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

class MyNeuralNetwork:
 
    def __init__(self,input_size):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand(1)
        self.weights2 = np.random.rand(input_size)
        self.bias2 = np.random.rand(1)

    #funkcje aktywacji
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    #zamiast lineara sprobowac ReLU 
    def linear(self,x):
        return x
    
    def relu(self, x):
        return np.maximum(0, x)

    #predykcja na podstawie mnozenia macierzy
    def predict_s(self, inputs):
        pr1 = np.dot(inputs, self.weights) + self.bias
        pr2 = np.dot(inputs, self.weights2) + self.bias2
        output = self.sigmoid(pr1)
        output2 = self.sigmoid(pr2)
        return output, output2

    def predict_l(self, inputs):
        pr1 = np.dot(inputs, self.weights) + self.bias
        pr2 = np.dot(inputs, self.weights2) + self.bias2
        output_lin = self.linear(pr1)
        output_lin2 = self.linear(pr2)
        return output_lin, output_lin2

    def predict_relu(self, inputs):
        pr1 = np.dot(inputs, self.weights) + self.bias
        pr2 = np.dot(inputs, self.weights2) + self.bias2
        output_relu = self.relu(pr1)
        output_relu2 = self.relu(pr2)
        return output_relu, output_relu2
    
    #obliczyc blad predykcji, aktualizowac wagi i bias   
    def training(self, X_train, y_train, cycles, learning_rate=0.001):
        for cycle in range(cycles):
            for inputs,target in zip(X_train,y_train):
                output1, output2 = self.predict_s(inputs)

                error1 = target - output1
                error2 = target - output2

                self.weights += learning_rate * error1 * output1 * (1 - output1) * inputs
                self.bias += learning_rate * error1 * output1 * (1 - output1)

                self.weights2 += learning_rate * error2 * output2 * (1 - output2) * inputs
                self.bias2 += learning_rate * error2 * output2 * (1 - output2)

        #mierzenie dokladnosci modeli
    """
    def model_accuracy(self, X_test, y_test):
        correct_predictions = 0
        total_samples = len(y_test)

        for inputs, target in zip(X_test, y_test):
            prediction, _ = self.predict_s(inputs)
            predicted_class = np.argmax(prediction)
            if predicted_class == target:
                correct_predictions += 1
        accuracy = correct_predictions / total_samples
        return accuracy    
    """

if __name__ == "__main__":
    iris=load_iris()
    X=iris.data
    y=iris.target

    #print(X)
    #preprocessing
    #scaler=StandardScaler()
    #X=scaler.fit_transform(X)
    X=StandardScaler().fit_transform(X)

    #PCA -> less is beter
    pca=PCA(3)
    best_comp=pca.fit_transform(X)
    #print(best_comp)
   
    X_train, X_test, y_train, y_test = train_test_split(best_comp, y, test_size=0.2, random_state=42)
    
    input_size=X_train.shape[1]
    neural_network = MyNeuralNetwork(input_size)

    #moj trening - 0.97/0.98 dla 2k cykli
    neural_network.training(X_train, y_train, cycles=100)
    #accuracy1 = neural_network.model_accuracy(X_test, y_test)
    #print("My model accuracy:",accuracy1)

    #gotowy model MLPC
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
    model.fit(X_train, y_train)
    #accuracy2 = model.score(X_test, y_test)
    #print("MLP model accuracy:",accuracy2)

    #inputs = np.array([0.6])
    sample_input=X_test[0]

    #prediction1, prediction2 = neural_network.predict_s(inputs)

    prediction_sigmoid = neural_network.predict_s(sample_input)
    print("My model - Sigmoid Prediction:", prediction_sigmoid)

    prediction_relu = neural_network.predict_relu(sample_input)
    print("My model - Relu Prediction:", prediction_relu)

    prediction_mlpc = model.predict([sample_input])
    print("MLP Classifier Prediction:", prediction_mlpc)

    # Uzyskanie predykcji dla obu modeli
    predictions_my_model_sig = []
    for input_data in X_test:
        prediction_sigmoid = neural_network.predict_s(input_data)
        predictions_my_model_sig.append(np.argmax(prediction_sigmoid))

    predictions_my_model_rel = []
    for input_data in X_test:
        prediction_relu = neural_network.predict_relu(input_data)
        predictions_my_model_rel.append(np.argmax(prediction_relu))

    predictions_mlpc = model.predict(X_test)

    print("My Model Sigmoid:")
    print(classification_report(y_test, predictions_my_model_sig))

    print("My Model Relu:")
    print(classification_report(y_test, predictions_my_model_rel))

    print("MLP Classifier:")
    print(classification_report(y_test, predictions_mlpc))
    

