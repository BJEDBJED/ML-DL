import torch
import torch.nn as nn
import torch.nn.functional as ff
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

class MLP(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(MLP,self).__init__()
        self.layer1=nn.Linear(input_size,hidden_size)
        self.layer2=nn.Linear(hidden_size,output_size)
    
    def forward(self,x):
        x=ff.relu(self.layer1(x))
        x=self.layer2(x)
        #x=ff.softmax(self.layer2(x),dim=1)
        return x
    
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

#train_dataset=X_train,y_train
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

input_size = 4  
hidden_size = 64  
output_size = 3  

model = MLP(input_size, hidden_size, output_size)

#criterion=nn.MSELoss() - dla regresji
criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


epochs = 100

for epoch in range(epochs):
    for batch in train_loader:
        X_batch, y_batch = batch

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print(f'Epoki: {epochs}, Strata: {loss.item():.4f}')

with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    print(f'Dokładność modelu: {accuracy * 100:.2f}%')
