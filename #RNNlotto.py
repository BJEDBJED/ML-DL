#RNNlotto
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.optim.lr_scheduler import StepLR

# Załaduj dane
file_path = 'C:/Users/USER/projects/neuralnetwork/lotto.csv'
data = pd.read_csv(file_path, header=None)
column_names = ['Serial Number', 'Date'] + [f'Result_{i}' for i in range(1, 7)]
data.columns = column_names
data_processed = data.drop(['Serial Number', 'Date'], axis=1)

# Skalowanie cech
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_processed)

# Tworzenie sekwencji danych
sequence_length = 5
X, y = [], []
for i in range(len(data_scaled) - sequence_length):
    X.append(data_scaled[i:i + sequence_length])
    y.append(data_scaled[i + sequence_length])
X = np.array(X)
y = np.array(y)

# Podział na zestawy treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Konwersja na tensory PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Definicja modelu RNN
class LottoRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LottoRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x): #TUUUUUUUUUUU
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Parametry modelu
input_size = 6
hidden_size = 64
output_size = 6
num_layers = 2

step_size = 10  # Co ile epok zmniejszać learning rate
gamma = 0.1     # Czynnik zmniejszenia learning rate

# Tworzenie instancji modelu, kryterium i optymalizatora
model = LottoRNN(input_size, hidden_size, output_size, num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

# Przygotowanie DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True)

# Trenowanie modelu
epochs = 100
for epoch in range(epochs):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        scheduler.step()
    if epoch % 10 == 0:
        scheduler.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Finalna strata
print(f'Final Loss: {loss.item():.4f}')

# Przewidywanie
model.eval()
last_sequence = X[-1].reshape(1, sequence_length, -1)
last_sequence = torch.tensor(last_sequence, dtype=torch.float32)
with torch.no_grad():
    predicted_sequence = model(last_sequence)
predicted_numbers = scaler.inverse_transform(predicted_sequence.cpu().numpy().reshape(-1, 6))
print("Predicted lottery numbers:", predicted_numbers)
