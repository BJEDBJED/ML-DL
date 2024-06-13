#6-13 - Job Recomendation with NN

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as Func
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os



data_json = pd.read_json('D:\\Datasets\\JJIT\\2023-09\\2023-09-01.json')
data_json = data_json.fillna(value=0)

data = []
for job in data_json.to_dict(orient='records'):
    for skill in job['skills']:
        if skill['name'] == 'Python':
            employment_type = job['employment_types'][0] if job['employment_types'] else None
            salary_from = employment_type['salary']['from'] if employment_type and 'salary' in employment_type and employment_type['salary'] else None
            salary_to = employment_type['salary']['to'] if employment_type and 'salary' in employment_type and employment_type['salary'] else None
            
            data.append({
                'title': job['title'],
                'city': job['city'],
                'experience_level': job['experience_level'],
                'salary_from': salary_from,
                'salary_to': salary_to,
                'python_level': skill['level']
            })

df = pd.DataFrame(data)

df = df[df['city'] == 'Kraków']

# missing salary
df = df.dropna(subset=['salary_from', 'salary_to'])

title = LabelEncoder()
experience = LabelEncoder()

df['title_encoded'] = title.fit_transform(df['title'])
df['experience_encoded'] = experience.fit_transform(df['experience_level'])


X = df[['python_level', 'experience_encoded','salary_from','salary_to']].values
y = df['title_encoded'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class MLP(nn.Module):
    def __init__(self,input_lay,hidden_lay,output_lay):
        super(MLP,self).__init__()
        self.layer1 = nn.Linear(input_lay, hidden_lay)
        self.dropout1 = nn.Dropout(0.5)
        self.layer2 = nn.Linear(hidden_lay, hidden_lay)
        self.dropout2 = nn.Dropout(0.5)
        self.layer3 = nn.Linear(hidden_lay, output_lay)

    def forward(self,x):
        x=Func.relu(self.layer1(x))
        x = self.dropout1(x)
        x = Func.relu(self.layer2(x))
        x = self.dropout2(x)
        x = self.layer3(x)
        return x
    
input_lay=4
hidden_lay=32
output_lay=len(df['title'].unique())

model=MLP(input_lay,hidden_lay,output_lay)
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)

num_epochs=50

for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

print("Done!")

model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    accuracy = accuracy_score(y_test_tensor, predicted)
    print(f'Test Accuracy: {accuracy}')
    print(classification_report(y_test_tensor, predicted))
    print(confusion_matrix(y_test_tensor, predicted))

predicted_titles = title.inverse_transform(predicted.numpy())
test_df = pd.DataFrame(X_test, columns=['python_level', 'experience_encoded', 'salary_from', 'salary_to'])
test_df['predicted_title'] = predicted_titles

# save to csv
output_path = os.path.join(os.getcwd(), 'predicted_jobs.csv')
test_df.to_csv(output_path, index=False)
print(f'Predicted jobs saved to: {output_path}')
print(test_df.head())