#6_12-JOBRECOM

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data_json = pd.read_json('D:\\Datasets\\JJIT\\2023-09\\2023-09-01.json')
data_json.fillna(value=0)

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

print(f'DataFrame size: {df.shape}')

df = df[df['city'] == 'Kraków']

# missing salary
df = df.dropna(subset=['salary_from', 'salary_to'])

title = LabelEncoder()
experience = LabelEncoder()


df['title_encoded'] = title.fit_transform(df['title'])
df['experience_encoded'] = experience.fit_transform(df['experience_level'])


X = df[['python_level', 'experience_encoded']]
y = df['title_encoded']


# samples amount check
if X.shape[0] == 0 or y.shape[0] == 0:
    raise ValueError("No samples")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy = {accuracy}')

input_data = pd.DataFrame({
    'python_level': [5],
    'experience_encoded': [experience.transform(['mid'])[0]]
})

predicted_job_encoded = model.predict(input_data)
predicted_job = title.inverse_transform(predicted_job_encoded)

print(f'Best job for someone with level 5 in Krakow is: {predicted_job[0]}')

job_details = df[df['title_encoded'] == predicted_job_encoded[0]].values[0]

print("Details of job:")
print(job_details)
