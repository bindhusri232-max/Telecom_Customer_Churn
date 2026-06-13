import pandas as pd
from sklearn.ensemble import RandomForestClassifier

data = {
    'MonthlyCharges': [500, 900, 300, 1000, 400],
    'Tenure': [24, 2, 36, 1, 48],
    'Churn': [0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

X = df[['MonthlyCharges', 'Tenure']]
y = df['Churn']

model = RandomForestClassifier()
model.fit(X, y)

result = model.predict([[850, 3]])

print("Prediction:", "Churn" if result[0] == 1 else "No Churn")