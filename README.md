import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = {
    'MonthlyCharges': [70, 90, 30, 100, 40],
    'Tenure': [12, 24, 6, 36, 8],
    'Churn': [1, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

X = df[['MonthlyCharges', 'Tenure']]
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))