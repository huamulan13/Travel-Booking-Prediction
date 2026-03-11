import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/travel_data.csv")

X = df.drop("booked", axis=1)
y = df["booked"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = joblib.load("model/model.pkl")

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))