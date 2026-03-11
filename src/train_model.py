import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# load dataset
# df = pd.read_csv("data/travel_data.csv")

df = pd.read_sql(query, conn)

X = df.drop("booked", axis=1)
y = df["booked"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()

model.fit(X_train, y_train)

joblib.dump(model, "model/model.pkl")

print("Model trained and saved!")