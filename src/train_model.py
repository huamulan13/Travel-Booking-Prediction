import pandas as pd
import psycopg2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# koneksi database
conn = psycopg2.connect(
    host="db",
    database="travel_db",
    user="tiket",
    password="tiket123"
)

# QUERY HARUS ADA
query = """
SELECT search_count, booking_history, price_sensitivity, booked
FROM travel_data
"""

# ambil data dari database
df = pd.read_sql(query, conn)

# fitur dan target
X = df[["search_count", "booking_history", "price_sensitivity"]]
y = df["booked"]

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# training model
model = LogisticRegression()
model.fit(X_train, y_train)

# simpan model
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model berhasil dilatih dan disimpan")

# import pandas as pd
# import joblib

# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier

# # load dataset
# # df = pd.read_csv("data/travel_data.csv")

# df = pd.read_sql(query, conn)

# X = df.drop("booked", axis=1)
# y = df["booked"]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model = RandomForestClassifier()

# model.fit(X_train, y_train)

# joblib.dump(model, "model/model.pkl")

# print("Model trained and saved!")