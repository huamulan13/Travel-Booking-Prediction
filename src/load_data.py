import pandas as pd
import psycopg2

conn = psycopg2.connect(
    host="db",
    database="travel",
    user="user",
    password="password"
)

query = """
SELECT
search_count,
booking_history,
price_sensitivity,
booked
FROM travel_data;
"""

df = pd.read_sql(query, conn)

print(df.head())