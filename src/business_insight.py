import pandas as pd

def generate_insight(df):

    avg_search = df.groupby("booked")["search_count"].mean()

    print("Average search count:")
    print(avg_search)

    if avg_search[1] > avg_search[0]:
        print("Insight: User yang sering mencari tiket lebih cenderung melakukan booking.")