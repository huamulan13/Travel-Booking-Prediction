import datetime

def log_prediction(data, prediction):

    with open("logs/prediction_logs.txt", "a") as f:
        f.write(f"{datetime.datetime.now()} | {data} | {prediction}\n")