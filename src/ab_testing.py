from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def run_ab_test(X_train, X_test, y_train, y_test):

    model_a = LogisticRegression()
    model_b = RandomForestClassifier()

    model_a.fit(X_train, y_train)
    model_b.fit(X_train, y_train)

    pred_a = model_a.predict(X_test)
    pred_b = model_b.predict(X_test)

    acc_a = accuracy_score(y_test, pred_a)
    acc_b = accuracy_score(y_test, pred_b)

    print("Model A Accuracy:", acc_a)
    print("Model B Accuracy:", acc_b)

    return model_a if acc_a > acc_b else model_b