from sklearn.metrics import mean_absolute_error, r2_score

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)

    return {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "R2": float(r2_score(y_test, y_pred))
    } 