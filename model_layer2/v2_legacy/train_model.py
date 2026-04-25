from sklearn.ensemble import RandomForestRegressor

def train_model(X, y):
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)
    return model 