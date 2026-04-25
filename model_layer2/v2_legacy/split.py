from sklearn.model_selection import train_test_split

def split_by_fluid(df, X, y, test_size=0.2, random_state=42):

    print("🔀 Split por fluido...")

    fluid_ids = df["fluid_id"].unique()

    train_ids, test_ids = train_test_split(
        fluid_ids,
        test_size=test_size,
        random_state=random_state
    )

    train_mask = df["fluid_id"].isin(train_ids)
    test_mask = df["fluid_id"].isin(test_ids)

    X_train = X[train_mask]
    X_test = X[test_mask]

    y_train = y[train_mask]
    y_test = y[test_mask]

    print(f"Fluidos treino: {len(train_ids)}")
    print(f"Fluidos teste: {len(test_ids)}")

    return X_train, X_test, y_train, y_test