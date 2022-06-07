from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import fetch_openml


# TODO will need a class that handles onehot encoding of subsets of features
# TODO and re-using categories


def get_dataset_model(args):
    ds = fetch_openml(**args)
    # Drop nans
    y = ds.target[ds.data.notnull().all(axis=1)]
    df = ds.data.dropna().reset_index(drop=True)

    # Encode (binary) output into 1/0
    y = LabelEncoder().fit_transform(y)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

    # Get categorical/numerical features
    cat_idx = df.select_dtypes(include=['object', 'bool', 'category']).columns
    num_idx = df.select_dtypes(include=['int64', 'float64']).columns

    # One-hot encoder for the categorical features
    ohe = OneHotEncoder()

    # StandardScaler for the numerical features
    scaler = StandardScaler()

    # Transform and normalize the data
    ct = ColumnTransformer([
        ("onehot", ohe, cat_idx),
        ("scaler", scaler, num_idx)
    ], remainder="passthrough")

    # Attach a GradientBoostingClassifier
    model = Pipeline([
        ("transform", ct),
        ("gbc", GradientBoostingClassifier())
    ])
    model.fit(X_train, y_train)

    print(f"Balanced accuracy: {balanced_accuracy_score(y_test, model.predict(X_test)):.2f}")
    return X_train, X_test, y_train, y_test, model
