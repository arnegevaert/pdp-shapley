import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class Preprocessor:
    def __init__(self, df: pd.DataFrame):
        # Get categorical/numerical features
        cat_idx = df.select_dtypes(include=['object', 'bool', 'category']).columns
        num_idx = df.select_dtypes(include=['int64', 'float64']).columns

        # Create preprocessor for each feature
        onehot_encoders = {feat_name: OneHotEncoder(sparse=False) for feat_name in cat_idx}
        scalers = {feat_name: StandardScaler() for feat_name in num_idx}
        self.preprocessors = {**onehot_encoders, **scalers}
        for key in self.preprocessors:
            self.preprocessors[key].fit(df[[key]])

    def __call__(self, df: pd.DataFrame):
        dfs = []
        for feat_name in df.columns:
            preproc = self.preprocessors[feat_name]
            part_df = preproc.transform(df[[feat_name]])
            if part_df.shape[1] > 1:
                columns = [f"{feat_name}_{category}" for category in preproc.categories_[0]]
                dfs.append(pd.DataFrame(part_df, columns=columns))
            else:
                dfs.append(pd.DataFrame(part_df, columns=[feat_name]))
        return pd.concat(dfs, axis=1)
