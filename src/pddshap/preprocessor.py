import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
import numpy as np
from typing import Union


class Preprocessor:
    def __init__(self, df: pd.DataFrame, categorical="onehot"):
        # Get categorical/numerical features
        self.cat_idx = df.select_dtypes(include=['object', 'bool', 'category']).columns
        self.num_idx = df.select_dtypes(include=['int64', 'float64']).columns
        self.columns = df.columns

        # Create preprocessor for each feature
        categorical_encoders = \
            {feat_name: OneHotEncoder(sparse=False) for feat_name in self.cat_idx} if categorical == "onehot" \
            else {feat_name: OrdinalEncoder() for feat_name in self.cat_idx}
        self.categorical = categorical
        scalers = {feat_name: StandardScaler() for feat_name in self.num_idx}
        self.preprocessors = {**categorical_encoders, **scalers}
        for key in self.preprocessors:
            self.preprocessors[key].fit(df[[key]])

    def __call__(self, df: Union[pd.DataFrame, np.ndarray]):
        if type(df) != pd.DataFrame:
            df = pd.DataFrame(df, columns=self.columns)
        dfs = []
        for feat_name in df.columns:
            preproc = self.preprocessors[feat_name]
            part_df = df[[feat_name]]
            if feat_name in self.cat_idx and df.dtypes[feat_name] != "category":
                # This means that the df is ordinal-encoded
                # Encode it back to the right categories
                categories = preproc.categories_[0]
                part_df = part_df.replace({idx: categories[idx] for idx in range(len(categories))})
            part_df = preproc.transform(part_df)
            if feat_name in self.cat_idx and self.categorical == "onehot":
                columns = [f"{feat_name}_{category}" for category in preproc.categories_[0]]
                dfs.append(pd.DataFrame(part_df, columns=columns))
            else:
                dfs.append(pd.DataFrame(part_df, columns=[feat_name]))
        return pd.concat(dfs, axis=1)
