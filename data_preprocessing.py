import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import glob
import os

def load_and_preprocess_data(file_paths):
    # Load data from all CSV files
    dfs = [pd.read_csv(file) for file in file_paths]
    data = pd.concat(dfs, ignore_index=True)

    # Extract required columns
    required_columns = ['url', 'domain', 'tld', 'ip', 'url_len', 'https', 'label']
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"One or more required columns are missing from the dataset. Required columns are: {required_columns}")

    X = data[['url', 'domain', 'tld', 'ip', 'url_len', 'https']]
    y = data['label']

    # Label encode the target variable
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Separate numeric and categorical features
    numeric_features = ['url_len']
    categorical_features = ['url', 'domain', 'tld', 'ip', 'https']

    # Preprocessing pipelines for numeric and categorical data
    numeric_transformer = StandardScaler()

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Apply preprocessing
    X = preprocessor.fit_transform(X)

    return X, y, label_encoder

if __name__ == '__main__':
    # This part of the code can be used for testing the preprocessing function independently
    file_paths = glob.glob(os.path.join(os.path.dirname(__file__), '../data/url/results/*.csv'))
    X, y, label_encoders = load_and_preprocess_data(file_paths)
    print(f"Processed data shapes: X.shape = {X.shape}, y.shape = {y.shape}")
    print("Preprocessing completed successfully.")
