import pandas as pd


def load_train_data(path="../data/raw/train.csv"):
    """Load the training dataset."""
    return pd.read_csv(path, parse_dates=True)


def load_test_data(path="../data/raw/test.csv"):
    """Load the test dataset."""
    return pd.read_csv(path)
