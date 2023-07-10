# ==========================
# Prepping and Encoding Data
# ==========================


import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

import sys

sys.path.append("..")
from data.csv_to_dataset import af


# Ordinal categories in descending order
duration_cat = ["> 240", "180-239", "120-179", "60-119", "< 60"]
votes_cat = [
    "> 100000",
    "50000-99999",
    "20000-49999",
    "10000-19999",
    "5000-9999",
    "2000-4999",
    "1000-1999",
    "500-999",
    "200-499",
    "100-199",
    "50-99",
    "20-49",
    "5-20",
]
start_cat = [
    "2020's",
    "2010's",
    "2000's",
    "1990's",
    "1980's",
    "1970's",
    "1960's",
    "1950's",
    "1877-1949",
]
rating_cat = [
    "Top-Rated",
    "Astounding",
    "Excellent",
    "Great",
    "Very Good",
    "Good",
    "Better than Average",
    "Average",
    "Worse than Average",
    "Bad",
    "Very Bad",
    "Worst-Rated",
]
income_cat = [
    "> $100,000,000",
    "$10,000,000 - $100,000,000",
    "$1,000,000 - $10,000,000",
    "$100,000 - $1,000,000",
    "$10,000 - $100,000",
    "$1,000 - $10,000",
    "< $1,000",
]

ord_cats_I = [duration_cat, votes_cat, start_cat, rating_cat] + [["1", "0"]] * 28
y_cat_I = [income_cat]

ord_cats_R = [duration_cat, votes_cat, income_cat, start_cat] + [["1", "0"]] * 28
y_cat_R = [rating_cat]


def data_split(af, target_index, test_size=1 / 3.0, random_state=1):
    """Separates the data from af into I/O columns based on the target_index.
    Returns training and testing data for the inputs and output"""

    # Splitting data into inputs (X) and outputs (y)
    data = af.values
    y = data[:, target_index : (target_index + 1)]
    X1 = data[:, :target_index]
    X2 = data[:, (target_index + 1) :]
    X = np.concatenate((X1, X2), axis=1)

    # Retrieving column names for the inputs and output
    cols = list(af.columns)
    input_cols = cols[:target_index] + cols[(target_index + 1) :]
    output_col = cols[target_index]

    # Splitting data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )  # random state = seeding

    return [X_train, X_test, y_train, y_test], input_cols, output_col


def transformer(ord_cats, y_cat):
    """Given a list-of-lists of input values and a list of output values,
    returns an input encoder and an output encoder"""

    # Ordinal encoder for the inputs
    col_transform_X = OrdinalEncoder(
        categories=ord_cats, handle_unknown="use_encoded_value", unknown_value=np.nan
    )

    # Ordinal encoder for the output
    col_transform_Y = OrdinalEncoder(
        categories=y_cat, handle_unknown="use_encoded_value", unknown_value=np.nan
    )

    return col_transform_X, col_transform_Y


def encode(split_data, col_transform_X, col_transform_Y):
    """Given training and testing data and transformers for the inputs and output,
    transforms all data based off the training data and returns the encoded data"""

    # Retrieving training and test data
    X_train, X_test, y_train, y_test = split_data

    # Fitting data to training data
    col_transform_X.fit(X_train)
    col_transform_Y.fit(y_train)

    # Transforming data using the fit transformer
    X_train_enc = col_transform_X.transform(X_train)
    X_test_enc = col_transform_X.transform(X_test)
    y_train_enc = col_transform_Y.transform(y_train)
    y_test_enc = col_transform_Y.transform(y_test)

    return X_test_enc, y_test_enc, X_train_enc, y_train_enc


def parse():
    # Setting up and retrieving main function arguments
    parser = argparse.ArgumentParser(
        description="Train (and save) hmm models for POS tagging"
    )
    parser.add_argument(
        "--rating",
        "-r",
        action="store_true",
        help="Output index becomes adjusted rating instead of gross_income",
    )
    args = parser.parse_args()

    # Define args-dependent variables
    target_index = 0
    ord_cats = []
    y_cat = []
    if args.rating:
        target_index = 4
        ord_cats = ord_cats_R
        y_cat = y_cat_R
    else:
        target_index = 2
        ord_cats = ord_cats_I
        y_cat = y_cat_I

    return target_index, ord_cats, y_cat


if __name__ == "__main__":
    """Splits and encodes the data into training and test features"""

    # Setting up and retrieving main function arguments
    target_index, ord_cats, y_cat = parse()

    # Make the training and test data
    data, input_cols, output_col = data_split(af, target_index)
    X_train, X_test, y_train, y_test = data

    # Fit transformers on training data then encode all data
    col_transform_X, col_transform_Y = transformer(ord_cats, y_cat)
    X_test_enc, y_test_enc, X_train_enc, y_train_enc = encode(
        data, col_transform_X, col_transform_Y
    )
