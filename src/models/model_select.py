# MUST BE RUN FROM THIS DIRECTORY!

# ==========================
# Modeling with Top Features
# ==========================

from features.build_features import (
    parse,
    data_split,
    transformer,
    encode
)
from data.csv_to_dataset import af
import numpy as np
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB

from models.model import modeling, stats

import sys
sys.path.append("..")


def chi2_features_select(features, X_train_enc, y_train_enc, k=3):
    """Returns a sorted list of features and a list of feature_indexes in descending order 
    based on the feature's chi-square value"""

    # Calculating chi-square for each input (or feature) using training data
    chis, _ = chi2(X_train_enc, y_train_enc.ravel())

    # Sorting features by chi-square in descending order
    fchi = [[chis[i], features[i], i] for i in range(len(chis))]
    fchi.sort()
    fchi.reverse()
    fchi = fchi[:k]

    f_inds = [[x[2], x[1]] for x in fchi]
    f_inds.sort()
    feats = [x[1] for x in f_inds]
    inds = [x[0] for x in f_inds]

    return feats, inds


if __name__ == "__main__":
    """Performing feature selection on the IMDB Movies Dataset with gross income or an
    adjusted rating score as our target value"""

    # Setting up and retrieving main function arguments
    target_index, ord_cats, y_cat = parse()

    # Make the training and test data
    data, input_cols, output_col = data_split(af, target_index)

    # Fit transformers on training data then encode all data
    col_transform_X, col_transform_Y = transformer(ord_cats, y_cat)
    X_test_enc, y_test_enc, X_train_enc, y_train_enc = encode(
        data, col_transform_X, col_transform_Y)

    # Picking the best 3 features based on their chi-square statistic
    features, inds = chi2_features_select(input_cols, X_train_enc, y_train_enc)
    X_train_enc = X_train_enc[:, inds]
    X_test_enc = X_test_enc[:, inds]

    # Fit and test a Logistic Regression model
    model_LR = modeling(X_train_enc, y_train_enc)
    statLine_LR = stats(model_LR, X_test_enc, y_test_enc,
                        output_col, select=True)

    # Fit and test a Categorical Naive Baynes model
    model_NB = modeling(
        X_train_enc, y_train_enc, classifier=CategoricalNB, max_iter=None
    )
    statLine_NB = stats(model_NB, X_test_enc, y_test_enc,
                        output_col, select=True)
