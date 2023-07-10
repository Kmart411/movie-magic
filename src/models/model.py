# MUST BE RUN FROM THIS DIRECTORY!

# ==============================
# Creating and Testing the Model
# ==============================

from features.build_features import parse, data_split, transformer, encode
from data.csv_to_dataset import af
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)

import sys

sys.path.append("..")


def modeling(X_train_enc, y_train_enc, classifier=LogisticRegression, max_iter=10000):
    """Returns a machine learning model fit on the given training data"""

    # Setup model
    model = None
    if classifier == LogisticRegression:
        model = classifier(max_iter=max_iter)
    else:
        model = classifier()

    # Fit on the training set; takes time to converge
    model.fit(X_train_enc, y_train_enc.ravel())
    return model


def stats(model, X_test_enc, y_test_enc, target, select=False):
    """Generates a txt file of statistics evaluating the given machine learning model"""

    # Predict on test set
    yhat = model.predict(X_test_enc)

    # Evaluate predictions through statistics
    acc = accuracy_score(y_test_enc, yhat)

    # Output categories are equally weighted for remaining metrics

    # Balanced accuracy = (recall + true_negative_rate) / 2
    b_acc = balanced_accuracy_score(y_true=y_test_enc, y_pred=yhat)

    # Calculates the precision, recall, and F1 score using the macro-average
    p, r, f1, _ = precision_recall_fscore_support(
        y_true=y_test_enc, y_pred=yhat, average="macro"
    )

    # Writing metrics (rounded to the nearest hundreth) to a txt file
    folder1 = ''
    if type(model) == CategoricalNB:
        folder1 = 'NB/'
    else:
        folder1 = 'LR/'

    folder2 = ''
    if select:
        folder2 = "select/"
    else:
        folder2 = "all/"

    fp = open("../../model_metrics/" + folder1 + folder2 +
              "mm_" + target + ".txt", "w", encoding="utf-8")
    fp.write("Accuracy: %.2f\n" % (acc * 100))
    fp.write("Balanced Accuracy: %.2f\n" % (b_acc * 100))
    fp.write("Precision: %.2f\n" % (p * 100))
    fp.write("Recall: %.2f\n" % (r * 100))
    fp.write("F1: %.2f\n" % (f1 * 100))
    fp.close()

    return acc, b_acc, p, r, f1


if __name__ == "__main__":
    """Making a logisitic regression model using features from the IMDB Movies Dataset with gross income or an
    adjusted rating score as our target value"""

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

    # Fit and test a Logistic Regression model
    model_LR = modeling(X_train_enc, y_train_enc)
    statLine_LR = stats(model_LR, X_test_enc, y_test_enc, output_col)

    # Fit and test a Categorical Naive Baynes model
    model_NB = modeling(
        X_train_enc, y_train_enc, classifier=CategoricalNB, max_iter=None
    )
    statLine_NB = stats(model_NB, X_test_enc, y_test_enc, output_col)
