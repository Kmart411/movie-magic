# MUST BE RUN FROM THIS DIRECTORY!

# ===========
# Graphs
# ===========


import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2

import sys

sys.path.append("..")
from data.csv_to_dataset import af

from features.build_features import (
    parse,
    data_split,
    transformer,
    encode,
    duration_cat,
    votes_cat,
    income_cat,
    start_cat,
    rating_cat,
)


def chi2_features(features, X_train_enc, y_train_enc):
    """Returns a sorted list of chi-square, feature pairs in descending order"""

    # Calculating chi-square for each input (or feature) using training data
    chis, _ = chi2(X_train_enc, y_train_enc.ravel())

    # Sorting features by chi-square in descending order
    fchi = [[chis[i], features[i]] for i in range(len(chis))]
    fchi.sort()
    fchi.reverse()
    return fchi


def plotting(af, target, chi_features):
    """Plots all inputs and their chi-square statistic relative to the target (output)
    and a percent frequency bar chart for each of the top 3 features"""

    inputs = plot_chi(target, chi_features)
    plot_freq(af, target, inputs)


def plot_chi(target, chi_features):
    """Plots all inputs and their chi-square statistic relative to the target (output).
    Returns a list of features sorted by their chi-square statistic"""

    # Getting the name for the output variable
    out_title = ""
    if target == "adjusted_rating":
        out_title = "Adjusted Rating"
    else:
        out_title = "Gross Income"

    # Separating list into features and chi-square values
    inputs = [x[1] for x in chi_features]
    chi = [x[0] for x in chi_features]

    # Horizontal Bar Plot
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.barh(inputs, chi)

    # Remove axes splines
    for s in ["top", "bottom", "left", "right"]:
        ax.spines[s].set_visible(False)

    # Remove x, y Ticks
    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")

    # Add padding between axes and labels
    ax.xaxis.set_tick_params(pad=5)
    ax.yaxis.set_tick_params(pad=10)

    # Add x, y gridlines
    ax.grid(visible=True, color="grey", linestyle="-.", linewidth=0.5, alpha=0.2)

    # Show top values
    ax.invert_yaxis()
    # Add annotation to bars
    for i in ax.patches:
        plt.text(
            i.get_width() + 0.2,
            i.get_y() + 0.5,
            str(round((i.get_width()), 2)),
            fontsize=10,
            fontweight="bold",
            color="grey",
        )

    # Add Titles
    ax.set_xlabel("Chi-square")
    ax.set_ylabel("Features")
    ax.set_title("Chi-square Values for " + out_title + " Features", loc="left")

    # Eponential scaling for chi-square values
    plt.xscale("symlog")

    # Save Plot
    plt.savefig("../../reports/figures/chi_square_" + target + ".png", format="png")
    plt.close()

    # List of features to be passed to plot_freq
    return inputs


def plot_freq(af, target, inputs, k=3):
    """Plots a percent frequency bar chart for each of the top 3 features"""

    # Getting the name for the output variable
    out_title = ""
    if target == "adjusted_rating":
        out_title = "Adjusted Rating"
    else:
        out_title = "Gross Income"

    # Selects the top k inputs
    inputs = inputs[:k]

    # Getting the values for the output variable
    out_cats = []
    if target == "adjusted_rating":
        out_cats = rating_cat
    else:
        out_cats = income_cat

    # Makes a plot for each feature
    for feature in inputs:

        # Getting the name and values for the input variable
        in_cats = []
        in_title = ""
        if feature == "duration":
            in_cats = duration_cat
            in_title = "Duration"
        elif feature == "votes":
            in_cats = votes_cat
            in_title = "Votes"
        elif feature == "gross_income":
            in_cats = income_cat
            in_title = "Gross Income"
        elif feature == "start_year":
            in_cats = start_cat
            in_title = "Start Year"
        elif feature == "adjusted_rating":
            in_cats = rating_cat
            in_title = "Adjusted Rating"
        else:
            in_cats = ["1", "0"]
            in_title = feature

        # Making a dictionary such that each key is an output value and
        # each value is a list such that the ith value of that list
        # is the percentage of seeing that output value given that
        # the input takes on the ith-ranked input value
        #
        # Ex: input = shape, input_categories = [square, circle], output = color
        #     dict['green'] = [0.5, 0.6]
        #     This implies that given the shape is a square, there is 50% probability
        #     that the object's color is green. And given the shape is a circle,
        #     this is a 60% chance.
        values = {}
        for out_cat in out_cats:
            af_out = af[af[target] == out_cat].reset_index(drop=True)
            for in_cat in in_cats:
                af_in = af[af[feature] == in_cat].reset_index(drop=True)
                af_io = af_out[af_out[feature] == in_cat].reset_index(drop=True)
                pFreq = (1.0 * af_io.shape[0]) / af_in.shape[0]

                if out_cat in values:
                    values[out_cat] += [pFreq]
                else:
                    values[out_cat] = [pFreq]

        label_locs = 0
        if len(in_cats) > 2:
            label_locs = np.arange(len(in_cats)) * int(
                len(in_cats) / 1.5
            )  # the x_axis label locations
        else:
            label_locs = np.arange(len(in_cats)) * 5 * int(
                len(in_cats) / 1.5
            )
        width = 0.25  # the width of the bars
        multiplier = 0

        # Making the multi bar chart
        fig, ax = plt.subplots(figsize=(32, 18))

        for attribute, measurement in values.items():
            offset = width * multiplier
            ax.bar(label_locs + offset, measurement, width, label=attribute)
            multiplier += 1

        # Setting the y-axis such that there's room for the legend
        ax.set_ylim(0, 1.1)

        # Addding labels
        ax.set_xlabel(in_title + " Categories", labelpad=10, fontdict={"fontsize": 30})
        ax.set_ylabel("% Frequency", labelpad=20, fontdict={"fontsize": 30})
        ax.set_title(
            "% Frequency of " + out_title + " Categories Over " + in_title,
            fontdict={"fontsize": 50},
        )
        ax.set_xticks(label_locs + width, in_cats)
        ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])
        ax.legend(loc="upper left", ncols=len(in_cats))

        plt.savefig(
            "../../reports/figures/freq_" + target + "_" + feature + ".png",
            format="png",
        )
        plt.close()


if __name__ == "__main__":
    """Performing feature selection on the IMDB Movies Dataset with gross income or an
    adjusted rating score as our target value"""

    # Setting up and retrieving main function arguments
    target_index, ord_cats, y_cat = parse()

    # Make the training and test data
    data, input_cols, output_col = data_split(af, target_index)

    # Fit transformers on training data then encode all data
    col_transform_X, col_transform_Y = transformer(ord_cats, y_cat)
    _, _, X_train_enc, y_train_enc = encode(data, col_transform_X, col_transform_Y)
    # Generate chi-square/frequency plots for all/top inputs
    fchi = chi2_features(input_cols, X_train_enc, y_train_enc)
    plotting(af, output_col, fchi)
