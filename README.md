This repository is a machine learning project, and it's main purpose is to discern what features are the most important for making a great motion picture (movie or TV show) or a profitable one. I attempted to do this by performing feauture selection on the IMDB Movies dataset (https://www.kaggle.com/datasets/ashishjangra27/imdb-movies-dataset). For profitability, the output variable is gross income (net profit except net losses were transformed to 0's in the raw data). For popularity, the output variable is the adjusted rating (the motion picture's rating on IMDb multiplied by the natural log of the number of ratings entered [votes in the raw data set]).

The repository is organized based on the Cookiecutter Data Science directory structure (https://drivendata.github.io/cookiecutter-data-science/).

The data files are too large to be uploaded to GitHub. They can be downloaded from this link (https://drive.google.com/drive/folders/1_Mb7aNu8fWZbMnw-A-cFtqNVI9X6wKa3). The data folder must be placed at the root of this repository in order for the programs to function.

The basic layout of the code structure goes as follows:
- make_csv.py transforms the raw data (movies.csv) into processed data (movies_filtered.csv)

- csv_to_dataset.py turns the processed data into a dataframe and discretizes numerical columns, making all of the data categorical.

- build_features.py splits and encodes the dataframe into training and test features

- visualize.py uses the training data to calulate and plot the chi-square statistic for each feature, sorts the list of features by chi-square, and for the top k (default 3) features, plots a multi bar chart of the percent frequency of the output variable's categories over that feature's categories

- model.py fits a choice classifier (default logistic regression) to the training data and is tested using the testing data. The accuracy, balanced_accuracy, recall, precision, and F1 score are written to a txt file in either the 'model_metrics/LR/all/' or 'model_metrics/NB/all/' section depending on whether a Logistic Regression or Naive Baynes classifier is used. Besides accuracy, these statistics equally weigh the output categories in their calculations, which is more telling as the data is imbalanced.

- model_select.py has the same functionality as model.py, however, you can trim training and test data to only include the top-k features (according to the chi-square statistic) before fitting the classifier. k defaults to 3 and the statistics from this model are written to a txt file in either the 'model_metrics/LR/select/' or 'model_metrics/NB/select/' section depending on whether a Logistic Regression or Naive Baynes classifier is used.   

Additionally, build_features.py, visualize.py, model.py, and model_select.py take in an optional argument -r (or --rating) that will run the files using the adjusted rating as the output variable. Otherwise, gross income is used.

All python files except build_features.py must be run from its own directory. This occurs because relative file paths are used to retrieve and save files.
Ex: the in-line command 'run data/make_csv.py' will not run as intended, but 'run make_csv.py' will.

The following warnings may occur, but they do not interfere with the program
- UserWarning in visualize.py
- UndefinedMetricWarning in model.py

There is also a paper covering my own procedure, results, and analysis of this subject under the 'reports' section.