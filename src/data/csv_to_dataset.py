# MUST BE RUN FROM THIS DIRECTORY!

# ================================
# Discretization of Numerical Data
# ================================

import pandas as pd


def read_and_bucket(filename):
    """Reads in data from movies_filtered.csv and discretizes numerical columns"""
    af = pd.read_csv(filename, delimiter=",")

    # Discretizing duration, votes, gross_income, start_year, and adjusted_rating
    d = list(af["duration"].astype("int64"))
    for i in range(af.shape[0]):
        x = d[i]
        if x < 60:
            d[i] = "< 60"
        elif x < 120:
            d[i] = "60-119"
        elif x < 180:
            d[i] = "120-179"
        elif x < 240:
            d[i] = "180-239"
        else:
            d[i] = "> 240"
    af["duration"] = d

    v = list(af["votes"].astype("int64"))
    for i in range(af.shape[0]):
        x = v[i]
        if x < 20:
            v[i] = "5-20"
        elif x < 50:
            v[i] = "20-49"
        elif x < 100:
            v[i] = "50-99"
        elif x < 200:
            v[i] = "100-199"
        elif x < 500:
            v[i] = "200-499"
        elif x < 1000:
            v[i] = "500-999"
        elif x < 2000:
            v[i] = "1000-1999"
        elif x < 5000:
            v[i] = "2000-4999"
        elif x < 10000:
            v[i] = "5000-9999"
        elif x < 20000:
            v[i] = "10000-19999"
        elif x < 50000:
            v[i] = "20000-49999"
        elif x < 100000:
            v[i] = "50000-99999"
        else:
            v[i] = "> 100000"
    af["votes"] = v

    g = list(af["gross_income"])
    for i in range(af.shape[0]):
        x = g[i]
        if x < 1000:
            g[i] = "< $1,000"
        elif x < 10000:
            g[i] = "$1,000 - $10,000"
        elif x < 100000:
            g[i] = "$10,000 - $100,000"
        elif x < 1000000:
            g[i] = "$100,000 - $1,000,000"
        elif x < 10000000:
            g[i] = "$1,000,000 - $10,000,000"
        elif x < 100000000:
            g[i] = "$10,000,000 - $100,000,000"
        else:
            g[i] = "> $100,000,000"
    af["gross_income"] = g

    sd = list(af["start_year"].astype("int64"))
    for i in range(af.shape[0]):
        x = sd[i]
        if x < 1950:
            sd[i] = "1877-1949"
        elif x < 1960:
            sd[i] = "1950's"
        elif x < 1970:
            sd[i] = "1960's"
        elif x < 1980:
            sd[i] = "1970's"
        elif x < 1990:
            sd[i] = "1980's"
        elif x < 2000:
            sd[i] = "1990's"
        elif x < 2010:
            sd[i] = "2000's"
        elif x < 2020:
            sd[i] = "2010's"
        else:
            sd[i] = "2020's"
    af["start_year"] = sd

    rt = list(af["adjusted_rating"])
    for i in range(af.shape[0]):
        x = rt[i]
        if x < -30:
            rt[i] = "Worst-Rated"
        elif x < -20:
            rt[i] = "Very Bad"
        elif x < -10:
            rt[i] = "Bad"
        elif x < 0:
            rt[i] = "Worse than Average"
        elif x == 0:
            rt[i] = "Average"
        elif x < 10:
            rt[i] = "Better than Average"
        elif x < 20:
            rt[i] = "Good"
        elif x < 30:
            rt[i] = "Very Good"
        elif x < 40:
            rt[i] = "Great"
        elif x < 50:
            rt[i] = "Excellent"
        elif x < 60:
            rt[i] = "Astounding"
        else:
            rt[i] = "Top-Rated"
    af["adjusted_rating"] = rt

    # Change all genre categories to strings
    af = af.astype(str)

    return af


movie_file = "../../data/processed/movies_filtered.csv"
af = read_and_bucket(movie_file)
