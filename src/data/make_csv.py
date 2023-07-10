# MUST BE RUN FROM THIS DIRECTORY!

# ==============================
# Filtering and Normalizing Data
# ==============================

import pandas as pd
import numpy as np


def create_filtered_csv(in_file, out_file):
    """Transforms the raw dataset (movies.csv) into a filtered dataset (movies_filtered.csv)"""

    # -----------------------
    # Read in data and filter
    # -----------------------

    df = pd.read_csv(in_file, delimiter=",")

    # (rrelevant media filter (leaves movies and TV shows, including direct-to-dvd)
    f = []
    for x in df["year"]:
        if "Podcast Series" in x or "Video Game" in x or "Music Video" in x:
            f += [False]
        else:
            f += [True]
    df = df[f]

    # Irrelevant shows filter
    df = df[
        ((df["votes"] != 0) & (df["votes"] != "0"))
        | ((df["gross_income"] != "0") & (df["gross_income"] != 0))
    ]

    # Removing non-standard ratings
    df = df[df.rating != 11.0]

    # Getting rid of non-years and creating start_year column
    year = list(df["year"])
    N = []
    for s in year:
        n = ""
        for char in s:
            if char.isdigit():
                if len(n) < 4:
                    n += char
                else:
                    break
        N += [n]

    df["start_year"] = N
    yf = [len(n) == 4 and int(n) != 2100 for n in N]  # 2100 marks an invalid year
    df = df[yf]

    df = df.reset_index(drop=True)

    # ----------------------
    # Fixing Esiting Columns
    # ----------------------

    # Normalizing gross_income (mixed types)
    lg = list(df["gross_income"])

    for i in range(df.shape[0]):
        xi = lg[i]
        if type(xi) == str:
            # for strings with the format $a.bcM
            if xi[0] == "$":
                lg[i] = (
                    int(xi[1]) * 10**6 + int(xi[3]) * 10**5 + int(xi[4]) * 10**4
                )

            # for standard number with commas
            else:
                xi = xi.replace(",", "")
                lg[i] = int(xi)

    df["gross_income"] = lg

    # Normalizing duration (number with commas followed by min)
    dur = list(df["duration"])
    dN = []
    for s in dur:
        dn = ""
        for char in s:
            if char.isdigit():
                dn += char
        dN += [dn]
    df["duration"] = dN

    # Normalizing votes (mixed types)
    vot = list(df["votes"])
    vN = []
    for s in vot:
        if type(s) == str:
            # Strings of decimals with commas
            if s[-2:] == ".0":
                vN += [s[:-2].replace(",", "")]
            else:
                vN += [s.replace(",", "")]
        else:
            # Int values
            vN += [str(s)]
    df["votes"] = vN

    # ------------------
    # Adding New Columns
    # ------------------

    # Shift rating so the average, 5.5, is 0 (-4.5). Then multiply by ln(votes) to get adjusted_rating
    df["adjusted_rating"] = (df.rating - 4.5) * np.log(df.votes.astype("int64"))

    # Adding a column for each genre
    gen = list(df["genre"])
    Lgen = [x.replace(" ", "").split(",") for x in gen]
    Sgen = []
    for cell in Lgen:
        for cat in cell:
            if cat not in Sgen:
                Sgen += [cat]
    Sgen.sort()

    Catsgen = {}
    for cat in Sgen:
        labels = []
        for cell in gen:
            if cat in cell:
                labels += ["1"]
            else:
                labels += ["0"]
        Catsgen[cat] = labels

    for cat in Catsgen:
        df[cat] = Catsgen[cat]

    # ----------------
    # Removing Columns
    # ----------------

    to_drop = [
        "id",
        "name",
        "year",
        "rating",
        "certificate",
        "genre",
        "directors_id",
        "directors_name",
        "stars_id",
        "stars_name",
        "description",
    ]
    df = df.drop(to_drop, axis=1)

    df.to_csv(out_file, index=False)


create_filtered_csv(
    "../../data/raw/movies.csv", "../../data/processed/movies_filtered.csv"
)
