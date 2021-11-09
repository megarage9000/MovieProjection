import numpy as np
import pandas as pd

def main():
    # Fetch important movie data
    files = ["title.basics.tsv.gz", "title.crew.tsv.gz", "title.principals.tsv.gz", "title.ratings.tsv.gz"]
    t_consts = getTconstsAndSave("title.basics.tsv.gz")
    for file in files:
        filterbyTConst(t_consts, file, "tconst")
    filterbyTConst(t_consts, "title.akas.tsv.gz", "titleId")


def getTconstsAndSave(file_name):
    data = pd.read_csv(file_name, sep='\t')
    movies_only = data[(data["titleType"] == "movie") | (data["titleType"] == "tvMovie")].drop(columns = ["endYear", "titleType"])
    movies_only.to_csv("filtered_"+file_name+".csv")
    movies_only.replace("\\N", np.nan)
    t_consts = movies_only["tconst"].to_numpy()
    return t_consts

def filterbyTConst(tconst_arr, file_name, col_name):
    frame = pd.read_csv(file_name, sep='\t')
    frame = frame[frame[col_name].isin(tconst_arr)]
    frame.replace("\\N", np.nan)
    frame.to_csv("filtered_"+file_name+".csv")

main()