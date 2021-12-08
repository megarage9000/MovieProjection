import numpy as np
import pandas as pd

def load_data(movie_data, movie_principals, names, movie_ratings):
    basics = movie_data[['tconst', 'primaryTitle', 'startYear', 'runtimeMinutes', 'genres']]
    principals = movie_principals[['tconst', 'nconst', 'category']]
    ratings = movie_ratings[['tconst', 'averageRating', 'numVotes']]
    principals = principals.merge(names, on=['nconst'])
    principals = principals[principals['category'] == principals['professions']]
    basics = basics.merge(principals, on=['tconst'])
    basics = basics.merge(ratings, on=['tconst'])
    return basics

def main():
    movie_data = pd.read_csv('../filtered_title.basics.tsv.gz.csv')
    movie_principals = pd.read_csv('../filtered_title.principals.tsv.gz.csv')
    movie_ratings = pd.read_csv('../filtered_title.ratings.tsv.gz.csv')
    names = pd.read_csv('../filtered_name.tsv.gz.csv')
    result_csv = load_data(movie_data, movie_principals, names, movie_ratings)
    result_csv = result_csv.fillna(0)
    print(result_csv.isna().sum())
    result_csv = result_csv[result_csv.columns.difference(['tconst', 'nconst', 'numVotes', 'primaryTitle', 'category'])]
    print(result_csv.columns.values)
    result_csv.to_csv('../movie_trends_data.csv', index=False)



main()