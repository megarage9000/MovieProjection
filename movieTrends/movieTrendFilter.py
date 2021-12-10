import numpy as np
import pandas as pd


categories = ['titleType', 'startYear', 'genres', 'language', 'region']
def main():
    movie_data = pd.read_csv('../title.basics.tsv.gz', sep='\t', na_values='\\N')
    ratings = pd.read_csv('../title.ratings.tsv.gz', sep='\t', na_values='\\N')
    akas = pd.read_csv('../title.akas.tsv.gz', sep='\t', na_values='\\N')
    akas['tconst'] = akas['titleId']
    movie_data = movie_data.merge(ratings, on=['tconst'])
    movie_data = movie_data.merge(akas, on=['tconst'])

    movie_data = movie_data[['titleType', 'startYear', 'genres', 'language', 'region', 'averageRating']]
    movie_data = movie_data.dropna()
    for category in categories:
        movie_data[category] = movie_data[category].astype(str).str.lower()

    print(movie_data.isna().sum())
    movie_data.to_csv('film_trend_data.csv', index=False)

# Do some feature engineering here!
main()
