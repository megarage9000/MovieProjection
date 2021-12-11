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

    # Removing NaN data
    movie_data_no_nan = movie_data.dropna()
    for category in categories:
        movie_data_no_nan[category] = movie_data_no_nan[category].astype(str).str.lower()
        movie_data[category] = movie_data[category].astype(str).str.lower()
    print(movie_data_no_nan.shape)
    movie_data_no_nan.to_csv('film_trend_data_no_nan.csv', index=False)

    # Including NaN data
    movie_data_with_nan = movie_data.fillna({
        'titleType': 'unknown',
        'startYear': 'unknown',
        'genres': 'unknown',
        'language': 'unknown',
        'region': 'unknown'
    })
    print(movie_data_with_nan.shape)
    movie_data_with_nan.to_csv('film_trend_data_with_nan.csv', index=False)


main()
