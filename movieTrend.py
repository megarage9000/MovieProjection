import numpy as np
import pandas as pd
from sklearn import linear_model.LinearRegression



def load_data(movie_data, movie_principals, names):
    basics = movie_data[['tconst', 'primaryTitle', 'startYear', 'runtimeMinutes', 'genres', 'isAdult']]

def main():
    movie_data = pd.read_csv('filtered_title.basics.tsv.gz.csv')
    movie_principals = pd.read_csv('filtered_title.principals.tsv.gz.csv')
    names = pd.read_csv('filtered_name.tsv.gz.csv')
    load_data(movie_data, movie_principals, names)
    print('hello world')
    

if __name__ == 'main':
    main()