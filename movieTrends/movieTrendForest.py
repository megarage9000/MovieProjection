
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import FeatureHasher, DictVectorizer
from sklearn.pipeline import make_pipeline
import numpy as np
import pandas as pd
import math

categorical_data = ['titleType', 'startYear', 'genres', 'language', 'region']


def main():
    film_data = pd.read_csv('film_trend_data.csv')

    test = film_data['genres'].unique().astype(str)
    print(test)
    print(len(test))



main()
