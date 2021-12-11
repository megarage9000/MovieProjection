import tensorflow as tf
import numpy as np
import pandas as pd
import sys


def main():
    showHelp()
    data, data_tf = getInput()
    print("Loading Tensorflow model...")
    model = tf.keras.models.load_model('models/NN_film_trends_without_na')
    result = model.predict(data_tf)
    print('Estimated rating for ' + str(data) + ': ' + str(result[0][0]))

def getInput():
    titleType = str(input("Enter a title type input: "))
    if titleType == 'test':
        print("Using default test data...")
        res = np.array(['fantasy,horror,thriller', 'ja', 'jp' ,'2001.0','tvmovie'])
        res_tf = np.expand_dims(res, axis=0)
        res_tf = np.split(res_tf, 5, axis=1)
        return res, res_tf
    print("- Entered title type = " + titleType)
    year = str(input("Enter a year input: "))
    print("- Entered year = " + year)
    genres = str(input("Enter a genres input(if multiple, at most 3 and separate by comma): "))
    print("- Entered genres = " + genres)
    language = str(input("Enter a language input: "))
    print("- Entered language = " + language)
    region = str(input("Enter a region input: "))
    print("- Entered region = " + region)
    data = np.array([genres, language, region, year, titleType])
    data_tf = packagevals(data)
    return data, data_tf


# need to a set of vals
def packagevals(data):
    data_for_tf = np.split(data, len(data), axis=0)
    return data_for_tf


def showHelp():
    print('Generating some example value types per category...')
    movie_data = pd.read_csv('film_trend_data.csv')
    uniqueTitleTypes = movie_data['titleType'].unique()
    uniqueYears = movie_data['startYear'].unique()
    uniqueGenres = movie_data['genres'].unique()
    uniqueLanguages = movie_data['language'].unique()
    uniqueRegions = movie_data['region'].unique()
    print('Some title type example input: ')
    print(uniqueTitleTypes)
    print('Some years example input: ')
    print(uniqueYears)
    print('Some genres example input: ')
    print(uniqueGenres)
    print('Some languages example input: ')
    print(uniqueLanguages)
    print('Some regions example input: ')
    print(uniqueRegions)


main()