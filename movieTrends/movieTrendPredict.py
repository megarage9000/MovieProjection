import tensorflow as tf
import numpy as np
import pandas as pd
import sys


def main():
    showHelp()
    getInput()


def getInput():
    titleType = input("Enter a title type input: ")
    print("- Entered title type = " + titleType)
    year = input("Enter a year input: ")
    print("- Entered year = " + year)
    genres = input("Enter a genres input(if multiple, at most 3 and separate by comma): ")
    print("- Entered genres = " + genres)
    genres = np.array(genres.lower().split(','))
    language = input("Enter a language input: ")
    print("- Entered language = " + language)
    region = input("Enter a region input: ")
    print("- Entered region = " + region)
    data = np.array([titleType, year, genres, language, region], dtype=object)
    print("Entered data = " + str(data))
    return data


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