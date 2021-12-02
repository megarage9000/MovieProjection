
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

tf.config.set_visible_devices([], 'GPU')

def getTrainModel(movie_data):
    X_data = movie_data[movie_data.columns.difference(['averageRating'])]
    y_data = movie_data[['averageRating']]
    X_train, X_valid, y_train, y_valid = train_test_split(X_data, y_data)
    inputs = keras.Input(shape=movie_data.shape)
    
    # layers 
    dense = layers.Dense(64, activation="relu")
    x = dense(inputs)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(10)(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name='testmodel')

def main():
    movie_data = pd.read_csv('movie_trends_data.csv')
    getTrainModel(movie_data)

main()