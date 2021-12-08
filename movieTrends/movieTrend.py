
import pandas as pd
import numpy as np
import sklearn.preprocessing
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Most code borrowed from "Hands-on Machine Learning with Scikit-learn, Keras, and Tensorflow 2nd Edition"
NUM_CAT = 1

def getTrainModel(movie_data):
    X_data = movie_data[movie_data.columns.difference(['averageRating'])]
    y_data = movie_data[['averageRating']]
    print(X_data)
    print(y_data)

    # Train test split
    X_train_full, X_test, y_train_full, y_test = train_test_split(X_data, y_data)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)
    model = createModel(X_train)
    model.summary()

    # Need to split up numerical and categorial data
    X_train_cat, X_train_num = X_train['genres'].to_numpy(), X_train[X_train.columns.difference(['genres'])].to_numpy
    X_valid_cat, X_train_num =
    # result = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))
    # test = model.evaluate(X_test, y_test)


def splitNumCat(data, category_names):
    print("yes")

def createModel(X_data):

    num_numerical = len(X_data.columns) - NUM_CAT
    numerical_inputs = keras.layers.Input(shape=[num_numerical])

    # Get categorical inputs
    genre_inputs = setupCategorialLayer(X_data, 'genres', 2, 3)
    all_inputs = keras.layers.concatenate([numerical_inputs, genre_inputs])
    rating_output = keras.layers.Dense(1)(all_inputs)
    movie_rating_model = keras.models.Model(inputs=[numerical_inputs, genre_inputs], outputs=[rating_output])
    movie_rating_model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))

    return movie_rating_model


def setupCategorialLayer(data, name, number_oov, dimension_size):
    category_values = data[name].unique()
    category_values = category_values.astype(str)
    indices = tf.range(len(category_values), dtype=tf.int64)
    table_init = tf.lookup.KeyValueTensorInitializer(category_values, indices)
    table = tf.lookup.StaticVocabularyTable(table_init, number_oov)

    categories = keras.layers.Input(shape=[], dtype=tf.string)
    category_indices = keras.layers.Lambda(lambda category: table.lookup(category))(categories)
    embedded_layer = keras.layers.\
        Embedding(input_dim=len(category_values) + number_oov, output_dim=dimension_size)\
        (category_indices)
    return embedded_layer


def main():
    movie_data = pd.read_csv('../filtered_title.basics.tsv.gz.csv')
    ratings = pd.read_csv('../filtered_title.ratings.tsv.gz.csv')
    movie_data = movie_data.merge(ratings, on=['tconst'])
    # Start with basic data first
    movie_data = movie_data[['startYear', 'runtimeMinutes', 'isAdult', 'genres', 'averageRating']]
    getTrainModel(movie_data)



main()
