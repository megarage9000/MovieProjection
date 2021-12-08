import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Most code borrowed from "Hands-on Machine Learning with Scikit-learn, Keras, and Tensorflow 2nd Edition"
NUM_CAT = 3


def getTrainModel(movie_data):
    X_data = movie_data[movie_data.columns.difference(['averageRating'])]
    y_data = movie_data[['averageRating']]

    # split data
    X_train_full, X_test, y_train_full, y_test = train_test_split(X_data, y_data)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)
    model = createModel(X_train)
    model.summary()

    X_train_cat, X_train_num, \
    X_valid_cat, X_valid_num, \
    X_test_cat, X_test_num = splitNumCat(X_train, X_valid, X_test, ['genres', 'startYear', 'isAdult'])

    # Scale the numerical data
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train_num)
    X_valid_num = scaler.transform(X_valid_num)
    X_test_num = scaler.transform(X_test_num)

    result = model.fit((X_train_num, np.split(X_train_cat, NUM_CAT, axis=1)), y_train, epochs=20,
                       validation_data=((X_valid_num, np.split(X_valid_cat, NUM_CAT, axis=1)), y_valid))
    test = model.evaluate((X_test_num, np.split(X_test_cat, NUM_CAT, axis=1)), y_test)
    print(test)

    return result

def splitNumCat(data_train, data_valid, data_test, category_names):
    # Split data to numerical and categorical data
    data_cat_train, data_num_train = data_train[category_names].to_numpy().astype(str), \
                                     data_train[data_train.columns.difference(category_names)].to_numpy()
    data_cat_valid, data_num_valid = data_valid[category_names].to_numpy().astype(str), \
                                     data_valid[data_valid.columns.difference(category_names)].to_numpy()
    data_cat_test, data_num_test = data_test[category_names].to_numpy().astype(str), \
                                   data_test[data_test.columns.difference(category_names)].to_numpy()
    return data_cat_train, data_num_train, data_cat_valid, data_num_valid, data_cat_test, data_num_test


def createModel(X_data):
    # Get numerical inputs
    num_numerical = len(X_data.columns) - NUM_CAT
    numerical_inputs = keras.layers.Input(shape=[num_numerical], dtype=tf.float64)
    layer_num_1 = keras.layers.Dense(30, activation="sigmoid")(numerical_inputs)

    # Get categorical inputs (genres, startYear, isAdult)
    genre_embed_layer, genre_inputs = setupCategorialLayer(X_data, 'genres', 2, 3)
    start_year_embed_layer, start_year_inputs = setupCategorialLayer(X_data, 'startYear', 2, 3)
    is_adult_embed_layer, is_adult_inputs = setupCategorialLayer(X_data, 'isAdult', 2, 3)
    embedded_layers = keras.layers.concatenate([genre_embed_layer, start_year_embed_layer, is_adult_embed_layer])

    # Apply layers on embedded
    embed_sigmoid_layer = keras.layers.Dense(30, activation="sigmoid")(embedded_layers)
    flattened_layer = keras.layers.Flatten()(embed_sigmoid_layer)
    # Concatenate the inputs
    all_inputs = keras.layers.concatenate([layer_num_1, flattened_layer])

    # Dense inputs to one
    rating_output = keras.layers.Dense(1)(all_inputs)
    movie_rating_model = keras.models.Model(inputs=[numerical_inputs, genre_inputs, start_year_inputs, is_adult_inputs], outputs=[rating_output])
    movie_rating_model.compile(loss=keras.losses.MeanAbsoluteError(), optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    return movie_rating_model


def setupCategorialLayer(data, name, number_oov, dimension_size):
    # Get all possible values for the category
    category_values = data[name].unique()
    category_values = category_values.astype(str)

    # Create indices for table
    indices = tf.range(len(category_values), dtype=tf.int64)

    # Create a table with category values linked to corresponding index
    table_init = tf.lookup.KeyValueTensorInitializer(category_values, indices)
    table = tf.lookup.StaticVocabularyTable(table_init, number_oov)

    # Create input layer for the category
    categories = keras.layers.Input(shape=[1], dtype=tf.string)

    # Create embedded layer for the categorical value
    category_indices = keras.layers.Lambda(lambda category: table.lookup(category))(categories)
    embedded_layer = keras.layers. \
        Embedding(input_dim=len(category_values) + number_oov, output_dim=dimension_size) \
        (category_indices)
    flatten_layer = keras.layers.Flatten()(embedded_layer)
    # Return the embedded layer along with the input layer
    return flatten_layer, categories


def main():
    movie_data = pd.read_csv('../filtered_title.basics.tsv.gz.csv')
    ratings = pd.read_csv('../filtered_title.ratings.tsv.gz.csv')
    movie_data = movie_data.merge(ratings, on=['tconst'])
    # Start with basic data first
    movie_data = movie_data[['startYear', 'runtimeMinutes', 'isAdult', 'genres', 'averageRating']]
    movie_data['startYear'].fillna("Unknown", inplace=True)
    movie_data['runtimeMinutes'].fillna(0, inplace=True)
    movie_data['genres'].fillna("Unknown", inplace=True)
    resulting_model = getTrainModel(movie_data)
    test_rating = resulting_model.predict(
        [2000, 160, 1, 'Action']
    )
    print("test rating = " + str(test_rating))


main()
