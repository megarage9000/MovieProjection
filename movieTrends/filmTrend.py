import math
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Code referenced from "Hands-on Machine Learning with Scikit-learn, Keras, and Tensorflow 2nd Edition"
# and Tensorflow documentation

CATEGORIES = ['titleType', 'startYear', 'genres', 'language', 'region']
NUM_CAT = len(CATEGORIES)

# Some Hyperparameters
NUM_EPOCHS = 15
LEARNING_RATE = 0.01

def getTrainModel(movie_data, num_epochs, model_name):
    X_data = movie_data[movie_data.columns.difference(['averageRating'])]
    y_data = movie_data[['averageRating']]

    # split data
    X_train_full, X_test, y_train_full, y_test = train_test_split(X_data, y_data)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)
    model = createModel(X_train)


    X_train = X_train.to_numpy().astype(str)
    X_valid = X_valid.to_numpy().astype(str)
    X_test = X_test.to_numpy().astype(str)

    # Create the exponential learning scheduler
    exp_decay_fn = exp_decay(lr0=LEARNING_RATE, s=num_epochs)
    lr_scheduler = keras.callbacks.LearningRateScheduler(exp_decay_fn)

    result = model.fit(np.split(X_train, NUM_CAT, axis=1), y_train, epochs=num_epochs,
                       validation_data=(np.split(X_valid, NUM_CAT, axis=1), y_valid), callbacks=[lr_scheduler])
    test = model.evaluate(np.split(X_test, NUM_CAT, axis=1), y_test)
    print(test)

    print('saving model...')
    model.save('models/' + model_name)

    return result


def createModel(X_data):
    # Create all the embedded layers
    categorical_inputs_names = CATEGORIES
    categorical_inputs = list()
    categorical_embedded_layers = list()
    for category in categorical_inputs_names:
        embed_layer, cat_input = setupCategorialLayer(X_data, category, 2)
        categorical_inputs.append(cat_input)
        categorical_embedded_layers.append(embed_layer)
    embedded_layers = keras.layers.concatenate(categorical_embedded_layers)

    # Apply hidden layers to the embedded layers
    hidden_layers = applyLayers(embedded_layers)

    # Dense inputs to one
    rating_output = keras.layers.Dense(1)(hidden_layers)

    movie_rating_model = keras.models.Model(inputs=[categorical_inputs], outputs=[rating_output])
    movie_rating_model.compile(loss=keras.losses.MeanAbsoluteError(),
                               optimizer=keras.optimizers.Nadam(learning_rate=LEARNING_RATE),
                               metrics=['accuracy'])

    return movie_rating_model


# From Hands-on-ML, exponential scheduling
def exp_decay(lr0, s):
    def exp_decay_fn(epoch):
        return lr0 * 0.1**(epoch / s)
    return exp_decay_fn


def setupCategorialLayer(data, name, number_oov):
    # Create input layer for these values
    cat_values = np.unique(data[name].astype(str).to_numpy())
    print('unique values for ' + name + ': ')
    print(cat_values)
    num_cat_values = len(cat_values)
    input_cat = tf.keras.layers.Input(shape=[], dtype=tf.string)
    cat_val_lookup = tf.keras.layers.StringLookup(
        vocabulary=cat_values,
        num_oov_indices=number_oov)(input_cat)
    vocab_size = num_cat_values
    # Create embedded layer for the values
    cat_embed_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=int(math.sqrt(vocab_size)))(cat_val_lookup)
    return cat_embed_layer, input_cat


def applyLayers(layers):
    # Scaled Exponential Linear Unit Layer
    batch = keras.layers.BatchNormalization()(layers)
    selu_layer = keras.layers.Dense(64,
                                    activation='selu',
                                    kernel_initializer="lecun_normal",
                                    use_bias=False)(batch)
    # Exponential Linear Unit Layer
    batch_2 = keras.layers.BatchNormalization()(selu_layer)
    elu_layer = keras.layers.Dense(64,
                                   activation='elu',
                                   kernel_initializer='he_normal',
                                   use_bias=False)(batch_2)
    # Dropout layer
    batch_3 = keras.layers.BatchNormalization()(elu_layer)
    drop = keras.layers.Dropout(rate=0.3)(batch_3)
    final_layer = keras.layers.Dense(32)(drop)
    return final_layer

def main():

    if_on_nan = input("Train with imputed NaN values(Y/N)? ")
    if_on_nan = if_on_nan.lower()
    if if_on_nan == 'y':
        movie_data = pd.read_csv('film_trend_data_with_nan.csv')
        print(movie_data.shape)
        resulting_model = getTrainModel(movie_data, 2, 'NN_film_trends_with_na')
    else:
        movie_data = pd.read_csv('film_trend_data_no_nan.csv')
        print(movie_data.shape)
        resulting_model = getTrainModel(movie_data, 15, 'NN_film_trends_without_na')


def processData():
    movie_data = pd.read_csv('film_trend_data.csv')
    return movie_data


main()
