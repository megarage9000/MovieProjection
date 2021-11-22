import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier


def is_adult_by_rating(data):
    X = data[['averageRating']].values
    y = data['isAdult'].values
    multi_classifiers(X, y)

def is_adult_by_num_rated(data):
    X = data[['numVotes']].values
    y = data['isAdult'].values
    multi_classifiers(X, y)


def is_adult_by_num_rated_and_rating(data):

    X = data.filter(['numVotes','averageRating']).to_numpy()
    y = data['isAdult'].values
    multi_classifiers(X, y)


def multi_classifiers(X, y):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)
    gauss_model = GaussianNB()
    gauss_model.fit(X_train, y_train)
    print("Gauss:", gauss_model.score(X_valid, y_valid))
    # calculated based on running the model on everything between 1 and 300
    neighbours_model = KNeighborsClassifier(56)
    neighbours_model.fit(X_train, y_train)
    print("K Neighbours:", neighbours_model.score(X_valid, y_valid))
    random_forest_model = RandomForestClassifier(n_estimators=100)
    random_forest_model.fit(X_train, y_train)
    print("Random Forest:", random_forest_model.score(X_valid, y_valid))
    artificial_intel = MLPClassifier(hidden_layer_sizes=(20))
    artificial_intel.fit(X_train, y_train)
    print("Machine Learning:", artificial_intel.score(X_valid, y_valid))
    sc = StackingClassifier(
        estimators=[
            ("gauss", gauss_model),
            ("kn", neighbours_model),
            ("rf", random_forest_model),
            ("ai", artificial_intel)]
    )
    sc.fit(X_train, y_train)
    print("StackingClassifier:", sc.score(X_valid, y_valid))


def equalize_adult_non_adult(data):
    adult = data[data["isAdult"] == 1]
    non_adult = data[data["isAdult"] == 0]

    fewest_rows = min(len(adult), len(non_adult))
    combined = pd.concat([adult.sample(fewest_rows), non_adult.sample(fewest_rows)])
    return combined


def get_ratings_from_file():
    ratings_file = "filtered_title.ratings.tsv.gz.csv"
    basics_file = "filtered_title.basics.tsv.gz.csv"
    basics = pd.read_csv(basics_file).filter(["tconst", 'primaryTitle', "isAdult"], axis=1)
    ratings = pd.read_csv(ratings_file).filter(["tconst", "averageRating", "numVotes"], axis=1)
    combined_adult_ratings = basics.join(ratings.set_index('tconst'), on='tconst').dropna()
    return combined_adult_ratings


def main():

    combined_adult_ratings = get_ratings_from_file()

    # Classifiers got confused when 98.6% of all results were non-adult classification
    #   and they just started guessing always true for that question
    equalized_rows = equalize_adult_non_adult(combined_adult_ratings)

    print("Asks various classifiers to decide if a movie is adult based on its rating")
    is_adult_by_rating(equalized_rows)
    print("\nAsks various classifiers to decide if a movie is adult based on how many people rated it")
    is_adult_by_num_rated(equalized_rows)
    print("\nAsks various classifiers to decide if a movie is adult based on how many people rated it and its rating")
    is_adult_by_num_rated_and_rating(equalized_rows)



main()