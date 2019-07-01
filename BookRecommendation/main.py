import pandas as pd
from BookRecommendation.recommendation_algorithms import popularity


def load_data():
    # Loads each of the .csv input files and creates a data frame consisting of user IDs
    books = pd.read_csv('./inputs/books.csv')
    book_tags = pd.read_csv('./inputs/book_tags.csv')
    ratings = pd.read_csv('./inputs/ratings.csv')
    tags = pd.read_csv('./inputs/tags.csv')
    to_read = pd.read_csv('./inputs/to_read.csv')
    users = ratings.user_id.unique()
    return books, book_tags, ratings, tags, to_read, users


def data_cleanup(ratings):
    # Cleans duplicates and uses the last known rating as the correct value
    ratings = ratings.drop_duplicates(subset=["user_id", "book_id"], keep="last")

    #TODO Remove foreign tags

    #TODO Combine similar tags

    return ratings


if __name__ == "__main__":
    books, book_tags, ratings, tags, to_read, users = load_data()
    ratings = data_cleanup(ratings)

    # Baseline models, not personalized in any way
    # Most popular
    recommendations = popularity.most_popular(books, ratings)
    print(recommendations.head(n=10))

    # Highest rated
    recommendations = popularity.highest_rated(books, ratings)
    print(recommendations.head(n=10))
