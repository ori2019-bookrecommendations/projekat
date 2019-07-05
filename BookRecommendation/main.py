import pandas as pd
from BookRecommendation.recommendation_algorithms import popularity, contentBasedFiltering


def load_data():
    # Loads each of the .csv input files and creates a data frame consisting of user IDs
    books = pd.read_csv('./inputs/books.csv')
    book_tags = pd.read_csv('./inputs/book_tags.csv')
    ratings = pd.read_csv('./inputs/ratings.csv')
    tags = pd.read_csv('./inputs/tags.csv')
    to_read = pd.read_csv('./inputs/to_read.csv')
    users = ratings.user_id.unique()
    return books, book_tags, ratings, tags, to_read, users


def data_cleanup(ratings, tags, book_tags):
    # Cleans duplicates and uses the last known rating as the correct value
    ratings = ratings.drop_duplicates(subset=["user_id", "book_id"], keep="last")

    # Cleans tags that contain non ascii characters
    tag_ids = tags[tags['tag_name'].map(lambda x: not all(ord(c) < 128 for c in x))].tag_id
    tags.drop(tag_ids, inplace=True)

    # Cleans rows that contain removed tag
    row_index = book_tags[book_tags['tag_id'].map(lambda x: x in tag_ids)].index
    book_tags.drop(row_index, inplace=True)


    #TODO Combine similar tags

    return ratings, tags, book_tags

def save(ratings, tags, book_tags):
    ratings.to_csv('./outputs/ratings.csv', index=False)
    tags.to_csv('./outputs/tags.csv', index=False)
    book_tags.to_csv('./outputs/book_tags.csv', index=False)


if __name__ == "__main__":

    books, book_tags, ratings, tags, to_read, users = load_data()
    ratings, tags, book_tags = data_cleanup(ratings, tags, book_tags)
    save(ratings, tags, book_tags)

    # Baseline models, not personalized in any way
    # Most popular
    recommendations = popularity.most_popular(books, ratings)
    print(recommendations.head(n=10))

    # Highest rated
    recommendations = popularity.highest_rated(books, ratings)
    print(recommendations.head(n=10))

