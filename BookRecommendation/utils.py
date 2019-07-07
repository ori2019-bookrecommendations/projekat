import pandas as pd
from sklearn.model_selection import train_test_split


def load_data():
    # Loads each of the .csv input files and creates a data frame consisting of user IDs
    books = pd.read_csv('./inputs/books.csv')
    book_tags = pd.read_csv('./outputs/book_tags.csv')
    ratings = pd.read_csv('./outputs/ratings.csv')
    tags = pd.read_csv('./outputs/tags.csv')
    to_read = pd.read_csv('./inputs/to_read.csv')
    users = ratings.user_id.unique()
    return books, book_tags, ratings, tags, to_read, users


def create_training_and_test_sets(ratings, split_percentage):
    # Splits the ratings into a training and test set
    train_df, test_df = train_test_split(ratings, stratify=ratings['user_id'], test_size=split_percentage)
    return train_df, test_df


def load_test_and_train_data():
    # Loads each of the .csv input files and creates a data frame consisting of user IDs
    test = pd.read_csv('./outputs/test_set.csv')
    train = pd.read_csv('./outputs/train_set.csv')
    return test, train


def data_cleanup(ratings, tags, book_tags):
    # Cleans duplicates and uses the last known rating as the correct value
    ratings = ratings.drop_duplicates(subset=["user_id", "book_id"], keep="last")

    # Cleans tags that contain non ascii characters
    tag_ids = tags[tags['tag_name'].map(lambda x: not all(ord(c) < 128 for c in x))].tag_id
    tags.drop(tag_ids, inplace=True)

    # Cleans rows that contain removed tag
    row_index = book_tags[book_tags['tag_id'].map(lambda x: x in tag_ids)].index
    book_tags.drop(row_index, inplace=True)

    return ratings, tags, book_tags


def save(ratings, tags, book_tags):
    ratings.to_csv('./outputs/ratings.csv', index=False)
    tags.to_csv('./outputs/tags.csv', index=False)
    book_tags.to_csv('./outputs/book_tags.csv', index=False)
