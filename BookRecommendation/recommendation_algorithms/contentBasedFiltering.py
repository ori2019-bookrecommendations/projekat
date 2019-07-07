from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk import word_tokenize
from nltk.stem import  WordNetLemmatizer, PorterStemmer
import numpy as np
import pandas as pd
import time
import pickle
from BookRecommendation.book import Book


class ContentBasedFiltering:

    def __init__(self):
        self.tags = pd.read_csv('./outputs/acc_tags.csv')
        self.results = {}

    def train(self):
        # Train the model

        # Create a TF-IDF matrix
        tf = TfidfVectorizer(tokenizer=LemmaTokenizer(),
                            analyzer='word',
                            ngram_range=(1,1),
                            max_features=1000,
                            min_df=0.005,
                            max_df=0.5)
        matrix = tf.fit_transform(self.tags['tags'])

        # Compute similarity between all products using SciKit Leanr's linear_kernel (equivalent to cosine similarity)
        cosine_similarities = linear_kernel(matrix, matrix)

        # Iterate through each item's similar items and store the 100 most-similar
        for id, tags in self.tags.iterrows():
            similar_indices = cosine_similarities[id].argsort()[:-100:-1]
            similar_books = [(cosine_similarities[id][i], self.tags['book_id'][i]) for i in similar_indices]
            # first item is the item itself, so remove it.
            flattened = sum(similar_books[1:], ())
            self.results[tags['book_id']] = flattened

    def recommend(self, books, book_ids, n=20):
        # Returns top n books with similar content
        similar_books = []

        # Combining results of every book in one list
        for id in book_ids:
            result = self.results[id][:2*n]
            for i in range (0, len(result), 2):
                similar_books.append((result[i], result[i+1]))

        # Sorting and extracting ones with max similarity
        sorted(similar_books, key=lambda x: x[0])
        similar_books = similar_books[:n]

        # Extracting ids of books
        id_list = []
        for i in similar_books:
            id_list.append(i[1])

        recommendations = books[books['book_id'].map(lambda x : x in id_list)]
        return convert(recommendations)

    def save(self, location):
        # Fully saves the model
        pickle.dump(self, open(location, 'wb'))


def convert(recommendations):
    result = []
    for id, book in recommendations.iterrows():
        b = Book(book['book_id'], book['title'], book['authors'], book['average_rating'], book['image_url'])
        result.append(b)
    return result


class LemmaTokenizer(object):

    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.stm = PorterStemmer()

    def __call__(self, doc):
        return [self.stm.stem(self.wnl.lemmatize(t)) for t in word_tokenize(doc)]


def get_tag_name(tag_id, tags):
    # Returns tag name based on tag id
    return {word for word in tags.loc[tag_id].tag_name.split('-') if word}


def accumulate_tags(books, book_tags, tags):
    # Accumulates all the tags of a book in a dict
    start = time.time()
    book_tags_dict = dict()
    for book_id, tag_id, _ in book_tags.values:
        tags_of_book = book_tags_dict.setdefault(book_id, set())
        tags_of_book |= get_tag_name(tag_id)

    goodread2id = {goodreads_book_id: book_id for book_id, goodreads_book_id in
                        books[['book_id', 'goodreads_book_id']].values}

    np_tags = np.array(
        sorted(
            [[0, "DUMMY"]] + [[goodread2id[id], " ".join(tags)] for id, tags in book_tags_dict.items()]))

    print("Accumulating tags completed, took %s seconds. " % (time.time() - start))

    df = pd.DataFrame(np_tags)
    df.columns = ['book_id', 'tags']
    df.to_csv('./outputs/acc_tags.csv', index=False)

    return np_tags


def load_model(location):
    # Loads the model
    infile = open(location, 'rb')
    obj = pickle.load(infile)
    infile.close()
    return obj
