import keras
import numpy as np
from surprise import Reader, Dataset, SVD, accuracy
from surprise.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import pickle
from BookRecommendation.book import Book


class SVDCollaborativeFiltering:

    # Based on Singular Value Decomposition (SVD) implementation built into surprise library
    # Uses a matrix factorization method to reduce a matrix into lower dimension parts simplifying the calculations

    def __init__(self, ratings):
        # Surprise library does not allow using data frames as training and test set values
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(ratings[['user_id', 'book_id', 'rating']], reader)

        self.train, self.test = train_test_split(data, test_size=.20)
        self.model = SVD()

    def test_model(self):
        # Checks the predicted values against the test set
        # Returns Root Mean Square Error (RMSE) accuracy
        predictions = self.model.test(self.test)
        return accuracy.mae(predictions, verbose=False), accuracy.rmse(predictions, verbose=False)

    def train_model(self):
        # Trains the model on the training set (80% of the total ratings data)
        self.model.fit(self.train)

    def predict(self, user_id, books, ratings, already_read=None):
        # Predicts recommended books for a given user

        # Gets all unread books
        if already_read is None:
            already_read = ratings[ratings['user_id'] == user_id]['book_id'].unique()

        prediction = books[['book_id', 'title', 'authors', 'average_rating', 'image_url']].copy()
        prediction = prediction[~prediction['book_id'].isin(already_read)]

        # Predicts a rating for each book and sorts them
        prediction['predict'] = prediction['book_id'].apply(lambda x: self.model.predict(user_id, x).est)
        prediction = prediction.sort_values('predict', ascending=False)
        return convert(prediction)

    def save(self, location):
        # Fully saves the model
        pickle.dump(self, open(location, 'wb'))

    @staticmethod
    def load(location):
        # Loads the model
        infile = open(location, 'rb')
        obj = pickle.load(infile)
        infile.close()
        return obj


class NeuralCollaborativeFiltering:

    # Based on NeuMF model proposed by a paper "Neural Collaborative Filtering":
    # https://dl.acm.org/citation.cfm?id=3052569

    # This is a deep learning model combining GMF (General Matrix Factorization) and MLP (Multi-layer perceptron)
    # predictions.

    def __init__(self, ratings, lfu=8, lfb=10, lfmf=3):
        # Hyper-parameters
        self.num_latent_factors_user = lfu
        self.num_latent_factors_book = lfb
        self.num_latent_factors_matrix_factorization = lfmf

        # Data frame input
        self.ratings = ratings
        self.num_users = len(self.ratings.user_id.unique())
        self.num_books = len(self.ratings.book_id.unique())
        self.model = self.create_model()

    def create_model(self):
        # Constructs the NeuMF model

        # Inputs
        book_input = keras.layers.Input(shape=[1], dtype='int32')
        user_input = keras.layers.Input(shape=[1], dtype='int32')

        # Matrix Factorization Embeddings
        book_embedding_mf = keras.layers.Embedding(
            input_dim=self.num_books + 1,
            output_dim=self.num_latent_factors_matrix_factorization,)(book_input)
        user_embedding_mf = keras.layers.Embedding(
            name='mlp_embedding_user',
            input_dim=self.num_users + 1,
            output_dim=self.num_latent_factors_user)(user_input)

        # Multi Layer Perceptron Embeddings
        book_embedding_mlp = keras.layers.Embedding(
            input_dim=self.num_books + 1,
            output_dim=self.num_latent_factors_book)(book_input)
        user_embedding_mlp = keras.layers.Embedding(
            input_dim=self.num_users + 1,
            output_dim=self.num_latent_factors_matrix_factorization)(user_input)

        # Matrix Factorization latent vectors
        book_vec_mf = keras.layers.Flatten()(book_embedding_mf)
        book_vec_mf = keras.layers.Dropout(0.2)(book_vec_mf)

        user_vec_mf = keras.layers.Flatten()(user_embedding_mlp)
        user_vec_mf = keras.layers.Dropout(0.2)(user_vec_mf)

        pred_mf = keras.layers.dot([book_vec_mf, user_vec_mf], axes=1, normalize=False)

        # Multi Layer Perceptron latent vectors
        book_vec_mlp = keras.layers.Flatten()(book_embedding_mlp)
        book_vec_mlp = keras.layers.Dropout(0.2)(book_vec_mlp)

        user_vec_mlp = keras.layers.Flatten()(user_embedding_mf)
        user_vec_mlp = keras.layers.Dropout(0.2)(user_vec_mlp)

        concat = keras.layers.concatenate([book_vec_mlp, user_vec_mlp])
        concat_dropout = keras.layers.Dropout(0.2)(concat)

        dense = keras.layers.Dense(200)(concat_dropout)
        dense_batch = keras.layers.BatchNormalization()(dense)
        dropout_1 = keras.layers.Dropout(0.2)(dense_batch)
        dense_2 = keras.layers.Dense(100)(dropout_1)
        dense_batch_2 = keras.layers.BatchNormalization()(dense_2)
        dropout_2 = keras.layers.Dropout(0.2)(dense_batch_2)
        dense_3 = keras.layers.Dense(50)(dropout_2)
        dense_4 = keras.layers.Dense(20, activation='relu')(dense_3)

        pred_mlp = keras.layers.Dense(1, activation='relu', name='Activation')(dense_4)

        # Combination
        combine_mlp_mf = keras.layers.concatenate([pred_mf, pred_mlp])
        result_combine = keras.layers.Dense(200)(combine_mlp_mf)
        deep_combine = keras.layers.Dense(100)(result_combine)

        # Final result
        result = keras.layers.Dense(1)(deep_combine)

        # Create model with two inputs, one output
        self.model = keras.Model([user_input, book_input], result)
        opt = keras.optimizers.Adam(lr=0.001)
        self.model.compile(optimizer=opt, loss='mean_squared_error')
        return self.model

    def test_model(self, test_set, batch):
        # Tests the model performance against a test data frame
        # Compares the predictions of the model with actual ratings
        prediction = self.model.predict([test_set.user_id, test_set.book_id], batch_size=batch)
        actual_rating = test_set.rating
        return mean_absolute_error(actual_rating, prediction), sqrt(mean_squared_error(actual_rating, prediction)),

    def train_model(self, training_set, batch, epochs_num):
        # Trains the model on a given training data frame
        # Depending on the hardware, different batch sizes will lead to faster fitting
        self.model.fit([training_set.user_id, training_set.book_id], training_set.rating,
                       batch_size=batch,
                       epochs=epochs_num,
                       verbose=1,
                       validation_split=0.1)

    def predict(self, user, books, ratings, prediction_num, already_read=None, graph=None):
        # Predicts top ten unread books for a given user ID
        if graph is None:
            if already_read is None:
                already_read = ratings[ratings['user_id'] == user]['book_id'].unique()

            book_data = np.array(list(set(books[~books['book_id'].isin(already_read)].book_id)))
            user_data = np.array([user for i in range(len(book_data))])

            predictions = self.model.predict([user_data, book_data], batch_size=512)
            predictions = np.array([item[0] for item in predictions])
            recommended_book_ids = (-predictions).argsort()

            recommended_books = books[books['book_id'].isin(recommended_book_ids)]
            result = []
            for _, book in recommended_books.iterrows():
                b = Book(book['book_id'], book['title'], book['authors'], book['average_rating'], book['image_url'])
                result.append(b)

            return result
        else:
            with graph.as_default():
                if already_read is None:
                    already_read = ratings[ratings['user_id'] == user]['book_id'].unique()

                book_data = np.array(list(set(books[~books['book_id'].isin(already_read)].book_id)))
                user_data = np.array([user for i in range(len(book_data))])

                predictions = self.model.predict([user_data, book_data], batch_size=512)
                predictions = np.array([item[0] for item in predictions])
                recommended_book_ids = (-predictions).argsort()[:prediction_num]

                recommended_books = books[books['book_id'].isin(recommended_book_ids)]
                result = []
                for _, book in recommended_books.iterrows():
                    b = Book(book['book_id'], book['title'], book['authors'], book['average_rating'], book['image_url'])
                    result.append(b)

                return result

    def save(self, location):
        # Fully saves the model
        self.model.save(location)

    def load(self, location):
        # Loads the model
        self.model = keras.models.load_model(location)


def convert(recommendations):
    result = []
    for _, book in recommendations.iterrows():
        b = Book(book['book_id'], book['title'], book['authors'], book['average_rating'], book['image_url'])
        result.append(b)

    return result
