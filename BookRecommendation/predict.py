import sys, ast
import pandas as pd
from BookRecommendation.recommendation_algorithms import popularity, contentBasedFiltering, collaborativeFiltering


def printRecommendations(recommendations, n=20):
    # Help function for printing
    for i in range(0, n):
        print("%d. %s" % (i + 1, recommendations[i]))


# predict.py -userID $userID -books $book1, $book2
if __name__ == "__main__":
    argv = sys.argv[1:]

    # check command line arguments
    if len(argv) != 4 and len(argv) != 2:
        sys.exit("Invalid arguments given. Valid calls:\n"
                 "If you want predictions for books: -userID $userID -books $book1, $book2,...\n"
                 "If you want predictions for users: -userID $userID")

    print("Loading necessary data")
    # Load data
    ratings = pd.read_csv('./inputs/ratings.csv')
    books = pd.read_csv('./inputs/books.csv')

    # Load models
    cbf = contentBasedFiltering.load_model('./inputs/cbf/cbf')

    ncf = collaborativeFiltering.NeuralCollaborativeFiltering(ratings, 20, 30, 6)
    ncf.load('./inputs/cf_models/ncf-model.cfm')

    svd = collaborativeFiltering.SVDCollaborativeFiltering(ratings)
    svd = svd.load('./inputs/cf_models/svd-model.cfm')

    if argv[0] != "-userID":
        sys.exit("Invalid argument %s!" % argv[0])
    try:
        user_id = int(argv[1])
    except ValueError:
        sys.exit("Invalid argument %s, expected int!" % argv[1])

    # Baseline models, not personalized in any way
    # Most popular
    recommendations = popularity.most_popular(books, ratings)
    print("-- Most popular --")
    printRecommendations(recommendations)
    print("-----------------------------------------------------------\n")

    # Highest rated
    recommendations = popularity.highest_rated(books, ratings)
    print("-- Highest rated --")
    printRecommendations(recommendations)
    print("-----------------------------------------------------------\n")

    try:
        if argv[2] != "-books":
            sys.exit("Invalid argument %s!" % argv[2])
        else:
            try:
                rec_books = [int(s) for s in argv[3].split(',')]
            except ValueError:
                sys.exit("Invalid argument %s, expected list of ints!" % argv[3])

            # Content based filtering
            recommendations = cbf.recommend(books, rec_books)
            print("-- Content based filtering --")
            printRecommendations(recommendations)
            print("-----------------------------------------------------------\n")

            # Neural Collaborative filtering
            recommendations = ncf.predict(1, books, ratings, 10, rec_books)
            print("-- Neural Collaborative filtering --")
            printRecommendations(recommendations)
            print("-----------------------------------------------------------\n")

            # SVD Collaborative filtering
            recommendations = svd.predict(user_id, books, ratings, rec_books)
            print("-- SVD Collaborative filtering --")
            printRecommendations(recommendations)
            print("-----------------------------------------------------------\n")
    except IndexError:
        # Neural Collaborative filtering
        recommendations = ncf.predict(1, books, ratings, 20)
        print("-- Neural Collaborative filtering --")
        printRecommendations(recommendations)
        print("-----------------------------------------------------------\n")

        # SVD Collaborative filtering
        recommendations = svd.predict(user_id, books, ratings)
        print("-- SVD Collaborative filtering --")
        printRecommendations(recommendations)
        print("-----------------------------------------------------------\n")

