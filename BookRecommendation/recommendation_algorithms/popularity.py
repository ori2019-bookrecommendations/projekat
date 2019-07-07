import pandas as pd
from BookRecommendation.book import Book


def most_popular(books, ratings):
    # Returns a list of books sorted by the sum of their ratings

    # Sums up book ratings
    recommendations = ratings.groupby('book_id')['rating'].sum()

    # Combines the 'title' and 'authors' fields with the already present data
    recommendations = pd.merge(recommendations, books[['book_id', 'title', 'authors', 'average_rating', 'image_url']], how='inner', on=['book_id'])

    # Sorts the values by rating
    recommendations = recommendations.sort_values(by=['rating'], ascending=False)

    return convert(recommendations)


def highest_rated(books, ratings):
    # Returns a list of books sorted by the average rating

    # Finds the average book rating
    recommendations = ratings.groupby('book_id')['rating'].mean()

    # Combines the 'title' and 'authors' fields with the already present data
    recommendations = pd.merge(recommendations, books[['book_id', 'title', 'authors', 'average_rating', 'image_url']], how='inner', on=['book_id'])

    # Sorts the values by rating
    recommendations = recommendations.sort_values(by=['rating'], ascending=False)
    return convert(recommendations)


def convert(recommendations):
    result = []
    for id, book in recommendations.iterrows():
        b = Book(book['book_id'], book['title'], book['authors'], book['average_rating'], book['image_url'])
        result.append(b)

    return result
