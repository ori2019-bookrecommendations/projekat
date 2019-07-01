import pandas as pd


def most_popular(books, ratings):
    # Returns a list of books sorted by the sum of their ratings

    # Sums up book ratings
    recommendations = ratings.groupby('book_id')['rating'].sum()

    # Combines the 'title' and 'authors' fields with the already present data
    recommendations = pd.merge(recommendations, books[['book_id', 'title', 'authors']], how='inner', on=['book_id'])

    # Sorts the values by rating
    recommendations = recommendations.sort_values(by=['rating'], ascending=False)
    return recommendations


def highest_rated(books, ratings):
    # Returns a list of books sorted by the average rating

    # Finds the average book rating
    recommendations = ratings.groupby('book_id')['rating'].mean()

    # Combines the 'title' and 'authors' fields with the already present data
    recommendations = pd.merge(recommendations, books[['book_id', 'title', 'authors']], how='inner', on=['book_id'])

    # Sorts the values by rating
    recommendations = recommendations.sort_values(by=['rating'], ascending=False)
    return recommendations
