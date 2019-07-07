from BookRecommendation import utils
from BookRecommendation.recommendation_algorithms import popularity, collaborativeFiltering, contentBasedFiltering
from flask import Flask, render_template, request, redirect, url_for, jsonify
import tensorflow as tf
from BookRecommendation.book import Book

app = Flask(__name__)


def search(search_param,):
    global results, books

    found_books = books[books['title'].str.contains(search_param)][:10]
    results = []
    for _, row in found_books.iterrows():
        book = Book(row['book_id'], row['title'], row['authors'], row['average_rating'], row['image_url'])
        results.append(book)
    return results


@app.route('/')
@app.route('/index')
def index():
    global results, selected_titles
    return render_template('index.html', result=results, selected=list(selected_titles))


@app.route('/mostPopular')
def most_popular():
    global results, selected_titles
    results = popularity.most_popular(books, ratings)[:num_predict]
    return render_template('index.html', result=results, selected=list(selected_titles))


@app.route('/highestRated')
def highest_rated():
    global results, selected_titles
    results = popularity.highest_rated(books, ratings)[:num_predict]
    return render_template('index.html', result=results, selected=list(selected_titles))


@app.route('/cbf')
def cbf_predict():
    global results, cbf, selected, selected_titles
    results = cbf.recommend(books, list(selected), num_predict)
    return render_template('index.html', result=results, selected=list(selected_titles))


@app.route('/svd')
@app.route('/svd/<user_id>')
def svd_predict(user_id=None):
    global results, svd, selected, selected_titles, graph

    if user_id is not None:
        user_id = int(user_id)
        rated = ratings[ratings['user_id'] == user_id]['book_id'].unique()
        read = books[books['book_id'].isin(rated)][:num_predict]

        already_read = []
        for _, book in read.iterrows():
            b = Book(book['book_id'], book['title'], book['authors'], book['average_rating'], book['image_url'])
            already_read.append(b)

        results = svd.predict(user_id, books, ratings, selected)[:num_predict]
        return render_template('index.html', result=results, selected=already_read)
    else:
        results = svd.predict(0, books, ratings, selected)[:num_predict]
        return render_template('index.html', result=results, selected=list(selected_titles))


@app.route('/neumf')
@app.route('/neumf/<user_id>')
def neumf_predict(user_id=None):
    global results, ncf, selected, graph

    if user_id is not None:
        user_id = int(user_id)
        rated = ratings[ratings['user_id'] == user_id]['book_id'].unique()
        read = books[books['book_id'].isin(rated)][:num_predict]

        already_read = []
        for _, book in read.iterrows():
            b = Book(book['book_id'], book['title'], book['authors'], book['average_rating'], book['image_url'])
            already_read.append(b)
        results = ncf.predict(user_id, books, ratings, num_predict, None, graph)
        return render_template('index.html', result=results, selected=already_read)
    else:
        results = ncf.predict(0, books, ratings, num_predict, selected, graph)
        return render_template('index.html', result=results, selected=selected_titles)


@app.route('/api/search', methods=['POST'])
def search_books():
    content = request.get_json()
    search(content['search'])
    return redirect(url_for('index'))


@app.route('/api/user', methods=['POST'])
def user():
    global user_id
    content = request.get_json()
    user_id = int(content['user_id'])
    return jsonify(success=True)


@app.route('/api/removeAll', methods=['POST'])
def remove_all():
    global selected, selected_titles
    selected = set()
    selected_titles = set()
    return jsonify(success=True)


@app.route('/api/selectBook', methods=['POST'])
def select_book():
    global selected, selected_titles
    content = request.get_json()
    selected.add(int(content['selected']))
    selected_titles.add(content['title'])
    print(selected)
    print(selected_titles)
    return jsonify(success=True)


# Global flask variables
num_predict = 12
books, book_tags, ratings, tags, to_read, users = utils.load_data()
results = popularity.most_popular(books, ratings)[:num_predict]
selected = set()
selected_titles = set()

# ContentBased Filtering
cbf = contentBasedFiltering.load_model('./inputs/cbf/cbf')
# Collaborative Filtering
ncf = collaborativeFiltering.NeuralCollaborativeFiltering(ratings, 20, 30, 6)
ncf.load('./inputs/cf_models/ncf-model.cfm')

svd = collaborativeFiltering.SVDCollaborativeFiltering(ratings)
svd = svd.load('./inputs/cf_models/svd-model.cfm')

graph = tf.get_default_graph()

if __name__ == "__main__":
    app.run(debug=False)
