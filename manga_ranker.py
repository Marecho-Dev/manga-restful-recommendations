import pickle
import psutil
import logging
from sklearn.metrics import pairwise_distances
# Set up logging configuration
import requests
from quart import Quart, jsonify
from prisma import Prisma, register
import pandas as pd
import re
from scipy.sparse import csr_matrix
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')
pkl_directory = "pkl_files/"
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

db = Prisma(
    http={
        'timeout': None,
    },
)
db.connect()
register(db)

app = Quart(__name__)


def clean_text(text):
    # Replace all non-alphanumeric characters (except spaces) with an empty string
    cleaned_text = re.sub(r"[^\w\s]", "", text)
    return cleaned_text


def tokenize_and_remove_stop_words(text):
    # Tokenize the text
    tokens = word_tokenize(text)

    # Convert the list of stop words to a set for faster processing
    stop_words = set(stopwords.words('english'))

    # Remove stop words from the list of tokens
    filtered_tokens = [token for token in tokens if token not in stop_words]

    return filtered_tokens


# 100,000 query limit so doing this in batches. Terms to lookup, sharding
@app.route('/manga_ranker/<int:user_id>')
async def user_recommendation(user_id, m_value=None):
    user_results = db.mangalist.find_many(where={"user_id": user_id, "rating": {"gt": 7}})
    manga_list = []
    for manga in user_results:
        print(manga)
        manga_list.append(manga.manga_id)
    manga_results = db.manga.find_many(where={"mal_id": {'in': manga_list}})
    print(manga_results)
    manga_string = ""
    for manga in manga_results:
        manga_string += manga.title + " " + manga.media_type + " " + manga.status + " " + manga.summary
        genres = db.genres.find_many(where={"mal_id": manga.mal_id})
        for genre in genres:
            print(genre)
            manga_string += " " + genre.genre_name
        print("------------------\n")
        print(manga)
    tokenized_manga_string = tokenize_and_remove_stop_words(clean_text(manga_string).lower())
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(tokenized_manga_string)
    url = f'https://restful-manga-recs.onrender.com/manga_recs/{user_id}'
    response = requests.get(url)
    rating_list = []
    if response.status_code == 200:
        data = response.json()
        for manga in data:
            rec_string = manga['title'] + ' ' + manga['status'] + ' ' + manga['summary'] + ' ' + manga['media_type']
            tokenized_rec_string = tokenize_and_remove_stop_words(clean_text(rec_string).lower())
            rec_string_joined = ' '.join(tokenized_rec_string)
            rec_vector = vectorizer.transform([rec_string_joined])
            cosine_sim_scores = cosine_similarity(rec_vector, tfidf_matrix)
            average_similarity = np.mean(cosine_sim_scores)
            rating_list.append([manga['mal_id'], manga['title'], manga['weighted_rating'],
                                average_similarity * manga['weighted_rating']])

    return sorted(rating_list, key=lambda x: x[3], reverse=True)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), threaded=True)
