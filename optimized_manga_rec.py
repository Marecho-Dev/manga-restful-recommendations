import pickle
import psutil
import logging
from sklearn.metrics import pairwise_distances
# Set up logging configuration
import numpy as np
from pandas.core.reshape import pivot
from quart import Quart, jsonify
from prisma import Prisma, register
import pandas as pd
import scipy as sp
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, hstack, csc_matrix
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Replace your print statements with logging.info
logging.info("Your message here")
process = psutil.Process()
mem_info = process.memory_info()
pkl_directory = "pkl_files/"
logging.info(f"Current memory usage: {mem_info.rss / 1024 / 1024} MB")

db = Prisma(
    http={
        'timeout': None,
    },
)
db.connect()
register(db)

app = Quart(__name__)


# 100,000 query limit so doing this in batches. Terms to lookup, sharding
@app.route('/manga_recs/<int:user_id>')
async def user_recommendation(user_id, m_value=None):
    logging.info('starting user_recommndation')
    # Use the Prisma client to query your database and return the results as JSON
    m_value = 10
    user_results = db.mangalist.find_many(where={"user_id": user_id})
    logging.info(f'calling users_results: {user_results}')
    total_results = []
    for result in user_results:
        total_results.append((result.user_id, result.manga_id, result.rating))
    logging.info(f'calling total results: {total_results}')
    user_df = pd.DataFrame(total_results)
    user_df.columns = ['user_id', 'manga_id', 'rating']
    logging.info(f'user_df.columns complete')
    user_list = []
    for file in os.listdir(pkl_directory):
        with open(f'pkl_files/{file}', "rb") as f:
            main_df = pickle.load(f)
        merged_df = pd.concat([main_df, user_df], ignore_index=True)
        merged_df = merged_df.sort_values(by='rating', ascending=False).drop_duplicates(subset=['user_id', 'manga_id'],
                                                                                        keep='first').reset_index(
            drop=True)
        main_df = merged_df
        logging.info(f'calling main_df: {main_df}')
        logging.info(f'pkl file opened and merged with previous df. ')
        # Create a mapping of user_id and manga_id to indices
        user_id_map = {user_id: i for i, user_id in enumerate(main_df['user_id'].unique())}
        manga_id_map = {manga_id: i for i, manga_id in enumerate(main_df['manga_id'].unique())}
        logging.info(f"Current memory usage in user_rec: {mem_info.rss / 1024 / 1024} MB")
        # Compute the rows, columns, and data for the CSR matrix
        rows = main_df['user_id'].map(user_id_map.get).values
        cols = main_df['manga_id'].map(manga_id_map.get).values
        data = main_df['rating'].values
        user_ids = list(user_id_map.keys())
        logging.info(f'user_id list: {user_ids}')
        # Create the CSR matrix
        main_piv_sparse = csr_matrix((data, (rows, cols)), shape=(len(user_id_map), len(manga_id_map)))
        logging.info(f'main_piv_sparse completed')
        logging.info(f"Current memory usage in user_rec: {mem_info.rss / 1024 / 1024} MB")
        # manga_similarity = cosine_similarity(main_piv_sparse)
        cosine_distance = pairwise_distances(main_piv_sparse, metric='cosine')
        manga_similarity = 1 - cosine_distance
        logging.info(f'calling manga_similiary: {manga_similarity}')
        # this gets pkled in my old file as manga pkl - this is the cosine similarity of similar users.
        manga_sim_df = pd.DataFrame(manga_similarity, index=user_ids, columns=user_ids)
        logging.info(f'calling manga_sim_df {manga_sim_df}')
        for user in manga_sim_df.sort_values(by=user_id, ascending=False).index[1:10]:
            logging.info(f'{user}, {round(manga_sim_df[user][user_id] * 100, 2)}% match\n')
            user_list.append((user, round(manga_sim_df[user][user_id] * 100, 2)))
    sorted_similar_user_list = sorted(user_list, key=lambda x: float(x[1]), reverse=True)
    user_ids_str = ', '.join([str(similar_user_id[0]) for similar_user_id in sorted_similar_user_list])
    logging.info(f'preparing to build manga_query')
    manga_query = f"""select m.mal_id, m.title,m.imageUrl, m.rating,                   
        count(ml.manga_id) as 'manga_count',
        avg(ml.rating) as 'average_rating', 
        (count(ml.manga_id) / 
        (count(ml.manga_id) + {m_value}))
         * avg(ml.rating) + 
         ({m_value} / (count(ml.manga_id) + {m_value})) 
         * (SELECT AVG(rating) FROM MangaList WHERE rating <> 0) AS 'weighted_rating'
        FROM MangaList ml
    JOIN Manga m ON ml.manga_id = m.mal_id
    WHERE ml.manga_id NOT IN (
        SELECT manga_id
        FROM MangaList
        WHERE user_id = '{user_id}'
    )
    AND ml.user_id IN ({user_ids_str})
    AND ml.rating <> 0
    GROUP BY ml.manga_id
    ORDER BY weighted_rating DESC;"""

    logging.info(manga_query)
    logging.info(f"Current memory usage: {mem_info.rss / 1024 / 1024} MB")
    manga_recs = db.query_raw(manga_query)
    logging.info(manga_recs)
    logging.info(f"Current memory usage: {mem_info.rss / 1024 / 1024} MB")
    return jsonify(manga_recs)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), threaded=True)
