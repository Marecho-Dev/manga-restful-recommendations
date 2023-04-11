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
    # print(df.head())
    # pivot = df.pivot_table(index=['user_id'], columns=['manga_id'], values='rating')
    # pivot.fillna(0, inplace=True)
    # user_piv_sparse = sp.sparse.csr_matrix(pivot.values)
    # print(user_piv_sparse)
    # print("----------------------------------------------------------------")
    # total_rows = db.mangalist.count()
    # print(total_rows)
    # batch_size = 100000
    # offset = 0
    # total_results = []
    # while offset < total_rows:
    #     results = db.mangalist.find_many(skip=offset, take=batch_size)
    #     for result in results:
    #         total_results.append((result.user_id, result.manga_id, result.rating))
    #     offset += batch_size
    # df = pd.DataFrame(total_results)
    # df.columns = ['user_id', 'manga_id', 'rating']
    # with open("mangalist_df.pkl", "wb") as f:
    #     pickle.dump(df, f)
    with open("mangalist_df.pkl", "rb") as f:
        main_df = pickle.load(f)
    merged_df = pd.concat([main_df, user_df], ignore_index=True)
    merged_df = merged_df.sort_values(by='rating', ascending=False).drop_duplicates(subset=['user_id', 'manga_id'],
                                                                                    keep='first').reset_index(drop=True)
    main_df = merged_df
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
    number = 1
    similar_users = ""
    user_list = []
    manga_list = []
    logging.info(f'preparing to build manga_query')
    manga_query = f"""select manga_id,                    
        count(manga_id) as 'manga_count',
        avg(rating) as 'average_rating', 
        (count(manga_id) / 
        (count(manga_id) + {m_value}))
         * avg(rating) + 
         ({m_value} / (count(manga_id) + {m_value})) 
         * (SELECT AVG(rating) FROM MangaList WHERE rating <> 0) AS 'weighted_rating'
        from (
            select * 
            from MangaList 
            where 
                manga_id not in (
                    select manga_id 
                    from MangaList 
                    where user_id = '{user_id}'
                    ) 
                    and ( """
    logging.info(f"Current memory usage: {mem_info.rss / 1024 / 1024} MB")
    for user in manga_sim_df.sort_values(by=user_id, ascending=False).index[1:101]:
        logging.info(f'#{number}: {user}, {round(manga_sim_df[user][user_id] * 100, 2)}% match\n')
        similar_users += f'#{number}: {user}, {round(manga_sim_df[user][user_id] * 100, 2)}% match\n'
        user_list.append(user)
        manga_query = manga_query + f"\n            user_id = '{user}'"

        if number < 100:
            manga_query = manga_query + ' OR '
        else:
            manga_query = manga_query + """) 
                and rating <> 0 
                ) as subquery 
                group by 
                manga_id order by 
                weighted_rating DESC;"""

        number += 1
    logging.info(manga_query)
    logging.info(f"Current memory usage: {mem_info.rss / 1024 / 1024} MB")
    manga_recs = db.query_raw(manga_query)
    logging.info(manga_recs)
    logging.info(f"Current memory usage: {mem_info.rss / 1024 / 1024} MB")
    return manga_recs


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT",5000)), threaded=True)
