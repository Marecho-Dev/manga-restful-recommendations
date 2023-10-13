import pickle
import psutil
import logging
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
# Set up logging configuration

from quart import Quart, jsonify
from prisma import Prisma, register
import pandas as pd

from scipy.sparse import csr_matrix
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

def get_user_likes(user_id):
    """
    Retrieve all the manga liked by the user from the database.

    Args:
    - user_id (int): The ID of the user.

    Returns:
    - list of tuples: Each tuple contains user_id, manga_id, and rating.
    """
    user_results = db.mangalist.find_many(where={"user_id": user_id})
    logging.info(f'Calling users_results: {user_results}')

    total_results = [(result.user_id, result.manga_id, result.rating) for result in user_results]

    return total_results

def create_dataframe(data, columns=['user_id', 'manga_id', 'rating']):
    df = pd.DataFrame(data, columns=columns)
    logging.info(f'{df.columns} complete')
    return df


def get_nearest_neighbors(user_id, binary_matrix, nn_model, k=5):
    """
    Return the k nearest neighbors to a given user_id based on a provided binary matrix.

    Parameters:
        user_id (int): ID of the user to find neighbors for.
        binary_matrix (pd.DataFrame): Binary user-item matrix.
        nn_model (NearestNeighbors): Trained NearestNeighbors model.
        k (int): Number of neighbors to return.

    Returns:
        list: List of user_ids of the k nearest neighbors.
    """

    # Retrieve the binary representation for the user
    user_vector = binary_matrix.loc[user_id].values.reshape(1, -1)

    # Find k nearest neighbors
    distances, indices = nn_model.kneighbors(user_vector, n_neighbors=k + 1)

    # Get user_ids of the nearest neighbors
    nearest_users = binary_matrix.index[indices.flatten()].tolist()

    # Exclude the user themselves (as they'll always be the closest match)
    nearest_users = nearest_users[1:]

    return nearest_users


# 100,000 query limit so doing this in batches. Terms to lookup, sharding
@app.route('/manga_recs/<int:user_id>')
async def user_recommendation(user_id, m_value=None):
    logging.info('starting user_recommndation')
    # Use the Prisma client to query your database and return the results as JSON
    m_value = 10
    #searches for user in database and returns all likes
    user_results = get_user_likes(user_id);
    #Creates dataframe for the username's provided manga list
    user_df = create_dataframe(user_results)
    #initalized empty array
    user_list = []
    for file in os.listdir(pkl_directory):
        # loads the current pkl file to main df
        with open(f'pkl_files/{file}', "rb") as f:
            main_df = pickle.load(f)
            print(main_df)
        #merged_df concats main_df and the user_df.
        merged_df = pd.concat([main_df, user_df], ignore_index=True)
        #sorts the merged_df by rating
        merged_df = merged_df.sort_values(by='rating', ascending=False).drop_duplicates(subset=['user_id', 'manga_id'],
                                                                                        keep='first').reset_index(
            drop=True)
        #sets main_df to merged_df
        logging.info('calling main_df = merged_df')
        main_df = merged_df
        # beginning the start of jaccards ---------------------------------------
        logging.info('start of jaccards')
        jac_main_df = main_df.copy()
        jac_main_df['rating'] = 1
        logging.info('creating binary_df')
        binary_df = main_df.pivot(index='user_id', columns="manga_id",values='rating').fillna(0).astype(int)
        logging.info('nearest neighbors')
        model = NearestNeighbors(metric='jaccard', algorithm='brute')
        model.fit(binary_df)
        nearest_neighbors_user = get_nearest_neighbors(user_id, binary_df, model, k=20)
        logging.info('printing out nearest_neighbors for the user')
        logging.info(nearest_neighbors_user)
        filtered_users = [529] + nearest_neighbors_user
        filtered_df = main_df[main_df['user_id'].isin(filtered_users)]
        main_df = filtered_df
        # end of jaccards dataframe modifying---------------------------------------
        # Create a mapping of user_id and manga_id to indices
        user_id_map = {user_id: i for i, user_id in enumerate(main_df['user_id'].unique())}
        manga_id_map = {manga_id: i for i, manga_id in enumerate(main_df['manga_id'].unique())}
        # Compute the rows, columns, and data for the CSR matrix
        rows = main_df['user_id'].map(user_id_map.get).values
        cols = main_df['manga_id'].map(manga_id_map.get).values
        data = main_df['rating'].values
        user_ids = list(user_id_map.keys())
        logging.info(f'user_id list: {user_ids}')
        # Create the CSR matrix
        main_piv_sparse = csr_matrix((data, (rows, cols)), shape=(len(user_id_map), len(manga_id_map)))
        logging.info(f'main_piv_sparse completed')
        logging.info(main_piv_sparse)
        logging.info(f"Current memory usage in user_rec: {mem_info.rss / 1024 / 1024} MB")
        # manga_similarity = cosine_similarity(main_piv_sparse)
        cosine_distance = pairwise_distances(main_piv_sparse, metric='cosine')
        manga_similarity = 1 - cosine_distance
        logging.info(f'calling manga_similiary: {manga_similarity}')
        # this gets pkled in my old file as manga pkl - this is the cosine similarity of similar users.
        manga_sim_df = pd.DataFrame(manga_similarity, index=user_ids, columns=user_ids)
        logging.info(f'calling manga_sim_df {manga_sim_df}')
        for user in manga_sim_df.sort_values(by=user_id, ascending=False).index[1:3]:
            logging.info(f'{user}, {round(manga_sim_df[user][user_id] * 100, 2)}% match\n')
            user_list.append((user, round(manga_sim_df[user][user_id] * 100, 2)))
    sorted_similar_user_list = sorted(user_list, key=lambda x: float(x[1]), reverse=True)
    user_ids_str = ', '.join([str(similar_user_id[0]) for similar_user_id in sorted_similar_user_list])
    logging.info(f'preparing to build manga_query')
    manga_query = f"""select m.mal_id, m.title,m.imageUrl, m.rating,                   
        m.media_type, m.author, m.status, m.summary, m.rank,
        count(ml.manga_id) as 'manga_count',
        avg(ml.rating) as 'average_rating', 
        (count(ml.manga_id) / 
        (count(ml.manga_id) + {m_value}))
         * avg(ml.rating) + 
         ({m_value} / (count(ml.manga_id) + {m_value})) 
         * (SELECT AVG(rating) FROM MangaList WHERE rating <> 0) AS 'weighted_rating',
         GROUP_CONCAT(DISTINCT g.genre_name) AS 'genres'
        FROM MangaList ml
    JOIN Manga m ON ml.manga_id = m.mal_id
    JOIN Genres mg ON mg.mal_id = m.mal_id
    JOIN Genre g ON g.id = mg.genre_id
    WHERE ml.manga_id NOT IN (
        SELECT manga_id
        FROM MangaList
        WHERE user_id = '{user_id}'
    )
    AND ml.user_id IN ({user_ids_str})
    AND ml.rating <> 0
    GROUP BY ml.manga_id, m.mal_id, m.title, m.imageUrl
    ORDER BY weighted_rating DESC;"""

    logging.info(manga_query)
    logging.info(f"Current memory usage: {mem_info.rss / 1024 / 1024} MB")
    manga_recs = db.query_raw(manga_query)
    logging.info(manga_recs)
    logging.info(f"Current memory usage: {mem_info.rss / 1024 / 1024} MB")
    return jsonify(manga_recs)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), threaded=True)
