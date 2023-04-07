import pickle

import numpy as np
from quart import Quart, jsonify
from prisma import Prisma, register
import pandas as pd
import scipy as sp
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, hstack, csc_matrix

db = Prisma(
    http={
        'timeout': None,
    },
)
db.connect()
register(db)

app = Quart(__name__)


# 100,000 query limit so doing this in batches. Terms to lookup, sharding
@app.route('/test/<int:user_id>')
async def user_recommendation(user_id):
    # Use the Prisma client to query your database and return the results as JSON
    m_value = 10
    user_results = db.mangalist.find_many(where={"user_id": user_id})
    print(user_results)
    total_results = []
    for result in user_results:
        total_results.append((result.user_id, result.manga_id, result.rating))
    df = pd.DataFrame(total_results)
    df.columns = ['user_id', 'manga_id', 'rating']
    print(df.head())
    pivot = df.pivot_table(index=['user_id'], columns=['manga_id'], values='rating')
    pivot.fillna(0, inplace=True)
    user_piv_sparse = sp.sparse.csr_matrix(pivot.values)
    print(user_piv_sparse)
    print("----------------------------------------------------------------")
    # --------------------------------------------------------------
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
    # print(df.head())
    # pivot = df.pivot_table(index=['user_id'], columns=['manga_id'], values='rating')
    # this gets pkld as well as manga_list pkl. This is a table of every manga_id as a column, each row is a user and
    # their rating is each of their ratings.
    # print(pivot)
    # pivot.fillna(0, inplace=True)
    # piv_sparse = sp.sparse.csr_matrix(pivot.values)
    # with open("sm_mangalist.pkl", "wb") as f:
    #     pickle.dump(piv_sparse, f)
    # -------------------------------------------------------------------------

    with open("sm_mangalist.pkl", "rb") as f:
        piv_sparse = pickle.load(f)

    manga_similarity = cosine_similarity(piv_sparse)
    print(manga_similarity)
    # this gets pkled in my old file as manga pkl - this is the cosine similarity of similar users.
    manga_sim_df = pd.DataFrame(manga_similarity, index=pivot.index, columns=pivot.index)
    print(manga_sim_df)
    number = 1
    similar_users = ""
    user_list = []
    manga_list = []
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

    for user in manga_sim_df.sort_values(by=user_id, ascending=False).index[1:101]:
        print(f'#{number}: {user}, {round(manga_sim_df[user][user_id] * 100, 2)}% match\n')
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
    print(manga_query)
    manga_recs = db.query_raw(manga_query)
    print(manga_recs)
    return manga_recs


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)
