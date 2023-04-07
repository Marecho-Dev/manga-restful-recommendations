from quart import Quart, jsonify
from prisma import Prisma, register
from prisma.models import User
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.metrics.pairwise import cosine_similarity


db = Prisma(
    http={
        'timeout': None,
    },
)
register(db)

app = Quart(__name__)


# 100,000 query limit so doing this in batches. Terms to lookup, sharding
@app.route('/recommendation/<int:user_id>')
async def user_recommendation(user_id):
    # Use the Prisma client to query your database and return the results as JSON
    total_rows = await db.mangalist.count()
    print(total_rows)
    batch_size = 100000
    offset = 0
    total_results = []
    while offset < total_rows:
        results = await db.mangalist.find_many(skip=offset, take=batch_size)
        for result in results:
            total_results.append((result.user_id, result.manga_id, result.rating))
        offset += batch_size
    df = pd.DataFrame(total_results)
    df.columns = ['user_id', 'manga_id', 'rating']
    print(df.head())
    pivot = df.pivot_table(index=['user_id'], columns=['manga_id'], values='rating')
    # this gets pkld as well as manga_list pkl. This is a table of every manga_id as a column, each row is a user and
    # their rating is each of their ratings.
    print(pivot)
    pivot.fillna(0, inplace=True)
    piv_sparse = sp.sparse.csr_matrix(pivot.values)
    print(piv_sparse)
    manga_similarity = cosine_similarity(piv_sparse)
    print(manga_similarity)
    # this gets pkled in my old file as manga pkl - this is the cosine similarity of similar users.
    manga_sim_df = pd.DataFrame(manga_similarity, index=pivot.index, columns=pivot.index)
    print(manga_sim_df)
    number = 1
    print('Recommended similar users {}:\n'.format(user_id))
    similar_users = ""
    user_list = []
    manga_list = []
    manga_query = f"""select manga_id,
    count(manga_id) as 'manga_count',
    avg(rating)*((count(manga_id)*.3)) as 'average' 
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


    for user in manga_sim_df.sort_values(by=user_id, ascending=False).index[1:11]:
        print(f'#{number}: {user}, {round(manga_sim_df[user][user_id] * 100, 2)}% match\n')
        similar_users += f'#{number}: {user}, {round(manga_sim_df[user][user_id] * 100, 2)}% match\n'
        user_list.append(user)
        manga_query = manga_query + f"\n            user_id = '{user}'"

        if number < 10:
            manga_query = manga_query + ' OR '
        else:
            manga_query = manga_query + """) 
            and rating <> 0 
            ) as subquery 
            group by 
            manga_id order by 
            average DESC;"""

        print(f'#{number}: {user}, {round(manga_sim_df[user][user_id] * 100, 2)}% match')
        number += 1
    print(manga_query)
    manga_recs = await db.query_raw(manga_query)
    print("calling manga recs query --------------------------------------------------------------------------")
    print(manga_recs)
    # for manga in manga_recs:
    #     manga_info= f"select * from mangas where mal_id = '{manga}"
    #     manga_info_res = db.execute_raw(manga_info)
    return manga_recs


# @app.route('/mangalist/<int:user_id>')
# async def getMangaList(user_id):
#   # Use the Prisma client to query your database and return the results as JSON
#   results = await db.mangalist.find_many(where={'user_id':user_id},include={'user':True, 'manga':True},)
#   mangas = []
#   for result in results:
#       mangas.append({'user_id': result.user_id,
#       'manga_id': result.manga_id,
#       'rating': result.rating,
#       'manga_title': result.manga.title,
#      'manga_url': result.manga.imageUrl,})
#   return jsonify(mangas)


if __name__ == '__main__':
    # Connect to the Prisma client before running the Flask app
    import asyncio

    loop = asyncio.get_event_loop()
    loop.run_until_complete(db.connect())
    app.run(debug=True)
