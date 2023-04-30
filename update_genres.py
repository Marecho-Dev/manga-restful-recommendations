import pickle
import numpy as np
from prisma.models import Genres
from quart import Quart, jsonify
from prisma import Prisma, register
import openpyxl
import json
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
    query_list = []
    with open('manga_insert_queries.txt', encoding='utf-8') as f:
        query_count = 0
        query = ''
        for line in f:
            print(query_count)
            query += line
            if query.endswith(';\n'):
                query_list.append(query.strip())
                query_count += 1
                query = ''


    for insert_query in query_list:
        print('------------------------------------------')
        print(insert_query)
        db.manga.query_raw(insert_query)
    # genre_results = db.genre.find_many()
    # print(genre_results)
    # wb = openpyxl.load_workbook('manga_genres.xlsx')
    # worksheet = wb.active
    # for row_index, row in enumerate(worksheet.iter_rows()):
    #     cell1 = row[0].value
    #     cell2 = row[1].value
    #     print(cell1)
    #     print(cell2)
    #     genres_str = cell2
    #     genres_str = genres_str.replace("'", "\"")
    #     genres = json.loads(genres_str)
    #     for genre in genres:
    #         genre_id = genre['id']
    #         genre_name = genre['name']
    #         print(f"Genre ID: {genre_id}, Genre Name: {genre_name}")
    #         genre_dict = {
    #             "mal_id": cell1,
    #             "genre_id": genre_id,
    #             "genre_name": genre_name,
    #         }
    #         db.genres.create(genre_dict)
    return 'test'


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)
