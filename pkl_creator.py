import pickle
import psutil
import logging
from sklearn.metrics import pairwise_distances
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
@app.route('/test/<int:user_id>')
async def user_recommendation(user_id):
    # Use the Prisma client to query your database and return the results as JSON
    total_rows = db.mangalist.count()
    batch_size = 40000
    offset = 0
    # this while loop goes through all the rows in the db in batches of the set batch_size
    while offset < total_rows:
        total_results = []
        results = db.mangalist.find_many(skip=offset, take=batch_size)
        for result in results:
            total_results.append((result.user_id, result.manga_id, result.rating))
        offset += batch_size
        # ------------------------offset change-------------------------------------------
        # making sure every batch has complete list of users. Prevents one batch having half of one user,
        # and another halving the half of that same user.
        last_user_array_count = 1
        last_user_db_count = db.mangalist.count(where={"user_id": total_results[-1][0]})
        while total_results[-1][0] == total_results[(-1 * (last_user_array_count + 1))][0]:
            last_user_array_count += 1
        if (last_user_db_count - last_user_array_count) > 1:
            results = db.mangalist.find_many(skip=offset, take=(last_user_db_count - last_user_array_count))
            for result in results:
                total_results.append((result.user_id, result.manga_id, result.rating))
            offset += (last_user_db_count - last_user_array_count)
        # ----------------------------offset end---------------------------------------

        df = pd.DataFrame(total_results)
        df.columns = ['user_id', 'manga_id', 'rating']
        with open(f"pkl_files/mangalist_df_{offset}.pkl", "wb") as f:
            pickle.dump(df, f)
    return 'complete'


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), threaded=True)
