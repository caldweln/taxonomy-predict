
from pymongo import MongoClient
import pandas as pd
import pickle
import time
import sys
import os.path

db_conn_str = 'mongodb://localhost:27017/'
db_database = "off"
db_table = "products"
#assumes a local instance of OFF database exists
#mongorestore --collection products --db off ../dump/off/products.bson

data_path = 'data/'
raw_data_file = 'raw_data.p'
raw_data_path = os.path.join(data_path, raw_data_file)
tbc_data_file = 'raw_data-tbc.p'
tbc_data_path = os.path.join(data_path, tbc_data_file)
feature_file = 'feature_data.p'
feature_file_path = os.path.join(data_path, feature_file)
tbc_feature_file = 'feature_data-tbc.p'
tbc_feature_file_path = os.path.join(data_path, tbc_feature_file)

#
# query objects for extracting uncategorized & categorized datasets
#
find_where_tbc = { \
    'categories_hierarchy':{'$exists':'true', '$eq':[]}, \
    'product_name':{'$exists':'true','$ne':''}, \
    'brands':{'$exists':'true','$ne':''}, \
    'quantity':{'$exists':'true','$ne':''}, \
    'lang':{'$exists':'true','$eq':'fr'}, \
    'ingredients_text':{'$exists':'true','$ne':''}
    }
find_where_categorized = { \
    'categories_hierarchy':{'$exists':'true', '$ne':[]}, \
    'product_name':{'$exists':'true','$ne':''}, \
    'brands':{'$exists':'true','$ne':''}, \
    'quantity':{'$exists':'true','$ne':''}, \
    'lang':{'$exists':'true','$eq':'fr'}, \
    'ingredients_text':{'$exists':'true','$ne':''}
    }
find_select_fields = {"_id":1,"product_name":1,"brands":1,"quantity":1,"ingredients_text":1,"categories_hierarchy":1}


def extract_db_data(db_conn_str, db_database, db_table, file_path, find_where, find_select):

    #
    # db connect
    #
    start_time = time.time()

    client = MongoClient()

    print "Connecting to: {0}".format(db_conn_str)

    client = MongoClient(db_conn_str)

    db = client[db_database]

    print "Found tables/collections: " + ', '.join( db.collection_names() )

    products = db[db_table]

    #
    # data query
    #
    data_raw  = []
    data_raw = list(products.find(find_where, find_select))

    print "Loaded {0} {1} records".format(len(data_raw), db_table)

    print "Time taken {0:.2f} seconds".format(time.time() - start_time)

    #
    # save result data
    #
    pickle.dump(data_raw, open(file_path, 'wb'))

    print "Saved data to {0}".format(file_path)

def extractFeatureData(raw_data_path, feature_file_path):
    #
    # save feature data
    #
    product_df = pd.DataFrame(pd.read_pickle(raw_data_path))

    product_df['feature_bag'] = product_df.product_name + ' ' + product_df.brands + ' ' + product_df.quantity + ' ' + product_df.ingredients_text

    product_df[['feature_bag','categories_hierarchy']].to_pickle(feature_file_path)

    print "Saved feature data to {0}".format(feature_file_path)


#
# check data already extracted
#
if not os.path.exists(data_path):
    os.makedirs(data_path)

if os.path.isfile(tbc_data_path):
    print "Re-using extracted data at: {0}".format(tbc_data_path)
else:
    extract_db_data(db_conn_str, db_database, db_table, tbc_data_path, find_where_tbc, find_select_fields)

if os.path.isfile(tbc_feature_file_path):
    print "Re-using extracted data at: {0}".format(tbc_feature_file_path)
else:
    extractFeatureData(tbc_data_path, tbc_feature_file_path)

if os.path.isfile(raw_data_path):
    print "Re-using extracted data at: {0}".format(raw_data_path)
else:
    extract_db_data(db_conn_str, db_database, db_table, raw_data_path, find_where_categorized, find_select_fields)

if os.path.isfile(feature_file_path):
    print "Re-using extracted data at: {0}".format(feature_file_path)
else:
    extractFeatureData(raw_data_path, feature_file_path)
