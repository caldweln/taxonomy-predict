
from pymongo import MongoClient
import pickle
import time

#########################
# data extract

db_conn_str = 'mongodb://localhost:27017/'
db_database = "off"
db_table = "products"
file_raw_data = '{0}-{1}.p'.format(db_database,  db_table)

#assumes a local instance of OFF database exists
#mongorestore --collection products --db off ../dump/off/products.bson


start_time = time.time()

client = MongoClient()

print "Connecting to: {0}".format(db_conn_str)

client = MongoClient(db_conn_str)

db = client[db_database]

print "Found tables/collections: " + ', '.join( db.collection_names() )

products = db[db_table]

data_raw  = []
data_raw = list(products.find({ \
    'categories_hierarchy':{'$exists':'true', '$ne':[]}, \
    'product_name':{'$exists':'true','$ne':''}, \
    'brands':{'$exists':'true','$ne':''}, \
    'quantity':{'$exists':'true','$ne':''}, \
    'ingredients_text':{'$exists':'true','$ne':''}
    }, \
    {"_id":1,"product_name":1,"brands":1,"quantity":1,"ingredients_text":1,"categories_hierarchy":1}))

print "Loaded {0} {1} records".format(len(data_raw), db_table)

print "Time taken {0:.2f} seconds: ".format(time.time() - start_time)

pickle.dump(data_raw, open(file_raw_data, 'wb'))

print "Saved {0} data to {1}".format(db_table, file_raw_data)
print "-------------------------------------------------------"
