
from pymongo import MongoClient
import pandas as pd
import pickle
import time
import sys
import os.path

class DataLayer:

    client = None
    db = None

    def __init__(self, db_conn_str, db_database):
        self.db_connect(db_conn_str, db_database)

    def db_connect(self, db_conn_str, db_database):
        #
        # db connect
        #
        self.client = MongoClient(db_conn_str)

        if db_database not in self.client.database_names():
            print("database list: " + str(self.client.database_names()))
            raise ValueError("requested database not found")

        self.db = self.client[db_database]

    def find_db_data(self, db_collection, file_path, find_where, find_select, db_conn_str=None, db_database=None):

        if db_conn_str is not None and db_database is not None:
            self.db_connect(db_conn_str, db_database)

        if db_collection not in self.db.collection_names():
            print("collection list: " + str(self.db.collection_names()))
            raise ValueError("requested collection not found")

        collection = self.db[db_collection]

        #
        # data query
        #
        db_documents  = []
        if find_select is not None and len(find_select) > 0:
            db_documents = list(collection.find(filter=find_where, projection=find_select))
        else:
            db_documents = list(collection.find(filter=find_where))

        print "Found {0} {1} documents".format(len(db_documents), db_collection)

        #
        # save result data
        #
        pickle.dump(db_documents, open(file_path, 'wb'))

        print "Saved documents to {0}".format(file_path)
