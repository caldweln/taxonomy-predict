import pandas as pd
import pickle
import os.path
from classes import data_layer
from etc import config_openfoodfacts as config


def bag_features(docs_path, features_path):
    #
    # save feature data
    #
    data_df = pd.DataFrame(pd.read_pickle(docs_path))

    data_df['feature_bag'] = data_df.product_name + ' ' + data_df.brands + ' ' + data_df.quantity + ' ' + data_df.ingredients_text

    data_df[['feature_bag','categories_hierarchy']].to_pickle(features_path)

    print(("Saved feature data to {0}".format(features_path)))


def run():
    #
    # connect to database
    #
    # assumes a local instance of database is running
    # mongorestore --collection products --db off ../dump/off/products.bson
    dl = data_layer.DataLayer(config.db['db_conn_str'], config.db['db_database'])

    categorized_docs_path = os.path.join(config.fs['data_path'], config.fs['categorized_docs'])
    uncategorized_docs_path = os.path.join(config.fs['data_path'], config.fs['uncategorized_docs'])
    categorized_features_path = os.path.join(config.fs['data_path'], config.fs['categorized_features'])
    uncategorized_features_path = os.path.join(config.fs['data_path'], config.fs['uncategorized_features'])

    #
    # check data already extracted
    #
    if not os.path.exists(config.fs['data_path']):
        os.makedirs(config.fs['data_path'])

    if os.path.isfile(uncategorized_docs_path):
        print(("Re-using docs at: {0}".format(uncategorized_docs_path)))
    else:
        dl.find_db_data(config.db['db_table'], uncategorized_docs_path, config.db['find_where_uncategorized'], config.db['find_select_fields'])

    if os.path.isfile(uncategorized_features_path):
        print(("Re-using features at: {0}".format(uncategorized_features_path)))
    else:
        bag_features(uncategorized_docs_path, uncategorized_features_path)

    if os.path.isfile(categorized_docs_path):
        print(("Re-using docs at: {0}".format(categorized_docs_path)))
    else:
        dl.find_db_data(config.db['db_table'], categorized_docs_path, config.db['find_where_categorized'], config.db['find_select_fields'])

    if os.path.isfile(categorized_features_path):
        print(("Re-using features at: {0}".format(categorized_features_path)))
    else:
        bag_features(categorized_docs_path, categorized_features_path)
