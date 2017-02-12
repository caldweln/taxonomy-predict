import pickle
import os.path
from classes import data_layer
from etc import config_openfoodfacts as config

def confirm():
    while True:
        yes = set(['yes','y', 'ye', ''])
        no = set(['no','n'])

        choice = input().lower()
        if choice in yes:
           return True
        elif choice in no:
           return False
        else:
           sys.stdout.write("Please respond with 'yes' or 'no'")

def run():
    #
    # connect to database
    #
    # assumes a local instance of database is running
    # mongorestore --collection products --db off ../dump/off/products.bson
    dl = data_layer.DataLayer(config.db['db_conn_str'], config.db['db_database'])

    prediction_results_path = os.path.join(config.fs['data_path'], config.fs['prediction_results'])

    #
    # check results data exists
    #
    if not os.path.exists(prediction_results_path):
        print("Prediction results not found")
        return

    results_dict = pickle.load( open( prediction_results_path, "rb" ) )

    #
    # confirm & action
    #
    print("You are about to update {0} records, are you sure you wish to continue? [yes/no] (yes)".format(len(results_dict)))

    if confirm():
        dl.update_db_data(config.db['db_table'], config.db['update_filter_field'], config.db['update_update_field'], results_dict)
    else:
        print("Exiting without action")
