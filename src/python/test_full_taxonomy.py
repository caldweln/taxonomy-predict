import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import  cross_validation, preprocessing

from classes import taxonomy as tax

db_conn_str = 'mongodb://localhost:27017/'
db_database = "off"
db_table = "products"
file_raw_data = '{0}-{1}.p'.format(db_database,  db_table)

pd.set_option('display.mpl_style', 'default')
pd.set_option('display.line_width', 5000)
pd.set_option('display.max_columns', 60)
figsize = (15, 5)

#########################
# data fetch

product_df = pd.DataFrame(pd.read_pickle(file_raw_data))

t =  tax.TaxonomyTree('Full OFF Taxonomy')

res = [t.add(row['categories_hierarchy']) for ix,row in product_df.iterrows()]

# Initialize classifiers
#t.initClassifiers('sklearn.linear_model','LogisticRegression',{'C':1,'class_weight':'balanced'})
# Print some info

t.describe()
