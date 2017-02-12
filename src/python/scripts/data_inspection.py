import os
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt

from scripts import extract_openfoodfacts as extractor
from etc import config_openfoodfacts as config

categorized_docs_path = os.path.join(config.fs['data_path'], config.fs['categorized_docs'])

pd.set_option('display.mpl_style', 'default')
pd.set_option('display.line_width', 5000)
pd.set_option('display.max_columns', 60)
figsize = (15, 5)

#
# data extract
#
extractor.run()

#
# data fetch
#
product_df = pd.DataFrame(pd.read_pickle(categorized_docs_path))
product_df.columns

categories_df = pd.DataFrame(product_df['categories_hierarchy'].tolist())

#
# data inspection
#

# categories_hierarchy

product_df['categories_hierarchy'].apply(len).plot(kind='hist',bins=25,title='Category Length Distribution')

print("categories_hierarchy length info")
product_df['categories_hierarchy'].apply(len).describe()
print("Top-level categories")
categories_df[0].value_counts()
categories_df[0].value_counts().plot.pie()

#problem: the long tail...
#try min count cats in hierarchy required for a valid hierarchy

#problem: unbalanced dist of instances
#need to sub-sample the data categories, most classifier support class_weight

product_df['feature_bag'] = product_df.product_name + ' ' + product_df.brands + ' ' + product_df.quantity + ' ' + product_df.ingredients_text
print("feature_bag length info")
product_df['feature_bag'].apply(len).describe()

#problem: small amount of products have large feature_bag length...
#should truncate to ~90%-ile length to save computation/memory
