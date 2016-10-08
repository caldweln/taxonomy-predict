
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import  cross_validation, preprocessing

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
categories_df = pd.DataFrame(product_df['categories_hierarchy'].tolist())

##########################
# data inspection

print "categories_hierarchy length info"
product_df['categories_hierarchy'].apply(len).describe()
print "Top-level categories"
categories_df[0].value_counts()
#problem: the long tail...

#idea: 5 cats in hierarchy required for a valid hierarchy
valid_cat_df = categories_df[categories_df[5].notnull()]
print "Top-level categories with a hierarchy depth > 5"
valid_cat_df[0].value_counts()
#idea: require at least n instances to train classifier
freq_tl_cats_counts = valid_cat_df[0].value_counts()[valid_cat_df[0].value_counts() > 100]
print "Plotting top-level categories with at least 100 instances"
freq_tl_cats_counts.plot.pie()
#problem: unbalanced dist of instances

#idea: need to sub-sample the data categories, will the classifier support class_weight??

#todo
# use dataframes to concat fields like below
# then filter data with above categories  

import sys
sys.exit()



ingredients = []
categories = []
cat_lens = []
res = [ingredients.append(d['product_name'] +' '+d['brands'] +' '+ d['quantity'] +' '+ d['ingredients_text']) for d in data['raw']]
res = [cat_lens.append(len(d['categories_hierarchy'])) for d in data['raw']]


print "Average length of docs.:", sum([len(x.split()) for x in ingredients])/float(len(ingredients)



##########################
# preprocessing


# train/test split
x_train_raw, x_test_raw, y_train, y_test = cross_validation.train_test_split(data['ingredients'],
                                                                data['categories'], train_size=0.75,random_state=123)
#print len(x_train_raw), len(x_test_raw)

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, HashingVectorizer, TfidfVectorizer

# try different vectorizers
my_representations = list()
for name, vectorizer in zip(['tf', 'hashing', 'tfidf', ],
                            [CountVectorizer(ngram_range=(1,1), stop_words='english'),
                                HashingVectorizer(n_features=10000, non_negative=True),
                                    TfidfVectorizer()]):
    vectorizer.fit(x_train_raw)
    my_representations.append({"name":name, "x_train":vectorizer.transform(x_train_raw), "x_test":vectorizer.transform(x_test_raw)})
    if name == 'tf':
        print len(vectorizer.vocabulary_)
