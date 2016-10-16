import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import  cross_validation, preprocessing
from sklearn import naive_bayes, linear_model, svm, ensemble, neighbors, metrics

from classes import taxonomy as tax
# Load preprocessed data, labelled with categories_hierarchy
product_df = pd.DataFrame(pd.read_pickle('feature_data.p'))

t = tax.TaxonomyTree('Multi-Level Taxonomy')
#print product_df
# Load taxonomy using DataFrame iterator
res = [t.add(row['categories_hierarchy']) for ix,row in product_df.iterrows()]
# Initialize classifiers
t.initClassifiers('sklearn.linear_model','LogisticRegression',{'C':1,'class_weight':'balanced'})
# Print some info
# TODO fix ascii print issue
#t.describe()
print "--------------"

min_per_class_sample_count = 3
min_class_count = 2

x_raw_all = product_df['feature_bag']
y_raw_all = product_df['categories_hierarchy']

t.fit(x_raw_all, y_raw_all)
