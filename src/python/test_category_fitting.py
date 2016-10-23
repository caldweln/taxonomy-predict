import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import  cross_validation, preprocessing
from sklearn import naive_bayes, linear_model, svm, ensemble, neighbors, metrics
from classes import taxonomy as tax
#
# Load data
#
#product_df = pd.DataFrame(data)
product_df = pd.DataFrame(pd.DataFrame(pd.read_pickle('feature_data.p')).sample(frac=0.05)).reset_index()
#
# Filter data with at least 5 categories, and only consider up to this category depth
#
categories_df = pd.DataFrame(product_df['categories_hierarchy'].tolist())
categories_df = categories_df[categories_df.columns[:5]].dropna()
features_df = pd.DataFrame(product_df[['feature_bag','categories_hierarchy']])[product_df.index.isin(categories_df.index)]
#
# Init tree of classifiers
#
t = tax.TaxonomyTree('Multi-Level Taxonomy')
#
#
# TODO migrate this to fit()
# Load taxonomy using DataFrame iterator
res = [t.add(row['categories_hierarchy']) for ix,row in features_df.iterrows()]

#
# Init multi-categorized features
#
arrays=categories_df.transpose().as_matrix()
vectorizer = CountVectorizer(ngram_range=(1,1), stop_words='english')
vectorizer.fit(features_df['feature_bag'])
feature_vectors = pd.DataFrame(vectorizer.transform(features_df['feature_bag']).toarray(), index= pd.MultiIndex.from_tuples(list(zip(*arrays))))

# TODO
# Split for validation
#
#x_train_raw, x_test_raw, y_train, y_test = cross_validation.train_test_split(x_raw_all, y_raw_all, train_size=0.75,random_state=123)

#
# Fit classifiers
#
t.fit(feature_vectors)
