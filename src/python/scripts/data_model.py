import os
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import  model_selection, preprocessing, metrics
from classes import taxonomy_predict as tp
from etc import config_openfoodfacts as config
from scripts import evaluate as ev

categorized_docs_path = os.path.join(config.fs['data_path'], config.fs['categorized_docs'])
uncategorized_docs_path = os.path.join(config.fs['data_path'], config.fs['uncategorized_docs'])
categorized_features_path = os.path.join(config.fs['data_path'], config.fs['categorized_features'])
uncategorized_features_path = os.path.join(config.fs['data_path'], config.fs['uncategorized_features'])
fitted_model_path = os.path.join(config.fs['data_path'], config.fs['fitted_model'])
fitted_vectorizer_path = os.path.join(config.fs['data_path'], config.fs['fitted_vectorizer'])

#
# Load data
#
#product_df = pd.DataFrame(pd.DataFrame(pd.read_pickle(categorized_features_path)).sample(frac=0.1)).reset_index()
product_df = pd.DataFrame(pd.DataFrame(pd.read_pickle(categorized_features_path))).reset_index()


#
# Filter data with at least 5 categories
#
categories_df = pd.DataFrame(product_df['categories_hierarchy'].tolist())
categories_df = categories_df[categories_df.columns[:5]].dropna()
features_df = pd.DataFrame(product_df['feature_bag'])[product_df.index.isin(categories_df.index)]

categories_df = pd.DataFrame(product_df['categories_hierarchy'].tolist())[product_df.index.isin(categories_df.index)]

#
# Vectorize features
#
if len(set(categories_df.index.tolist()) - set(features_df.index.tolist())) > 0:
    raise ValueError('ERROR - data mismatch to fit model')

vectorizer = CountVectorizer(ngram_range=(1,1), stop_words='english', strip_accents='unicode')
feature_vectors = pd.DataFrame(vectorizer.fit_transform(features_df['feature_bag']).toarray(), index=categories_df.index)
pickle.dump(vectorizer, open(fitted_vectorizer_path, 'wb'))

#
# Split for validation
#
x_train, x_test, y_train, y_test = model_selection.train_test_split(feature_vectors, categories_df, train_size=0.75,random_state=123)

#
# Init/Fit tree of classifiers
#
t = tp.TreeOfClassifiers('Multi-Level Taxonomy', config.op['classifier_module'], config.op['classifier_name'], config.op['classifier_params'])
t.fit(x_train, y_train)

#
# save fitted model
#
pickle.dump(t, open(fitted_model_path, 'wb'))

#
# score on validation data
#
preds = t.predict(x_test)

#
# convert y_test to list of lists, and rtrim None values
#
y_test_list = list(map(lambda l: list(filter(None,l)),y_test.values.tolist()))

#
# evaluate predictions
#
scores = list(map(lambda p,y: (len(y),*ev.score_class_pred(p,y)),preds,y_test_list))

scores_by_depth = pd.DataFrame(scores, columns=['Length','Precision','Recall','F-score']).groupby('Length').agg([np.mean]) #.plot()

print(scores_by_depth)
