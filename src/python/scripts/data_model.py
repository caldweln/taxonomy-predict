import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import  model_selection, preprocessing, metrics
import decimal
from classes import taxonomy_predict as tp
from etc import config_openfoodfacts as config

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

miss_count = 0
level_miss_counts = []
for i in range(0,len(preds)):
    if abs(cmp(preds[i],y_test.iloc[i].tolist())) > 0:
        miss_count += 1
    for j in range(0,len(preds[i])):
        while len(level_miss_counts) < len(preds[i]):
            level_miss_counts.append(0)
        if preds[i][j] != y_test.iloc[i].tolist()[j]:
            for k in range(j,len(preds[i])):
                level_miss_counts[k] += 1
            break # one bad prediction causes misses for rest of prediction chain

miss_count = decimal.Decimal(miss_count)
pred_count = decimal.Decimal(len(preds))
print "Perfect predictions: {0:.2f}%".format(100-(miss_count*100/pred_count))
for i in range(0,len(level_miss_counts)):
    miss_count = decimal.Decimal(level_miss_counts[i])
    print "Accuracy at level {0}: {1:.2f}%".format(i, 100-(miss_count*100/pred_count))
