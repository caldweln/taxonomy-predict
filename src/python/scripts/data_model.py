import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import  cross_validation, preprocessing, metrics
import decimal
from classes import taxonomy_predict as tp

data_path = 'data/'
feature_file = 'feature_data.p'
feature_file_path = os.path.join(data_path, feature_file)

#
# Load data
#
#product_df = pd.DataFrame(pd.DataFrame(pd.read_pickle(feature_file_path)).sample(frac=0.1)).reset_index()
product_df = pd.DataFrame(pd.DataFrame(pd.read_pickle(os.path.join(data_path, 'feature_data.p')))).reset_index()


#
# Filter data with at least 5 categories, and only consider up to this category depth
#
categories_df = pd.DataFrame(product_df['categories_hierarchy'].tolist())
categories_df = categories_df[categories_df.columns[:5]].dropna()
features_df = pd.DataFrame(product_df['feature_bag'])[product_df.index.isin(categories_df.index)]

#
# Strip out non-ASCII
#
categories_df = categories_df.applymap(lambda x: x.encode('ascii','ignore'))
features_df = features_df.applymap(lambda x: x.encode('ascii','ignore'))

#
# Init multi-categorized features
#
arrays=categories_df.transpose().as_matrix()
vectorizer = CountVectorizer(ngram_range=(1,1), stop_words='english')
vectorizer.fit(features_df['feature_bag'])
feature_vectors = pd.DataFrame(vectorizer.transform(features_df['feature_bag']).toarray(), index= pd.MultiIndex.from_tuples(list(zip(*arrays))))

#
# Split for validation
#
x_train, x_test, y_train, y_test = cross_validation.train_test_split(feature_vectors, categories_df, train_size=0.75,random_state=123)

#
# Init/Fit tree of classifiers
#
t = tp.TreeOfClassifiers('Multi-Level Taxonomy')
t.fit(x_train)

preds = t.predict(x_test)


#
# Accuracy Score
#
miss_count = 0
level_miss_counts = []
for i in range(0,len(preds)):
    miss_count += abs(cmp(preds[i],y_test.iloc[i].tolist()))
    for j in range(0,len(preds[i])):
        if len(level_miss_counts) <= j:
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
