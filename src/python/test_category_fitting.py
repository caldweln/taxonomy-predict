import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import  cross_validation, preprocessing, metrics
from classes import taxonomy as tax

#
# Load data
#

#data = {'categories_hierarchy':\
#[['plant-based-foods','frozen','greens','salads'],\
#['plant-based-foods','frozen','greens','salads'],\
##['plant-based-foods','frozen','greens','legumes'],\
#['plant-based-foods','frozen','greens','legumes']],
#'feature_bag':\
#["leafy crispy iceberg lettuce",
#"leafy spinnach ",
#"juicy eggplant",
#"juicy courgettes"]
#}

#product_df = pd.DataFrame(data)
#product_df = pd.DataFrame(pd.DataFrame(pd.read_pickle('feature_data.p')).sample(frac=0.1, random_state=8)).reset_index()
product_df = pd.DataFrame(pd.DataFrame(pd.read_pickle('feature_data.p'))).reset_index()


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
t = tax.TaxonomyTree('Multi-Level Taxonomy')
t.fit(x_train)

preds = t.predict(x_test)


#TODO
# Accuracy Score
#

#print "Score: %0.3f"%(metrics.accuracy_score(y_test, preds))
