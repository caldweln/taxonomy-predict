import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import  cross_validation, preprocessing
from classes import taxonomy_predict as tp

#
# Load data

data = {'categories_hierarchy':\
[['plant-based-foods','frozen','greens','salads'],\
['plant-based-foods','frozen','greens','salads'],\
['plant-based-foods','frozen','greens','legumes'],\
['plant-based-foods','frozen','greens','legumes']],
'feature_bag':\
["leafy crispy iceberg lettuce",
"leafy spinnach ",
"juicy eggplant",
"juicy courgettes"]
}

product_df = pd.DataFrame(data)

categories_df = pd.DataFrame(product_df['categories_hierarchy'].tolist())
features_df = pd.DataFrame(product_df['feature_bag'])

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
t.fit(feature_vectors)

#
#
#
t.describe()
