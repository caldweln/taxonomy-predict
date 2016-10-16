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

import sys
sys.exit()



# Get location/classifier pairs from taxonomy
locationClassifiers = t.getLocationClassifiers()
# For each pair, using location filter relevant product data for classifier training
for locClsfr in locationClassifiers:
    location = locClsfr[0]
    classifier = locClsfr[1]
    rowixs = []
    product_df['nxtCategory'] = ''

    #
    # Which of the data applies to me
    #
    for ix,row in product_df.iterrows():
        if len(row['categories_hierarchy']) > len(location) and cmp(row['categories_hierarchy'][:len(location)],location)==0:
            rowixs += [ix]
            product_df['nxtCategory'][ix] = row['categories_hierarchy'][len(location)]
    filtered_products_df = None
    filtered_products_df = product_df[product_df.index.isin(rowixs)]

    #
    # Split into train data for fitting classifier & test data for getting accuracy results
    #
    x_train_raw, x_test_raw, y_train, y_test = cross_validation.train_test_split(filtered_products_df.feature_bag,
                                                                    filtered_products_df.nxtCategory, train_size=0.75,random_state=123)

    #
    # Which of the data that applies to me, is sufficient in numbers
    #

    y_train=y_train.groupby(y_train).filter(lambda x: len(x) > min_per_class_sample_count)
    x_train_raw=x_train_raw[x_train_raw.index.isin(y_train.index)]



    if len(y_train.value_counts()) > min_class_count:


        #
        # Vectorize what's left
        # And fit the classifier
        #


        vectorizer = CountVectorizer(ngram_range=(1,1), stop_words='english')
        vectorizer.fit(x_train_raw)
        x_train = vectorizer.transform(x_train_raw)
        x_test = vectorizer.transform(x_test_raw)
        classifier.fit(x_train, y_train)
        preds = classifier.predict(x_test)
        print "%s:\tAccuracy: %0.3f\tF1 macro: %0.3f"%(location,
                    metrics.accuracy_score(y_test, preds), metrics.f1_score(y_test, preds, average='macro'))
    else:
        print "Skipping "+str(location)+" due to insufficient class count"
        #TODO flag a default prediction for this node/classifier
        #remove this classifier



print "--------------"
