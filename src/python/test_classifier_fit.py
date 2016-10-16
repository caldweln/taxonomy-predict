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


# Get location/classifier pairs from taxonomy
locationClassifiers = t.getLocationClassifiers()
# For each pair, using location filter relevant product data for classifier training
for locClsfr in locationClassifiers:
    location = locClsfr[0]
    classifier = locClsfr[1]
    rowixs = []
    product_df['nxtCategory'] = ''
    for ix,row in product_df.iterrows():
        if len(row['categories_hierarchy']) > len(location) and cmp(row['categories_hierarchy'][:len(location)],location)==0:
            rowixs += [ix]
            product_df['nxtCategory'][ix] = row['categories_hierarchy'][len(location)]
    filtered_products_df = None
    filtered_products_df = product_df[product_df.index.isin(rowixs)]
    print "\nTraining ["+str(location)+"] classifier on the following samples"
    #print(filtered_products_df)

    x_train_raw, x_test_raw, y_train, y_test = cross_validation.train_test_split(filtered_products_df.feature_bag,
                                                                    filtered_products_df.nxtCategory, train_size=0.75,random_state=123)


    print "Before removing small counts..."
    print "x_train_raw"
    print x_train_raw.describe()
    print "y_train"
    print y_train.describe()

    #TODO
    # Handle issue where filtered and split dataset no longer has the sample count for classification
    y_train=y_train.groupby(y_train).filter(lambda x: len(x) > min_per_class_sample_count)
    x_train_raw=x_train_raw[x_train_raw.index.isin(y_train.index)]


    print "After...."
    print "x_train_raw"
    print x_train_raw.describe()
    print "y_train"
    print y_train.describe()

    if len(y_train.value_counts()) > min_class_count:

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
