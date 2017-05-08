# taxonomy-predict

A tool to predict a suitable place for a product in an existing taxonomy.

Using data extracted from a product database, taxonomy-predict fits a tree of classifiers to the already categorized products.

This tree can then be used to classify un-categoried products.

This is a work in progress, preliminary results below.

# Setup

See [setup.txt](https://github.com/caldweln/taxonomy-predict/blob/master/setup.txt)

# Results

Training on a dataset of 55K products, where 5% is reserved for validation, achieved the following :

![results](https://cloud.githubusercontent.com/assets/9846264/25824831/920867ee-3438-11e7-9a96-fbb99d92981c.png)

![category length dist](https://cloud.githubusercontent.com/assets/9846264/20651407/dded64d8-b4dc-11e6-8dc5-488c51be4294.png)


| Classifier | Train Time |
|------------|------------|
| LogisticRegression | 14m44s |
| LinearSVC | 13m42s |
| RandomForestClassifier | 57m14s |
| MultinomialNB | 15m02s |

## Configuration

The classifier to be used, file locations and database settings can be configured at  [etc/config_openfoodfacts](https://github.com/caldweln/taxonomy-predict/blob/master/src/python/etc/config_openfoodfacts.py).



Results were achieved with the following classifier configurations:

```
classifier_module='sklearn.linear_model',
classifier_name='LogisticRegression',
classifier_params={'C':1,'class_weight':'balanced'}

classifier_module='sklearn.svm',
classifier_name='LinearSVC',
classifier_params={'C':1,'class_weight':'balanced'}

classifier_module='sklearn.ensemble',
classifier_name='RandomForestClassifier',
classifier_params={'n_estimators':10}

classifier_module='sklearn.naive_bayes',
classifier_name='MultinomialNB',
classifier_params={'alpha':1}

```


## Notes
- results obtained on a [Open Food Facts](http://world.openfoodfacts.org/data) mongodb data dump
- only product category hierarchies of length of at least 5 are considered
- LogisticRegression requires about 10Gb of RAM on OFF data
  - however others may use considerably more


# Disclaimer

No warranties, provided 'AS-IS'.
