# taxonomy-predict

A tool to predict a suitable place for a product in an existing taxonomy.

Using data extracted from a product database, taxonomy-predict fits a tree of classifiers to the already categorized products. This tree can then be used to classify un-categoried products.

This is a work in progress, preliminary results below.

# Setup

See [setup.txt](https://github.com/caldweln/taxonomy-predict/blob/master/setup.txt)

# Results

| Classifier \ Accuracy | @Level 0 | @Level 1 | @Level 2 | @Level 3 | @Level 4 |
| ----------------------|----------|----------|----------|----------|----------|
| LogisticRegression    |  88.67%  |  85.16%  |  77.69%  |  71.83%  |  71.83%  |
| LinearSVC             |  87.85%  |  84.44%  |  76.79%  |  71.09%  |  65.32%  |
| RandomForestClassifier|  87.48%  |  83.94%  |  75.07%  |  69.00%  |  62.56%  |
| KNeighborsClassifier  |  82.26%  |  77.84%  |  68.21%  |  61.78%  |  55.09%  |
| MultinomialNB         |  80.91%  |  74.10%  |  62.29%  |  54.15%  |  45.23%  |


## Classifier configuration

LogisticRegression, params={'C':1,'class_weight':'balanced'}

LinearSVC, params={'C':1,'class_weight':'balanced'}

RandomForestClassifier,params={'n_estimators':100}

KNeighborsClassifier, params={'n_neighbors':4}

MultinomialNB, params={'alpha':1}



## Notes

 - training
  - considered only products marked with lang='fr'
  - all non-ASCII characters are dropped
  - only product category hierarchies of length of 5 are considered
    - shorter hierarchies are dropped
    - longer hierarchies are truncated
  - RAM up to 32Gb required for some classifiers
 - validation
  - a validation set of 25% of the dataset is used to calculate results
