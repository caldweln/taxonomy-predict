
db_conn_str = 'mongodb://localhost:27017/'
db_database = "off"
db_table = "products"
file_raw_data = '{0}-{1}.p'.format(db_database,  db_table)


#########################
# data fetch

data_raw_list = pd.read_pickle(file_raw_data))

ingredients_list = []
categories_list = []
res = [ingredients_list.append(d['product_name'] +' '+d['brands'] +' '+ d['quantity'] +' '+ d['ingredients_text']) for d in data_raw_list]
res = [categories_list.append(d['categories_hierarchy'][-1]) for d in data_raw_list]


##########################
# preprocessing

from sklearn import  cross_validation, preprocessing
import scipy.sparse as sp

# train/test split
x_train_raw, x_test_raw, y_train, y_test = cross_validation.train_test_split(ingredients_list,
                                                                categories_list, train_size=0.75,random_state=123)
#print len(x_train_raw), len(x_test_raw)

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, HashingVectorizer, TfidfVectorizer

# try different vectorizers
my_representations = list()
for name, vectorizer in zip(['tf', 'hashing', 'tfidf', ],
                            [CountVectorizer(ngram_range=(1,1), stop_words='english'),
                                HashingVectorizer(n_features=10000, non_negative=True),
                                    TfidfVectorizer()]):
    vectorizer.fit(x_train_raw)
    my_representations.append({"name":name, "x_train":vectorizer.transform(x_train_raw), "x_test":vectorizer.transform(x_test_raw)})
    if name == 'tf':
        print len(vectorizer.vocabulary_)


###########################
# learning



from sklearn import naive_bayes, linear_model, svm, ensemble, neighbors, metrics
from sklearn.ensemble import RandomForestClassifier

# configure
learners = [{"name":"LR", "model":linear_model.LogisticRegression(C=1,class_weight='balanced')},
            {"name":"SVM", "model":svm.LinearSVC(C=1,class_weight='balanced')},
            {"name":"5-NN", "model":neighbors.KNeighborsClassifier(n_neighbors=5)},
            {"name":"Rochio", "model":neighbors.NearestCentroid()},
            {"name":"N.B.", "model":naive_bayes.MultinomialNB(alpha=1)},
            {"name":"R.F.", "model":RandomForestClassifier(n_estimators = 100)}]
# fit and test
for representation in my_representations:
    print "\tRepresentation:", representation["name"]
    for learner in learners:
        learner['model'].fit(representation["x_train"], y_train)
        preds = learner['model'].predict(representation["x_test"])
        print "%s:\tAccuracy: %0.3f\tF1 macro: %0.3f"%(learner['name'],
                    metrics.accuracy_score(y_test, preds), metrics.f1_score(y_test, preds, average='macro'))
    print "----------------"
