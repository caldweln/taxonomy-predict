#########################
# data fetch

#assumes a local instance of OFF database exists
#mongorestore --collection products --db off ../dump/off/products.bson

from pymongo import MongoClient
import pickle

client = MongoClient()
client = MongoClient('mongodb://localhost:27017/')

db = client['off']

print db.collection_names()


products = db['products']

data = {'raw':[],'ingredients':[], 'categories':[]} # categories_hierarchy and ingredients_text,product_name,brands,quantity
#test = {'raw':[],'ingredients':[], 'categories':[]} # subset of train
#todo = {'raw':[],'ingredients':[], 'categories':[]} # no categories_hierarchy

data['raw'] = list(products.find({ \
    'categories_hierarchy':{'$exists':'true', '$ne':[]}, \
    'product_name':{'$exists':'true','$ne':''}, \
    'brands':{'$exists':'true','$ne':''}, \
    'quantity':{'$exists':'true','$ne':''}, \
    'ingredients_text':{'$exists':'true','$ne':''}
    }, \
    {"_id":1,"product_name":1,"brands":1,"quantity":1,"ingredients_text":1,"categories_hierarchy":1}))

res = [data['ingredients'].append(d['product_name'] +' '+d['brands'] +' '+ d['quantity'] +' '+ d['ingredients_text']) for d in data['raw']]
res = [data['categories'].append(d['categories_hierarchy'][-1]) for d in data['raw']]


#todo['raw'] = list(products.find({ \
#    'categories_hierarchy':{'$exists':'true','$eq':[]}, \
#    'product_name':{'$exists':'true','$ne':''}, \
#    'brands':{'$exists':'true','$ne':''}, \
#    'quantity':{'$exists':'true','$ne':''}, \
#    'ingredients_text':{'$exists':'true','$ne':''}, \
#    'lang':{'$exists':'true','$eq':'en'} \
#    }, \
#    {"_id":1,"product_name":1,"brands":1,"quantity":1,"ingredients_text":1}))




##########################
# preprocessing

from sklearn import  cross_validation, preprocessing
import scipy.sparse as sp

# train/test split
x_train_raw, x_test_raw, y_train, y_test = cross_validation.train_test_split(data['ingredients'],
                                                                data['categories'], train_size=0.75,random_state=123)
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


##########################
# data inspection


from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

print "Number of Products:", len(data['ingredients'])
print "Average length of docs.:", sum([len(x.split()) for x in data['ingredients']])/float(len(data['ingredients']))
print "Different classes:", len(set(data['categories']))

# print "Instances per class:", Counter(labels)
plt.hist(Counter(data['categories']).values())#np.arange(0, 215, 10), color='r', alpha=0.4)
plt.grid()
plt.title("Category system description: instances/class")
plt.xlabel("Instances")
plt.ylabel("Number of Classes")
plt.show()



import sys
sys.exit()


###########################
# learning



from sklearn import naive_bayes, linear_model, svm, ensemble, neighbors, metrics
from sklearn.ensemble import RandomForestClassifier

# configure
learners = [{"name":"LR", "model":linear_model.LogisticRegression(C=1)},
            {"name":"SVM", "model":svm.LinearSVC(C=1)},
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
