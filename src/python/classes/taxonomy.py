import pandas as pd
import numpy as np
import codecs
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import  cross_validation, preprocessing
from sklearn import naive_bayes, linear_model, svm, ensemble, neighbors, metrics

class TaxonomyTree:

    def __init__(self, description):
        self.description = description
        self.root = TaxTreeNode([])
        self.root.isRoot = True

    def add(self, nodeLabelChain):
        # pass to root node
        self.root.add(TaxTreeNode(nodeLabelChain))

    def initClassifiers(self, moduleName, classifierName, params):
        self.root.initClassifier(moduleName, classifierName, params)

    def getLocationClassifiers(self):
        locClsfrs = self.root.getLocationClassifiers()
        clsfrCount = self.root.getClassifierCount()
        if len(locClsfrs) != clsfrCount:
            raise ValueError('getLocationClassifiers / getClassifierCount differ on counts')
        return locClsfrs

    def fit(self, x_raw_all, y_raw_all):
        self.root.fit(x_raw_all, y_raw_all)

    def predict(self, x_data):
        #
        # TODO
        # return self.root.predict(x_data)

    def describe(self):
        print "Taxonomy: " + self.description + " contains "+str(self.root.getDescendentCount()+1)+" nodes, "+str(self.root.getClassifierCount())+" of which have a classifier"
        self.root.describe()
        print "Taxonomy: " + self.description + " contains "+str(self.root.getDescendentCount()+1)+" nodes, "+str(self.root.getClassifierCount())+" of which have a classifier"

class TaxTreeNode:
    def __init__(self, location):
        self.location = location                # list of parent node labels, in order from top to bottom
        self.children = {}                      # dict of taxonomy tree node, key: last entry in location
        self.isParent = False
        self.isRoot = False
        self.classifier = None

    def add(self, node):
        self.isParent = True
        if node.location is None or len(node.location) == 0:
            raise ValueError('trying to add a node without location')
        elif len(node.location) <= len(self.location):
            # fix location and keep at this level
            fixedLocation = self.location + [node.location[-1]]
            node.location = fixedLocation
            print "Warning: fixing bad location to "+node.getLocationStr()

        if cmp(node.location[:-1], self.location) == 0:
            # keep at this level
            self.children[node.location[-1]] = node
        else:
            # pass down to next level
            dstNodeLabel = node.location[len(self.location)]
            if not self.children.has_key(dstNodeLabel):
                self.children[dstNodeLabel] = TaxTreeNode(self.location + [dstNodeLabel])
            self.children[dstNodeLabel].add(node)

    def initClassifier(self, moduleName, classifierName, params):
        for child in self.children.values():
            child.initClassifier(moduleName, classifierName, params)
        if len(self.children) > 1:
            module = __import__(moduleName, fromlist=['dummy'])
            classifierClass = getattr(module, classifierName)
            self.classifier = classifierClass(**params)

    def getDescendentCount(self):
        num_desc = len(self.children)
        for child in self.children.values():
            num_desc += child.getDescendentCount()
        return num_desc

    def getClassifierCount(self):
        num_clsf = (1 if len(self.children) > 1 else 0)
        for child in self.children.values():
            num_clsf += child.getClassifierCount()
        return num_clsf

    def getLocationClassifiers(self):
        result = []
        if self.classifier is not None:
            result.append([self.location,self.classifier])
        for child in self.children.values():
            descendentResults = child.getLocationClassifiers()
            if len(descendentResults) > 0:
                result += (descendentResults)
        return result

    def fit(self, x_raw_all, y_raw_all):
        min_per_class_sample_count = 10
        min_class_count = 2
        rowixs = []
        #
        # Which of the data applies to this node
        #
        y_all = pd.DataFrame(y_raw_all)
        y_all['label'] = ''
        for ix,row in y_raw_all.iteritems():
            if len(row) > len(self.location) and cmp(row[:len(self.location)],self.location)==0:
                y_all['label'][ix] = row[len(self.location)]
                rowixs.append(ix)
        x_raw = x_raw_all[x_raw_all.index.isin(rowixs)]
        y = y_all['label'][y_all.index.isin(rowixs)]
        #
        # Split into train data for fitting classifier & test data for getting accuracy stat
        #
        #
        #TODO
        #push data split ot outside of class
        #
        x_train_raw, x_test_raw, y_train, y_test = cross_validation.train_test_split(x_raw, y, train_size=0.75,random_state=123)
        #
        # Which of this train data is sufficient in numbers
        #
        y_train=y_train.groupby(y_train).filter(lambda x: len(x) > min_per_class_sample_count)
        x_train_raw=x_train_raw[x_train_raw.index.isin(y_train.index)]
        #
        # Is there any classes remaining
        #
        pruned_child_keys = pd.unique(y_train.ravel())

        if len(pruned_child_keys) <= 0:
            #prune this entire branch
            print "Pruning "+self.getLocationStr()+", not enough samples for any children!"
            return

        if len(pruned_child_keys) > min_class_count and self.classifier != None:
            #
            # Vectorize and fit the classifier
            #
            print "Fitting: "+self.getLocationStr()
            #
            # TODO
            # push vectorizer config out of class
            #
            vectorizer = CountVectorizer(ngram_range=(1,1), stop_words='english')
            vectorizer.fit(x_train_raw)
            x_train = vectorizer.transform(x_train_raw)
            x_test = vectorizer.transform(x_test_raw)
            self.classifier.fit(x_train, y_train)
            preds = self.classifier.predict(x_test)
            print "%s:\tAccuracy: %0.3f\tF1 macro: %0.3f"%(self.getLocationStr(),
                        metrics.accuracy_score(y_test, preds), metrics.f1_score(y_test, preds, average='macro'))
            self.local_accuracy = metrics.accuracy_score(y_test, preds)
            self.local_f1_score = metrics.f1_score(y_test, preds, average='macro')
            #TODO
            #global metrics, taking into account loss in upper layers
        else:
            self.prune(y_train.value_counts().index[0])
            print "Skipping: "+self.getLocationStr()+", defaulting prediction to ["+self.getPredictDefaultStr()+"]"
        #
        # fit descendent classifiers
        #
        x_raw = None
        y_all = None
        y = None
        for child_key in pruned_child_keys:
            if self.children.has_key(child_key):
                self.children[child_key].fit(x_raw_all, y_raw_all)
        for child in self.children.values():
            if child.location[-1] not in pruned_child_keys:
                child.prune()

    def prune(self, default_predict=None):
        self.classifier = None
        self.default_predict = default_predict

    def predict(self, x_data):
        #
        # TODO
        #
        # check if fitted
        # check x_data
        # x_data : {array-like, sparse matrix} Samples
        #
        # recursive predict


    def getLocationStr(self):
        return str(self.location).encode('ascii', 'ignore')

    def getPredictDefaultStr(self):
        return self.predict_default.encode('ascii', 'ignore')


    def describe(self):
        if self.isRoot:
            print "<ROOT NODE> ["+str(self.classifier)+"] parent of " + str(len(self.children)) + " children"
        elif self.isParent:
            print "'" + self.location[-1].encode('ascii', 'ignore') + "' ["+str(self.classifier)+"] at " + self.getLocationStr() + " parent of " + str(len(self.children)) + " children"
        else:
            print "'" + self.location[-1].encode('ascii', 'ignore') + "' ["+str(self.classifier)+"] at " + self.getLocationStr()

        for child in self.children.values():
            child.describe()
