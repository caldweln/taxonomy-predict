import pandas as pd
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
            fixedLocation = self.location + node.location[-1]
            print "Warning: fixing bad location ["+str(node.location)+"] -> ["+str(fixedLocation)+"]"
            node.location = fixedLocation

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

        min_per_class_sample_count = 30
        min_class_count = 2
        rowixs = []
        #
        # Which of the data applies to this node
        #
        y_all = pd.DataFrame(y_raw_all)
        for ix,row in y_raw_all.iteritems():
            if len(row) > len(self.location) and cmp(row[:len(self.location)],self.location)==0:
                y_all[ix] = row[len(self.location)]
                rowixs += [ix]
                #product_df['nxtCategory'][ix] = row['categories_hierarchy'][len(self.location)]

        x_raw = x_raw_all[x_raw_all.index.isin(rowixs)]
        y = y_all[y_all.index.isin(rowixs)]

        #
        # Split into train data for fitting classifier & test data for getting accuracy stat
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
            return

        if len(pruned_child_keys) > min_class_count:
            #
            # Vectorize and fit the classifier
            #
            vectorizer = CountVectorizer(ngram_range=(1,1), stop_words='english')
            vectorizer.fit(x_train_raw)
            x_train = vectorizer.transform(x_train_raw)
            x_test = vectorizer.transform(x_test_raw)
            self.classifier.fit(x_train, y_train)
            preds = self.classifier.predict(x_test)
            print "%s:\tAccuracy: %0.3f\tF1 macro: %0.3f"%(self.location,
                        metrics.accuracy_score(y_test, preds), metrics.f1_score(y_test, preds, average='macro'))
            self.local_accuracy = metrics.accuracy_score(y_test, preds)
            self.local_f1_score = metrics.f1_score(y_test, preds, average='macro')
            #TODO
            #global metrics, taking into account loss in upper layers
        else:
            self.classifier = None
            self.predict_default = y_train.value_counts().index[0]
            print "Insufficient class count at ["+str(self.location)+"] defaulting prediction to ["+self.predict_default+"]"


        for child_key in pruned_child_keys:
            self.children[child_key].fit(x_raw, y)



    def describe(self):
        if self.isRoot:
            print "<ROOT NODE> ["+str(self.classifier)+"] parent of " + str(len(self.children)) + " children"
        elif self.isParent:
            print "'" + self.location[-1] + "' ["+str(self.classifier)+"] at " + str(self.location) + " parent of " + str(len(self.children)) + " children"
        else:
            print "'" + self.location[-1] + "' ["+str(self.classifier)+"] at " + str(self.location)

        for child in self.children.values():
            child.describe()
