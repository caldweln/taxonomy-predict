import pandas as pd
import numpy as np
import codecs
from sklearn import naive_bayes, linear_model, svm, ensemble, neighbors, metrics
from exceptions import NotFittedError

class TreeOfClassifiers:

    def __init__(self, description, moduleName='sklearn.linear_model',classifierName='LogisticRegression',params={'C':1,'class_weight':'balanced'}):
        self.description = description
        self.root = TreeNode([])
        self.root.isRoot = True

        self.moduleName = moduleName
        self.classifierName = classifierName
        self.params = params

    def fit(self, x_all, y_all):
        print "Building tree of classifiers"
        res = [self.root.add(TreeNode(list(row.dropna()))) for ix,row in y_all.iterrows()]
        self.root.initClassifier(self.moduleName, self.classifierName, self.params)
        print "Fitting to data"
        self.root.fit(x_all, y_all)


    def predict(self, x_data):
        results = []
        res = [results.append(self.root.predict(row)) for ix,row in x_data.iterrows()]
        return results

    def getLocationClassifiers(self):
        locClsfrs = self.root.getLocationClassifiers()
        clsfrCount = self.root.getClassifierCount()
        if len(locClsfrs) != clsfrCount:
            raise ValueError('getLocationClassifiers / getClassifierCount differ on counts')
        return locClsfrs

    def get_tree_depth(self):
        return self.root.max_dist_to_leaf()


    def describe_levels(self):
        for l in range(1,self.get_tree_depth()+1):
            self.getLevelStats(l)

    def getLevelStats(self, level_num):
        nodes=self.root.seekNodesAt(level_num)
        total_acc = 0
        total_f1_score = 0
        total_count = 0
        for node in nodes:
            if node.local_accuracy is not None:
                total_count += 1
                total_acc += node.local_accuracy
                total_f1_score += node.local_f1_score
        if total_count > 0:
            total_acc /= total_count
            total_f1_score /= total_count
            print "Level "+str(level_num)+", "+str(total_count)+"/"+str(len(nodes))
            print "Avg Accuracy: "+str(total_acc)
            print "Avg F1 Score: "+str(total_f1_score)
        else:
            print "Level "+str(level_num)+", "+str(total_count)+"/"+str(len(nodes))


    def describe(self):
        print "Taxonomy: " + self.description + " contains "+str(self.root.getDescendentCount()+1)+" nodes, "+str(self.root.getClassifierCount())+" of which have a classifier"
        self.root.describe()
        print "Taxonomy: " + self.description + " contains "+str(self.root.getDescendentCount()+1)+" nodes, "+str(self.root.getClassifierCount())+" of which have a classifier"

class TreeNode:
    def __init__(self, location):
        self.location = location                # list of parent node labels, in order from top to bottom
        self.children = {}                      # dict of taxonomy tree node, key: last entry in location
        self.isParent = False
        self.isRoot = False
        self.classifier = None
        self.isFitted = False
        self.local_accuracy = None
        self.local_f1_score = None
        self.min_per_class_sample_count = 2
        self.min_class_count = 2
        self.default_predict = None

    def seekNodesAt(self, level_to_go):
        if level_to_go<=0:
            return self.children.values()
        else:
            result = []
            for child in self.children.values():
                result += child.seekNodesAt(level_to_go-1)
            return result

    def add(self, node):
        self.isParent = True
        if node.location is None or len(node.location) == 0 or len(node.location) <= len(self.location):
            raise ValueError('trying to add a node with bad/missing location')

        if cmp(node.location[:-1], self.location) == 0:
            # keep at this level
            if not self.children.has_key(node.location[-1]):
                #print "Adding child at "+self.getLocationStr()
                self.children[node.location[-1]] = node
        else:
            # pass down to next level
            dstNodeLabel = node.location[len(self.location)]
            if not self.children.has_key(dstNodeLabel):
                self.add(TreeNode(self.location + [dstNodeLabel]))
            self.children[dstNodeLabel].add(node)

    def initClassifier(self, moduleName, classifierName, params):
        for child in self.children.values():
            child.initClassifier(moduleName, classifierName, params)
        if len(self.children) == 1:
            self.default_predict = self.children.keys()[0]
        elif len(self.children) > 1:
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
                result.extend(descendentResults)
        return result

    def fit(self, x_all, y_all):
        if len(self.children) <= 0:
            return
        #
        # fit child classifiers
        #
        for child in self.children.values():
            child.fit(x_all, y_all) #maybe just pass x_fit
        #
        # Which of the data applies to this node
        #
        if len(self.location) > 1:
            y_local = y_all.groupby(y_all.columns[:len(self.location)].tolist()).get_group(tuple(self.location))
            x_local = x_all[x_all.index.isin(y_local.index)]
        elif len(self.location) == 1:
            y_local = y_all.groupby(y_all.columns[0]).get_group(self.location[0])
            x_local = x_all[x_all.index.isin(y_local.index)]
        else:
            y_local = y_all
            x_local = x_all

        #
        #
        y_fit=y_local.groupby(list(y_local.columns[:len(self.location)+1])).filter(lambda x: len(x) >= self.min_per_class_sample_count)
        x_fit=x_local[x_local.index.isin(y_fit.index)]
        #
        # Is there any classes remaining
        #
        #
        pruned_child_keys = y_fit[len(self.location)].dropna().unique()

        if len(pruned_child_keys) <= 0:
            #print "Pruning "+self.getLocationStr()+", not enough samples for any children!"
            self.prune()
            return #prune this entire branch


        if len(pruned_child_keys) >= self.min_class_count:
            #
            # fit the classifier
            #
            #print "Fitting: "+self.getLocationStr()
            if len(set(x_fit.index.tolist()) - set(y_fit.index.tolist())) > 0:
                raise ValueError('BAD ERROR - data mismatch to fit classifier')

            self.classifier.fit(x_fit, y_fit[len(self.location)])
            self.isFitted = True
            self.local_accuracy = 1.0 #metrics.accuracy_score(y_test, preds)
            self.local_f1_score = 1.0 #metrics.f1_score(y_test, preds, average='macro')

        else:
            self.prune(pruned_child_keys[0])
            #print "Skipping: "+self.getLocationStr()+", defaulting prediction to 1/"+str(len(pruned_child_keys))+" ["+self.getPredictDefaultStr()+"]"

    def prune(self, default_predict=None):
        self.classifier = None
        self.default_predict = default_predict
        if default_predict is not None and self.children.has_key(default_predict):
            prunes = self.children.keys()
            #print str(prunes.remove(default_predict)) + " PRUNED FROM CHILDREN LIST"
            self.children = {default_predict:self.children[default_predict]} # removing other children
        else:
            #print str(self.children.keys()) + " PRUNED FROM CHILDREN LIST"
            self.children = {}

    def predict(self, x_data):
        #TODO validate x_data
        pred_results = []

        if len(self.children) == 0:
            return pred_results
        if len(self.children) == 1:
            pred_results.append(self.default_predict)
        if len(self.children) > 1:
            if self.isFitted:
                pred_results.append(self.classifier.predict(x_data.as_matrix().reshape(1,-1))[0])
            else:
                pred_results.append(self.default_predict)

        if pred_results[-1] not in self.children:
            raise ValueError("ERROR: predicting unknown child: "+str(pred_results[-1]))

        desc_preds = self.children[pred_results[-1]].predict(x_data)
        if desc_preds is not None:
            pred_results.extend(desc_preds)
        return pred_results

    def getLocationStr(self):
        return str(self.location).encode('ascii', 'ignore')

    def toASCII(self, _str):
        return _str.encode('ascii', 'ignore')


    def getPredictDefaultStr(self):
        return self.default_predict.encode('ascii', 'ignore')

    def max_dist_to_leaf(self):
        if len(self.children) <= 0:
            return 0
        else:
            min_dist = 1
            result = min_dist
            for child in self.children.values():
                result = max(result, min_dist + child.max_dist_to_leaf())
            return result

    def describe(self):
        if self.isRoot:
            print "<ROOT NODE> ["+str(self.classifier)+"] parent of " + str(len(self.children)) + " children"
        elif self.isParent:
            print "'" + self.location[-1].encode('ascii', 'ignore') + "' ["+str(self.classifier)+"] at " + self.getLocationStr() + " parent of " + str(len(self.children)) + " children"
        else:
            print "'" + self.location[-1].encode('ascii', 'ignore') + "' ["+str(self.classifier)+"] at " + self.getLocationStr()

        for child in self.children.values():
            child.describe()
