
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

    def describe(self):
        if self.isRoot:
            print "<ROOT NODE> ["+str(self.classifier)+"] parent of " + str(len(self.children)) + " children"
        elif self.isParent:
            print "'" + self.location[-1] + "' ["+str(self.classifier)+"] at " + str(self.location) + " parent of " + str(len(self.children)) + " children"
        else:
            print "'" + self.location[-1] + "' ["+str(self.classifier)+"] at " + str(self.location)

        for child in self.children.values():
            child.describe()
