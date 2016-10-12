
class TaxonomyTree:

    def __init__(self, description):
        self.description = description
        self.root = TaxTreeNode(['root'])
        self.root.isRoot = True

    def add(self, nodeLabelChain):
        # pass to root node
        self.root.add(TaxTreeNode(['root'] + nodeLabelChain))

    def initClassifiers(self, moduleName, classifierName, params):
        self.root.initClassifier(moduleName, classifierName, params)

    def describe(self):
        print "Taxonomy: " + self.description
        self.root.describe()

class TaxTreeNode:
    def __init__(self, parentChain):
        self.parentChain = parentChain          # list of parent node labels, in order from top to bottom
        self.children = {}                      # dict of taxonomy tree node, key: last entry in parentChain
        self.isParent = False
        self.isRoot = False
        self.classifier = None

    def add(self, node):
        self.isParent = True
        if node.parentChain is None or len(node.parentChain) <= len(self.parentChain):
            # fix parentChain and keep at this level
            node.parentChain = self.parentChain + node.parentChain[-1]

        if cmp(node.parentChain[:-1], self.parentChain) == 0:
            # keep at this level
            self.children[node.parentChain[-1]] = node
        else:
            # pass down to next level
            dstNodeLabel = node.parentChain[len(self.parentChain)]
            if not self.children.has_key(dstNodeLabel):
                self.children[dstNodeLabel] = TaxTreeNode(self.parentChain + [dstNodeLabel])
            self.children[dstNodeLabel].add(node)

    def initClassifier(self, moduleName, classifierName, params):
        for child in self.children.values():
            child.initClassifier(moduleName, classifierName, params)
        if len(self.children) > 1:
            module = __import__(moduleName, fromlist=['dummy'])
            classifierClass = getattr(module, classifierName)
            self.classifier = classifierClass(**params)

    def describe(self):
        if self.isRoot:
            print "'" + self.parentChain[-1] + "' ["+str(self.classifier)+"] parent of " + str(len(self.children)) + " children"
        elif self.isParent:
            print "'" + self.parentChain[-1] + "' ["+str(self.classifier)+"] at " + str(self.parentChain) + " parent of " + str(len(self.children)) + " children"
        else:
            print "'" + self.parentChain[-1] + "' ["+str(self.classifier)+"] at " + str(self.parentChain)

        for child in self.children.values():
            child.describe()
