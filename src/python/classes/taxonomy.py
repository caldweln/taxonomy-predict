class TaxonomyTree:

    def __init__(self, description):
        self.description = description
        self.root = TaxTreeNode('root',[])
        self.root.isRoot = True

    def add(self, nodeLabelChain):
        self.root.add(TaxTreeNode(nodeLabelChain.pop(), ['root'] + nodeLabelChain))

    def describe(self):
        print "Taxonomy: " + self.description
        self.root.describe()

class TaxTreeNode:
    def __init__(self, label, parentChain):
        self.label = label
        self.parentChain = parentChain
        self.children = {}
        self.isParent = False
        self.isRoot = False
        self.distance = 0

    def add(self, node):
        self.isParent = True

        if node.parentChain is None or len(node.parentChain) == 0 or node.parentChain[-1] == self.label:
            self.children[node.label] = node
        else:
            nxtParent = node.descend()

            if not self.children.has_key(nxtParent):
                self.children[nxtParent] = TaxTreeNode(nxtParent, self.parentChain + [self.label])

            self.children[nxtParent].add(node)

    def descend(self):
        self.distance += 1
        nxtParent = self.parentChain[self.distance]
        return nxtParent

    def describe(self):
        if self.isRoot:
            print "'" + self.label + "' parent of " + str(len(self.children)) + " children"
        elif self.isParent:
            print "'" + self.label + "' child of " + str(self.parentChain) + " and parent of " + str(len(self.children)) + " children"
        else:
            print "'" + self.label + "' child of " + str(self.parentChain)

        for child in self.children.values():
            child.describe()
