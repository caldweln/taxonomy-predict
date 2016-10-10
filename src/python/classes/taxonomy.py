class TaxonomyTree:

    def __init__(self, description):
        self.description = description
        self.root = TaxTreeNode('root',[])
        self.root.isRoot = True

    def add(self, nodeLabel, parents):
        self.root.add(TaxTreeNode(nodeLabel, parents))

    def describe(self):
        print "Taxonomy: " + self.description
        self.root.describe()

class TaxTreeNode:
    def __init__(self, label, parents):
        self.label = label
        self.parents = parents
        self.children = []
        self.isParent = False
        self.isRoot = False

    def add(self, node):
        if node.parents is None or len(node.parents) == 0:
            raise ValueError('node has no parent!')
        elif self.label in node.parents: #quietly exit otherwise
            self.children.append(node)
            self.isParent = True
        else:
            for child in self.children:
                child.add(node)

    def describe(self):
        if self.isRoot:
            print "'" + self.label + "' parent of " + str(len(self.children)) + " children"
        elif self.isParent:
            print "'" + self.label + "' child of " + str(self.parents) + " and parent of " + str(len(self.children)) + " children"
        else:
            print "'" + self.label + "' child of " + str(self.parents)

        for child in self.children:
            child.describe()
