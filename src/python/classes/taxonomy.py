class TaxonomyTree:

    def __init__(self, description):
        self.description = description
        self.root = TaxTreeNode('root',[])
        self.root.isRoot = True

    def add(self, nodeLabelChain):
        # pass to root node 
        self.root.add(TaxTreeNode(['root'] + nodeLabelChain))

    def describe(self):
        print "Taxonomy: " + self.description
        self.root.describe()

class TaxTreeNode:
    def __init__(self, parentChain):
        self.parentChain = parentChain          # list of parent node labels, in order from top to bottom 
        self.children = {}                      # dict of taxonomy tree node, key: last entry in parentChain
        self.isParent = False
        self.isRoot = False

    def add(self, node):
        self.isParent = True
        if node.parentChain is None or len(node.parentChain) <= len(self.parentChain):
            # fix parentChain and keep at this level
            node.parentChain = self.parentChain
            
        if cmp(node.parentChain, self.parentChain) == 0:
            # keep at this level
            self.children[node.parentChain[-1]] = node
        else:
            # pass down to next level
            dstNodeLabel = node.parentChain[len(self.parentChain)+1]
            if not self.children.has_key(dstNodeLabel):
                self.children[dstNodeLabel] = TaxTreeNode(self.parentChain + [dstNodeLabel])
            self.children[nxtParent].add(node)

    def describe(self):
        if self.isRoot:
            print "'" + self.parentChain[-1] + "' parent of " + str(len(self.children)) + " children"
        elif self.isParent:
            print "'" + self.parentChain[-1] + "' at " + str(self.parentChain) + " parent of " + str(len(self.children)) + " children"
        else:
            print "'" + self.parentChain[-1] + "' at " + str(self.parentChain)

        for child in self.children.values():
            child.describe()
