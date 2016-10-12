class TaxonomyTree:

    def __init__(self, description):
        self.description = description
        self.root = TaxTreeNode('root',[])
        self.root.isRoot = True

    def add(self, nodeLabelChain):
        # pass to root node 
        self.root.add(TaxTreeNode(nodeLabelChain.pop(), ['root'] + nodeLabelChain))

    def describe(self):
        print "Taxonomy: " + self.description
        self.root.describe()

class TaxTreeNode:
    def __init__(self, label, parentChain):
        self.label = label                      # taxonomy tree node label (category name), unique amongst siblings 
        self.parentChain = parentChain          # list of node labels, in order from top to bottom 
        self.children = {}                      # dict of taxonomy tree node, key:node.label 
        self.isParent = False
        self.isRoot = False

    def add(self, node):
        self.isParent = True
        if node.parentChain is None or len(node.parentChain) <= len(self.parentChain):
            # fix parentChain and keep at this level
            node.parentChain = self.parentChain + [self.label]
            
        if cmp(node.parentChain, self.parentChain + [self.label]) == 0:
            # keep at this level
            self.children[node.label] = node
        else:
            # pass down to next level
            dstNodeLabel = node.parentChain[len(self.parentChain)+1]
            if not self.children.has_key(dstNodeLabel):
                self.children[dstNodeLabel] = TaxTreeNode(dstNodeLabel, self.parentChain + [self.label])
            self.children[nxtParent].add(node)

    def describe(self):
        if self.isRoot:
            print "'" + self.label + "' parent of " + str(len(self.children)) + " children"
        elif self.isParent:
            print "'" + self.label + "' child of " + str(self.parentChain) + " and parent of " + str(len(self.children)) + " children"
        else:
            print "'" + self.label + "' child of " + str(self.parentChain)

        for child in self.children.values():
            child.describe()
