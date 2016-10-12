from classes import taxonomy as tax
#from sklearn import linear_model

t = tax.TaxonomyTree('Simple Three Level Taxonomy')
# Sample Category hierarchies to build Taxonomy
t.add(['plant-based-foods','frozen','greens'])
t.add(['plant-based-foods','frozen','roots'])
# Initialize classifiers
t.initClassifiers('sklearn.linear_model','LogisticRegression',{'C':1,'class_weight':'balanced'})
t.describe()
