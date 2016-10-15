from classes import taxonomy as tax

t = tax.TaxonomyTree('Simple Multi-Level Taxonomy')
# Sample Category hierarchies to build Taxonomy
t.add(['plant-based-foods','fresh','greens'])
t.add(['plant-based-foods','frozen','roots'])
t.add(['plant-based-foods','frozen','fruit'])
t.add(['plant-based-foods','frozen','greens','salads'])
t.add(['plant-based-foods','frozen','greens','legumes'])
# Initialize classifiers
t.initClassifiers('sklearn.linear_model','LogisticRegression',{'C':1,'class_weight':'balanced'})
# Print some info
t.describe()
