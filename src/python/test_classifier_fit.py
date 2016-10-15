import pandas as pd
from classes import taxonomy as tax

t = tax.TaxonomyTree('Simple Multi-Level Taxonomy')
# Sample Category hierarchies to build Taxonomy
categories = {'categories_hierarchy':\
[['plant-based-foods','fresh','greens'],\
['plant-based-foods','frozen','roots'],\
['plant-based-foods','frozen','fruit'],\
['plant-based-foods','frozen','greens','salads'],\
['plant-based-foods','frozen','greens','legumes']]}
# Load sample data into DataFrame
product_df = pd.DataFrame(categories)
# Load taxonomy using DataFrame iterator
res = [t.add(row['categories_hierarchy']) for ix,row in product_df.iterrows()]
# Initialize classifiers
t.initClassifiers('sklearn.linear_model','LogisticRegression',{'C':1,'class_weight':'balanced'})
# Print some info
t.describe()
print "--------------"
# Get location/classifier pairs from taxonomy
locationClassifiers = t.getLocationClassifiers()
# For each pair, using location filter relevant product data for classifier training
for locClsfr in locationClassifiers:
    location = locClsfr[0]
    classifier = locClsfr[1]
    rowixs = []
    for ix,row in product_df.iterrows():
        if len(row['categories_hierarchy']) > len(location) and cmp(row['categories_hierarchy'][:len(location)],location)==0:
            rowixs += [ix]
    filtered_products_df = product_df[product_df.index.isin(rowixs)]
    print "\nTraining ["+str(location)+"] classifier on the following samples"
    print(filtered_products_df)
print "--------------"
