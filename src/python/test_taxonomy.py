from classes import taxonomy as tax
t = tax.TaxonomyTree('Simple Three Level Taxonomy')
t.add(['plant-based-foods','frozen','greens'])
t.add(['plant-based-foods','frozen','roots'])
t.describe()
