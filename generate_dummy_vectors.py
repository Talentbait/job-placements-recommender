import numpy as np
import json

# unique_placements = {}
# with open('unique_placements.json') as json_file:
#     unique_placements = json.load(json_file)
#     print("Succesfully loaded dataset")

# ids_to_placement_dict = {}
# placements_labels = []
# for k, v in unique_placements.items():
#     if 'language' in v:
#         if v['language'] == 'de':
#             ids_to_placement_dict[v['id']] = {
#                 'name':v['name'],
#                 'type':v['type'],
#                 'url':k,
#                 'description':v['description'],
#                 'language':v['language']
#             }
#             placements_labels.append('__placement__' + str(v['id']))

with open('Starspace/dummy_vectors.tsv', 'w') as output_file:
    for label in ['__jobtitle__Erzieher','__jobtitle__Vertriebsmitarbeiter','__jobtitle__None','__jobtitle__Elektrotechniker']:
        dummy = np.random.normal(loc=0, scale=0.001, size=300)
        output_file.write(label.replace("\n","") + '\t' + "\t".join(["%.6f" % a for a in dummy]) + '\n')
