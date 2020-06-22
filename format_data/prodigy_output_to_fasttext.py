import json
import pandas as pd
from format_train_file import clean_description

def get_validation_set():
    with open('../Starspace/datasets/new_validation_set.json') as json_file:
        validation_set = json.load(json_file)
    print("Succesfully loaded dataset")
    return validation_set

def get_ids_to_placement_dict():
    unique_placements = {}
    with open('../unique_placements_updated.json') as json_file:
        unique_placements = json.load(json_file)
    ids_to_placement_dict = {}
    for k, v in unique_placements.items():
        if 'language' in v:
            if v['language'] == 'de':
                ids_to_placement_dict[v['id']] = {
                    'name':v['name'],
                    'type':v['type'],
                    'url':k,
                    'description':v['description'],
                    'language':v['language']
                }
    return ids_to_placement_dict

ids_to_placement_dict = get_ids_to_placement_dict()

validation_set = get_validation_set()

labeled_data = []
with open('placement-labels-final.json') as data_path:
    data = json.load(data_path)
    labeled_data = data['data']
all_keys = ids_to_placement_dict.keys()

for job in validation_set.keys():
    for example in labeled_data:
        example_id = example['meta']['id']
        if example_id in validation_set[job]:
            validation_set[job].remove(example_id)

job_to_label_dict = {
    'Software-Entwickler':'None',
    'Elektrotechniker':'Elektrotechniker',
    'Erzieher':'Erzieher',
    'Wirtschaftswissenschaftler':'Vertriebsmitarbeiter',
    'Bankkaufmann':'None',
    'Auszubildende':'None',
    'Busfahrer':'None',
    'Krankenpfleger':'Erzieher',
    'Architekten':'None',
    'Personalreferent':'None',
    'Vertriebsmitarbeiter':'Vertriebsmitarbeiter'
} 

# with open('../Starspace/datasets/test_unseen_examples_from_same_categories_with_none.txt','w') as output:
#     for job, val_set in validation_set.items():
#         if job in ['Elektrotechniker','Vertriebsmitarbeiter','Erzieher','Busfahrer']:
#             for example_id in val_set:
#                 placement = ids_to_placement_dict[example_id]
#                 output.write(clean_description(placement['description'])[0] + ' __jobtitle__' + job_to_label_dict[job] + "\n")

with open('../Starspace/datasets/placement_to_jobtitle_classifier_normalized_examples_trainfile.txt','w') as output:
    labeled_data_dict = {}
    for example in labeled_data:
        example_id = example['meta']['id']
        answer = example['answer']
        # print(answer)
        if answer == 'accept':
            tag = example['accept']
            if len(tag):
                label = tag[0]
                labeled_data_dict[example_id] = {
                    'name':ids_to_placement_dict[example_id]['name'],
                    'description':ids_to_placement_dict[example_id]['description'],
                    'label':label
                }
            else:
                labeled_data_dict[example_id] = {
                    'name':ids_to_placement_dict[example_id]['name'],
                    'description':ids_to_placement_dict[example_id]['description'],
                    'label':'None'
                }
                output.write(clean_description(example['text'])[0] + " __jobtitle__" + str(label) + "\n")
        elif answer == 'reject':
            # output.write(clean_description(example['text'])[0] + " __jobtitle__None\n")
            pass
    for example_id in validation_set['Busfahrer']:
        example = ids_to_placement_dict[example_id]
        output.write(clean_description(example['description'])[0] + " __jobtitle__None\n")

data = pd.DataFrame.from_dict(labeled_data_dict).T

print(data)
print(data['label'].value_counts())