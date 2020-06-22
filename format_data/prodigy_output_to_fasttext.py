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

used_data = {}
with open('placement_to_jobtitle_classifier_normalized_examples_trainfile_2.json') as data_path:
    data = json.load(data_path)
    used_data = data

# with open('../unique_placements_dataset.json','w') as file_path:
#     json.dump(ids_to_placement_dict,file_path,indent=4)

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

# with open('../Starspace/datasets/placement_to_jobtitle_classifier_normalized_examples_testfile_2.txt','w') as output:
#     for job, val_set in validation_set.items():
#         for example_id in val_set:
#             if example_id not in used_data.keys():
#                 placement = ids_to_placement_dict[example_id]
#                 output.write(clean_description(placement['description'])[0] + ' __jobtitle__' + job_to_label_dict[job] + "\n")

# with open('../Starspace/datasets/placement_to_jobtitle_classifier_normalized_examples_trainfile_2.txt','w') as output:
#     labeled_data_dict = {}
#     for example in labeled_data:
#         example_id = example['meta']['id']
#         answer = example['answer']
#         # print(answer)
#         if answer == 'accept':
#             tag = example['accept']
#             if len(tag):
#                 label = tag[0]
#                 labeled_data_dict[example_id] = {
#                     'name':ids_to_placement_dict[example_id]['name'],
#                     'description':ids_to_placement_dict[example_id]['description'],
#                     'label':label
#                 }
#                 output.write(clean_description(example['text'])[0] + " __jobtitle__" + str(label) + "\n")
#         elif answer == 'reject':
#             # output.write(clean_description(example['text'])[0] + " __jobtitle__None\n")
#             pass
#     for job in [k for k, v in job_to_label_dict.items() if v == 'None']:
#         for example_id in validation_set[job][:4]:
#             example = ids_to_placement_dict[example_id]
#             output.write(clean_description(example['description'])[0] + " __jobtitle__None\n")
#             labeled_data_dict[example_id] = {
#                 'name':ids_to_placement_dict[example_id]['name'],
#                 'description':ids_to_placement_dict[example_id]['description'],
#                 'label':'None'
#             }

# data = pd.DataFrame.from_dict(labeled_data_dict).T

# print(data)
# print(data['label'].value_counts())

# data.to_json("placement_to_jobtitle_classifier_normalized_examples_trainfile_2.json",indent=2,orient='index')