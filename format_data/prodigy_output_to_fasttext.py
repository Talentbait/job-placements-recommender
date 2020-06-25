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
    with open('../placement_pipeline_status_v01.json') as json_file:
        unique_placements = json.load(json_file)
    # ids_to_placement_dict = {}
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
    return unique_placements

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
with open('placement_to_jobtitle_classifier_normalized_examples_trainfile_4.json') as data_path:
    data = json.load(data_path)
    used_data = data

# with open('../unique_placements_dataset.json','w') as file_path:
#     json.dump(ids_to_placement_dict,file_path,indent=4)

labels_dict = {
    'Erzieher':'__label__Erzieher __label__Erzieherinnen __label__Kita-Erzieher __label__Pädagogen __label__Erziehern __label__Kindererzieher __label__Kinderpfleger __label__Erzieherin __label__Kindergärtner __label__Heilerzieher __label__Kindergartenpädagogen __label__Lehrer __label__Arbeitserzieher __label__ErzieherInnen __label__Kita-Erzieherinnen __label__Sozialpädagogen __label__Horterzieher __label__Erziehers __label__Sozialarbeiter __label__Kindergärtnerinnen',
    'Elektrotechniker':'__label__Elektrotechniker  __label__Elektroingenieur __label__Nachrichtentechniker __label__Elektrotechnik-Ingenieur __label__Elektriker __label__Elektrotechnikingenieur __label__Energietechniker __label__Elektromechaniker __label__Elektrotechnikerin __label__Elektro-Ingenieur __label__Informationstechniker __label__Ingenieur __label__Elektrotechnik __label__Elektrotechnikermeister __label__Elektrobetriebstechniker __label__Elektronikingenieur __label__Elektrotechnikern __label__Elektroniker __label__elektrotechniker __label__Elektroingenieure',
    'Vertriebsmitarbeiter':'__label__Vertriebsmitarbeiter __label__Außendienstmitarbeiter __label__Vetriebsmitarbeiter __label__Vertriebsmitarbeitern __label__Vertriebsmitarbeiterin __label__Vertriebsingenieure __label__Vertriebsprofi __label__Vertriebsmitarbeiters __label__Außendienst __label__Vertriebler __label__Vertriebsspezialisten __label__Kundenbetreuer __label__Vertriebsingenieur __label__Vertriebsleiter __label__Außendienst-Mitarbeiter __label__Anwendungstechniker __label__Vertriebsmanager __label__Vertriebsassistenten __label__Kundenberater __label__Sales-Mitarbeiter',
    'None':'__label__None0 __label__None1 __label__None2 __label__None3 __label__None4 __label__None5 __label__None6 __label__None7 __label__None8 __label__None9 __label__None10 __label__None11 __label__None12 __label__None13 __label__None14 __label__None15 __label__None16 __label__None17 __label__None18 __label__None19'
}

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

# labeled_data_dict = {}
# with open('../Starspace/datasets/testfile_jobtitle_classifier_v01_5.txt','w') as output:
#     for job, val_set in validation_set.items():
#         for example_id in val_set:
#             if example_id not in used_data.keys() and job not in ['Bankkaufmann','Personalreferent','Software-Entwickler','Auszubildende']:
#                 placement = ids_to_placement_dict[example_id]
#                 output.write(clean_description(placement['description'])[0] + ' __jobtitle__' + job_to_label_dict[job] + "\n")
#                 labeled_data_dict[example_id] = {
#                     'name':ids_to_placement_dict[example_id]['name'],
#                     'description':ids_to_placement_dict[example_id]['description'],
#                     'label':job_to_label_dict[job]
#                 }

# with open('../Starspace/datasets/trainfile_jobtitle_classifier_v02.txt','w') as output:
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
#                 output.write(clean_description(example['text'])[0] + " __jobtitle__" + label + "\n")
#     for job in [k for k, v in job_to_label_dict.items() if v == 'None']:
#         for example_id in validation_set[job][:4]:
#             example = ids_to_placement_dict[example_id]
#             output.write(clean_description(example['description'])[0] + " __jobtitle__None\n")
#             labeled_data_dict[example_id] = {
#                 'name':ids_to_placement_dict[example_id]['name'],
#                 'description':ids_to_placement_dict[example_id]['description'],
#                 'label':'None'
#             }

with open('../Starspace/datasets/trainfile_useful_placements_classifier.txt','w') as output:
    labeled_data_dict = {}
    descriptions_examples = {}
    for placement_id, placement_info in ids_to_placement_dict.items():
        if 'classified' in placement_info:
            description = placement_info['description']
            labeled_data_dict[placement_id] = {
                'name':placement_info['name'],
                'description':description
            }
            if 'label' in placement_info:
                label = placement_info['label']
                labeled_data_dict[placement_id]['label'] = "useful"
                output.write(clean_description(description)[0] + " __label__useful\n")
            else:
                labeled_data_dict[placement_id]['label'] = "dump"
                output.write(clean_description(description)[0] + " __label__dump\n")
    # for job in [k for k, v in job_to_label_dict.items() if v == 'None']:
    #     for placement_info_id in validation_set[job][:4]:
    #         placement_info = placement_info
    #         output.write(clean_description(placement_info['description'])[0] + " __jobtitle__None\n")
    #         labeled_data_dict[example_id] = {
    #             'name':placement_info['name'],
    #             'description':placement_info['description'],
    #             'label':'None'
    #         }
    for job_title, descriptions_list in descriptions_examples.items():
        print(job_title)
        output.write("\t".join([clean_description(a)[0].replace("\t"," ") for a in descriptions_list]) + "\n")

        
data = pd.DataFrame.from_dict(labeled_data_dict).T

print(data)
print(data['label'].value_counts())

# data.to_json("placement_to_jobtitle_classifier_normalized_examples_trainfile_4.json",indent=2,orient='index')