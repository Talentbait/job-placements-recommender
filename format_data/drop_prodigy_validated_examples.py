import json
import io

def get_validation_set():
    with open('../Starspace/datasets/new_validation_set.json') as json_file:
        validation_set = json.load(json_file)
    print("Succesfully loaded dataset")
    return validation_set 

def get_ids_to_placement_dict():
    unique_placements = {}
    with open('../unique_placements_with_german_descriptions.json') as json_file:
        unique_placements = json.load(json_file)
    return unique_placements

#Keys are placements ids and we have the info from that placement
ids_to_placement_dict = get_ids_to_placement_dict()

#Keys are jobtitles and we have array of ids
validation_set = get_validation_set()

#Prodigy validated data 
labeled_data = []
with open('placement-labels-final.json') as data_path:
    data = json.load(data_path)
    labeled_data = data['data']


for job_title,placement_ids in validation_set.items():
    for placement_id in placement_ids:
        placement_id = str(placement_id)
        ids_to_placement_dict[placement_id]['classified'] = True
        ids_to_placement_dict[placement_id]['label'] = job_title

for prodigy_example in labeled_data:
    current_placement = ids_to_placement_dict[str(prodigy_example['meta']['id'])] 

    if 'accept' in prodigy_example and len(prodigy_example['accept']):
        current_job_title = prodigy_example['accept'][0]
        current_job_title = prodigy_example['accept'][0]
        current_placement['classified'] = True
        current_placement['prodigy'] = True
        current_placement['v00'] = {
            current_job_title: prodigy_example['meta']['score']
        }
    else:
        current_placement['classified'] = False
        current_placement['prodigy'] = True

    ids_to_placement_dict[str(prodigy_example['meta']['id'])] = current_placement

with io.open('../placement_pipeline_status.json','w',encoding="UTF-8") as file_path:
    json.dump(ids_to_placement_dict,file_path,indent=2,ensure_ascii=False)
