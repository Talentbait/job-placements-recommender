import pandas as pd
import json

new_data = pd.read_csv('validation_set.csv')
new_data.fillna(0,inplace=True)

last_used_id = 156457

new_placements = {}
validation_set_dict = {}
for idx, row in new_data.drop_duplicates('name').iterrows():
    if row['id'] == 0:
        last_used_id = last_used_id + 1
        new_placements[row['url']] = {
            'name': row['name'],
            'id': last_used_id,
            'type': row['type'],
            'description': row['description'],
            'language': row['language']
        }
        row['id'] = last_used_id
    if row['job_title'] in validation_set_dict:
        validation_set_dict[row['job_title']].append(int(row['id']))
    else:
        validation_set_dict[row['job_title']] = [int(row['id'])]
# with open('new_placements.json','w') as file_path:
#     json.dump(new_placements,file_path,indent=4)

with open('new_validation_set.json','w') as file_path:
    json.dump(validation_set_dict,file_path,indent=4)