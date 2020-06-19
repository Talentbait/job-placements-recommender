import json
import pandas as pd

unique_placements = {}
with open('unique_placements.json') as json_file:
    unique_placements = json.load(json_file)

new_descriptions = pd.read_csv('placements_with_new_description.csv')

print(new_descriptions.head())

with open('unique_placements_updated.json','w') as output:
    for idx, placement in new_descriptions.iterrows():
        unique_placements[placement['url']]['description'] = placement['new_description']
    json.dump(unique_placements,output,indent=4)