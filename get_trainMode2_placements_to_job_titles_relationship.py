import streamlit as st
import numpy as np
import pandas as pd
import json
from tqdm import tqdm

st.title('Format the predictions from trainMode 2')

##############################################################################################
# Useful dicts
#Build dict of ids to names, urls and types

emoji_dict = {
    'Software-Entwickler': '🧑‍💻',
    'Elektrotechniker': '🔌',
    'Erzieher': '👶',
    'Wirtschaftswissenschaftler': '💱',
    'Bankkaufmann': '🏦',
    'Auszubildende': '👩‍🎓',
    'Busfahrer': '🚌',
    'Krankenpfleger': '👩‍⚕️',
    'Architekten': '📐',
    'Personalreferent': '👩‍💼',
    'Vertriebsmitarbeiter': '💵'
}

model_versions = {
    'v01':(11421,'v01',4),
    'v02':(8858,'v02',4),
    'v02_1':(8858,'v02',4),
    'v03':(8858,'v02',4),
    'v02_extended':(9088,'v02_extended',11)
}

version = 'v02_extended'
rec_step = model_versions[version][0]

##############################################################################################
# PATHS
labels_path = f'Starspace/datasets/labels_for_tab_separated_descriptions_spaced_{model_versions[version][1]}.txt'

default_jobs = ['Software-Entwickler','Elektrotechniker','Erzieher','Wirtschaftswissenschaftler','Bankkaufmann','Auszubildende','Busfahrer','Krankenpfleger','Architekten','Personalreferent','Vertriebsmitarbeiter']

@st.cache(allow_output_mutation=True)
def get_ids_to_placements():
    unique_placements = []
    with open('unique_placements.json') as json_file:
        unique_placements = json.load(json_file)
    print("Succesfully loaded dataset")

    placements_ids = []
    with open(labels_path,'r') as textfile:
        for idx, line in tqdm(enumerate(textfile),desc="Getting labels"):
                line = line.split()[0]
                placements_ids.append(int(line.replace('__placement__','')))

    ids_to_placement_dict = {}
    for k, v in unique_placements.items():
        if v['id'] in placements_ids:
            ids_to_placement_dict[v['id']] = {
                'name':v['name'],
                'type':v['type'],
                'url':k,
                'description':v['description'],
                'language':v['language']
            }

    def getPlacementFromId(placement_id):
        return ids_to_placement_dict[int(placement_id)]

    default_jobs_recommendations = []
    default_jobs_recommendations_score = []
    with open(f'Starspace/datasets/default_recommendations_{version}.txt') as recommendations_file:
        lines = recommendations_file.readlines()
        for i in range(model_versions[version][2]):
            default_jobs_recommendations.append([int(line.split()[0]) for line in lines[i*rec_step:rec_step*(i+1)-2]])
            default_jobs_recommendations_score.append([float(line.split()[1]) for line in lines[i*rec_step:rec_step*(i+1)-2]])

    job_recs_dict = {}
    for idx, job in enumerate(default_jobs):
        rec_placements_info = {}
        default_jobs_labels = [placements_ids[placement] for placement in default_jobs_recommendations[idx]]
        for rank, rec_id in enumerate(default_jobs_labels):
            info = getPlacementFromId(rec_id).copy()
            info['score'] = round(default_jobs_recommendations_score[idx][rank],4)
            info['rank'] = rank + 1
            rec_placements_info[rec_id] = info
        job_recs_dict[job] = rec_placements_info

    for idx, job in enumerate(default_jobs):
        default_jobs_labels = [placements_ids[placement] for placement in default_jobs_recommendations[idx]]
        for placement_id in ids_to_placement_dict.keys():
            ids_to_placement_dict[placement_id][job] = job_recs_dict[job][placement_id]['score']

    return ids_to_placement_dict

ids_to_placement_dict = get_ids_to_placements()

job_df = pd.DataFrame(ids_to_placement_dict).T
job_df = job_df.fillna(0)
st.write(job_df)
# job_df.to_csv('placements_and_similarity_score_with_job_titles.csv')

# with open('prodigy_validation_input_file_with_options.jl','w') as output_file:
for placement_id, placement in ids_to_placement_dict.items():
    jobs = ('Software-Entwickler','Elektrotechniker','Erzieher','Wirtschaftswissenschaftler','Bankkaufmann','Auszubildende','Busfahrer','Krankenpfleger','Architekten','Personalreferent','Vertriebsmitarbeiter')
    sorted_jobs = {k:placement[k] for k in jobs if k in placement}
    placement_sorted_keys = sorted(sorted_jobs, key=sorted_jobs.get, reverse=True)
    placement_dict = {
        'text':placement['description'],
        'meta': placement,
        'options':[{'id':r,'text':emoji_dict[r] + r} for r in placement_sorted_keys[:3]]
    }
    placement_dict['meta']['id'] = placement_id
        # for key in ['description','language','type']:
        #     del placement_dict['meta'][key]
        # output_file.write(json.dumps(placement_dict) + '\n')
