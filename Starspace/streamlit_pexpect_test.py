import pexpect_test
import pandas as pd
import streamlit as st
import re
import json
from tqdm import tqdm

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

st.title("Test of pexpect")

@st.cache(allow_output_mutation=True)
def get_ids_to_placement_dict():
    unique_placements = {}
    with open('../unique_placements_updated.json') as json_file:
        unique_placements = json.load(json_file)
    ids_to_placement_dict = {}
    for k, v in tqdm(unique_placements.items(),desc="Get ids to placements dict"):
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

def my_hash_func(a):
    return 2

@st.cache(hash_funcs={pexpect_test.PythonPredictor: my_hash_func})
def init_query_predict():
    return pexpect_test.PythonPredictor(config=0)

predictor = init_query_predict()

job_input = st.text_input('Enter a job title:','Busfahrer')

def get_predictions(job):
    output = predictor.predict(job)

    placements = [a for a in output.split('\n') if len(a) > 5]

    placements_info = {}
    for placement in placements:
        prob = float(re.search(r"[0-9]+\[(.*?)\].*",placement).group(1))
        placement_id = int(re.search(".*__placement__([0-9]+).*",placement).group(1))
        # print(placement_id)
        placements_info[placement_id] = ids_to_placement_dict[placement_id]
        placements_info[placement_id]['score'] = prob
    
    return placements_info

recomendations_for_selection = get_predictions(job=job_input)

recs = pd.DataFrame.from_dict(recomendations_for_selection).T

st.write(recs[['name','type','description','score']])

# included_examples = {}
# for job in ['Vertriebsmitarbeiter','Erzieher','Elektrotechniker']:
#     current_job_examples = get_predictions(job)
#     for placement_id, placement in current_job_examples.items():
#         if placement_id not in included_examples:
#             # jobs = ('Software-Entwickler','Elektrotechniker','Erzieher','Wirtschaftswissenschaftler','Bankkaufmann','Auszubildende','Busfahrer','Krankenpfleger','Architekten','Personalreferent','Vertriebsmitarbeiter')
#             # sorted_jobs = {k:placement[k] for k in jobs if k in placement}
#             # placement_sorted_keys = sorted(sorted_jobs, key=sorted_jobs.get, reverse=True)
#             placement_dict = {
#                 'label':placement['name'],
#                 'text':placement['description'],
#                 'meta': placement,
#                 'options':[{'id':r,'text':emoji_dict[r] + r} for r in ['Vertriebsmitarbeiter','Erzieher','Elektrotechniker']]
#             }
#             placement_dict['meta']['id'] = placement_id
#             placement_dict['meta']['job_title'] = job
#             for key in ['description','language','type','name']:
#                 del placement_dict['meta'][key]
#             included_examples[placement_id] = placement_dict
#         else:
#             if placement['score'] > included_examples[placement_id]['meta']['score']:
#                 included_examples[placement_id]['meta']['job_title'] = job
#                 included_examples[placement_id]['meta']['score'] = placement['score']
# #                 output_file.write(json.dumps(placement_dict) + '\n')
# #                 included_examples.append(placement_id)

# with open('prodigy_validation_input_file_700.jl','w') as output_file:
#     for placement_id, placement in included_examples.items():
#         output_file.write(json.dumps(placement) + '\n')

# print(len(list(included_examples.keys())))