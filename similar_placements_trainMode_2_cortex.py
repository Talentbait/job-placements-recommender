try:
    from macos import starwrap as sw
except:
    import starwrap as sw
import numpy as np
from numpy import dot
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
import streamlit as st
import pandas as pd
import json
import utils
from tqdm import tqdm
import plotly.express as px
import recommender_metrics as rc
import plotly.graph_objects as go
from operator import itemgetter
from format_data import format_train_file
from Starspace.validate_mannually_annotated_placements import get_mean_rank
from Starspace.validate_mannually_annotated_placements import get_validation_ranks
from Starspace.validate_mannually_annotated_placements import get_dropped_elements
from Starspace.validate_mannually_annotated_placements import get_hit_at_k
from copy import deepcopy
import model_holder as pexpect_test

# @st.cache()
# def get_unique_placements():
#     with open('unique_placements.json') as json_file:
#         unique_placements = json.load(json_file)
#     print("Succesfully loaded dataset")
#     return unique_placements

@st.cache()
def get_validation_set():
    with open('Starspace/datasets/new_validation_set.json') as json_file:
        validation_set = json.load(json_file)
    print("Succesfully loaded dataset")
    return validation_set
##############################################################################################
# PATHS
base_model_path = 'Starspace/models/trainMode0_v01'
placement_embeddings_path = 'Starspace/models/trainMode0_v01.tsv'

##############################################################################################
# Useful dicts
#Build dict of ids to names, urls and types
@st.cache()
def get_ids_to_placement_dict():
    unique_placements = {}
    with open('placement_pipeline_status_v01.json') as json_file:
        unique_placements = json.load(json_file)
    return unique_placements

validation_set = get_validation_set()
ids_to_placement_dict = get_ids_to_placement_dict()

# def getNameFromURl(url):
#     return unique_placements[url]['name']

def getNameFromId(placement_id):
    return ids_to_placement_dict[int(placement_id)]['name']

def getPlacementFromId(placement_id):
    return ids_to_placement_dict[int(placement_id)]

def clean_description(description):
    return format_train_file.clean_description(description)[0]


st.title("Placement recommender system (with trainMode 0)")
st.subheader("How does this work?")
st.write(f"""> * Train a model with StarSpace on top of a pretrained model on a larger corpora (FastText German model with 2M vocabulary).
> * Get the input Job title and predict the recommendations for it.
using model in {base_model_path}
""")

def get_sim(a,b):
    return dot(a, b)#/(norm(a)*norm(b))

def my_hash_func(a):
    return 2

@st.cache(hash_funcs={pexpect_test.PythonPredictor: my_hash_func})
def init_query_predict():
    return pexpect_test.PythonPredictor(config=0)

placement_classifier = init_query_predict()


@st.cache()
def get_placement_embeddings():
    with open(placement_embeddings_path) as input_file:
        vectors = {}
        for idx, line in tqdm(enumerate(input_file),desc="Getting vectors"):
            if idx > 2000000:
                line = line.split()
                vectors[int(line[0].replace("__placement__",""))] = np.fromstring(' '.join(line[1:]), sep=' ')
    return vectors

# placements_embeddings = get_placement_embeddings()




text1 = st.text_input('text1','Busfahrer')
text2 = st.text_input('text2','busfahrer')

st.subheader("Nearest neighbors")
st.write(f"The 20 nearest neighbors for **{text1}** are the following:")

related_predicts = 9000
print(f"Predicted {related_predicts} domain related placements.")

output = placement_classifier.predict({'jobTitle':text1})

domain_related_placement_table = pd.DataFrame(output['placements'])

st.write(domain_related_placement_table[['name','type','description','score']])

default_jobs = ['Software-Entwickler','Elektrotechniker','Erzieher','Wirtschaftswissenschaftler','Bankkaufmann','Auszubildende','Busfahrer','Krankenpfleger','Architekten','Personalreferent','Vertriebsmitarbeiter']


def get_similarity_for_default():
    jobs_recommendations = []
    for job in tqdm(default_jobs,desc="Getting recommendations"):
        try:
            output = placement_classifier.predict({'jobTitle':job})['placements']
        except:
            print(job)
        jobs_recommendations.append([a['id'] for a in output])
    return jobs_recommendations

default_jobs_recommendations = get_similarity_for_default()

def get_personalisation_at_k(k):
    pers = round(rc.personalization([sublist[:k] for sublist in default_jobs_recommendations])*100,2)
    return pers

@st.cache()
def get_personalisation(calculations):
    print("Calculating personalization at k...")
    personalization_record = []
    for i in tqdm(range(calculations),desc="Personalization"):
        pers = round(rc.personalization([sublist[:i+1] for sublist in default_jobs_recommendations])*100,2)
        personalization_record.append(pers)
    print("Done calculating personalization at k.")
    return personalization_record

st.subheader("Metrics at K")
st.write("Pre-model loaded from " + base_model_path)
st.write("Placements embeddings grabbed from " + placement_embeddings_path)
st.write("Recommendations made for " + " ".join(default_jobs))

st.write("**Personalization**")
personalization_tuple = [(a,get_personalisation_at_k(a)) for a in [1,5,10,20,50]]
st.write("\t".join(f"Pers@{a[0]}: **{a[1]}%**" for a in personalization_tuple))
st.write("Copy: " + ",".join([f"{a[1]}%" for a in personalization_tuple]))

st.write("**Mean rank**")
mean_ranks = []
for job in range(len(default_jobs)):
    default_jobs_labels = default_jobs_recommendations[job]
    mean_rank = get_mean_rank(validation_set[default_jobs[job]],default_jobs_labels)
    mean_rank = round(mean_rank,2)
    st.write(default_jobs[job] + ":", mean_rank)
    mean_ranks.append(mean_rank)

# st.write("Copy: " + "\n".join([f"{a}" for a in mean_ranks]))
st.write("Copy: " + str(round(sum(mean_ranks)/len(mean_ranks))))

st.write("**Hit@K**")
jobs_hits = []
for job in range(len(default_jobs)):
    default_jobs_labels = default_jobs_recommendations[job]
    job_hits = [(a,get_hit_at_k(validation_set[default_jobs[job]],default_jobs_labels,a)[0]) for a in [1,5,10,20,50]]
    st.write(default_jobs[job] + ": " + "\t".join(f"Hit@{a[0]}: **{a[1]}**" for a in job_hits))
    jobs_hits.append([a[1] for a in job_hits])

mean_hits = np.array(jobs_hits)
mean_hits = mean_hits.mean(axis=0)

st.write("Copy: " + ",".join([f"{round(a,2)}" for a in mean_hits]))

st.subheader("Validation set for each job title")
for idx, job in enumerate(default_jobs):
    validation_ranks = get_validation_ranks(validation_set[job],default_jobs_recommendations[idx])
    validation_placements = {}
    print(validation_ranks)
    st.write(f"Validation set for **{job}**.")
    for placement_id in validation_set[job]:
        if placement_id in validation_ranks:
            validation_placements[placement_id] = default_jobs_recommendations_info[job][placement_id]
            validation_placements[placement_id]['rank'] = int(validation_ranks[placement_id])
        else:
            validation_placements[placement_id] = ids_to_placement_dict[placement_id]
            print(placement_id)
    job_df = pd.DataFrame(validation_placements).T
    job_df = job_df.fillna(0).sort_values(by=['rank'])
    st.write(job_df[['name','description','rank','score','type']])