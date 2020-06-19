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

@st.cache()
def get_unique_placements():
    with open('unique_placements.json') as json_file:
        unique_placements = json.load(json_file)
    print("Succesfully loaded dataset")
    return unique_placements

@st.cache()
def get_validation_set():
    with open('Starspace/datasets/new_validation_set.json') as json_file:
        validation_set = json.load(json_file)
    print("Succesfully loaded dataset")
    return validation_set

model_versions = {
    'v01':(11421,'v01',4),
    'v02':(8858,'v02',4),
    'v02_1':(8858,'v02',4),
    'v03':(8858,'v02',4),
    'v02_extended':(9088,'v02_extended',11)
}

model_selected = st.sidebar.selectbox(
    label = 'Which version of the model to display',
    options = list(model_versions),
    index = 4
)

version = model_selected
rec_step = model_versions[model_selected][0]

##############################################################################################
# PATHS
base_model_path = f'Starspace/models/tab_separated_descriptions_spaced_{version}'
labels_path = f'Starspace/datasets/labels_for_tab_separated_descriptions_spaced_{model_versions[model_selected][1]}.txt'
placement_embeddings_path = f'Starspace/models/placements_vectors_trainMode2_{version}.tsv'
basedoc_path = f'Starspace/datasets/complete_descritpions_spaced_{version}.txt'

##############################################################################################
# Useful dicts
#Build dict of ids to names, urls and types
def get_ids_to_placement_dict():
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

validation_set = get_validation_set()
unique_placements = get_unique_placements()
ids_to_placement_dict = get_ids_to_placement_dict()

def getNameFromURl(url):
    return unique_placements[url]['name']

def getNameFromId(placement_id):
    return ids_to_placement_dict[int(placement_id)]['name']

def getPlacementFromId(placement_id):
    return ids_to_placement_dict[int(placement_id)]

def clean_description(description):
    return format_train_file.clean_description(description)[0]


def get_sim(a,b):
    return dot(a, b)#/(norm(a)*norm(b))

@st.cache()
def init_placement_classifier(): 
    print("Starspace: init (placement classification model)")
    arg = sw.args()
    arg.dim = 300
    arg.trainMode = 2
    arg.label = "__placement__"
    arg.fileFormat = 'labelDoc'
    arg.basedoc = basedoc_path
    model = sw.starSpace(arg)
    print("Starspace: loading from saved model (placement classification model)")
    model.initFromSavedModel(base_model_path)
    model.loadBaseDocs()
    print("Placement classification model loaded succesfully")
    return model

# placement_classifier = init_placement_classifier()

@st.cache()
def get_embeddings_labels():
    labels = []
    with open(labels_path,'r') as textfile:
        for idx, line in tqdm(enumerate(textfile),desc="Getting labels"):
            line = line.split()[0]
            labels.append(int(line.replace('__placement__','')))
    return labels

@st.cache()
def get_placement_embeddings():
    if version == 'v01':
        vectors = np.genfromtxt(placement_embeddings_path,delimiter='\t',usecols=[a + 1 for a in range(300)])
    else:
        vectors = np.genfromtxt(placement_embeddings_path,delimiter='\t',usecols=range(300))
    return vectors

@st.cache()
def get_recommendations_from_file():
    default_jobs_recommendations = []
    default_jobs_recommendations_score = []
    with open(f'Starspace/datasets/default_recommendations_{version}.txt') as recommendations_file:
        lines = recommendations_file.readlines()
        for i in range(model_versions[model_selected][2]):
            default_jobs_recommendations.append([int(line.split()[0]) for line in lines[i*rec_step:rec_step*(i+1)-2]])
            default_jobs_recommendations_score.append([float(line.split()[1]) for line in lines[i*rec_step:rec_step*(i+1)-2]])
    return default_jobs_recommendations, default_jobs_recommendations_score

st.title("Placement recommender system (with trainMode 2)")
st.subheader("How does this work?")
st.write(f"""> * Train a model with StarSpace on top of a pretrained model on a larger corpora (FastText German model with 2M vocabulary).
> * Get the input Job title and predict the recommendations for it.
using model in {base_model_path}
""")

placements_ids = get_embeddings_labels()
placements_embeddings = get_placement_embeddings()
default_jobs_recommendations, default_jobs_recommendations_score = get_recommendations_from_file()
default_jobs = ['Bankkaufmann','Krankenschwester','Busfahrer','Elektrotechniker']
if model_versions[model_selected][2] == 11:
    default_jobs = ['Software-Entwickler','Elektrotechniker','Erzieher','Wirtschaftswissenschaftler','Bankkaufmann','Auszubildende','Busfahrer','Krankenpfleger','Architekten','Personalreferent','Vertriebsmitarbeiter']
placements_embeddings_df = pd.DataFrame(placements_embeddings)
# st.write(default_jobs_recommendations)
# model = init_placement_classifier()

# default_jobs_recommendations_labels = 

def return_job_title(a):
    return default_jobs[a]

selected_default = st.selectbox(
    label = 'Choose a jobtitle',
    options = range(model_versions[model_selected][2]),
    index = 2,
    format_func = return_job_title
)
# text1 = st.text_input('text1','Busfahrer')
# text2 = st.text_input('text2','busfahrer')

# w3 = (np.array(model.getDocVector(text1,' ')))
# w4 = (np.array(model.getDocVector(text2,' ')))
# sim = get_sim(w3,w4.T)
# st.write(f"The similarity between **{text1}** and **{text2}** is **{round(sim[0][0]*100,2)}%**")

# if not np.any(w3):
#     st.write(f"{text1} has no embedding, please try other word.")
# if not np.any(w4):
#     st.write(f"{text2} has no embedding, please try other word.")

st.subheader("Nearest neighbors")
st.write(f"The 20 nearest neighbors for **{default_jobs[selected_default]}** are the following:")

selection_info = [getPlacementFromId(placements_ids[rec_id]) for rec_id in default_jobs_recommendations[selected_default]]
domain_related_placement_info = []
for idx, placement in enumerate(selection_info):
    placement['score'] = str(round(default_jobs_recommendations_score[selected_default][idx]*100,2)) + "%"
    placement['rank'] = idx + 1
    domain_related_placement_info.append(placement)

domain_related_placement_table = pd.DataFrame(domain_related_placement_info)
domain_related_placement_table['description'] = domain_related_placement_table['description'].apply(lambda x: clean_description(x))
st.write(domain_related_placement_table[['name','type','description','score','rank']])

@st.cache()
def get_default_job_titles_recommendations():
    job_recs_dict = {}
    for idx, job in enumerate(default_jobs):
        rec_placements_info = {}
        default_jobs_labels = [placements_ids[placement] for placement in default_jobs_recommendations[idx]]
        for rank, rec_id in enumerate(default_jobs_labels):
            info = getPlacementFromId(rec_id).copy()
            info['score'] = str(round(default_jobs_recommendations_score[idx][rank]*100,2)) + "%"
            info['rank'] = rank + 1
            rec_placements_info[rec_id] = info
        job_recs_dict[job] = rec_placements_info
    return job_recs_dict

def get_personalisation_at_k(k):
    pers = round(rc.personalization([sublist[:k] for sublist in default_jobs_recommendations])*100,2)
    return pers

def get_intra_list_similarity_at_k(k,job):
    ils = round(rc._single_list_similarity(default_jobs_recommendations[job][:k],placements_embeddings_df)*100,2)
    return ils

@st.cache()
def get_personalisation(calculations):
    print("Calculating personalization at k...")
    personalization_record = []
    for i in tqdm(range(calculations),desc="Personalization"):
        pers = round(rc.personalization([sublist[:i+1] for sublist in default_jobs_recommendations])*100,2)
        personalization_record.append(pers)
    print("Done calculating personalization at k.")
    return personalization_record

@st.cache()
def get_intra_list_similarity(calculations):
    print("Calculating intralist similarity at k...")
    intra_list_similarities = []
    for job in range(len(default_jobs)):
        intra_list_similarity_record = []
        for i in tqdm(range(calculations-1),desc="Intra list similarity"):
            in_sim = round(rc._single_list_similarity(default_jobs_recommendations[job][:i+2],placements_embeddings_df)*100,2)
            intra_list_similarity_record.append(in_sim)
        intra_list_similarities.append(intra_list_similarity_record)
    print("Done calculating intralist similarity at k.")
    return intra_list_similarities

st.subheader("Metrics at K")
st.write("Pre-model loaded from " + base_model_path)
st.write("Placements embeddings grabbed from " + placement_embeddings_path)
st.write("Recommendations made for " + " ".join(default_jobs))

st.write("**Personalization**")
personalization_tuple = [(a,get_personalisation_at_k(a)) for a in [1,5,10,20,50]]
st.write("\t".join(f"Pers@{a[0]}: **{a[1]}%**" for a in personalization_tuple))
st.write("Copy: " + ",".join([f"{a[1]}%" for a in personalization_tuple]))

st.write("**Intralist similarity**")
jobs_ils = []
for job in range(len(default_jobs)):
    job_ils = [(a,get_intra_list_similarity_at_k(a,job)) for a in [2,5,10,20]]
    jobs_ils.append([a[1] for a in job_ils])
    st.write(default_jobs[job] + ": " + "\t".join(f"ILS@{a[0]}: **{a[1]}%**" for a in job_ils))

mean_ils = np.array(jobs_ils)
mean_ils = mean_ils.mean(axis=0)

st.write("Copy: " + ",".join([f"{round(a,2)}%" for a in mean_ils]))

st.write("**Mean rank**")
mean_ranks = []
for job in range(len(default_jobs)):
    default_jobs_labels = [placements_ids[placement] for placement in default_jobs_recommendations[job]]
    mean_rank = get_mean_rank(validation_set[default_jobs[job]],default_jobs_labels)
    mean_rank = round(mean_rank,2)
    st.write(default_jobs[job] + ":", mean_rank)
    mean_ranks.append(mean_rank)

st.write("Copy: " + "\n".join([f"{a}" for a in mean_ranks]))
st.write("Copy: " + str(round(sum(mean_ranks)/len(mean_ranks))))

default_jobs_recommendations_info = get_default_job_titles_recommendations()

st.write("**Hit@K**")
jobs_hits = []
for job in range(len(default_jobs)):
    default_jobs_labels = [placements_ids[placement] for placement in default_jobs_recommendations[job]]
    job_hits = [(a,get_hit_at_k(validation_set[default_jobs[job]],default_jobs_labels,a)[0]) for a in [1,5,10,20,50]]
    st.write(default_jobs[job] + ": " + "\t".join(f"Hit@{a[0]}: **{a[1]}**" for a in job_hits))
    jobs_hits.append([a[1] for a in job_hits])

mean_hits = np.array(jobs_hits)
mean_hits = mean_hits.mean(axis=0)

st.write("Copy: " + ",".join([f"{round(a,2)}" for a in mean_hits]))

st.subheader("Validation set for each job title")
for idx, job in enumerate(default_jobs):
    print(job)
    default_jobs_labels = [placements_ids[placement] for placement in default_jobs_recommendations[idx]]
    validation_ranks = get_validation_ranks(validation_set[job],default_jobs_labels)
    validation_placements = {}
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
    st.write(job_df[['name','type','description','score','rank']])

st.subheader(f"TrainMode = 2 ({version})")

if st.checkbox("Watch metrics for " + ", ".join(default_jobs)):
    st.subheader("Personalization")
    st.write("The personalization between receomendations is th percentage of unique placements recommended for each selected placement. That is, the bigger the personalization, the less placements they share at k recomendations.")

    print("Getting length")
    calculations = len(default_jobs_recommendations[0]) if st.checkbox('Calculate personalization up to full recommendation set.') else 1200
    print("Length done")

    # list1 = ["4","5","6","1","2","3","7","8","10","9","0"]
    # list2 = ["0","1","2","3","4","5","6","7","8","9","10"]
    # list3 = ["4","5","6","1","2","3","7","8","10","9","0"]

    print(len(default_jobs_recommendations))
    default_recommendations_df = pd.DataFrame(default_jobs_recommendations).T
    default_recommendations_df.columns = default_jobs

    personalization_record = get_personalisation(calculations)

    fig = px.line(x = [i+1 for i in range(calculations)],y=personalization_record, labels={'x':'Recommendations', 'y':'Personalization %'})
    st.plotly_chart(fig)

    intra_list_calculations = 200

    intra_list_similarities = get_intra_list_similarity(intra_list_calculations)
    fig = go.Figure()
    for job in range(len(default_jobs)):
        fig.add_trace(go.Scatter(x = [i + 1 for i in range(intra_list_calculations-1)],y=intra_list_similarities[job],name=default_jobs[job],mode='lines')) 
    st.plotly_chart(fig)

