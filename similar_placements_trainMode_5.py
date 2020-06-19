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
from Starspace.validate_mannually_annotated_placements import get_mean_rank
from Starspace.validate_mannually_annotated_placements import get_validation_ranks

@st.cache()
def get_unique_placements():
    with open('unique_placements.json') as json_file:
        unique_placements = json.load(json_file)
    print("Succesfully loaded dataset")
    return unique_placements

@st.cache()
def get_validation_set():
    with open('Starspace/datasets/validation_set.json') as json_file:
        validation_set = json.load(json_file)
    print("Succesfully loaded dataset")
    return validation_set

##############################################################################################
# PATHS
base_model_path = 'collaborative_filtering/models/german_tabbed_vectors.tsv'
placement_embeddings_path = 'collaborative_filtering/models/labeled_placement_embeddings_word_count_6_190.tsv'

##############################################################################################
# Useful dicts
#Build dict of ids to names, urls and types
def get_ids_to_placement_dict():
    ids_to_placement_dict = {}
    for k, v in tqdm(unique_placements.items(),desc="Get ids to placements dict"):
        ids_to_placement_dict[v['id']] = {
            'name':v['name'],
            'type':v['type'],
            'url':k,
            'description':v['description'],
            'language':'' if 'language' not in v else v['language']
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
    return utils.clean_description(description)


st.title("Placement recommender system (with trainMode 5)")
st.subheader("How does this work?")
st.write("""> * Have an embedding model to embed descriptions and input (FastText German model with 2M vocabulary).
> * Get the input Job title and embed it with the same model.
> * Get the k nearest neighbors for the input’s embedding.
**Additions:**
* Removed placements with less than 6 words (5%) and with more than 190 (10%) (text_classification/plain_text_word_count_6_190.txt)
""")

def get_sim(a,b):
    return dot(a, b)#/(norm(a)*norm(b))

@st.cache()
def init_placement_classifier(): 
    print("Starspace: init (placement classification model)")
    arg = sw.args()
    arg.label = '__placement__'
    arg.dim = 300
    model = sw.starSpace(arg)
    print("Starspace: loading from saved model (placement classification model)")
    # model.initFromSavedModel('collaborative_filtering/models/german_vectors_and_placement_embeddings.tsv')
    model.initFromTsv(base_model_path)
    print("Placement classification model loaded succesfully")
    return model

@st.cache()
def get_embeddings_labels():
    labels = []
    with open('collaborative_filtering/models/placement_labels_word_count_6_190.txt','r') as textfile:
        for line in textfile:
            placement_id = int(line.replace('__placement__','').replace('\n',''))
            labels.append(placement_id)
    return labels

@st.cache()
def get_placement_embeddings():
    vectors = np.genfromtxt(placement_embeddings_path,delimiter='\t',usecols=[a + 1 for a in range(300)])
    return vectors

placements_ids = get_embeddings_labels()
placements_embeddings = get_placement_embeddings()

model = init_placement_classifier()

text1 = st.text_input('text1','Busfahrer')
text2 = st.text_input('text2','busfahrer')

w3 = (np.array(model.getDocVector(text1,' ')))
w4 = (np.array(model.getDocVector(text2, ' ')))
sim = get_sim(w3,w4.T)
st.write(f"The similarity between **{text1}** and **{text2}** is **{round(sim[0][0]*100,2)}%**")

if not np.any(w3):
    st.write(f"{text1} has no embedding, please try other word.")
if not np.any(w4):
    st.write(f"{text2} has no embedding, please try other word.")

st.subheader("Nearest neighbors")
st.write(f"The 20 nearest neighbors for **{clean_description(text1)}** are the following:")

@st.cache()
def get_similarity_for_selection(selection):
    print('calculating similarities.')
    similarity_scores = cos_sim(selection,placements_embeddings)[0]
    print('done calculating similarities')
    return similarity_scores

similarity_scores = get_similarity_for_selection(w3)

sorted_similarity_scores = np.argsort(similarity_scores)[::-1]
predicted_nn_names = [getNameFromId(placements_ids[a]) for a in sorted_similarity_scores]
predicted_nn_info = [getPlacementFromId(placements_ids[a]) for a in sorted_similarity_scores]
predicted_nn_info_table = pd.DataFrame(predicted_nn_info)
# tqdm.pandas(desc="cleaning descriptions")
print("Everything sorted")
predicted_nn_info_table['cleaned_description'] = predicted_nn_info_table['description']#.progress_apply(lambda x: clean_description(x))
predicted_nn_info_table['score'] = [similarity_scores[a] for a in sorted_similarity_scores]
st.write(predicted_nn_info_table[['name','type','cleaned_description','score','description']])


@st.cache()
def get_similarity_for_default():
    jobs_embeddings = default_jobs_vectors
    print("Calculating similarities")
    job_recs_dict = {}
    jobs_recommendations = []
    for job, job_embedding in tqdm(enumerate(jobs_embeddings),desc="Getting recommendations"):
        similarity_scores = cos_sim(job_embedding,placements_embeddings)[0]
        sorted_similarity_scores = np.argsort(similarity_scores)[::-1]
        jobs_recommendations.append(sorted_similarity_scores)
        default_jobs_labels = [placements_ids[placement] for placement in sorted_similarity_scores]
        scores_dict = [similarity_scores[a] for a in sorted_similarity_scores]
        rec_placements_info = {}
        for rank, rec_id in enumerate(default_jobs_labels):
            info = getPlacementFromId(rec_id).copy()
            info['score'] = str(round(scores_dict[rank]*100,2)) + "%"
            info['rank'] = rank + 1
            rec_placements_info[rec_id] = info
        job_recs_dict[default_jobs[job]] = rec_placements_info
    return jobs_recommendations, job_recs_dict

default_jobs = ['Bankkaufmann','Krankenschwester','Busfahrer','Elektrotechniker']
placements_embeddings_df = pd.DataFrame(placements_embeddings)
default_jobs_vectors = []
for job in default_jobs:
    print("Getting embeddings for " + job)
    job_embedding = (np.array(model.getDocVector(job,' ')))
    default_jobs_vectors.append(job_embedding)

print("This works")
default_jobs_recommendations, default_jobs_recommendations_info = get_similarity_for_default()

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
st.write("\t".join(f"Pers@{a}: **{get_personalisation_at_k(a)}%**" for a in [1,5,10,20,50]))

st.write("**Intralist similarity**")
for job in range(len(default_jobs)):
    st.write(default_jobs[job] + ": " + "\t".join(f"ILS@{a}: **{get_intra_list_similarity_at_k(a,job)}%**" for a in [2,5,10,20]))

st.write("**Mean rank**")
for job in range(len(default_jobs)):
    default_jobs_labels = [placements_ids[placement] for placement in default_jobs_recommendations[job]]
    mean_rank = get_mean_rank(validation_set[default_jobs[job]]['placements'],default_jobs_recommendations[job])
    st.write(default_jobs[job] + ":", round(mean_rank,2))

st.subheader("Validation set for each job title")
for idx, job in enumerate(default_jobs):
    job_validation_set = validation_set[job]['placements']
    default_jobs_labels = [placements_ids[placement] for placement in default_jobs_recommendations[idx]]
    validation_ranks = get_validation_ranks(job_validation_set,default_jobs_labels)
    validation_placements = {}
    st.write(f"Validation set for **{job}**.")
    for placement_id in job_validation_set:
        if placement_id in default_jobs_recommendations_info[job]:
            validation_placements[placement_id] = default_jobs_recommendations_info[job][placement_id]
            validation_placements[placement_id]['rank'] = validation_ranks[placement_id]
        else:
            validation_placements[placement_id] = getPlacementFromId(placement_id)
    job_df = pd.DataFrame(validation_placements).T
    job_df = job_df.fillna(0)
    st.write(job_df[['name','type','description','score','rank']])

st.subheader("TrainMode = 5")

if st.checkbox("Watch metrics for " + ", ".join(default_jobs)):
    st.subheader("Personalization")
    st.write("The personalization between receomendations is th percentage of unique placements recommended for each selected placement. That is, the bigger the personalization, the less placements they share at k recomendations.")

    print("Getting length")
    calculations = len(default_jobs_recommendations[0]) if st.checkbox('Calculate personalization up to full recommendation set.') else 1200
    print("Length done")

    default_recommendations_df = pd.DataFrame(default_jobs_recommendations).T
    default_recommendations_df.columns = default_jobs

    personalization_record = get_personalisation(calculations)

    fig = px.line(x = [i+1 for i in range(calculations)],y=personalization_record, labels={'x':'Recommendations', 'y':'Personalization %'})
    st.plotly_chart(fig)

    intra_list_calculations = 200

    intra_list_similarities = get_intra_list_similarity(intra_list_calculations)
    for job in range(len(default_jobs)):
        fig.add_trace(go.Scatter(x = [i + 1 for i in range(intra_list_calculations-1)],y=intra_list_similarities[job],name=default_jobs[job],mode='lines')) 
    st.plotly_chart(fig)