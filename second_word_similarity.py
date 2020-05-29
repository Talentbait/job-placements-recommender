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

with open('unique_placements.json') as json_file:
    unique_placements = json.load(json_file)
print("Succesfully loaded dataset")

##############################################################################################
# Useful dicts
ids_to_placement_dict = {}
#Build dict of ids to names, urls and types
for k, v in unique_placements.items():
    ids_to_placement_dict[v['id']] = {
        'name':v['name'],
        'type':v['type'],
        'url':k,
        'description':v['description'],
        'language':'' if 'language' not in v else v['language']
    }

def getNameFromURl(url):
    return unique_placements[url]['name']

def getNameFromId(placement_id):
    return ids_to_placement_dict[int(placement_id)]['name']

def getPlacementFromId(placement_id):
    return ids_to_placement_dict[int(placement_id)]

def clean_description(description):
    return utils.clean_description(description)


st.title("Job title - Placement recommender system")
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
    model.initFromTsv('collaborative_filtering/models/german_tabbed_vectors.tsv')
    print("Placement classification model loaded succesfully")
    return model

@st.cache()
def get_embeddings_labels():
    labels = []
    with open('collaborative_filtering/models/placement_labels_word_count_6_190.txt','r') as textfile:
        for line in textfile:
            labels.append(line.replace('__placement__','').replace('\n',''))
    return labels

@st.cache()
def get_placement_embeddings():
    vectors = np.genfromtxt('collaborative_filtering/models/labeled_placement_embeddings_word_count_6_190.tsv',delimiter='\t',usecols=[a + 1 for a in range(300)])
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
tqdm.pandas(desc="cleaning descriptions")
predicted_nn_info_table['cleaned_description'] = predicted_nn_info_table['description']#.progress_apply(lambda x: clean_description(x))
predicted_nn_info_table['score'] = [similarity_scores[a] for a in sorted_similarity_scores]
st.write(predicted_nn_info_table[['name','type','cleaned_description','score','description']])

default_jobs = ['Bankkaufmann','Krankenschwester','Busfahrer','Elektrotechniker']
if st.checkbox("Watch metrics for " + ", ".join(default_jobs)):
    default_jobs_vectors = []
    for job in default_jobs:
        print("Getting embeddings for " + job)
        job_embedding = (np.array(model.getDocVector(job,' ')))
        default_jobs_vectors.append(job_embedding)

    print("This works")

    @st.cache()
    def get_similarity_for_default(jobs_embeddings):
        print("Calculating similarities")
        jobs_recommendations = []
        for job in tqdm(jobs_embeddings,desc="Getting recommendations"):
            similarity_scores = cos_sim(job,placements_embeddings)[0]
            sorted_similarity_scores = list(np.argsort(similarity_scores)[::-1])
            predicted_nn_names = [getNameFromId(placements_ids[a]) for a in sorted_similarity_scores]
            jobs_recommendations.append(predicted_nn_names)
        return jobs_recommendations

    print("...")
    default_jobs_recommendations = get_similarity_for_default(default_jobs_vectors)
    print("...")

    st.subheader("Personalization")
    st.write("The personalization between receomendations is th percentage of unique placements recommended for each selected placement. That is, te bigger the personalization, the less placements they share at k recomendations.")

    print("Getting length")
    calculations = len(default_jobs_recommendations[0]) if st.checkbox('Calculate personalization up to full recommendation set.') else 1200
    print("Length done")

    @st.cache()
    def get_personalisation_at_k(calculations):
        print("Calculating personalization at k...")
        personalization_record = []
        for i in tqdm(range(calculations),desc="Personalization"):
            pers = len(set(default_jobs_recommendations[0][:i+1]).intersection(set(default_jobs_recommendations[1][:i+1])).intersection(set(default_jobs_recommendations[2][:i+1])).intersection(set(default_jobs_recommendations[3][:i+1])))
            pers = round(1-(pers/(i+1)/3),4)*100
            personalization_record.append(pers)
        print("Done calculating personalization at k.")
        return personalization_record

    personalization_record = get_personalisation_at_k(calculations)

    fig = px.line(x = [i+1 for i in range(calculations)],y=personalization_record, labels={'x':'Recommendations', 'y':'Personalization %'})
    st.plotly_chart(fig)

    st.write(len(default_jobs_recommendations[0]))