from tqdm import tqdm
import streamlit as st
import json
from operator import itemgetter
import numpy as np
try:
    from macos import starwrap as sw
except:
    import starwrap as sw
import recommender_metrics as rc
import plotly.express as px
from numpy import linalg as LA

with open('collaborative_filtering/unique_placements_dict.json') as json_file:
    unique_placements = json.load(json_file)
print("Succesfully loaded dataset")

st.title("Placement Recommender")
st.subheader("Choose a site to see on which other placements we should show the ad")

ids_to_placement_dict = {}
#Build dict of ids to names, urls and types
for k, v in unique_placements.items():
    ids_to_placement_dict[v['id']] = {
        'name':v['name'],
        'type':v['type'],
        'url':k
    }

def getNameFromURl(url):
    return unique_placements[url]['name']

def getNameFromId(id):
    return ids_to_placement_dict[int(id)]['name']

def getPlacementFromId(id):
    return ids_to_placement_dict[int(id)]

placement_selected = st.selectbox(
    label='',
    options = list(unique_placements.keys())[1:1001],
    format_func = getNameFromURl,
    index=0
)
print("selectbox displayed")

selected_placement_id = unique_placements[placement_selected]['id']
selected_placement_name = unique_placements[placement_selected]['name']
st.write(f"People who clicked the ad on **{selected_placement_name}** would also click on these other placements")

@st.cache()
def init_starspace(): 
    print("Starspace: init")
    arg = sw.args()
    model = sw.starSpace(arg)
    print("Starspace: loading from saved model")
    model.initFromSavedModel('collaborative_filtering/models/collaborative_system')
    print("Model loaded succesfully")
    return model

model = init_starspace()

predicts = 1000
dict_obj = model.predictTags('__label__'+str(selected_placement_id), predicts)
dict_obj = sorted( dict_obj.items(), key = itemgetter(1), reverse = True )
print(f"Predicted {predicts} similar placements.")
recommendations_list = []
recommendations = []

print("Processing predicted tags")
for tag, prob in dict_obj:
    tag_placement_id = tag.replace("__label__", "")
    recommendations_list.append(getNameFromId(tag_placement_id))
    recommendations.append(getPlacementFromId(tag_placement_id))

print("Getting top 10 placements for each placement type")
recommended_sites = [placement['name'] for placement in recommendations if placement['type']=='Site'][:10]
recommended_channels = [placement['name'] for placement in recommendations if placement['type']=='YouTube channel'][:10]
recommended_apps = [placement['name'] for placement in recommendations if placement['type']=='Mobile application'][:10]

st.write("Top **YouTube channels** that they would also click")
st.write(recommended_channels)
st.write("Top **mobile applications** that they would also click")
st.write(recommended_apps)
st.write("Top **sites** that they would also click")
st.write(recommended_sites)

print("Everything done!")

############# METRICS #############
st.subheader("Evaluating the results")
st.write("For some metrics, like personalizatio and coverage, we should compare 2 or more recommendation sets. So first, let's define a second placement to get recommendations for.")

second_placement = st.selectbox(
    label='',
    options = list(unique_placements.keys())[1:501],
    format_func = getNameFromURl,
    index=5
)

second_placement_id = unique_placements[second_placement]['id']
secondd_placement_name = unique_placements[second_placement]['name']

dict_obj = model.predictTags('__label__'+str(second_placement_id), predicts)
dict_obj = sorted( dict_obj.items(), key = itemgetter(1), reverse = True )

recommendations_second_placement = []

print("Processing predicted tags")
for tag, prob in dict_obj:
    tag_placement_id = tag.replace("__label__", "")
    recommendations_second_placement.append(getPlacementFromId(tag_placement_id)['name'])

@st.cache()
def get_personalisation_at_k():
    print("Calculating personalization at k...")
    personalization_record = []
    for i in tqdm(range(predicts),desc="Personalization"):
        pers = len(set(recommendations_second_placement[:i+1]).intersection(set(recommendations_list[:i+1])))
        pers = round(1-(pers/(i+1)/2),4)*100
        personalization_record.append(pers)
    print("Done calculating personalization at k.")
    return personalization_record

personalization_record = get_personalisation_at_k()

st.subheader("Personalization")
st.write("The personalization between receomendations is th percentage of unique placements recommended for each selected placement. That is, te bigger the personalization, the less placements they share at k recomendations.")
fig = px.line(x = [i+1 for i in range(predicts)],y=personalization_record, labels={'x':'Recommendations', 'y':'Personalization %'})
st.plotly_chart(fig)

@st.cache()
def get_coverage_at_k():
    print("Calculating coverage at k...")
    coverage_record = []
    for i in tqdm(range(predicts),desc="coverage"):
        cover = len(set(recommendations_second_placement[:i+1]).union(set(recommendations_list[:i+1])))
        cover = round(cover/predicts/2,4)*100
        coverage_record.append(cover)
    print("Done calculating coverage at k.")
    return coverage_record

coverage_record = get_coverage_at_k()
st.subheader("Coverage")
st.write("The personalization between receomendations is th percentage of unique placements recommended for each selected placement. That is, te bigger the personalization, the less placements they share at k recomendations.")
fig = px.line(x = [i+1 for i in range(predicts)],y=coverage_record, labels={'x':'Recommendations', 'y':'Coverage %'})
st.plotly_chart(fig)


# placement_vectors = pd.from_csv('collaborative/filtering')
