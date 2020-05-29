from tqdm import tqdm
import streamlit as st
import json
from operator import itemgetter
import numpy as np
try:
    from macos import starwrap as sw
except:
    import starwrap as sw
import plotly.express as px
import re
import pandas as pd
import utils


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

##############################################################################################
# Job title classification
st.title("Job Title Calssification")

st.subheader("Enter a job title to see which are the more domain related placements")
job_title = clean_description(st.text_input(
    label = 'Enter a job title here',
    value = 'Busfahrer'
))
st.write("The preprocessed input will be: **" + job_title +"**")
@st.cache()
def init_placement_classifier(): 
    print("Starspace: init (placement classification model)")
    arg = sw.args()
    arg.label = '__placement__'
    arg.dim = 300
    arg.trainMode = 0
    model = sw.starSpace(arg)
    print("Starspace: loading from saved model (placement classification model)")
    model.initFromTsv('collaborative_filtering/models/german_vectors_and_placement_embeddings.tsv')
    print("Placement classification model loaded succesfully")
    return model

placement_classifier = init_placement_classifier()

related_predicts = 20
domain_related_predictions_dict = placement_classifier.predictTags(job_title, related_predicts)
domain_related_predictions_dict = sorted(domain_related_predictions_dict.items(), key = itemgetter(1), reverse = True )
print(f"Predicted {related_predicts} domain related placements.")
domain_related_placement_names = []
domain_related_placement_info = []
print("Processing predicted tags")
print(len(domain_related_predictions_dict))
for tag, prob in domain_related_predictions_dict:
    tag_placement_id = tag.replace("__placement__", "")
    if prob > 0:
        info = getPlacementFromId(tag_placement_id)
        info['score'] = str(round(prob*100,2)) + "%"
        domain_related_placement_names.append(getNameFromId(tag_placement_id))
        domain_related_placement_info.append(info)

# When there is no match from the job title with YouTube placements, then there's nothing else to do
if  not len(domain_related_placement_names) > 0:
    st.write(f"There were no matching placements for {job_title}.")
    st.subheader(f"Unfortunately, as there are no domain related placements found for {job_title}, the rest of the recommender system has no use. Please try entering another job title")
else:
    domain_related_placement_table = pd.DataFrame(domain_related_placement_info)
    domain_related_placement_table['description'] = domain_related_placement_table['description'].apply(lambda x: clean_description(x))
    st.write(domain_related_placement_table[['name','type','description','score']])

    ##############################################################################################
    # Collaborative model    

    st.title("Placement Recommender")

    # st.subheader("Choose a site to see on which other placements we should show the ad")

    # placement_selected = st.selectbox(
    #     label='',
    #     options = list(unique_placements.keys())[1:1001],
    #     format_func = getNameFromURl,
    #     index=0
    # )
    # print("selectbox displayed")

    similar_placements = domain_related_placement_info[:5]
    placement_selected = domain_related_placement_info[0]
    
    selected_placement_url = placement_selected['url']
    selected_placement_id = unique_placements[selected_placement_url]['id']
    selected_placement_name = placement_selected['name']
    selected_placement_type = placement_selected['type']
    st.write(f"People who clicked the ad on **{selected_placement_name}** (which is a {selected_placement_type}) would also click on these other placements:")

    if st.checkbox('Display more information from this placement?'):
        st.write(placement_selected)

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

    predicts = 3000
    collaborative_input_query = '__label__' + ' __label__'.join(str(unique_placements[placement['url']]['id']) for placement in similar_placements)
    print(collaborative_input_query)
    dict_obj = model.predictTags(collaborative_input_query, predicts)
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
    second_most_similar_placement = domain_related_placement_info[1]
    
    second_placement_url = second_most_similar_placement['url']
    second_placement_id = unique_placements[selected_placement_url]['id']
    second_placement_name = second_most_similar_placement['name']
    second_placement_type = second_most_similar_placement['type']

    st.subheader("Evaluating the results")
    st.write(f"For some metrics, like personalizatio and coverage, we should compare 2 or more recommendation sets. So we will compare these recommendatiosn to the ones generated for **{second_placement_name}**, which was the second most domain similar placement.")


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
