import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
from PIL import Image
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from spacy.lang.de.stop_words import STOP_WORDS as de_stop
from spacy.lang.en.stop_words import STOP_WORDS as en_stop
import format_train_file
import json
import os
from PIL import Image

# def cleaned_description(description):
#     description,cleaning_info = format_train_file.clean_description(description)
#     return description


class FileReference:
    def __init__(self, filename):
        self.filename = filename

def hash_file_reference(file_reference):
    filename = file_reference.filename
    return (filename, os.path.getmtime(filename))

@st.cache(hash_funcs={FileReference: hash_file_reference})
def get_unique_placements(file_path):
    with open(file_path) as json_file:
        unique_placements_german_dict = json.load(json_file)
        print("Succesfully loaded dataset")
        return unique_placements_german_dict
    
current_dataset_path = "replacements_for_all_descriptions.json"
print("Loading current_dataset from " + current_dataset_path)
unique_placements_german_dict = get_unique_placements(current_dataset_path)

st.title("Data Analysis - Unique placements")
st.subheader("Here we are going to explore the unique placements where ads were placeds")
st.markdown("<br>", unsafe_allow_html=True)

unique_placements_german_df = pd.DataFrame.from_dict(unique_placements_german_dict,orient='index')
if st.checkbox('Just use German and english descriptions'):
    unique_placements_german_df = unique_placements_german_df[(unique_placements_german_df['language']=='de')|(unique_placements_german_df['language']=='en')]
st.write(unique_placements_german_df)

#-------------------------------------------------------------------#
#--------------------   Language distribution  ---------------------#
#-------------------------------------------------------------------#
language_distribution = unique_placements_german_df['language'].value_counts()
st.subheader('Language of retrieved placements distribution')
st.write(f'From the **{len(unique_placements_german_dict)}** placements that we already have information. **{language_distribution[0] + language_distribution[1]}** are in English and German and the other **{len(unique_placements_german_df)-(language_distribution[0] + language_distribution[1])}** are in other languages')
fig = px.pie(pd.DataFrame(language_distribution).reset_index(),names='index',values='language')
fig.update_traces(textposition='inside', textinfo='percent+label')
st.plotly_chart(fig)

#-------------------------------------------------------------------#
#----------------------   Type distribution  -----------------------#
#-------------------------------------------------------------------#
type_distribution = unique_placements_german_df['type'].value_counts()
# type_distribution = type_distribution.rename(columns={'index':'type','type':'count'})

st.subheader('Type of retrieved placements distribution')
st.write(f'From the **{len(unique_placements_german_dict)}** placements that we already have information. **{type_distribution[0]}** placements correspond to Mobile application and the other **{type_distribution[1]}** placements correspond to YouTube channels')
fig = px.pie(pd.DataFrame(type_distribution).reset_index(),names='index',values='type')
fig.update_traces(textposition='inside', textinfo='percent+label')
st.plotly_chart(fig)
#-------------------------------------------------------------------#
#----------------------   Description's info  ----------------------#
#-------------------------------------------------------------------#

if st.checkbox('Just look at german descriptions'): 
    unique_placements_german_df = unique_placements_german_df[unique_placements_german_df['language'] == 'de']

st.subheader("Description's analysis")
st.write("We need to have an insight of what we'll be working with so here we're going to get some useful metrics from the various descritpions that we have.")

unique_placements_german_df['description_len'] = unique_placements_german_df['cleaned_description'].apply(lambda x: len(x) if x else 0)
unique_placements_german_df['description_word_count'] = unique_placements_german_df['cleaned_description'].apply(lambda x: len(x.split()) if x else 0)

st.write(f"From all the descriptions that we have gathered, the mean character lenght is {round(unique_placements_german_df.description_len.mean())}")

measurable_options = list(unique_placements_german_df.columns)
for element in ['name','id','type','description','language','cleaned_description']:
    measurable_options.remove(element)

box_plot_label = st.selectbox(
    label = 'Choose a category to display its boxplot',
    options = measurable_options,
    index = 2
)

# grouping_attribute = 'language'
# if st.checkbox('Choose a grouping attribute?'):
grouping_attribute = st.selectbox(
    label = 'Choose a category to group data for the boxplot',
    options = ['language','type'],
    index = 0
)


fig = px.box(data_frame=unique_placements_german_df,x=grouping_attribute,y=box_plot_label,labels={grouping_attribute:box_plot_label,box_plot_label:grouping_attribute})
st.plotly_chart(fig)


st.subheader("Outlayers")


nbins = unique_placements_german_df[box_plot_label].max() + 1
if st.checkbox('More detailed histogram'):
    fig = px.histogram(unique_placements_german_df, x=box_plot_label,nbins=int(nbins))
    st.plotly_chart(fig)
else:
    fig = px.histogram(unique_placements_german_df, x=box_plot_label)
    st.plotly_chart(fig)

percentiles = [0.05,0.125,0.25,0.5,0.75,0.8,0.85,0.9,0.91,0.92,0.93,0.94,0.95]

quantile_count = [round(len(unique_placements_german_df)*percentile) for percentile in percentiles] +[len(unique_placements_german_df)]
quantile_count = [j-i for i, j in zip(quantile_count[:-1], quantile_count[1:])]
quantile_count = np.concatenate((np.zeros(5),quantile_count))
info_df = unique_placements_german_df[box_plot_label].describe(percentiles=percentiles).to_frame()
info_df['placement_count'] = quantile_count.tolist()
st.write(info_df)

st.subheader("Placements that will not be considered")

dump_percentage = st.slider(
        'Select range of att data to discriminate',
        0,100,(0,95),step=1,format='%d%%'
    )

quantile_low = unique_placements_german_df[box_plot_label].quantile(dump_percentage[0]/100)
quantile_high = unique_placements_german_df[box_plot_label].quantile(dump_percentage[1]/100)
if st.checkbox(f'Input {box_plot_label} count'):
    max_of_label = int(unique_placements_german_df[box_plot_label].max())
    max_slected = 196 if max_of_label > 196 else max_of_label
    dump_word_count = st.slider(
        'Select range of att data to discriminate',
        0,max_of_label,(1,max_slected),step=1
    )
    quantile_low = dump_word_count[0]
    quantile_high = dump_word_count[1]
    dump_percentage = [round(100*len(unique_placements_german_df[unique_placements_german_df[box_plot_label]<quantile_low])/len(unique_placements_german_df),2),round(100*len(unique_placements_german_df[unique_placements_german_df[box_plot_label]<quantile_high])/len(unique_placements_german_df),2)]

st.write(f"The placements that have less than {quantile_low} {box_plot_label}, or more than {quantile_high} are shown below. These values correspond to what the {dump_percentage[0]}% of the descrptions with fewer {box_plot_label} ({round(dump_percentage[0]/100*len(unique_placements_german_df))} descriptions) and the {round(100-dump_percentage[1],2)}% of the descrptions with more {box_plot_label} ({round((1 - dump_percentage[1]/100)*len(unique_placements_german_df))} descriptions)")

discriminated_placements = unique_placements_german_df[(unique_placements_german_df[box_plot_label]<quantile_low)|(unique_placements_german_df[box_plot_label]>quantile_high)]

new_diplay_order = ['name',box_plot_label,'type','cleaned_description','description','pre_preprocessing_words']+list(discriminated_placements.columns.values)

st.write(discriminated_placements[list(dict.fromkeys(new_diplay_order))])

type_distribution = discriminated_placements['type'].value_counts()
fig = px.pie(pd.DataFrame(type_distribution).reset_index(),names='index',values='type')
fig.update_traces(textposition='inside', textinfo='percent+label')
st.plotly_chart(fig)
########################## WORD CLOUD ##########################

print("Counting words...")
token_pattern = '(?ui)\\b[a-z]{2,}\\b'
cv = CountVectorizer(lowercase = False,token_pattern = token_pattern)
@st.cache()
def get_cleaned_descriptions():
    descriptions = unique_placements_german_df['cleaned_description']
    return descriptions

descriptions = get_cleaned_descriptions()

@st.cache()
def get_word_count():
    descriptions = unique_placements_german_df['cleaned_description']
    x = cv.fit_transform(descriptions)
    word_count = (np.sort(x.toarray().sum(axis=0)))[::-1]
    word_count_idx = (np.argsort(x.toarray().sum(axis=0)))[::-1]
    feature_names = np.array(cv.get_feature_names())
    top_words = feature_names[word_count_idx]
    word_frequency = pd.DataFrame({
        'word': top_words,
        'count': word_count
    })
    print("Count ended.")
    return word_frequency
word_frequency = get_word_count()

st.subheader('Word cloud of most frequent words troughout the descriptions')

@st.cache()
def load_word_cloud():
    im = Image.open('word_cloud_frequent_words.png')
    return im

# Comment for heroku
print("Creating wordcloud...")
wc = WordCloud(width=800, height=400, max_words=130,scale=10,background_color="white")
wc.generate_from_frequencies(word_frequency.set_index('word').to_dict()['count'])
print("Done.")
wc.to_file('word_cloud_frequent_words.png')
plt.axis("off")
plt.imshow(wc,interpolation='bilinear')
st.pyplot()

# Uncomment for Heroku
# st.image(load_word_cloud(),use_column_width=True)

# print(type(word_frequency))
# print(word_frequency)

word_frequency.to_csv('word_count.csv')
print('saved word_count.csv to directory.')

word_frequency = pd.read_csv('word_count.csv',index_col=0)
print('loaded word_count.csv from directory.')

if st.checkbox('View word count...'):
    st.write('A more detailed view of the words')
    st.write(word_frequency)
