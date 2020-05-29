import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
from PIL import Image
import utils
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from spacy.lang.de.stop_words import STOP_WORDS as de_stop


st.title("Data Analysis - Unique placements")
st.subheader("Here we are going to explore the unique placements where ads were placeds")
st.markdown("<br>", unsafe_allow_html=True)

unique_placements_dict = utils.load_unique_placements()
#-------------------------------------------------------------------#
#--------------------------   Dataset  -----------------------------#
#-------------------------------------------------------------------#
# st.subheader("Dataset")

# st.write(data)

# missing_data = data[(data['youtube id'].notnull())&(data['view_count'].isnull())]
# missing_data = pd.concat([missing_data,data[(data['app_id'].notnull())&(data['description'].isnull())]],axis=0)

# st.write(f"There are {data[data['description'].isna()].shape[0]} null descriptions from the {data.shape[0]} unique placements.")

# if st.checkbox('Show pending placements'):
#     st.subheader('Placements with missing data')
#     st.write('These are the YouTube channels and mobile applications that still need to be scrapped. Some YouTube channels where banned and is not possible to retrieve that information.')
#     st.write(missing_data)

# data = data.drop(index=list(data[data['description'].isnull()].index))

# #-------------------------------------------------------------------#
# #----------------------  Type distribution  ------------------------#
# #-------------------------------------------------------------------#

unique_placements_german_dict = utils.load_unique_placements_by_language('de')
unique_placements_german_df = pd.DataFrame.from_dict(unique_placements_german_dict,orient='index')
st.write(unique_placements_german_df)

type_distribution = unique_placements_german_df['type'].value_counts()
# type_distribution = type_distribution.rename(columns={'index':'type','type':'count'})

st.subheader('Type of retrieved placements distribution')
st.write(f'From the placements that we already have information. **{type_distribution[0]}** placements correspond to YouTube channels and the other **{type_distribution[1]}** placements correspond to mobile applications')
fig = px.pie(pd.DataFrame(type_distribution).reset_index(),names='index',values='type')
fig.update_traces(textposition='inside', textinfo='percent+label')
st.plotly_chart(fig)

#-------------------------------------------------------------------#
#-------------------  Language of descriptions  --------------------#
#-------------------------------------------------------------------#

# st.subheader("Descriptions' languages")

# val_count = pd.DataFrame(data['language'].value_counts()).reset_index()
# val_count = val_count.rename(columns={'index':'language','language':'count'})

# prev_size = data.shape[0]
# curr_size = data[(data['language'] == 'en')|(data['language'] == 'de')].shape[0]
# st.write(f"Working just with german and english descriptions drops {prev_size-curr_size} placements that don't have information yet. ({round((1-curr_size/prev_size)*100,2)}% of the original placements). We'll be just left with {curr_size} placements")

# fig = px.pie(pd.DataFrame(val_count).reset_index(),names='language',values='count')
# fig.update_traces(textposition='inside', textinfo='percent+label')
# st.plotly_chart(fig)


#-------------------------------------------------------------------#
#----------------------   Description's info  ----------------------#
#-------------------------------------------------------------------#


st.subheader("Description's analysis")
st.write("We need to have an insight of what we'll be working with so here we're going to get some useful metrics from the various descritpions that we have.")



unique_placements_german_df['description_len'] = unique_placements_german_df['description'].apply(lambda x: len(x) if x else 0)
unique_placements_german_df['description_word_count'] = unique_placements_german_df['description'].apply(lambda x: len(x.split()) if x else 0)

st.write(f"From all the descriptions that we have gathered, the mean character lenght is {round(unique_placements_german_df.description_len.mean())}")

fig = px.box(data_frame=unique_placements_german_df,x='language',y='description_word_count')
st.plotly_chart(fig)


st.subheader("Outlayers")

fig = px.histogram(unique_placements_german_df, x="description_word_count")
st.plotly_chart(fig)

percentiles = [0.05,0.125,0.25,0.5,0.75,0.8,0.85,0.9,0.91,0.92,0.93,0.94,0.95]

quantile_count = [round(len(unique_placements_german_df)*percentile) for percentile in percentiles] +[len(unique_placements_german_df)]
quantile_count = [j-i for i, j in zip(quantile_count[:-1], quantile_count[1:])]
quantile_count = np.concatenate((np.zeros(5),quantile_count))
info_df = unique_placements_german_df['description_word_count'].describe(percentiles=percentiles).to_frame()
info_df['placement_count'] = quantile_count.tolist()
st.write(info_df)

st.subheader("Placements that will not be considered")

dump_percentage = st.slider(
        'Select range of att data to discriminate',
        0,100,(0,95),step=1,format='%d%%'
    )

quantile_low = unique_placements_german_df['description_word_count'].quantile(dump_percentage[0]/100)
quantile_high = unique_placements_german_df['description_word_count'].quantile(dump_percentage[1]/100)
if st.checkbox('Input word count'):
    dump_word_count = st.slider(
        'Select range of att data to discriminate',
        0,int(unique_placements_german_df['description_word_count'].max()),(1,196),step=1
    )
    quantile_low = dump_word_count[0]
    quantile_high = dump_word_count[1]

st.write(f"The placements that have less than {quantile_low} words, or more than {quantile_high} are shown below. These values correspond to what the {dump_percentage[0]}% of the descrptions that are the shortest ({round(dump_percentage[0]/100*len(unique_placements_german_df))} descriptions) and the {dump_percentage[1]}% of the descrptions that are the longest ({round((1 - dump_percentage[1]/100)*len(unique_placements_german_df))} descriptions)")

discriminated_placements = unique_placements_german_df[(unique_placements_german_df['description_word_count']<quantile_low)|(unique_placements_german_df['description_word_count']>quantile_high)]

st.write(discriminated_placements)

fig = utils.distribution_pie_chart(discriminated_placements,'type')
st.plotly_chart(fig)

########################## WORD CLOUD ##########################

print("Counting words...")
token_pattern = '(?ui)\\b[a-z]{2,}\\b'
cv = CountVectorizer(lowercase = True, stop_words = de_stop,token_pattern = token_pattern)
@st.cache()
def get_cleaned_descriptions():
    descriptions = unique_placements_german_df['description'].apply(utils.clean_description)
    return descriptions

descriptions = get_cleaned_descriptions()

@st.cache()
def get_word_count():
    descriptions = unique_placements_german_df['description'].apply(utils.clean_description)
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
print("Creating wordcloud...")
wc = WordCloud(width=800, height=400, max_words=130,scale=10,background_color="white")
wc.generate_from_frequencies(word_frequency.set_index('word').to_dict()['count'])
# wc.to_file('wordcloud_of_frequent_words_v1.png')
print("Done.")
plt.axis("off")
plt.imshow(wc,interpolation='bilinear')
st.pyplot()
if st.checkbox('View word count...'):
    st.write('A more detailed view of the words')
    st.write(word_frequency)