import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
from PIL import Image


st.title("Data Analysis - Unique placements")
st.subheader("Here we are going to explore the unique placements where ads were placeds")
st.markdown("<br>", unsafe_allow_html=True)

@st.cache()
def load_data(persist= True):
    data = pd.read_csv("saves/cleaned_placements_dataset_landetect.csv", index_col = 0,low_memory=False) 
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis="columns", inplace=True)
    # data = data.drop(columns=['avg. cost','currency code','german','international','foreign'])
    return data

data = load_data()

#-------------------------------------------------------------------#
#--------------------------   Dataset  -----------------------------#
#-------------------------------------------------------------------#
st.subheader("Dataset")

st.write(data)

missing_data = data[(data['youtube id'].notnull())&(data['view_count'].isnull())]
missing_data = pd.concat([missing_data,data[(data['app_id'].notnull())&(data['description'].isnull())]],axis=0)

st.write(f"There are {data[data['description'].isna()].shape[0]} null descriptions from the {data.shape[0]} unique placements.")

if st.checkbox('Show pending placements'):
    st.subheader('Placements with missing data')
    st.write('These are the YouTube channels and mobile applications that still need to be scrapped. Some YouTube channels where banned and is not possible to retrieve that information.')
    st.write(missing_data)

data = data.drop(index=list(data[data['description'].isnull()].index))

type_distribution = data['type'].value_counts()
appstore_distribution = data['app_store'].value_counts()

#-------------------------------------------------------------------#
#----------------------  Type distribution  ------------------------#
#-------------------------------------------------------------------#

st.subheader('Type of retrieved placements distribution')
st.write(f'From the placements that we already have information. **{type_distribution[0]}** placements correspond to YouTube channels and the other **{type_distribution[1]}** placements correspond to mobile applications, from which {appstore_distribution[0]} are from the Google Playstore and the other {appstore_distribution[1]} are from itunes.')
plt.pie(type_distribution,labels=['YouTube channels','Mobile applications'], explode=[0.05, 0.05], pctdistance=0.70,autopct='%1.1f%%')

st.pyplot()

#-------------------------------------------------------------------#
#-------------------  Language of descriptions  --------------------#
#-------------------------------------------------------------------#

st.subheader("Descriptions' languages")

val_count = pd.DataFrame(data['language'].value_counts()).reset_index()
val_count = val_count.rename(columns={'index':'language','language':'count'})

prev_size = data.shape[0]
curr_size = data[(data['language'] == 'en')|(data['language'] == 'de')].shape[0]
st.write(f"Working just with german and english descriptions drops {prev_size-curr_size} placements that don't have information yet. ({round((1-curr_size/prev_size)*100,2)}% of the original placements). We'll be just left with {curr_size} placements")

fig = px.pie(pd.DataFrame(val_count).reset_index(),names='language',values='count')
fig.update_traces(textposition='inside', textinfo='percent+label')
st.plotly_chart(fig)

#-------------------------------------------------------------------#
#----------------------  Applications labels  ----------------------#
#-------------------------------------------------------------------#

st.subheader("Application's labels")
st.write('From the google playstore and also from itunes, we also have some app tag that clasify them by its content. Here you can have a look of how they are categorized.')

app_store = st.selectbox(
    "Choose the store you'll like to check",
    ['google','itunes']
)

app_genre_distribution = pd.DataFrame(data[data['app_store']==app_store]['app_genre'].value_counts()).reset_index()
app_genre_distribution = app_genre_distribution.rename(columns={'index':'category','app_genre':'count'})
app_genre_labels = app_genre_distribution.keys().tolist()
if not st.checkbox('Display bar chart horizontally'):
    fig = px.bar(pd.DataFrame(app_genre_distribution).reset_index(),x='category',y='count',orientation='v')
else:
    fig = px.bar(pd.DataFrame(app_genre_distribution).reset_index(),y='category',x='count',orientation='h')
st.plotly_chart(fig)

if st.checkbox(f'View complete list of app categories for {app_store}'):
    st.write(app_genre_distribution['category'].unique().tolist())

#-------------------------------------------------------------------#
#----------------------   Description's info  ----------------------#
#-------------------------------------------------------------------#

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\S*.com\S*", "", text)
    text = re.sub(r"\S*.de\S*", "", text)
    text = re.sub(r"\S*www.\S*", "", text)
    text = text.lower()
    text = re.sub(r'[0-9]{2,}', '', text)
    # sentence = re.sub(r'[^a-zßäöü\s]', '', text)
    text = re.sub(r"[^a-z0-9äöü _.,!ß?\-\"'/$]*", '', text)
    return re.sub(r'\s{2,}', ' ', text)

st.subheader("Description's analysis")
st.write("We need to have an insight of what we'll be working with so here we're going to get some useful metrics from the various descritpions that we have.")

descriptions_df = data[['placement_name','description','type','app_store']]

descriptions_df['description_len'] = descriptions_df['description'].apply(lambda x: len(x) if x else 0)

st.write(f"From all the descriptions that we have gathered, the mean character lenght is {round(descriptions_df.description_len.mean())}")

fig = px.box(data_frame=descriptions_df,x='type',y='description_len')
st.plotly_chart(fig)