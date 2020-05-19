import streamlit as st
import data_analysis
from recommender_systems_prev_versions import recommender_system_nlp_feature_extraction as rs_with_nlp
from recommender_systems_prev_versions import recommender_system_model_building as rs_with_2_jobs
import pandas as pd
import numpy as np
import pydeck as pdk
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
from langdetect import detect
from langdetect import detect_langs
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from spacy.lang.en.stop_words import STOP_WORDS as en_stop
from spacy.lang.de.stop_words import STOP_WORDS as de_stop
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from recommender_systems_prev_versions import recommender_metrics as rc 
from PIL import Image
import tqdm
from recommender_systems_prev_versions import recommender_system_model_building_3_jobs as rs_with_3_jobs
from recommender_systems_prev_versions import recommender_system_nlp_similarity as rs_sim_nlp

st.sidebar.header('Job placements recommender for google ads')

model_view = st.sidebar.selectbox(
    'Which section would you like to view?',
    ('Data analysis','Second iteration (2 jobs)', 'Second iteration (3 jobs)', 'Third iteration (with NLP processing)','Using nlp similarity calc'),
    0
)

st.sidebar.markdown("**What is this project about?**")
st.sidebar.markdown("Google Ads is one advertising platform used by Talentbait. Also, it provides us with performance metrics for each job ad that can be used to optimize the advertisement campaigns.",unsafe_allow_html=True)
st.sidebar.markdown("This recommender system learns from previous campaigns and suggests where the ad should be placed directly with manual placing.")
st.sidebar.markdown("*The goal is to give the system an input job title and it will output the top N recommendations for websites and/or youtube channels to place the ad and have more efficient metrics.*")

if model_view == 'Data analysis':
    data_analysis.data_analysis()
elif model_view == 'Second iteration (2 jobs)':
    rs_with_2_jobs.recommender_system_model_building()
elif model_view == 'Second iteration (3 jobs)':
    rs_with_3_jobs.rs_with_3_jobs()
elif model_view == 'Third iteration (with NLP processing)':
    rs_with_nlp.rs_with_nlp()
elif model_view == 'Using nlp similarity calc':
    rs_sim_nlp.rs_similarity_with_pos_filtering()
else:
    st.error('Please select a file to show.')

st.balloons()