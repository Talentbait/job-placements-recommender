import streamlit as st 
import json
import re
from tqdm import tqdm
import pandas as pd


st.title('Search for exact matching descriptions')
st.write('Enter some Keywords to look for in the placments we have and retrieve descriptions that matched most to the input.')

unique_placements_dict = {}
with open("unique_placements.json",'r') as json_file:
    unique_placements_dict = json.load(json_file)

desired_languages = st.multiselect(
    label = 'Choose languague(s) to work with.',
    options = ['en','de'],
    default = ['de']
)

matches_dict = {}

search_keywords = st.text_input('Keywords to look for, separated by commas','Busfahrer, Führerschein, Bus, Omnibus')

keywords_list = search_keywords.split(', ')

keywords_re = [re.compile(r'(?i)' + re.escape(keyword)) for keyword in keywords_list]

for url, placement in tqdm(unique_placements_dict.items(),desc='Looking in the placements'):
    if 'language' in placement:
        if any(desired_language in placement['language'] for desired_language in desired_languages):
            text_from_placement = str(placement['description']) + str(placement['name']) + (str(placement['keywords']) if 'keywords' in placement else '')
            unique_placements_dict[url]['total_matches'] = 0
            for keyword_idx, keyword_re in enumerate(keywords_re):
                matces = len(re.findall(keyword_re,text_from_placement))
                unique_placements_dict[url][keywords_list[keyword_idx]] = matces
                unique_placements_dict[url]['total_matches'] = unique_placements_dict[url]['total_matches'] + matces
        # else:
        #     del 

matches_table = pd.DataFrame.from_dict(unique_placements_dict,orient='index').reset_index(drop=True)

st.write(matches_table)