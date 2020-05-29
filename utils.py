import json
from tqdm import tqdm
import re
import plotly.express as px
import pandas as pd
from spacy.lang.de.stop_words import STOP_WORDS as de_stop

def load_unique_placements():
    with open('unique_placements.json') as json_file:
        unique_placements = json.load(json_file)
        print("Succesfully loaded dataset")
        return unique_placements

def load_unique_placements_with_cleaning_info():
    with open('replacement_counts.json') as json_file:
        unique_placements = json.load(json_file)
        print("Succesfully loaded dataset")
        return unique_placements

def load_unique_placements_by_language(selected_language):
    unique_placements_language_dict = {}
    unique_placements_dict = load_unique_placements()
    for placement, values in tqdm(unique_placements_dict.items(),desc='Filtering descriptions by "' + selected_language + '"'):
        if 'language' in values:
            language = values['language']
            if language == selected_language:
                unique_placements_language_dict[placement] = values
    return unique_placements_language_dict

def clean_description(description):
    description = re.sub(r'\s+', ' ', description)         # Removing all spaces
    # description = description.lower()
    description = re.sub(r"\S+@\S+\.\S+", "EMAIL", description)   #Removing emails
    description = re.sub(r"(?i)(\S+)?[a-zA-Z0-9]+\.(com|org|de|net|ly|tv)(\.[a-z]{2,3})?(\S+)?", "URL", description)   #Removing urls
    description = re.sub(r"(?i)http\S+", "URL", description)   #Removing urls
    # description = re.sub(r"\S+__\S+", "", description)  #Removing double undescores
    # description = re.sub(r'[^a-zA-Z0-9äöüß\s]', '', description)
    description = re.sub(r"(?i)[^a-z0-9äöüß\s?!.,\"#$%'()*+\-/:;<=>@[\\\]^_`{|}~]", '', description)
    description = re.sub(r"[0-9][\.,][0-9]","",description)
    description = re.sub(r"[0-9]+"," NUMMER ",description)
    description = re.sub(r"[\"#$%'()*+\-/:;<=>@[\\\]^_`{|}~▬]{2,}", '',description)
    description = re.sub(r"([?!\.,])\1+", r' \1 ',description)
    description = re.sub(r"([?!\.,])", r' \1 ',description)
    description = re.sub(r"[\"#$%'()*+/:;<=>@[\\\]^_`{|}~]", r" \g<0> ",description)
    stop_words_regex = re.compile(r'(?i)' + r'\b%s\b' % r'\b|\b'.join(map(re.escape, de_stop)))
    description = re.sub(stop_words_regex,'',description) 
    # replacements['stop_words'] = len(re.findall(stop_words_regex,description.lower()))
    description = re.sub(r'\s+', ' ', description)
    return description

def clean_description_with_replacement_info(description):
    replacements = {}
    description, replacements['space_truncated'] = re.subn(r'\s+', ' ', description)         # Removing all spaces
    replacements['total_characters'] = len(description) - len(re.findall(r'\s',description))    # Counting total characters minus the white characters
    replacements['pre_preprocessing_words'] = len(description.split())                          # Counting words (separated by blank characters)
    replacements['uppercase_letters'] = len(re.findall(r'[A-ZÖÄÜß]',description))               # Count characters that are uppercase
    replacements['uppercase_first_letter_words'] = len(re.findall(r"\b[A-ZÖÄÜ][a-zöäüß]\S*\b",description))     # Count words that contain a capital letter at the beginning
    replacements['uppercase_loss_ratio'] = replacements['uppercase_first_letter_words']/replacements['pre_preprocessing_words']
    replacements['uppercase_words'] = len(re.findall(r"\b\S*[A-ZÖÄÜ]\S*\b",description))        # Count words that contain at leas one capital letter
    replacements['uppercase_complete_words'] = len(re.findall(r"\b[A-ZÖÄÜ]+\b",description))    # Count words that are completely uppercased
    replacements['acronyms'] = len(re.findall(r"\b[A-ZÖÄÜ\.]+\b",description))                  # Count acronyms A.A.A.
    # description = description.lower()
    description, replacements['email'] = re.subn(r"\S+@\S+\.\S+", "EMAIL", description)   #Removing emails
    description, replacements['url'] = re.subn(r"(?i)(\S+)?[a-zA-Z0-9]+\.(com|org|de|net|ly|tv)(\.[a-z]{2,3})?(\S+)?", "URL", description)   #Removing urls
    description, added_url_replacements = re.subn(r"(?i)http\S+", "URL", description)   #Removing urls
    replacements['url'] = replacements['url'] + added_url_replacements
    # description = re.sub(r"\S+__\S+", "", description)  #Removing double undescores
    # description, replacements['weird_characters'] = re.subn(r'[^a-zA-Z0-9äöüß\s]', '', description)
    description, replacements['weird_characters'] = re.subn(r"(?i)[^a-z0-9äöüß\s?!.,\"#$%'()*+\-/:;<=>@[\\\]^_`{|}~]", '', description)
    description, replacements['removed_decimal_separators'] = re.subn(r"[0-9][\.,][0-9]","",description)
    description, replacements['nummer'] = re.subn(r"[0-9]+"," NUMMER ",description)
    description, replacements['consecutive_nonalpha_characters'] = re.subn(r"[\"#$%'()*+\-/:;<=>@[\\\]^_`{|}~▬]{2,}", '',description)
    description, replacements['consecutive_punctuation_characters'] = re.subn(r"([?!\.,])\1+", r' \1 ',description)
    description, replacements['spaced_punctuation_characters'] = re.subn(r"([?!\.,])", r' \1 ',description)
    description, replacements['spaced_symbols'] = re.subn(r"[\"#$%'()*+/:;<=>@[\\\]^_`{|}~]", r" \g<0> ",description)
    stop_words_regex = re.compile(r'(?i)' + r'\b%s\b' % r'\b|\b'.join(map(re.escape, de_stop)))
    description, replacements['stop_words'] = re.subn(stop_words_regex,'',description) 
    # replacements['stop_words'] = len(re.findall(stop_words_regex,description.lower()))
    description = re.sub(r'\s+', ' ', description)
    replacements['removed_characters'] = replacements['total_characters'] - len(description) + len(re.findall(r'\s',description))
    replacements['post_preprocessing_words'] = len(description.split()) if description else 0
    replacements['single_characters'] = len(re.findall(r"(?i)\W\S\W",description))
    replacements['character_count_cleaned'] = len(description)
    replacements['cleaned_description'] = description
    # description_spacy = nlp(description)
    return description, replacements

def distribution_pie_chart(df,column):
    type_distribution = df[column].value_counts()
    fig = px.pie(pd.DataFrame(type_distribution).reset_index(),names='index',values=column)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig
