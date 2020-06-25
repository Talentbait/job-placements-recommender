# -*- coding: utf-8 -*-
import json
import random
from tqdm import tqdm
import numpy as np
import re
import spacy
import os, sys
from pathlib import Path
from spacy.lang.de.stop_words import STOP_WORDS as de_stop
from spacy.lang.en.stop_words import STOP_WORDS as en_stop
import spacy

print("Entered")

# de_stop = []
# with open('de_stop.txt','r') as textfile:
#     for line in textfile:
#         de_stop.append(line.replace('\n',''))
# stop_words = list(de_stop) + list(en_stop)
# # placements_and_info = pd.read_csv('complete_datasets/cleaned_placements_dataset_landetect.csv',index_col=0)
# stop_words_regex = re.compile(r'(?i)' + r'\b%s\b' % r'\b|\b'.join(map(re.escape, stop_words)))

def clean_description(description):
    replacements = {}
    description, replacements['space_truncated'] = re.subn(r'\s+', ' ', description)         # Removing all spaces
    replacements['total_characters'] = len(description) - len(re.findall(r'\s',description))    # Counting total characters minus the white characters
    replacements['pre_preprocessing_words'] = len(description.split())                          # Counting words (separated by blank characters)
    replacements['uppercase_letters'] = len(re.findall(r'[A-ZÖÄÜß]',description))               # Count characters that are uppercase
    replacements['uppercase_first_letter_words'] = len(re.findall(r"\b[A-ZÖÄÜ][a-zöäüß]\S*\b",description))     # Count words that contain a capital letter at the beginning
    replacements['uppercase_loss_ratio'] = replacements['uppercase_first_letter_words']/replacements['pre_preprocessing_words'] if replacements['pre_preprocessing_words'] > 0 else 0
    replacements['uppercase_words'] = len(re.findall(r"\b\S*[A-ZÖÄÜ]\S*\b",description))        # Count words that contain at leas one capital letter
    replacements['uppercase_complete_words'] = len(re.findall(r"\b[A-ZÖÄÜ]+\b",description))    # Count words that are completely uppercased
    replacements['acronyms'] = len(re.findall(r"\b[A-ZÖÄÜ\.]+\b",description))                  # Count acronyms A.A.A.
    description, replacements['impressum'] = re.subn(r"Impressum.*","",description)
    # description = description.lower()
    description, replacements['email'] = re.subn(r"\S+@\S+\.\S+", "EMAIL", description)   #Removing emails
    description, replacements['url'] = re.subn(r"uni    (\S+)?[a-zA-Z0-9]+\.(com|org|de|net|ly|tv)(\.[a-z]{2,3})?(\S+)?", "URL", description)   #Removing urls
    description, added_url_replacements = re.subn(r"(?i)http\S+", "URL", description)   #Removing urls
    replacements['url'] = replacements['url'] + added_url_replacements
    # description = re.sub(r"\S+__\S+", "", description)  #Removing double undescores
    # description, replacements['weird_characters'] = re.subn(r'[^a-zA-Z0-9äöüß\s]', '', description)
    description, replacements['weird_characters'] = re.subn(r"(?i)[^a-z0-9äöüß\s?!.,\"#$%'()*+\-/:;<=>@[\\\]^_`{}~]", '', description)
    description, replacements['removed_decimal_separators'] = re.subn(r"[0-9][\.,][0-9]","",description)
    description, replacements['nummer'] = re.subn(r"[0-9]+"," NUMMER ",description)
    description = re.sub(r"\s+"," ",description)
    description = re.sub(r'( NUMMER)+'," NUMMER",description)
    # description, replacements['nummer'] = re.subn(r"[0-9]+","",description) 
    description, replacements['consecutive_nonalpha_characters'] = re.subn(r"[?!,\"#$%'()*+\-/:;<=>@[\\\]^_`{|}~▬]{2,}", '',description)
    description, replacements['consecutive_punctuation_characters'] = re.subn(r"([\"?!\.,\-])\1+", r' \1 ',description)
    description, replacements['spaced_punctuation_characters'] = re.subn(r"([\"?!\.,])", r' \1 ',description)
    description, replacements['spaced_symbols'] = re.subn(r"[\"#$%'()*+/:;<=>@[\\\]^_`{|}~]", r" \g<0> ",description)
    # description, replacements['stop_words'] = re.subn(stop_words_regex,'',description) 
    # replacements['stop_words'] = len(re.findall(stop_words_regex,description.lower()))
    description = re.sub(r'\s+', ' ', description)
    replacements['removed_characters'] = replacements['total_characters'] - len(description) + len(re.findall(r'\s',description))
    replacements['post_preprocessing_words'] = len(description.split()) if description else 0
    replacements['single_characters'] = len(re.findall(r"(?i)\W\S\W",description))
    replacements['character_count_cleaned'] = len(description)
    replacements['cleaned_description'] = description
    description = re.sub(r'\s+', ' ', description)         # Removing all spaces
    # description_spacy = nlp(description)
    return description, replacements

def clean_app_name(name):
    name = name.replace('Mobile App: ','')
    name = name.replace('mobileapp::','')
    return name




def main():
    # from spacy.lang.de.stop_words import STOP_WORDS as de_stop
    # with open('de_stop.txt','w') as output:
    #     for word in list(de_stop):
    #         output.write(str(word) + '\n')
    train_file_path = "Starspace/datasets/" + (sys.argv[1] if len(sys.argv) > 1 else "tab_separated_descriptions_spaced_v02_extended.txt")
    basedoc_path = "Starspace/datasets/" + (sys.argv[2] if len(sys.argv) > 2 else "complete_descriptions_spaced_v02_extended.txt")
    labels_file_path = "Starspace/datasets/" + (sys.argv[3] if len(sys.argv) > 3 else "labels_for_tab_separated_descriptions_spaced_v02_extended.txt")
    replacement_count_current_file = sys.argv[4] if len(sys.argv) > 4 else 'format_data/replacement_counts_tab_separated_spaced_extended.json'

    nlp = spacy.load('de_core_news_sm')
    print('de_core_news_sm loaded')

    unique_placements_dict = {}
    with open("unique_placements_updated.json",'r') as json_file:
        unique_placements_dict = json.load(json_file)

    replacement_counts = {}
    unique_placements_dict_cleaned_info = {}
    replacements_total = {
        'count':0
    }
    missing = []
    with open(train_file_path,'w') as output, open(replacement_count_current_file,'w') as count_file, open(labels_file_path,'w') as label_output, open(basedoc_path,'w') as complete_file:
        for _, placement in tqdm(enumerate(unique_placements_dict.keys()),desc='Generating dataset unsplitted'):
            description = unique_placements_dict[placement]['description']
            placement_id = unique_placements_dict[placement]['id']
            placement_name = unique_placements_dict[placement]['name']
            if 'language' in unique_placements_dict[placement]:
                language = unique_placements_dict[placement]['language']
                if language == 'de':# and unique_placements_dict[placement]['type'] != 'Mobile application':
                    # cleaned_description, replacements = clean_description(description)
                    # if replacements['post_preprocessing_words'] > 5:# and replacements['post_preprocessing_words'] < 190:
                    line = nlp(description)
                    cleaned_sentences = [sent.string.strip() for sent in line.sents if len(sent.string.strip()) > 10]
                    # cleaned_sentences = [sentence for sentence in cleaned_sentences if len(sentence) > 10]
                    cleaned_sentences = [clean_description(sentence)[0] for sentence in cleaned_sentences]
                    if len(cleaned_sentences) > 1:
                        # line = line + " __placement__" + str(placement_id)
                        label_output.write("__placement__" + str(placement_id) + '\n')
                        # label_output.write("__placement__" + placement_name.replace(" ","_") + " " + cleaned_description + "\n")
                        output.write("\t".join(cleaned_sentences) + "\n")
                        cleaned_description, replacements = clean_description(description)
                        complete_file.write(cleaned_description + "\n")
                        # output.write(line + "\n")
                        replacement_counts[placement] = {
                            'name': unique_placements_dict[placement]['name'],
                            'id': unique_placements_dict[placement]['id'],
                            'description': description,
                            'cleaned': cleaned_description,
                            'language': language,
                            'replacements':replacements
                        }
                        unique_placements_dict_cleaned_info[placement] = unique_placements_dict[placement]
                        unique_placements_dict_cleaned_info[placement].update(replacements)
                        replacements_total['count'] = replacements_total['count'] + 1
                        for label in replacements:
                            if label != 'cleaned_description':
                                if label in replacements_total:
                                    replacements_total[label] = replacements_total[label] + replacements[label]
                                else:
                                    replacements_total[label] = replacements[label]
                else:
                    missing.append(placement_id)
        json.dump(unique_placements_dict_cleaned_info,count_file,indent=2)
        print('Done saving replacement_count file in ' + str(replacement_count_current_file))
        print('Done saving train file in ' + str(train_file_path))

    print("\n".join([str(a) for a in missing]))
    print(json.dumps(replacements_total,indent=2))

    
    total_characters = replacements_total['total_characters']
    removed_characters = replacements_total['removed_characters']
    remaining_corpora = removed_characters/total_characters
    total_words = replacements_total['post_preprocessing_words']
    initial_words = replacements_total['pre_preprocessing_words']
    single_character_words = replacements_total['single_characters']
    non_single_character_words = total_words - initial_words - single_character_words

    print(f"\nRemoved characters: {round(remaining_corpora*100,2)}%.")
    print(f"Added words: {total_words - initial_words} ({round(total_words/initial_words*100-100,2)}%).")
    print(f"Single character words: {single_character_words} ({round(single_character_words/total_words*100,2)}%).")
    print(f"Added non single character words: {non_single_character_words} ({round(non_single_character_words/total_words*100,2)}%).")
    for label in ['url','nummer','email','uppercase_first_letter_words','uppercase_words','uppercase_complete_words','acronyms','stop_words']:
        print(f"{label}: {replacements_total[label]} ({round(replacements_total[label]/initial_words*100,2)}%).")

if __name__ == "__main__":
    main()