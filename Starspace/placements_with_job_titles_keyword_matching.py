import json
import pandas as pd
import re
from tqdm import tqdm

unique_placements_dict = {}
with open('unique_placements.json') as json_file:
    unique_placements_dict = json.load(json_file)
print("Succesfully loaded dataset")

unique_placements = pd.read_json('unique_placements.json',orient='index')

keywords_df = pd.read_csv('datasets/keywords_for_new_job_titles.csv',low_memory=False)
job_title_keywords = {}
for idx, row in keywords_df.iterrows():
    job_title_keywords[row['job_title']] = re.compile(r'(?i)' + r'%s' % r'|'.join(map(re.escape, row['keywords'].split(', '))))
print("Succesfully loaded keywords for job titles")

with open('datasets/placements_with_keyword_matching_with_new_jobs.csv','w') as output: 
    output.write(",".join(list(unique_placements.columns)) + ',' + ",".join(list(job_title_keywords.keys())) + '\n')
    for idx, placement in tqdm(unique_placements.iterrows(),desc="Calculating placement matches"):
        if placement['language'] == 'de':
            text_from_placement = str(placement['description']) + str(placement['name']) + str(placement['keywords'])
            output.write(",".join(['"' + str(a).replace('"',"'") + '"' for a in placement.values.tolist()]))
            for job_title, keywords in job_title_keywords.items():
                output.write("," + str(len(re.findall(keywords,text_from_placement))))
            output.write("\n")

# with open('datasets/placements_with_keyword_matching_faster.csv','w') as output: 
#     output.write(",".join(list(unique_placements.columns)) + ',' + ",".join(list(job_title_keywords.keys())) + '\n')
#     for k, v in tqdm(unique_placements.items(),desc="Calculating placement matches"):
#         for key in ['language','keywords']
#             unique_placements_dict[k]['language'] = '' if 'language' not in placement
#         if placement['language'] == 'de':
#             text_from_placement = str(placement['description']) + str(placement['name']) + str(placement['keywords'])
#             output.write(",".join(['"' + str(a).replace('"',"'") + '"' for a in placement.values.tolist()]))
#             for job_title, keywords in job_title_keywords.items():
#                 output.write("," + str(len(re.findall(keywords,text_from_placement))))
#             output.write("\n")