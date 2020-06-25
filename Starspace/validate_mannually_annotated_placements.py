import pandas as pd
import streamlit as st
import json

@st.cache()
def get_validation_set_from_json():
    with open('Starspace/datasets/validation_set.json') as json_file:
        validation_set = json.load(json_file)
    print("Succesfully loaded dataset")
    return validation_set

@st.cache()
def get_validation_set_from_csv():
    validation_set_df = pd.read_csv("datasets/job_title_validation_set_v02.csv")
    return validation_set_df

def get_dropped_elements(validation_set,recommendations):
    return [validation_example for validation_example in validation_set if validation_example not in recommendations]

def get_mean_rank(validation_set,recommendations):
    recommendations_rank_dict = {}
    for rank, recommendation in enumerate(recommendations):
        recommendations_rank_dict[recommendation] = rank + 1

    rank_sum = 0.0
    count = 0
    for val_example in validation_set:
        if val_example in recommendations_rank_dict:
            rank_sum = rank_sum + recommendations_rank_dict[val_example]
            count = count + 1
    
    mean_rank = rank_sum/count if rank_sum > 0 else 0

    return mean_rank

def get_validation_ranks(validation_set,recommendations):
    recommendations_rank_dict = {}
    for rank, recommendation in enumerate(recommendations):
        recommendations_rank_dict[recommendation] = rank + 1

    rank_dict = {}
    for val_example in validation_set:
        if val_example in recommendations_rank_dict:
            rank_dict[val_example] = recommendations_rank_dict[val_example]

    return rank_dict

def get_hit_at_k(validation_set,recommendations,k):
    rank_dict = get_validation_ranks(validation_set,recommendations)
    hits = 0
    placements = []
    for val_example in validation_set:
        if val_example in rank_dict:
            if rank_dict[val_example] - 1 < k:
                hits = hits + 1
                placements.append(val_example)
    return hits, placements

    
def main():
    st.header('Validate with manually annotated placments')
    st.subheader('Validation Set')
    st.write('Here we have the annotated placments that we wpuld expect to appear in the recommendation set when looking for their job titles.')

    validation_set_df = get_validation_set_from_csv()

    validation_set_dict = {}
    for idx, row in validation_set_df.iterrows():
        placments = [value for value in row.values if type(value) is type("String") and value.startswith("__placement__")]
        validation_set_dict[row['job_title']] = {
            'placements':[int(placement.split()[0].replace("__placement__","")) for placement in placments],
            'related_keywords':row['job_title_related_keywords']
        }

    st.write(validation_set_df.set_index('job_title'))

    with open("datasets/validation_set.json", "w") as json_file:
        json.dump(validation_set_dict,json_file,indent=2)

if __name__ == "__main__":
    main()