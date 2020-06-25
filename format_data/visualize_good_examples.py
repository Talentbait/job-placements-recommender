import json
import io
import pandas as pd
import streamlit as st

st.title("Visualize useful placements")
st.subheader("Possible placements that could be in one of our 11 current jobtitles")

st.subheader("BAD EXAMPLES")
placements = {}
with io.open("bad_examples.jsonl") as input_file:
    for line in input_file:
        placement = json.loads(line)
        placement_id = placement['id']
        placements[placement_id] = {
            'name': placement['name'],
            'description':placement['description'],
            'score': placement['v02']['useful'],
            'type': placement['type'],
            'classified': placement.get('classified',False),
            'label': placement.get('label','')
        }

data = pd.DataFrame.from_dict(placements,orient="columns").T
unlabeled_data = data[data['label']==""]
st.write(data)
st.write(data['type'].value_counts())


st.subheader("GOOD EXAMPLES")
placements = {}
with io.open("good_examples.jsonl") as input_file:
    for line in input_file:
        placement = json.loads(line)
        placement_id = placement['id']
        placements[placement_id] = {
            'name': placement['name'],
            'description':placement['description'],
            'score': placement['v02']['useful'],
            'type': placement['type'],
            'classified': placement.get('classified',False),
            'label': placement.get('label','')
        }

data = pd.DataFrame.from_dict(placements,orient="columns").T
unlabeled_data = data[data['label']==""]
st.write(data)
st.write(data['type'].value_counts())

# st.write(list(placements.keys()))

# with io.open("../placement_pipeline_status_v01_01.json") as json_file, io.open("../placement_pipeline_status_v01_02.json","w",encoding="UTF-8") as output:
#     all_placements = json.load(json_file)
#     for good_id in placements.keys():
#         all_placements[good_id]['good_example'] = True

#     json.dump(all_placements,output,ensure_ascii=False,indent=4)