import json
import io
import streamlit as st
from placement_to_jobtitle_classifier import PythonPredictor as classifier

st.title("Classify all placements to a jobtitle")

placement_pipeline_status = {}
with io.open('../placement_pipeline_status.json') as file_path:
    placement_pipeline_status = json.load(file_path)

def hash_for_cortex(a):
    return 1

@st.cache(hash_funcs={classifier: hash_for_cortex})
def load_classifier():
    return classifier(0)

model = load_classifier()

# for placement_id, placement_info in placement_pipeline_status.items():
#     description = placement_info['description']
#     labels = model.predict(description)['labels']
#     placement_pipeline_status[placement_id]['v01'] = {}
#     for label in labels:
#         placement_pipeline_status[placement_id]['v01'][label['label']] = label['score']
    

# with io.open('test.json') as file_path:
#     json.dump(placement_pipeline_status,file_path,indent=2)

labels_predicted = model.predict("Eine Schule, fünf neue Referendare, aber nur zwei freie Lehrerstellen. Bei „Krass Schule - Die jungen Lehrer“ kämpfen hoch motivierte Lehramtsanwärter um ihren persönlichen Traumjob. Doch abseits von Lehrprobe und Unterrichtsvorbereitung erleben die jungen Referendare hinter den Mauern dieser Schule die härteste Zeit ihres Ausbildungsdienstes. Gesponnene Intrigen, schwierige Schüler und verbotene Liebschaften verlangen ihnen alles ab. Der Druck ist hoch - und manche drohen daran zu zerbrechen.\n\nhttp://www.RTLZWEI.de/impressum")['labels']

st.write(labels_predicted)