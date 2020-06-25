import io
import re
import json
from format_train_file import clean_description
import numpy as np

# # Parse the output of query_predict to placement_pipeline_status
# with io.open("output.txt") as input_file, \
#     io.open("../placement_pipeline_status_v01.json") as placements_file, \
#     io.open("../placement_pipeline_status_v01_01.json",'w',encoding="UTF-8") as output, \
#     io.open("good_examples_descriptions.txt","w") as basedoc, \
#     io.open("labels_good_examples_descriptions.txt","w") as basedoc_labels:
#     predictions = input_file.read().split("\n\n")
#     placements = json.load(placements_file)
#     for prediction_idx, placement_id in enumerate(placements.keys()):
#         pred_dict = {}
#         for line in predictions[prediction_idx].split("\n"):
#             if "__label__" in line:
#                 label = line.split("__label__")[1].replace(" ","")
#                 try:
#                     score = float(re.search(r"\[([-0-9\.e]+)\]",line).group(1))
#                 except:
#                     print(line)
#                 pred_dict[label] = score
#         placements[placement_id]['v01_01'] = pred_dict
#         if max(pred_dict, key=pred_dict.get) == 'useful':
#             placements[placement_id]['id'] = placement_id
#             basedoc.write(clean_description(placements[placement_id]['description'])[0] + "\n")
#             basedoc_labels.write("__placement__" + placement_id + "\n")
#             # json_output.write(json.dumps(placements[placement_id]) + "\n")



#     # # Add the validated label to the placements that had it missing
#     # for placement_id, placement_info in placements.items():
#     #     if 'classified' in placement_info and 'prodigy' in placement_info:
#     #         if placement_info['classified'] and placement_info['prodigy']:
#     #             placements[placement_id]['label'] = list(placement_info['v00'].keys())[0]

#     json.dump(placements,output,indent=2,ensure_ascii=False)

# # Output sample descriptions to classify
# with  io.open("../placement_pipeline_status_v00.json") as placements_file, io.open("../Starspace/datasets/all_descriptions_input.txt",'w') as output:
#     placements = json.load(placements_file)
#     for placement_info in placements.values():
#         description = clean_description(placement_info['description'])[0]
#         # if len(description):
#         output.write(description + "\n")


