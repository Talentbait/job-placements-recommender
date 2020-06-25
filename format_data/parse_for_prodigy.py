import json
import io
import random

with io.open("../placement_pipeline_status_v01_01.json", "r") as input_file:
    placement_pipeline_status = json.load(input_file)

placement_ids = {}

not_classified_count = 0

for placement_id, placement_detail in placement_pipeline_status.items():
    placement_id = str(placement_id)
    if 'good_example' in placement_detail:
        if "classified" not in placement_detail:
            not_classified_count += 1
            scores = sorted(placement_detail["v01"].items(
            ), key=lambda x: x[1], reverse=True)

            if not placement_ids.get(placement_id, ""):
                placement_ids[placement_id] = {
                    "jobTitle": scores[0][0],
                    "score": scores[0][1]
                }

            elif placement_ids.get(placement_id, "") and placement_ids[placement_id]["jobTitle"] != scores[0][0] and placement_ids[placement_id]["score"] < scores[0][1]:
                placement_ids[placement_id] = {
                    "jobTitle": scores[0][0],
                    "score": scores[0][1]
                }

top_200_Erzieher = list(map(lambda x: x[0], sorted(
    filter(lambda x: x[1]["jobTitle"] == "Erzieher", placement_ids.items()),
    key=lambda x: x[1]["score"], reverse=True
)))[:200]

top_200_vertrieb = list(map(lambda x: x[0], sorted(
    filter(lambda x: x[1]["jobTitle"] ==
           "Vertriebsmitarbeiter", placement_ids.items()),
    key=lambda x: x[1]["score"], reverse=True
)))[:200]

top_200_electro = list(map(lambda x: x[0], sorted(
    filter(lambda x: x[1]["jobTitle"] ==
           "Elektrotechniker", placement_ids.items()),
    key=lambda x: x[1]["score"], reverse=True
)))[:200]


print(not_classified_count)
print(len(set(top_200_Erzieher).intersection(
    set(top_200_vertrieb)).intersection(top_200_electro)))
# print(top_200_Erzieher)

output = []
for placement_id in top_200_electro + top_200_Erzieher + top_200_vertrieb:
    place_dets = placement_pipeline_status[placement_id]
    scoring = sorted(place_dets["v01"].items(),
                     key=lambda x: x[1], reverse=True)
    to_append = {
        "label": place_dets["name"],
        "text": place_dets["description"],
        "meta": {
            "url": place_dets["url"],
            "score": scoring[0][1],
            "id": placement_id,
            "job_title": scoring[0][0]
        },
        "choice_auto_accept": True,
        "options": [
            {"id": "Vertriebsmitarbeiter", "text": "\ud83d\udcb5 Vertriebsmitarbeiter"}, 
            {"id": "Erzieher", "text": "\ud83d\udc76 Erzieher"}, 
            {"id": "Elektrotechniker", "text": "\ud83d\udd0c Elektrotechniker"},
            {"id": "Other", "text": "Other job titles"}
        ]
    }
    if scoring[0][0] == "Vertriebsmitarbeiter":
        to_append["options"][0]["style"] = {"background": "#a6f7ab"}
    elif scoring[0][0] == "Erzieher":
        to_append["options"][1]["style"] = {"background": "#a6f7ab"}
    elif scoring[0][0] == "Elektrotechniker":
        to_append["options"][2]["style"] = {"background": "#a6f7ab"}

    output.append(json.dumps(to_append)+"\n")

with io.open('./output.jsonl', mode='w') as output_file:
    random.shuffle(output)
    output_file.writelines(output)
    # writer.write_all(output)
# with io.open("./output.json", "w") as output_file:
#     json.dump(output, output_file)
