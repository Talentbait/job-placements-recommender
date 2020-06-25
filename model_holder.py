import pexpect
import os
import json
import io
import re

required_files = ["models/trainMode2_light_test", "datasets/tab_separated_descriptions_spaced_v02_extended.txt", "datasets/labels_for_tab_separated_descriptions_spaced_v02_extended.txt", "others/unique_placements_with_german_descriptions.json"]

class PythonPredictor:
    def __init__(self,config):
        print("Loading placement mapping")
        with io.open("placement_pipeline_status_v01.json", "r", encoding="UTF-8") as input_file:
            self.placement_mapping = json.load(input_file)

        child = pexpect.spawn("Starspace/query_predict_placement_id Starspace/models/jobtitle_classifier_v02_1 9000 Starspace/datasets/tab_separated_descriptions_spaced_v02_extended.txt Starspace/datasets/labels_for_tab_separated_descriptions_spaced_v02_extended.txt 9086",timeout=450)
        # child = pexpect.spawn("query_predict models/trainMode0_v01 5",timeout=80)
        child.expect("STARSPACE-2018-2")
        print("Loading model...")

        child.expect(f"Enter some text:")
        self.child = child
        print("Succesfully initialized model")

    def predict(self,payload):
        job_title = str(payload.get("jobTitle", ""))
        print(f"Getting predictions for {job_title}")

        if len(job_title) > 950:
            print("Text is to long")
            return ""

        if not job_title:
            print(f"No job title found")
            return ""
        
        self.child.send(job_title+"\n")
        self.child.expect(f"Enter")

        output = self.child.before.decode("utf-8")
        placement_predictions = self.parse_output(output)

        return {"placements": placement_predictions}

    def parse_output(self,output_str):
        placement_predictions = []
        placement_prediction_lines = output_str.split("\r\n")

        for placement_prediction_line in placement_prediction_lines:
            if "__placement__" in placement_prediction_line:
                placement_id = placement_prediction_line.split("__placement__")[1]
                placement_detail = self.placement_mapping.get(placement_id, "")
                score = re.search(r"\[([0-9.]+)\]",placement_prediction_line).group(1)
                if placement_detail:
                    placement_predictions.append({
                        "id": placement_id,
                        "name": placement_detail["name"],
                        "type": placement_detail["type"],
                        "description": placement_detail["description"],
                        "score": score
                    })

        return placement_predictions