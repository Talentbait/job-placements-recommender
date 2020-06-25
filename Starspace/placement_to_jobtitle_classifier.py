import pexpect
from subprocess import Popen, PIPE
import os
import json
import io
import re
import subprocess

required_files = ["models/placement_to_jobtitle_classifier_normalized_examples_3", "models/trainMode0_v01"]

class PythonPredictor:
    def __init__(self,config):

        child = pexpect.spawn(f"./query_predict {required_files[0]} 4",timeout=400)
        child.expect("STARSPACE-2018-2")
        print("Loading model...")
        child.expect(f"Enter some text:")
        print(child.before.decode("utf-8"))
        print(child.after.decode("utf-8"))
        self.child = child
        print("Succesfully initialized model")

    def predict(self,payload):
        description = payload
        self.child.send(description + '\n')
        self.child.expect("Enter some text:")

        output = self.child.before.decode("utf-8")
        # print(output)
        labels_predictions = self.parse_output(output)

        return {"labels": output}

    def parse_output(self,output_str):
        labels_predictions = []
        labels_predictions_lines = output_str.split("\r\n")

        for labels_predictions_line in labels_predictions_lines:
            label_tag = "__jobtitle__" #"__label__"
            if label_tag in labels_predictions_line:
                label = labels_predictions_line.split(label_tag)[1]
                # placement_detail = self.placement_mapping.get(placement_id, "")
                # score = re.search(r"\[([0-9.]+)\]",labels_predictions_line).group(1)
                # if placement_detail:
                #     labels_predictions.append({
                #         "url": placement_detail["url"],
                #         "name": placement_detail["name"],
                #         "type": placement_detail["type"],
                #         "score": score
                #     })
                # labels_predictions.append({
                #     'label':label,
                #     'score':score
                # })
                labels_predictions.append(label)

        return labels_predictions