import json
from format_train_file import clean_description

labeled_data = []
with open('prodigy_output_example.json') as data_path:
    data = json.load(data_path)
    labeled_data = data['labeled_data']

with open('fasttext_format.txt','w') as output:
    for example in labeled_data:
        answer = example['answer']
        # print(answer)
        if answer == 'accept':
            label = example['accept'][0]
            output.write(clean_description(example['text'])[0] + " __jobtitle__" + str(label) + "\n")
        elif answer == 'reject':
            output.write(clean_description(example['text'])[0] + " __jobtitle__None\n")