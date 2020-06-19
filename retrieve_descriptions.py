import json

with open('unique_placements.json') as json_file:
    unique_placements = json.load(json_file)

ids_to_placement_dict = {}
for k, v in unique_placements.items():
    if 'language' in v:
        if v['language'] == 'de':
            ids_to_placement_dict[v['id']] = {
                'name':v['name'],
                'type':v['type'],
                'url':k,
                'description':v['description'],
                'language':v['language']
            }

with open('placements_to_add_a_sentence.txt') as document, open('placements_with_description_and_url.csv','w') as output:
    output.write('id,current_description,keywords,url\n')
    for line in document:
        line = line.replace('\n','')
        if line[0] in '123456789':
            output.write(line + ',"' + ids_to_placement_dict[int(line)]['description'] + '","')
            output.write(ids_to_placement_dict[int(line)]['keywords'] if 'keywords' in ids_to_placement_dict[int(line)] else '')
            output.write('",' + ids_to_placement_dict[int(line)]['url'] + '\n')