import numpy as np
import json

# unique_placements = {}
# with open('unique_placements.json') as json_file:
#     unique_placements = json.load(json_file)
#     print("Succesfully loaded dataset")

# ids_to_placement_dict = {}
# placements_labels = []
# for k, v in unique_placements.items():
#     if 'language' in v:
#         if v['language'] == 'de':
#             ids_to_placement_dict[v['id']] = {
#                 'name':v['name'],
#                 'type':v['type'],
#                 'url':k,
#                 'description':v['description'],
#                 'language':v['language']
#             }
#             placements_labels.append('__placement__' + str(v['id']))

labels = "__label__Erzieher __label__Erzieherinnen __label__Kita-Erzieher __label__Pädagogen __label__Erziehern __label__Kindererzieher __label__Kinderpfleger __label__Erzieherin __label__Kindergärtner __label__Heilerzieher __label__Kindergartenpädagogen __label__Lehrer __label__Arbeitserzieher __label__ErzieherInnen __label__Kita-Erzieherinnen __label__Sozialpädagogen __label__Horterzieher __label__Erziehers __label__Sozialarbeiter __label__Kindergärtnerinnen __label__Vertriebsmitarbeiter __label__Außendienstmitarbeiter __label__Vetriebsmitarbeiter __label__Vertriebsmitarbeitern __label__Vertriebsmitarbeiterin __label__Vertriebsingenieure __label__Vertriebsprofi __label__Vertriebsmitarbeiters __label__Außendienst __label__Vertriebler __label__Vertriebsspezialisten __label__Kundenbetreuer __label__Vertriebsingenieur __label__Vertriebsleiter __label__Außendienst-Mitarbeiter __label__Anwendungstechniker __label__Vertriebsmanager __label__Vertriebsassistenten __label__Kundenberater __label__Sales-Mitarbeiter __label__Elektrotechniker __label__Elektroingenieur __label__Nachrichtentechniker __label__Elektrotechnik-Ingenieur __label__Elektriker __label__Elektrotechnikingenieur __label__Energietechniker __label__Elektromechaniker __label__Elektrotechnikerin __label__Elektro-Ingenieur __label__Informationstechniker __label__Ingenieur __label__Elektrotechnik __label__Elektrotechnikermeister __label__Elektrobetriebstechniker __label__Elektronikingenieur __label__Elektrotechnikern __label__Elektroniker __label__elektrotechniker __label__Elektroingenieure __label__None0 __label__None1 __label__None2 __label__None3 __label__None4 __label__None5 __label__None6 __label__None7 __label__None8 __label__None9 __label__None10 __label__None11 __label__None12 __label__None13 __label__None14 __label__None15 __label__None16 __label__None17 __label__None18 __label__None19"

with open('Starspace/dummy_vectors.tsv', 'w') as output_file:
    for label in labels.split():
        dummy = np.random.normal(loc=0, scale=0.001, size=300)
        output_file.write(label.replace("\n","") + '\t' + "\t".join(["%.6f" % a for a in dummy]) + '\n')
