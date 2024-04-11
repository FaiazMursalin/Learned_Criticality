import random
from itertools import combinations
import xml.etree.ElementTree as ET
base_path = "/home/faiaz/Documents/msu_courses/semester2/Machine_Learning/pmpl/src/Examples"
filepath = f"{base_path}/BasicPRM1.12345678.map"

points = []
with open(filepath, "r") as inf:
    lines = inf.readlines()[5:-1]
    for line in lines:
        # if line.startswith("#") or len(line.strip()) == 0 or line == "\n":
        #     continue
        # else:
        line_data = line.split()
        point = line_data[2:4]
        point = (float(point[0]), float(point[1]))
        points.append({"x": point[0], "y": point[1]})
        # print(line.split()[2:4])


random.shuffle(points)
combs = list(combinations(points, 2))

# print(len(combs))


xml_filepath = f"{base_path}/CfgExamples.xml"
# Reading XML file
def xml_to_dict(element):
    result = {}
    if element.attrib:
        result.update(element.attrib)
    if element.text:
        result["_text"] = element.text.strip()
    for child in element:
        child_result = xml_to_dict(child)
        if child.tag in result:
            if not isinstance(result[child.tag], list):
                result[child.tag] = [result[child.tag]]
            result[child.tag].append(child_result)
        else:
            result[child.tag] = child_result
    return result

def parse_xml_to_dict(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    return {root.tag: xml_to_dict(root)}

def dict_to_xml(element_name, dictionary):
    element = ET.Element(element_name)
    for key, value in dictionary.items():
        if key == '_text':
            element.text = value
        elif isinstance(value, dict):
            sub_element = dict_to_xml(key, value)
            element.append(sub_element)
        elif isinstance(value, list):
            for item in value:
                sub_element = dict_to_xml(key, item)
                element.append(sub_element)
        else:
            element.set(key, str(value))
    return element

def write_dict_to_xml(dictionary, file_path):
    root_element_name = list(dictionary.keys())[0]
    root_element = dict_to_xml(root_element_name, dictionary[root_element_name])
    tree = ET.ElementTree(root_element)
    tree.write(file_path, encoding='utf-8', xml_declaration=True)

xml_dict = parse_xml_to_dict(xml_filepath)

for i, points in enumerate(random.sample(combs, 200)):
    xml_dict_copy = xml_dict.copy()
    xml_dict_copy['MotionPlanning']['Library']['Solver']['baseFilename'] = f"BasicPRM_ZigZag_{i}"
    xml_dict_copy['MotionPlanning']['Problem']['Task']['StartConstraints']['CSpaceConstraint']['point'] = f"{round(points[0]['x'], 2)} {round(points[0]['y'], 2)}"
    xml_dict_copy['MotionPlanning']['Problem']['Task']['GoalConstraints']['CSpaceConstraint'][
        'point'] = f"{round(points[1]['x'], 2)} {round(points[1]['y'], 2)}"
    write_dict_to_xml(xml_dict_copy, f"{base_path}/Cfg_Gen_ZigZag_{i}.xml")

# # Writing XML file
# new_tree = ET.ElementTree(root)
# new_tree.write('new_example.xml')
