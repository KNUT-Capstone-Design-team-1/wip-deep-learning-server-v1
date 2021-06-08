import json

def make_json_file(pill_shape, pill_text, pill_line=False):
    drugData = {
        "drug_name" : "",
        "drug_type" : "",
        "drug_shape" : "",
        "drug_color" : "",
        "drug_line" : "없음"
    }
    #drugData['drug_shape'] = pill_shape
    drugData["drug_name"] = pill_text
    # drugData["drug_line"] = pill_line
    try:
        with open('pill_data.json', 'w') as pill_data:
            json.dump(drugData, pill_data, indent='\t')
        return True
    except:
        return False
