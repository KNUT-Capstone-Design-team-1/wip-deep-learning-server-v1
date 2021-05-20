import requests, json

#주소에 알약 검색 서버 주소 json에 json파일
#POST 형식으로 JSON데이터 전송
def SendJson():
    try:
        with open('pill_data.json', 'r') as pill_json:
            jsonData = json.load(pill_json)
            res = requests.post('http://3.37.82.154:8080/ml', json=jsonData)
        return True
    except:
        return False

def MakeJson(class_data, ocr_data=None):
    #if class_data[0] == 0: drugShape = '원형'
    #else: drugShape = '타원형'
    drugData = {
        "drug_name" : "ID",
        "drug_type" : "정제",
        "drug_shape" : "원형",
        "drug_color" : "하양",
        "drug_line" : "없음"
    }
    #drugData['drug_shape'] = drugShape
    try:
        with open('pill_data.json', 'w') as pill_data:
            json.dump(drugData, pill_data, indent='\t')
        return True
    except:
        return False
