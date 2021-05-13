import requests, json

#주소에 알약 검색 서버 주소 json에 json파일
#POST 형식으로 JSON데이터 전송
def __init__(self):
    return None
def SendJson():
    try:
        with open('pill_image.json', 'r') as pill_json:
            jsonData = json.load(pill_json)
            res = requests.post('http://52.79.235.20:5000/test', json=jsonData)
        return True
    except:
        return False
