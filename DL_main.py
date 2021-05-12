from flask import Flask, request, jsonify
import json, base64
from PIL import Image
from io import BytesIO

app = Flask(__name__)

@app.route("/data", methods=['POST'])
def Json_receive():
  params = request.get_json()                                               # json 데이터 받기
  with open('pill_image.json', 'w') as pill_file:                           # json 데이터 저장
    json.dump(params, pill_file)
  if(SaveImage(params)):                                                    # 알약 이미지 저장 여부 확인
    print("image write")
    return "Success"
  else:
    print("image write failed")
    return "image write failed"

def SaveImage(imjson):                                                      # 이미지 저장 함수
  try:
    pillImage = Image.open(BytesIO(base64.b64decode(imjson['img_base64']))) # base64데이터를 이미지로 변환
    pillImage.save("pill_image/test.png", 'PNG')
    return True
  except:
    return False
  


if __name__ == '__main__':
  app.run(host="0.0.0.0", port="5000")
