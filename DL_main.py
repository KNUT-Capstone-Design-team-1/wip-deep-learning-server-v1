from flask import Flask, request, jsonify
import json, base64
from PIL import Image
from io import BytesIO

app = Flask(__name__)

@app.route("/data", methods=['POST'])
def GetJson():
  params = request.get_json()                                               # receive json data
  with open('pill_image.json', 'w') as pill_file:                           # write json data
    json.dump(params, pill_file)
  if(WriteImage(params)):                                                   # check write image 
    print("image write")
    return "Success"
  else:
    print("image write failed")
    return "image write failed"

def WriteImage(imjson):                                                     # image write
  try:
    pillImage = Image.open(BytesIO(base64.b64decode(imjson['img_base64']))) # base64 data to image
    pillImage.save("pill_image/test.png", 'PNG')
    return True
  except:
    return False
  
@app.route("/pilldata", methods=['GET'])
def SendJson():
  with open('pill_image.json', 'r') as pill_json:
    jsonData = json.load(pill_json)
    return jsonify(jsonData)

if __name__ == '__main__':
  app.run(host="0.0.0.0", port="5000")
