from flask import Flask, request, jsonify
import json, base64
from PIL import Image
from io import BytesIO

app = Flask(__name__)

@app.route("/data", methods=['POST'])
def Json_receive():
  params = request.get_json() # receive json data
  with open('pill_image.json', 'w') as pill_file: # write json file
    json.dump(params, pill_file)
  if(SaveImage(params)):
    print("image write")
    return "Success"
  else:
    print("image write failed")
    return "image write failed"

def SaveImage(imjson):
  try:
    pillImage = Image.open(BytesIO(base64.b64decode(imjson['img_pill'])))
    pillImage.save("pill_image/test.png", 'PNG')
    return True
  except:
    return False
  


if __name__ == '__main__':
  app.run(host="0.0.0.0", port="5000")
