from flask import Flask, request, jsonify
import json, base64
from PIL import Image
from io import BytesIO
import makejson
import detect_text, text_recog, shape_classification

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

@app.route("/data", methods=['POST'])
def get_json():
  # receive json data
  params = request.get_json()

  # check write image 
  if(WriteImage(params)):
    crop_files = detect_text.detect_text_img()
    pill_text = text_recog.img_text_recog(crop_files)
    pill_shape = shape_classification.detect_pill_shape()
    
    if(makejson.make_json_file(pill_shape=pill_shape, pill_text=pill_text)):
      print("make json")
    
    with open('pill_data.json', 'r') as pill_data:
      json_data = json.load(pill_data)
    
    return jsonify(json_data)
  
  else:
    print("image write failed")
    return "image write failed"

# image write
def WriteImage(imjson):
  try:
    # base64 data to image
    pillImage = Image.open(BytesIO(base64.b64decode(imjson['img_base64'])))
    pillImage.save("pill_image/pill_img.png", 'PNG')
    
    return True
  
  except:
    return False

if __name__ == '__main__':
  app.run(host="0.0.0.0", port="5000")
