from flask import Flask, request, jsonify
import json

app = Flask(__name__)

@app.route("/data", methods=['POST'])
def Json_receive():
  params = request.get_json() #json데이터를 받는다.
  with open('pill_image.json', 'w') as make_file:
    json.dump(params, make_file, ensure_ascii=False, inden="\t")
  print("test")
  return "test"



if __name__ == '__main__':
  app.run(host="0.0.0.0", port="5000")
