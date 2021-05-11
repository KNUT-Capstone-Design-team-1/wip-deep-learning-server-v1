from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/data", methods=['POST'])
def Json_receive():
  params = request.get_json()
  for key in params.keys():
    print(key)
  return params.keys()



if __name__ == '__main__':
  app.run(host="0.0.0.0", port="5000")
