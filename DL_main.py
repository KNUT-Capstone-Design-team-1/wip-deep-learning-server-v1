from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/data", methods=['GET', 'POST'])
def Json_recive():
  pill_image = request.get_json() #json데이터 받는 코드
  return jsonify(pill_image)



if __name__ == '__main__':
  app.run(host="0.0.0.0", port="5000")
