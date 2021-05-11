from flask import Flask, request

app = Flask(__name__)

@app.route("/data", methods=['GET', 'POST'])
def Json_recive():
  Pill_image = request.get_json() #json데이터 받는 코드
  return "test"



if __name__ == '__main__':
  app.run(host="0.0.0.0", port="5000")
