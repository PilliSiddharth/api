from flask import Flask, jsonify, request

from fer import FER
import matplotlib.pyplot as plt
import requests
import io


app = Flask(__name__)


@app.route('/get_data/',methods=['POST'])
def return_emotion():

    req = request.get_json()
    img_link = req['url']

    response = requests.get(img_link).content
    print(response)
    img = plt.imread(io.BytesIO(response), format='JPG')

    detector = FER(mtcnn=True)
    emotion, score = detector.top_emotion(img)
    print(emotion)

    return emotion

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)