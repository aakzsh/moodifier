from flask import Flask, render_template, request
import cv2
import numpy as np
from keras.models import load_model
import os

port = int(os.environ.get('PORT', 5000))

UPLOAD_FOLDER = 'uploadFile'
app = Flask(__name__)
app.config["DEBUG"] = True
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def after():

    if request.method == 'POST':
        img = request.files['file1']
        img.save('static2/file.jpg')

        image = cv2.imread('static2/file.jpg', 0)

        image = cv2.resize(image, (48, 48))

        image = np.reshape(image, (1, 48, 48, 1))

        model = load_model('model.h5')

        prediction = model.predict(image)

    label_map = ['Angry', 'Neutral', 'Scared', 'Happy', 'Sad', 'Surprised']

    prediction = np.argmax(prediction)
    print(f"{prediction} this is your prediction")
    final_prediction = label_map[prediction]
    if (prediction == 0):
        return render_template('landing_anger.html')
    elif (prediction == 1):
        return render_template('landing_neutral.html')
    elif (prediction == 2):
        return render_template('landing_scary.html')
    elif (prediction == 3):
        return render_template('landing_happy.html')
    elif (prediction == 4):
        return render_template('landing_sad.html')
    elif (prediction == 5):
        return render_template('landing_surprise.html')

    return render_template('after.html', data=final_prediction)
#should be working

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port, debug=True)
