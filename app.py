from flask import Flask, render_template, request
import cv2
import numpy as np
from keras.models import load_model
import os


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
        if 'file' not in request.files:
            return print('Nothing to see homie')
        file = request.files['file']
        file.filename = "file"
        path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(path)
        print('file uploaded')
    image = cv2.imread('uploadFile/file', 0)

    image = cv2.resize(image, (48, 48))

    image = np.reshape(image, (1, 48, 48, 1))

    model = load_model('model.h5')

    prediction = model.predict(image)

    label_map = ['Angry', 'Neutral', 'Scared', 'Happy', 'Sad', 'Surprised']

    prediction = np.argmax(prediction)
    print(f"{prediction} this is your prediction")
    os.remove('uploadFile/file')
    final_prediction = label_map[prediction]
    if (final_prediction == 'Angry'):
        return render_template('anger.html')
    elif (final_prediction == 'Neutral'):
        return render_template('neutral.html')
    elif (final_prediction == 'Scared'):
        return render_template('fear.html')
    elif (final_prediction == 'Happy'):
        return render_template('happiness.html')
    elif (final_prediction == 'Sad'):
        return render_template('sadness.html')
    elif (final_prediction == 'Surprised'):
        return print('there is nothing for you here')

    return render_template('after.html', data=final_prediction)
#should be working

if __name__ == "__main__":
    app.run(debug=True)
